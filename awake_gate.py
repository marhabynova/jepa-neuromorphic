"""
Awake Gate (RESEARCH_NOTES.md §43, §69 update) - first execution of the plan
that was judged "most elegant" back in §43c but blocked until two
prerequisites cleared: SpikingEncoderV2 validated on hxtorch (§36i, closed)
and the FPGA-vs-TinyTapeout decision (§42c, decided). Both are now done.

Idea: distill the ALREADY-TRAINED, ALREADY-WINNING DenseJEPA's own
prediction-error signal into a small, cheap SNN "gate". The gate watches raw
input and tries to predict "will the expensive dense encoder's
prediction-error be high on this sample" WITHOUT running the dense encoder.
If the gate is right, the dense encoder only needs to wake up for genuinely
"surprising" input - the rest of the time it can stay idle.

This is a FIRST PASS, single seed, explicitly preliminary (matches the
project's own established pattern - §30/31 were single-seed-preliminary
until §34/35 made them 3-seed-robust; don't cite this run's numbers as final
without a multi-seed re-run first if this result is promising).

Honest framing per §69's AGI calibration: this is an efficiency/
compute-allocation mechanism, not a complexity-class-crossing one. Value is
in itself (measurable duty-cycle/recall numbers against defined baselines),
not in any AGI claim.
"""
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from real_data_shd import load_shd, make_loader, FLAT_DIM
from dense_jepa import DenseJEPA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
HIDDEN_DIM = 128
REPR_DIM = 64
BATCH_SIZE = 64
JEPA_STEPS = 300
JEPA_LR = 3e-3
CONTEXT_FRAC = 0.6

GATE_HIDDEN = 16          # deliberately tiny vs DenseJEPA's 128 - "gate must be cheap"
GATE_STEPS = 500
GATE_LR = 1e-3
SURPRISE_PERCENTILE = 80  # top 20% of prediction-error = "surprising"


def split_context_target(x, context_frac=CONTEXT_FRAC):
    T = x.shape[1]
    k = int(T * context_frac)
    return x[:, :k, :], x[:, k:, :]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GateNet(nn.Module):
    """Minimal spiking gate: 1 spiking hidden layer + linear scalar readout.
    First layer is event-driven/cheap (spiking); readout is a plain linear
    layer so the output is a continuous regression target, not a spike."""

    def __init__(self, in_dim, hidden_dim=GATE_HIDDEN, beta=0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, in_dim)
        batch, seq_len, _ = x.shape
        mem1 = self.lif1.init_leaky()
        score_accum = x.new_zeros(batch, 1)
        for t in range(seq_len):
            cur1 = self.fc1(x[:, t, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            score_accum = score_accum + self.readout(spk1)
        return (score_accum / seq_len).squeeze(-1)


def train_dense_jepa(train_loader):
    torch.manual_seed(SEED)
    model = DenseJEPA(FLAT_DIM, HIDDEN_DIM, REPR_DIM).to(DEVICE)
    opt = torch.optim.Adam(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()), lr=JEPA_LR
    )
    data_iter = iter(train_loader)
    for step in range(JEPA_STEPS):
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, _ = next(data_iter)
        x = x.to(DEVICE)
        context, target = split_context_target(x)
        loss, _, _ = model(context, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()
        if (step + 1) % 100 == 0:
            print(f"  [DenseJEPA] step {step+1}/{JEPA_STEPS} loss={loss.item():.4f}")
    return model


@torch.no_grad()
def collect_surprise_dataset(dense_model, loader, n_batches):
    """For each sample: raw context window (gate input) + per-sample
    prediction-error (gate's regression target), using the FROZEN,
    already-trained DenseJEPA as the labeling oracle."""
    dense_model.eval()
    contexts, errors = [], []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        x = x.to(DEVICE)
        context, target = split_context_target(x)
        ctx_repr = dense_model.context_encoder(context)
        pred_repr = dense_model.predictor(ctx_repr)
        target_repr = dense_model.target_encoder(target)
        per_sample_err = ((pred_repr - target_repr) ** 2).mean(dim=1)  # (batch,)
        contexts.append(context.cpu())
        errors.append(per_sample_err.cpu())
    dense_model.train()
    return torch.cat(contexts, dim=0), torch.cat(errors, dim=0)


def train_gate(gate, train_contexts, train_errors):
    # Normalize regression targets for stable training; keep the scaler to
    # invert later only if needed (we only need rank/threshold, not raw scale).
    err_mean, err_std = train_errors.mean(), train_errors.std() + 1e-8
    targets = (train_errors - err_mean) / err_std

    opt = torch.optim.Adam(gate.parameters(), lr=GATE_LR)
    n = train_contexts.shape[0]
    for step in range(GATE_STEPS):
        idx = torch.randint(0, n, (BATCH_SIZE,))
        x = train_contexts[idx].to(DEVICE)
        y = targets[idx].to(DEVICE)
        pred = gate(x)
        loss = torch.mean((pred - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            print(f"  [Gate] step {step+1}/{GATE_STEPS} mse={loss.item():.4f}")
    return err_mean, err_std


def pearson_corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (torch.sqrt((a ** 2).sum()) * torch.sqrt((b ** 2).sum()) + 1e-12)


@torch.no_grad()
def evaluate(gate, test_contexts, test_errors):
    gate.eval()
    preds = []
    for i in range(0, test_contexts.shape[0], BATCH_SIZE):
        x = test_contexts[i:i + BATCH_SIZE].to(DEVICE)
        preds.append(gate(x).cpu())
    gate_scores = torch.cat(preds, dim=0)

    corr = pearson_corr(gate_scores, test_errors).item()

    # Mandatory trivial baselines (project convention - a learned mechanism
    # must beat a naive statistic of the raw input, not just random chance).
    trivial_spike_count = test_contexts.sum(dim=(1, 2))
    trivial_l2norm = test_contexts.pow(2).sum(dim=(1, 2)).sqrt()
    corr_trivial_count = pearson_corr(trivial_spike_count, test_errors).item()
    corr_trivial_norm = pearson_corr(trivial_l2norm, test_errors).item()

    true_thresh = np.percentile(test_errors.numpy(), SURPRISE_PERCENTILE)
    is_surprising = test_errors >= true_thresh
    n_surprising = is_surprising.sum().item()

    # Threshold the GATE's own score at the matching percentile (we don't
    # get to peek at true labels at decision time - only the gate's own
    # score distribution is available to pick an operating point).
    gate_thresh = np.percentile(gate_scores.numpy(), SURPRISE_PERCENTILE)
    gate_wakes = gate_scores >= gate_thresh

    duty_cycle = gate_wakes.float().mean().item()
    recall = (gate_wakes & is_surprising).sum().item() / max(n_surprising, 1)

    # Random-gate baseline at the SAME duty cycle, computed analytically:
    # expected recall of a random selector = duty_cycle (no information).
    random_expected_recall = duty_cycle

    return {
        "n_test": test_contexts.shape[0],
        "correlation": corr,
        "trivial_spike_count_correlation": corr_trivial_count,
        "trivial_l2norm_correlation": corr_trivial_norm,
        "n_surprising": n_surprising,
        "duty_cycle": duty_cycle,
        "gate_recall": recall,
        "random_expected_recall": random_expected_recall,
    }


if __name__ == "__main__":
    print("=== Loading SHD ===")
    train_set = load_shd(train=True)
    eval_set = load_shd(train=False)
    train_loader = make_loader(train_set, BATCH_SIZE, shuffle=True)
    eval_loader = make_loader(eval_set, BATCH_SIZE, shuffle=False)
    gate_train_loader = make_loader(train_set, BATCH_SIZE, shuffle=True)

    print("\n=== Step 1: train DenseJEPA (the oracle the gate will distill) ===")
    dense_model = train_dense_jepa(train_loader)
    n_dense_params = count_params(dense_model.context_encoder)

    print("\n=== Step 2: collect surprise-labeling dataset from the frozen DenseJEPA ===")
    train_contexts, train_errors = collect_surprise_dataset(dense_model, gate_train_loader, n_batches=64)
    test_contexts, test_errors = collect_surprise_dataset(dense_model, eval_loader, n_batches=16)
    print(f"  train: {train_contexts.shape[0]} samples, test: {test_contexts.shape[0]} samples")
    print(f"  train prediction-error: mean={train_errors.mean():.4f} std={train_errors.std():.4f}")

    print("\n=== Step 3: train the gate (distillation) ===")
    torch.manual_seed(SEED)
    gate = GateNet(FLAT_DIM).to(DEVICE)
    n_gate_params = count_params(gate)
    train_gate(gate, train_contexts, train_errors)

    print("\n=== Step 4: evaluate against baselines ===")
    result = evaluate(gate, test_contexts, test_errors)

    print(f"\n{'='*60}")
    print(f"DenseJEPA context_encoder params: {n_dense_params:,}")
    print(f"Gate params:                      {n_gate_params:,}  ({n_dense_params / n_gate_params:.1f}x smaller)")
    print(f"{'='*60}")
    print(f"Test set size: {result['n_test']}")
    print(f"Gate score <-> true prediction-error correlation: {result['correlation']:.4f}")
    print(f"  trivial baseline (raw input total spike count) correlation: {result['trivial_spike_count_correlation']:.4f}")
    print(f"  trivial baseline (raw input L2 norm) correlation:           {result['trivial_l2norm_correlation']:.4f}")
    print(f"Surprise definition: top {100 - SURPRISE_PERCENTILE}% prediction-error "
          f"({result['n_surprising']} of {result['n_test']} samples)")
    print(f"Gate duty cycle (fraction of time dense encoder is woken): {result['duty_cycle']*100:.1f}%")
    print(f"  -> dense encoder can SLEEP {(1-result['duty_cycle'])*100:.1f}% of the time")
    print(f"Gate recall of truly-surprising samples:   {result['gate_recall']*100:.1f}%")
    print(f"Random-gate expected recall (same duty cycle, 0 info): {result['random_expected_recall']*100:.1f}%")
    print(f"Always-on baseline: duty_cycle=100%, recall=100% (reference point, not a competitor)")
    lift = result['gate_recall'] - result['random_expected_recall']
    print(f"\nGate lift over random at matched duty cycle: {lift*100:+.1f}pp")
    print("NOTE: single seed, first pass - preliminary, needs multi-seed re-run before citing as robust.")
