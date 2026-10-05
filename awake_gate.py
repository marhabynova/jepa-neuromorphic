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
SEEDS = [0, 1, 2]  # §70d called 1-seed "preliminary" - matches §34/§35's 3-seed bar
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


def run_seed(seed):
    torch.manual_seed(seed)
    train_set = load_shd(train=True)
    eval_set = load_shd(train=False)
    train_loader = make_loader(train_set, BATCH_SIZE, shuffle=True)
    eval_loader = make_loader(eval_set, BATCH_SIZE, shuffle=False)
    gate_train_loader = make_loader(train_set, BATCH_SIZE, shuffle=True)

    dense_model = train_dense_jepa(train_loader)
    n_dense_params = count_params(dense_model.context_encoder)

    train_contexts, train_errors = collect_surprise_dataset(dense_model, gate_train_loader, n_batches=64)
    test_contexts, test_errors = collect_surprise_dataset(dense_model, eval_loader, n_batches=16)

    gate = GateNet(FLAT_DIM).to(DEVICE)
    n_gate_params = count_params(gate)
    train_gate(gate, train_contexts, train_errors)

    result = evaluate(gate, test_contexts, test_errors)
    result["n_dense_params"] = n_dense_params
    result["n_gate_params"] = n_gate_params
    result["lift_pp"] = (result["gate_recall"] - result["random_expected_recall"]) * 100
    return result


if __name__ == "__main__":
    import statistics

    all_results = []
    for seed in SEEDS:
        print(f"\n{'='*20} seed={seed} {'='*20}")
        r = run_seed(seed)
        all_results.append(r)
        print(f"  correlation={r['correlation']:.4f} (trivial L2norm={r['trivial_l2norm_correlation']:.4f}) "
              f"duty_cycle={r['duty_cycle']*100:.1f}% recall={r['gate_recall']*100:.1f}% "
              f"(random={r['random_expected_recall']*100:.1f}%) lift={r['lift_pp']:+.1f}pp")

    def summarize(key, scale=1.0):
        vals = [r[key] * scale for r in all_results]
        mean = statistics.mean(vals)
        spread = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mean, spread, vals

    corr_mean, corr_spread, corr_vals = summarize("correlation")
    trivial_mean, trivial_spread, _ = summarize("trivial_l2norm_correlation")
    recall_mean, recall_spread, recall_vals = summarize("gate_recall", 100)
    lift_mean, lift_spread, lift_vals = summarize("lift_pp")

    print(f"\n{'='*60}")
    print(f"SUMMARY across {len(SEEDS)} seeds (mean +/- stdev)")
    print(f"{'='*60}")
    print(f"DenseJEPA context_encoder params: {all_results[0]['n_dense_params']:,}")
    print(f"Gate params:                      {all_results[0]['n_gate_params']:,} "
          f"({all_results[0]['n_dense_params'] / all_results[0]['n_gate_params']:.1f}x smaller)")
    print(f"Gate<->true-error correlation: {corr_mean:.4f} +/- {corr_spread:.4f} (per-seed: {[f'{v:.4f}' for v in corr_vals]})")
    print(f"Trivial L2-norm baseline correlation: {trivial_mean:.4f} +/- {trivial_spread:.4f}")
    print(f"Gate recall @ matched duty cycle: {recall_mean:.1f}% +/- {recall_spread:.1f}% (per-seed: {[f'{v:.1f}' for v in recall_vals]})")
    print(f"Lift over random: {lift_mean:+.1f}pp +/- {lift_spread:.1f}pp (per-seed: {[f'{v:+.1f}' for v in lift_vals]})")
    worst_lift = min(lift_vals)
    print(f"Worst-seed lift: {worst_lift:+.1f}pp (if this is still clearly >0, the effect is robust, not a lucky seed)")
