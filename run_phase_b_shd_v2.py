"""
Corrected re-run per Opus audit round 4 (RESEARCH_NOTES.md §33f-bis). Fixes
BOTH bugs together, as explicitly recommended (fixing only one leaves the
other still confounding the result):

  (a) extract_embeddings now slices to CONTEXT LENGTH (first 12 of 20 bins),
      matching what context_encoder was actually trained on. The original
      run_phase_b_shd.py fed the FULL 20-bin sequence at eval time, which is
      length-robust for Dense (mean-pool) and Spiking-v2 (rate-code /
      seq_len) but not for GRU-dense (final hidden state only) -- this was
      the root cause of §32's now-reversed "GRU is worst" claim.

  (b) linear probe trained to convergence: embeddings standardized (zero
      mean, unit variance per dim) before probing, steps raised to 3000, and
      train_acc is printed so convergence can be checked rather than assumed
      (Spiking-v2's embeddings have ~3x the std of Dense's, which made the
      unstandardized 300-step probe converge at very different rates per
      encoder -- this is why §30/§31's gap magnitudes were retracted).

  (c) count_params restored (run_phase_a.py had this, run_phase_b_shd.py
      silently dropped it -- which is how the 2.9x GRU/Spiking parameter
      mismatch went unnoticed).

Loops 3 seeds (0, 1, 2) and reports mean +/- spread, since a single-seed
number was explicitly flagged as not good enough to cite as a magnitude.
"""
import statistics
import torch

from real_data_shd import load_shd, make_loader, FLAT_DIM
from dense_jepa import DenseJEPA
from gru_jepa import GRUJepa
from spiking_jepa_v2 import SpikingJEPA_V2
from collapse_diagnostics import diagnose
from downstream_probe import train_linear_probe, TrivialConstantEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_DIM = 128
REPR_DIM = 64
NUM_CLASSES = 20
BATCH_SIZE = 64
STEPS = 300
LR = 3e-3
SEEDS = [0, 1, 2]
PROBE_STEPS = 3000
CONTEXT_FRAC = 0.6


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def split_context_target(x, context_frac=CONTEXT_FRAC):
    T = x.shape[1]
    k = int(T * context_frac)
    return x[:, :k, :], x[:, k:, :]


def train_jepa(model, is_spiking, train_loader):
    opt = torch.optim.Adam(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()), lr=LR
    )
    data_iter = iter(train_loader)
    for step in range(STEPS):
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, _ = next(data_iter)
        x = x.to(DEVICE)
        context, target = split_context_target(x)
        if is_spiking:
            loss, pred_repr, target_repr, sr, layer_rates = model(context, target)
        else:
            loss, pred_repr, target_repr = model(context, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()


@torch.no_grad()
def extract_embeddings_context_length(model, is_spiking, loader, n_batches):
    """Fix (a): slice to context length before encoding, matching training."""
    embs, labels = [], []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x = x.to(DEVICE)
        context, _ = split_context_target(x)  # <-- the fix: context-length only
        if is_spiking:
            repr_out, _, _ = model.context_encoder(context)
        else:
            repr_out = model.context_encoder(context)
        embs.append(repr_out.cpu())
        labels.append(y)
    return torch.cat(embs, dim=0), torch.cat(labels, dim=0)


def probe_to_convergence(train_emb, train_lab, test_emb, test_lab, num_classes):
    """Fix (b): standardize + more steps, report train_acc so convergence is checkable."""
    mean = train_emb.mean(dim=0, keepdim=True)
    std = train_emb.std(dim=0, keepdim=True) + 1e-6
    train_std = (train_emb - mean) / std
    test_std = (test_emb - mean) / std
    result = train_linear_probe(train_std, train_lab, test_std, test_lab, num_classes=num_classes, device=DEVICE, steps=PROBE_STEPS)
    return result


def run_condition(name, is_spiking, seed, model_cls=None):
    torch.manual_seed(seed)
    train_set = load_shd(train=True)
    eval_set = load_shd(train=False)
    train_loader = make_loader(train_set, BATCH_SIZE, shuffle=True)
    eval_loader = make_loader(eval_set, BATCH_SIZE, shuffle=False)
    probe_train_loader = make_loader(train_set, BATCH_SIZE, shuffle=False)

    if is_spiking:
        model = SpikingJEPA_V2(FLAT_DIM, HIDDEN_DIM, REPR_DIM).to(DEVICE)
    else:
        model_cls = model_cls or DenseJEPA
        model = model_cls(FLAT_DIM, HIDDEN_DIM, REPR_DIM).to(DEVICE)
    n_params = count_params(model.context_encoder)

    train_jepa(model, is_spiking, train_loader)

    train_emb, train_lab = extract_embeddings_context_length(model, is_spiking, probe_train_loader, n_batches=32)
    test_emb, test_lab = extract_embeddings_context_length(model, is_spiking, eval_loader, n_batches=16)

    diag = diagnose(train_emb, label=f"{name} seed={seed} (context-length, standardized probe)")
    real_result = probe_to_convergence(train_emb, train_lab, test_emb, test_lab, NUM_CLASSES)

    trivial = TrivialConstantEncoder(REPR_DIM).to(DEVICE)
    with torch.no_grad():
        trivial_train = trivial(train_emb.to(DEVICE)).cpu()
        trivial_test = trivial(test_emb.to(DEVICE)).cpu()
    trivial_result = probe_to_convergence(trivial_train, train_lab, trivial_test, test_lab, NUM_CLASSES)

    return {
        "name": name, "seed": seed, "n_params": n_params,
        "collapse_suspected": diag["collapse_suspected"],
        "probe_train_acc": real_result["train_acc"], "test_acc": real_result["test_acc"],
        "trivial_acc": trivial_result["test_acc"],
    }


if __name__ == "__main__":
    conditions = [
        ("Dense", False, None),
        ("GRU-dense (control)", False, GRUJepa),
        ("Spiking-v2", True, None),
    ]
    all_results = {name: [] for name, _, _ in conditions}

    for name, is_spiking, model_cls in conditions:
        print(f"\n{'='*20} {name} {'='*20}")
        for seed in SEEDS:
            r = run_condition(name, is_spiking, seed, model_cls)
            all_results[name].append(r)
            print(f"  seed={seed} n_params={r['n_params']:,} collapse={r['collapse_suspected']} "
                  f"probe_train_acc={r['probe_train_acc']:.4f} test_acc={r['test_acc']:.4f} "
                  f"(convergence check: train_acc should be notably > test_acc if overfit, "
                  f"and should plateau not still be climbing)")

    print(f"\n{'='*20} SUMMARY (mean +/- stdev across {len(SEEDS)} seeds) {'='*20}")
    for name, _, _ in conditions:
        accs = [r["test_acc"] for r in all_results[name]]
        mean_acc = statistics.mean(accs)
        spread = statistics.stdev(accs) if len(accs) > 1 else 0.0
        n_params = all_results[name][0]["n_params"]
        n_collapsed = sum(1 for r in all_results[name] if r["collapse_suspected"])
        print(f"{name}: n_params={n_params:,} test_acc={mean_acc:.4f} +/- {spread:.4f} "
              f"(per-seed: {[f'{a:.4f}' for a in accs]}) collapsed {n_collapsed}/{len(SEEDS)}")

    dense_mean = statistics.mean([r["test_acc"] for r in all_results["Dense"]])
    gru_mean = statistics.mean([r["test_acc"] for r in all_results["GRU-dense (control)"]])
    spiking_mean = statistics.mean([r["test_acc"] for r in all_results["Spiking-v2"]])
    print(f"\nDense - GRU-dense gap: {(dense_mean-gru_mean)*100:+.2f}pp")
    print(f"GRU-dense - Spiking-v2 gap: {(gru_mean-spiking_mean)*100:+.2f}pp")
    print(f"Dense - Spiking-v2 gap: {(dense_mean-spiking_mean)*100:+.2f}pp")
