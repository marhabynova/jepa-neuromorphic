"""
Hyperparameter sweep for spiking_jepa_v2.py (snn.Synaptic) on SHD, §30.
Targets the exact axes flagged as never tested in the original §16-17 sweep
(RESEARCH_NOTES.md §24: "sumbu yang disweep (beta, LR) BUKAN sumbu yang
paling mungkin kausal -- threshold/readout-type/EMA-decay gak pernah
divariasikan"). This sweep fixes that gap: alpha, beta, AND threshold
(the dead-neuron-risk knob), not just beta/LR.

Logs every single run to CSV -- the original grid_sweep.py never saved
results to disk (print-only), a gap noted in this project's own memory.
"""
import copy
import csv
import itertools
import statistics
import time

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from real_data_shd import load_shd, make_loader, FLAT_DIM
from dense_jepa import DenseJEPA
from collapse_diagnostics import diagnose
from downstream_probe import train_linear_probe, TrivialConstantEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_DIM = 128
REPR_DIM = 64
NUM_CLASSES = 20
BATCH_SIZE = 64
STEPS = 200  # trimmed from 300 given sweep size (16 combos x 2 = 32 runs)
LR = 3e-3

ALPHAS = [0.7, 0.9]
BETAS = [0.7, 0.9]
THRESHOLDS = [0.5, 1.0]
SEEDS = [0, 1]

CSV_PATH = "grid_sweep_v2_results.csv"


class SpikingEncoderSweep(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, alpha, beta, threshold):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.lif1 = snn.Synaptic(alpha=alpha, beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = snn.Synaptic(alpha=alpha, beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.lif3 = snn.Synaptic(alpha=alpha, beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.out_features = out_dim

    def forward(self, x):
        batch, seq_len, _ = x.shape
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()
        syn3, mem3 = self.lif3.init_synaptic()
        spk3_accum = torch.zeros(batch, self.out_features, device=x.device)
        n_spikes = x.new_zeros(())
        n_possible = x.new_zeros(())
        for t in range(seq_len):
            cur1 = self.fc1(x[:, t, :])
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            cur2 = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            cur3 = self.fc3(spk2)
            spk3, syn3, mem3 = self.lif3(cur3, syn3, mem3)
            spk3_accum = spk3_accum + spk3
            n_spikes = n_spikes + spk1.sum() + spk2.sum() + spk3.sum()
            n_possible = n_possible + spk1.numel() + spk2.numel() + spk3.numel()
        return spk3_accum / seq_len, n_spikes / n_possible


class Predictor(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x):
        return self.net(x)


def split_context_target(x, context_frac=0.6):
    T = x.shape[1]
    k = int(T * context_frac)
    return x[:, :k, :], x[:, k:, :]


def run_one(alpha, beta, threshold, seed, train_loader, eval_loader, probe_train_loader):
    torch.manual_seed(seed)
    context_encoder = SpikingEncoderSweep(FLAT_DIM, HIDDEN_DIM, REPR_DIM, alpha, beta, threshold).to(DEVICE)
    target_encoder = copy.deepcopy(context_encoder).to(DEVICE)
    for p in target_encoder.parameters():
        p.requires_grad = False
    predictor = Predictor(REPR_DIM, HIDDEN_DIM).to(DEVICE)
    opt = torch.optim.Adam(list(context_encoder.parameters()) + list(predictor.parameters()), lr=LR)

    data_iter = iter(train_loader)
    spike_rates = []
    for step in range(STEPS):
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, _ = next(data_iter)
        x = x.to(DEVICE)
        context, target = split_context_target(x)
        ctx_repr, sr = context_encoder(context)
        spike_rates.append(sr.item())
        pred_repr = predictor(ctx_repr)
        with torch.no_grad():
            target_repr, _ = target_encoder(target)
        loss = torch.mean((pred_repr - target_repr) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p_t, p_c in zip(target_encoder.parameters(), context_encoder.parameters()):
                p_t.data.mul_(0.99).add_(p_c.data, alpha=0.01)

    avg_sr = sum(spike_rates) / len(spike_rates)

    @torch.no_grad()
    def extract(loader, n_batches):
        embs, labels = [], []
        it = iter(loader)
        for _ in range(n_batches):
            try:
                xb, yb = next(it)
            except StopIteration:
                break
            xb = xb.to(DEVICE)
            r, _ = context_encoder(xb)
            embs.append(r.cpu())
            labels.append(yb)
        return torch.cat(embs, dim=0), torch.cat(labels, dim=0)

    train_emb, train_lab = extract(probe_train_loader, 32)
    test_emb, test_lab = extract(eval_loader, 16)
    diag = diagnose(train_emb, label=f"a={alpha} b={beta} thr={threshold} seed={seed}")
    real_result = train_linear_probe(train_emb, train_lab, test_emb, test_lab, num_classes=NUM_CLASSES, device=DEVICE)
    trivial = TrivialConstantEncoder(REPR_DIM).to(DEVICE)
    with torch.no_grad():
        trivial_train = trivial(train_emb.to(DEVICE)).cpu()
        trivial_test = trivial(test_emb.to(DEVICE)).cpu()
    trivial_result = train_linear_probe(trivial_train, train_lab, trivial_test, test_lab, num_classes=NUM_CLASSES, device=DEVICE)

    return {
        "alpha": alpha, "beta": beta, "threshold": threshold, "seed": seed,
        "spike_rate": avg_sr, "collapse_suspected": diag["collapse_suspected"],
        "erank_S": diag["rank"]["effective_rank_S"],
        "test_acc": real_result["test_acc"], "trivial_acc": trivial_result["test_acc"],
    }


if __name__ == "__main__":
    train_set = load_shd(train=True)
    eval_set = load_shd(train=False)
    train_loader = make_loader(train_set, BATCH_SIZE, shuffle=True)
    eval_loader = make_loader(eval_set, BATCH_SIZE, shuffle=False)
    probe_train_loader = make_loader(train_set, BATCH_SIZE, shuffle=False)

    combos = list(itertools.product(ALPHAS, BETAS, THRESHOLDS, SEEDS))
    print(f"Total runs: {len(combos)}")

    results = []
    t0 = time.time()
    for i, (alpha, beta, threshold, seed) in enumerate(combos):
        print(f"\n--- Run {i+1}/{len(combos)}: alpha={alpha} beta={beta} threshold={threshold} seed={seed} ---")
        r = run_one(alpha, beta, threshold, seed, train_loader, eval_loader, probe_train_loader)
        print(f"  spike_rate={r['spike_rate']:.4f} collapse={r['collapse_suspected']} "
              f"test_acc={r['test_acc']:.4f} trivial={r['trivial_acc']:.4f} erank_S={r['erank_S']:.2f}")
        results.append(r)

    elapsed = time.time() - t0
    print(f"\nTotal sweep time: {elapsed:.1f}s")

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved all {len(results)} results to {CSV_PATH}")

    n_not_collapsed = sum(1 for r in results if not r["collapse_suspected"])
    print(f"\n{n_not_collapsed}/{len(results)} combos did NOT show collapse")
    not_collapsed = [r for r in results if not r["collapse_suspected"]]
    if not_collapsed:
        best = max(not_collapsed, key=lambda r: r["test_acc"])
        print(f"Best non-collapsed combo: alpha={best['alpha']} beta={best['beta']} "
              f"threshold={best['threshold']} seed={best['seed']} -> test_acc={best['test_acc']:.4f}")
    best_overall = max(results, key=lambda r: r["test_acc"])
    print(f"Best test_acc overall (any combo): {best_overall}")
