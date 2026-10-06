# SUPERSEDED (RESEARCH_NOTES.md §73, Opus audit): evaluate_condition() here
# measured against a probe-training set that was re-shuffled on every call,
# injecting more noise (sd ~1.4-3.2pp) than the effects being compared -
# this script's numbers are NOT reliable (see §73 for the full finding).
# Also now broken as-is: evaluate_condition's signature changed to take a
# fixed batch list, not a loader. Use sleep_consolidation_corrected.py,
# which fixes the noise bug AND adds the missing control conditions
# (DECAY_ONLY, NODECAY, RANDOMWAKE) this script never had.

"""
Multi-seed confirmation of sleep_consolidation_oja.py's single-seed
directional result: error-modulated Oja's rule at eta=1e-4 (highest overall,
balanced retention+acquisition) beat Static 50.7% vs 48.0% on seed=0. Same
bar as every other positive single-seed result in this project before it's
allowed to be called real (§70d->f, §71b->c already burned this lesson in
hard - don't skip it here just because the mechanism is new and exciting).
"""
import copy
import statistics
import torch

from real_data_shd import load_shd, make_loader, FLAT_DIM
from dense_jepa import DenseJEPA
from awake_gate import GateNet, split_context_target, collect_surprise_dataset, train_gate
from sleep_consolidation import (
    filter_by_classes, train_jepa_steps, evaluate_condition,
    KNOWN_CLASSES, NEW_CLASSES, JEPA_STEPS, BATCH_SIZE, DAY_STREAM_BATCHES,
)
from sleep_consolidation_oja import oja_modulated_update

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2]
ETA = 1e-4


def run_seed(seed):
    torch.manual_seed(seed)
    train_set = load_shd(train=True)
    eval_set = load_shd(train=False)
    known_train = filter_by_classes(train_set, KNOWN_CLASSES)
    new_train = filter_by_classes(train_set, NEW_CLASSES)
    known_eval = filter_by_classes(eval_set, KNOWN_CLASSES)
    new_eval = filter_by_classes(eval_set, NEW_CLASSES)

    known_train_loader = make_loader(known_train, BATCH_SIZE, shuffle=True)
    new_train_loader = make_loader(new_train, BATCH_SIZE, shuffle=True)
    known_eval_loader = make_loader(known_eval, BATCH_SIZE, shuffle=False)
    new_eval_loader = make_loader(new_eval, BATCH_SIZE, shuffle=False)
    mixed_probe_loader = make_loader(torch.utils.data.ConcatDataset([known_train, new_train]),
                                      BATCH_SIZE, shuffle=True)

    base_model = DenseJEPA(FLAT_DIM, 128, 64).to(DEVICE)
    train_jepa_steps(base_model, known_train_loader, JEPA_STEPS)

    gate_train_contexts, gate_train_errors = collect_surprise_dataset(base_model, known_train_loader, n_batches=64)
    gate = GateNet(FLAT_DIM).to(DEVICE)
    train_gate(gate, gate_train_contexts, gate_train_errors)
    gate_thresh = torch.quantile(gate(gate_train_contexts.to(DEVICE)).detach().cpu(), 0.80).item()

    day_batches = []
    it = iter(new_train_loader)
    for _ in range(DAY_STREAM_BATCHES):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(new_train_loader)
            x, y = next(it)
        day_batches.append((x, y))

    static_model = copy.deepcopy(base_model)
    static_r = evaluate_condition("STATIC", static_model, mixed_probe_loader, known_eval_loader, new_eval_loader)

    online_model = copy.deepcopy(base_model)
    opt = torch.optim.Adam(
        list(online_model.context_encoder.parameters()) + list(online_model.predictor.parameters()), lr=3e-3
    )
    for x, _ in day_batches:
        x = x.to(DEVICE)
        context, target = split_context_target(x)
        loss, _, _ = online_model(context, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        online_model.update_target()
    online_r = evaluate_condition("ONLINE_NAIVE", online_model, mixed_probe_loader, known_eval_loader, new_eval_loader)

    woken_pairs = []
    for x, _ in day_batches:
        x = x.to(DEVICE)
        context, target = split_context_target(x)
        with torch.no_grad():
            scores = gate(context)
        mask = scores >= gate_thresh
        if mask.any():
            woken_pairs.append((context[mask], target[mask]))
    oja_model = copy.deepcopy(base_model)
    if woken_pairs:
        woken_context = torch.cat([c for c, _ in woken_pairs], dim=0)
        woken_target = torch.cat([t for _, t in woken_pairs], dim=0)
        with torch.no_grad():
            ctx_repr = base_model.context_encoder(woken_context)
            pred_repr = base_model.predictor(ctx_repr)
            tgt_repr = base_model.target_encoder(woken_target)
            sample_errors = ((pred_repr - tgt_repr) ** 2).mean(dim=1).cpu()
        oja_modulated_update(oja_model.context_encoder, woken_context, sample_errors, eta=ETA)
    oja_r = evaluate_condition(f"SLEEP_OJA(eta={ETA:.0e})", oja_model, mixed_probe_loader, known_eval_loader, new_eval_loader)

    return static_r, online_r, oja_r


if __name__ == "__main__":
    all_static, all_online, all_oja = [], [], []
    for seed in SEEDS:
        print(f"\n{'='*20} seed={seed} {'='*20}")
        s, o, oj = run_seed(seed)
        all_static.append(s)
        all_online.append(o)
        all_oja.append(oj)

    def summarize(results, key):
        vals = [r[key] * 100 for r in results]
        mean = statistics.mean(vals)
        spread = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mean, spread, vals

    print(f"\n{'='*70}")
    print(f"SUMMARY across {len(SEEDS)} seeds (mean +/- stdev)")
    print(f"{'='*70}")
    for label, results in [("STATIC", all_static), ("ONLINE_NAIVE", all_online), (f"SLEEP_OJA(eta={ETA:.0e})", all_oja)]:
        ret_mean, ret_sd, _ = summarize(results, "known_acc")
        acq_mean, acq_sd, _ = summarize(results, "new_acc")
        ov_mean, ov_sd, ov_vals = summarize(results, "overall_test_acc")
        print(f"{label:<22} retention={ret_mean:5.1f}%+/-{ret_sd:4.1f}  "
              f"acquisition={acq_mean:5.1f}%+/-{acq_sd:4.1f}  overall={ov_mean:5.1f}%+/-{ov_sd:4.1f}  "
              f"(per-seed: {[f'{v:.1f}' for v in ov_vals]})")

    oja_overall = [r["overall_test_acc"] for r in all_oja]
    static_overall = [r["overall_test_acc"] for r in all_static]
    deltas = [(oj - st) * 100 for oj, st in zip(oja_overall, static_overall)]
    print(f"\nSleep(Oja) - Static delta per seed: {[f'{d:+.1f}pp' for d in deltas]}")
    print(f"Mean delta: {statistics.mean(deltas):+.1f}pp, worst seed: {min(deltas):+.1f}pp")
    print("(worst-seed delta still clearly positive = robust; if any seed flips negative, same")
    print(" lucky-seed pattern as §71b->§71c, not a real effect)")
