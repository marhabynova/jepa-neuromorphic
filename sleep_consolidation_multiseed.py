# SUPERSEDED (RESEARCH_NOTES.md §73, Opus audit): evaluate_condition() here
# measured against a probe-training set that was re-shuffled on every call,
# injecting more noise (sd ~1.4-3.2pp) than the effects being compared -
# this script's numbers are NOT reliable (see §73 for the full finding).
# Also now broken as-is: evaluate_condition's signature changed to take a
# fixed batch list, not a loader. Use sleep_consolidation_corrected.py,
# which fixes the noise bug AND adds the missing control conditions
# (DECAY_ONLY, NODECAY, RANDOMWAKE) this script never had.

"""
Multi-seed (3-seed) confirmation of §71b's single-seed positive finding:
at a properly-calibrated eta=1e-3 (real Hebbian signal, not destructive
like the original 5e-3, not decay-floor-dominated like <=1e-4), Sleep
Consolidation beat Static by +2.1pp overall on seed=0. Same bar as Awake
Gate's 1-seed->3-seed progression (§70d->§70f) - don't cite a single-seed
sweep pick as a real finding.
"""
import copy
import statistics
import torch

from real_data_shd import load_shd, make_loader, FLAT_DIM
from dense_jepa import DenseJEPA
from awake_gate import GateNet, split_context_target, collect_surprise_dataset, train_gate
from sleep_consolidation import (
    filter_by_classes, train_jepa_steps, hebbian_update, evaluate_condition,
    KNOWN_CLASSES, NEW_CLASSES, JEPA_STEPS, BATCH_SIZE, DAY_STREAM_BATCHES,
    HEBBIAN_DECAY,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2]
CALIBRATED_ETA = 1e-3  # §71b: the one sweep point with real, non-destructive Hebbian signal


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

    # STATIC
    static_model = copy.deepcopy(base_model)
    static_r = evaluate_condition("STATIC", static_model, mixed_probe_loader, known_eval_loader, new_eval_loader)

    # ONLINE_NAIVE
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

    # SLEEP @ calibrated eta
    woken_contexts = []
    for x, _ in day_batches:
        x = x.to(DEVICE)
        context, _ = split_context_target(x)
        with torch.no_grad():
            scores = gate(context)
        mask = scores >= gate_thresh
        if mask.any():
            woken_contexts.append(context[mask])
    sleep_model = copy.deepcopy(base_model)
    if woken_contexts:
        night_batch = torch.cat(woken_contexts, dim=0)
        hebbian_update(sleep_model.context_encoder, night_batch, eta=CALIBRATED_ETA, decay=HEBBIAN_DECAY)
    sleep_r = evaluate_condition(f"SLEEP(eta={CALIBRATED_ETA:.0e})", sleep_model, mixed_probe_loader, known_eval_loader, new_eval_loader)

    return static_r, online_r, sleep_r


if __name__ == "__main__":
    all_static, all_online, all_sleep = [], [], []
    for seed in SEEDS:
        print(f"\n{'='*20} seed={seed} {'='*20}")
        s, o, sl = run_seed(seed)
        all_static.append(s)
        all_online.append(o)
        all_sleep.append(sl)

    def summarize(results, key):
        vals = [r[key] * 100 for r in results]
        mean = statistics.mean(vals)
        spread = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mean, spread, vals

    print(f"\n{'='*70}")
    print(f"SUMMARY across {len(SEEDS)} seeds (mean +/- stdev)")
    print(f"{'='*70}")
    for label, results in [("STATIC", all_static), ("ONLINE_NAIVE", all_online), (f"SLEEP(eta={CALIBRATED_ETA:.0e})", all_sleep)]:
        ret_mean, ret_sd, ret_vals = summarize(results, "known_acc")
        acq_mean, acq_sd, acq_vals = summarize(results, "new_acc")
        ov_mean, ov_sd, ov_vals = summarize(results, "overall_test_acc")
        print(f"{label:<20} retention={ret_mean:5.1f}%+/-{ret_sd:4.1f}  "
              f"acquisition={acq_mean:5.1f}%+/-{acq_sd:4.1f}  overall={ov_mean:5.1f}%+/-{ov_sd:4.1f}  "
              f"(per-seed overall: {[f'{v:.1f}' for v in ov_vals]})")

    sleep_overall = [r["overall_test_acc"] for r in all_sleep]
    static_overall = [r["overall_test_acc"] for r in all_static]
    deltas = [(sl - st) * 100 for sl, st in zip(sleep_overall, static_overall)]
    print(f"\nSleep - Static delta per seed: {[f'{d:+.1f}pp' for d in deltas]}")
    print(f"Mean delta: {statistics.mean(deltas):+.1f}pp, worst seed: {min(deltas):+.1f}pp")
    print("(if worst-seed delta is still clearly positive, the eta=1e-3 effect is robust, not a lucky seed)")
