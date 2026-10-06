"""
Corrected, definitive re-run per the §73 Opus audit's prioritized fix list.
Two things were wrong in §71-§72, not one:

1. MEASUREMENT BUG: evaluate_condition() used to redraw the probe-training
   set fresh (shuffle=True) on every call, injecting more noise (sd
   ~1.4-3.2pp per call) than any effect being compared. Fixed in
   sleep_consolidation.py (draw_fixed_batches + extract_embeddings_from_
   batches) - every condition below sees the SAME probe-training data
   within a seed.

2. MISSING CONTROLS: §71-§72 only ever compared SLEEP(some rule) against
   STATIC and ONLINE_NAIVE. That can't separate "the Hebbian term helps",
   "the decay term hurts", or "the gate's sample SELECTION matters at all"
   - three different, confusable claims. This run adds:
     - DECAY_ONLY (eta=0): isolates the synaptic-downscaling term alone.
     - NODECAY (decay=0): isolates the Hebbian/Oja term alone.
     - RANDOMWAKE: same update rule, but on a RANDOM subset of the day
       stream instead of the gate's selection - tests whether the gate's
       selection does anything at all, not just whether consolidation
       helps.
   Also restores eta=1e-4, which §71b excluded via a tautological
   "decay-floor" argument (||decay*W||/||W|| == decay by construction, for
   ANY eta) - it's actually the best-balanced point between the two terms.

Everything here is single-shot (one night pass), same limitation the audit
flagged (Issue 3.3: a single rank-1 update may be structurally close to
invisible to a refit-per-condition linear probe) - this run does NOT fix
that, it only fixes the measurement noise and adds the missing controls.
That's an explicit, separate limitation for any future follow-up.
"""
import copy
import statistics
import torch

from real_data_shd import load_shd, make_loader, FLAT_DIM
from dense_jepa import DenseJEPA
from awake_gate import GateNet, split_context_target, collect_surprise_dataset, train_gate
from sleep_consolidation import (
    filter_by_classes, train_jepa_steps, hebbian_update, evaluate_condition,
    draw_fixed_batches, KNOWN_CLASSES, NEW_CLASSES, JEPA_STEPS, BATCH_SIZE,
    DAY_STREAM_BATCHES, HEBBIAN_DECAY, PROBE_TRAIN_BATCHES,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2]
ETA_CANDIDATES = [1e-4, 1e-3]  # 1e-4 restored per audit; 1e-3 kept for comparison


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
    probe_train_batches = draw_fixed_batches(mixed_probe_loader, PROBE_TRAIN_BATCHES)

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

    def eval_m(name, model):
        return evaluate_condition(name, model, probe_train_batches, known_eval_loader, new_eval_loader)

    results = {}

    # STATIC
    results["STATIC"] = eval_m("STATIC", copy.deepcopy(base_model))

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
    results["ONLINE_NAIVE"] = eval_m("ONLINE_NAIVE", online_model)

    # Gate-selected wake set
    woken_contexts = []
    for x, _ in day_batches:
        x = x.to(DEVICE)
        context, _ = split_context_target(x)
        with torch.no_grad():
            scores = gate(context)
        mask = scores >= gate_thresh
        if mask.any():
            woken_contexts.append(context[mask])
    gate_night_batch = torch.cat(woken_contexts, dim=0)
    n_woken = gate_night_batch.shape[0]

    # Random-selected wake set, SAME SIZE as the gate's selection
    all_context = torch.cat([split_context_target(x.to(DEVICE))[0] for x, _ in day_batches], dim=0)
    perm = torch.randperm(all_context.shape[0])[:n_woken]
    random_night_batch = all_context[perm]

    # SLEEP at each eta candidate (gate-selected, standard rule)
    for eta in ETA_CANDIDATES:
        m = copy.deepcopy(base_model)
        hebbian_update(m.context_encoder, gate_night_batch, eta=eta, decay=HEBBIAN_DECAY)
        results[f"SLEEP(eta={eta:.0e})"] = eval_m(f"SLEEP(eta={eta:.0e})", m)

    # DECAY_ONLY: eta=0, isolates the synaptic-downscaling term
    m = copy.deepcopy(base_model)
    hebbian_update(m.context_encoder, gate_night_batch, eta=0.0, decay=HEBBIAN_DECAY)
    results["DECAY_ONLY"] = eval_m("DECAY_ONLY", m)

    # NODECAY: decay=0, isolates the Hebbian term (at eta=1e-3)
    m = copy.deepcopy(base_model)
    hebbian_update(m.context_encoder, gate_night_batch, eta=1e-3, decay=0.0)
    results["NODECAY(eta=1e-3)"] = eval_m("NODECAY(eta=1e-3)", m)

    # RANDOMWAKE: same rule (eta=1e-3, standard decay), random selection instead of gate
    m = copy.deepcopy(base_model)
    hebbian_update(m.context_encoder, random_night_batch, eta=1e-3, decay=HEBBIAN_DECAY)
    results["RANDOMWAKE(eta=1e-3)"] = eval_m("RANDOMWAKE(eta=1e-3)", m)

    results["_n_woken"] = n_woken
    return results


if __name__ == "__main__":
    all_results = {}
    for seed in SEEDS:
        print(f"\n{'='*20} seed={seed} {'='*20}")
        r = run_seed(seed)
        print(f"  n_woken={r['_n_woken']}")
        for k, v in r.items():
            if k == "_n_woken":
                continue
            all_results.setdefault(k, []).append(v)

    def summarize(results, key):
        vals = [r[key] * 100 for r in results]
        mean = statistics.mean(vals)
        spread = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mean, spread, vals

    print(f"\n{'='*80}")
    print(f"SUMMARY across {len(SEEDS)} seeds (mean +/- stdev), probe-noise FIXED, controls ADDED")
    print(f"{'='*80}")
    static_overall = [r["overall_test_acc"] for r in all_results["STATIC"]]
    for label, results in all_results.items():
        ret_mean, ret_sd, _ = summarize(results, "known_acc")
        acq_mean, acq_sd, _ = summarize(results, "new_acc")
        ov_mean, ov_sd, ov_vals = summarize(results, "overall_test_acc")
        deltas = [(r["overall_test_acc"] - s) * 100 for r, s in zip(results, static_overall)]
        delta_str = f"Δvs STATIC: mean={statistics.mean(deltas):+.2f}pp worst={min(deltas):+.2f}pp" if label != "STATIC" else ""
        print(f"{label:<22} retention={ret_mean:5.1f}%+/-{ret_sd:4.1f}  "
              f"acquisition={acq_mean:5.1f}%+/-{acq_sd:4.1f}  overall={ov_mean:5.1f}%+/-{ov_sd:4.1f}  {delta_str}")

    print("\nNote: STATIC itself varies seed-to-seed (base model + probe draw differ per seed) -")
    print("the per-seed PAIRED delta (same seed's STATIC) is the right comparison, not the pooled means.")
