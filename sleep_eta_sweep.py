"""
Follow-up to §71a: the Opus-flagged lead ("critical finding in the Hebbian
magnitudes") was confirmed by direct measurement - a single hebbian_update()
at eta=5e-3 changed ||W2|| by 93.8% (nearly doubled in one step). This
sweeps eta over several orders of magnitude on COPIES of the SAME trained
base model + gate + night batch (apples-to-apples, only eta varies), to
find a properly-calibrated scale before re-testing the reward-modulation
hypothesis from §71 (project priority: reject the cheap hypothesis first).

decay is kept fixed at the original §71 value (1e-2) throughout - this
sweep isolates eta specifically, since eta was identified as the dominant
term at the original setting.
"""
import copy
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
SEED = 0
ETA_SWEEP = [5e-3, 1e-3, 1e-4, 1e-5, 1e-6]  # 5e-3 is §71's original (destructive) setting


if __name__ == "__main__":
    torch.manual_seed(SEED)
    print("=== Loading & splitting SHD by class (same as §71) ===")
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

    print("\n=== Train base DenseJEPA + Gate once (reused across the whole sweep) ===")
    base_model = DenseJEPA(FLAT_DIM, 128, 64).to(DEVICE)
    train_jepa_steps(base_model, known_train_loader, JEPA_STEPS)

    gate_train_contexts, gate_train_errors = collect_surprise_dataset(base_model, known_train_loader, n_batches=64)
    gate = GateNet(FLAT_DIM).to(DEVICE)
    train_gate(gate, gate_train_contexts, gate_train_errors)
    gate_thresh = torch.quantile(gate(gate_train_contexts.to(DEVICE)).detach().cpu(), 0.80).item()

    print("\n=== Build ONE fixed 'day stream' + 'night batch' (reused across the whole sweep) ===")
    day_batches = []
    it = iter(new_train_loader)
    for _ in range(DAY_STREAM_BATCHES):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(new_train_loader)
            x, y = next(it)
        day_batches.append((x, y))

    woken_contexts = []
    for x, _ in day_batches:
        x = x.to(DEVICE)
        context, _ = split_context_target(x)
        with torch.no_grad():
            scores = gate(context)
        mask = scores >= gate_thresh
        if mask.any():
            woken_contexts.append(context[mask])
    night_batch = torch.cat(woken_contexts, dim=0)
    print(f"  night_batch: {night_batch.shape[0]} samples")

    print("\n=== Baselines (recomputed fresh in this run, same seed) ===")
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

    print(f"\n=== Eta sweep (decay fixed at {HEBBIAN_DECAY}) ===")
    sweep_results = []
    for eta in ETA_SWEEP:
        sleep_model = copy.deepcopy(base_model)
        net = sleep_model.context_encoder.net
        w1_before, w2_before = net[0].weight.detach().clone(), net[2].weight.detach().clone()

        hebbian_update(sleep_model.context_encoder, night_batch, eta=eta, decay=HEBBIAN_DECAY)

        d1 = (net[0].weight.detach() - w1_before).norm().item() / w1_before.norm().item() * 100
        d2 = (net[2].weight.detach() - w2_before).norm().item() / w2_before.norm().item() * 100
        print(f"\n  --- eta={eta:.0e} --- (||dW1||={d1:.1f}% of ||W1||, ||dW2||={d2:.1f}% of ||W2||)")
        r = evaluate_condition(f"SLEEP(eta={eta:.0e})", sleep_model, mixed_probe_loader, known_eval_loader, new_eval_loader)
        r["eta"] = eta
        r["dW1_pct"] = d1
        r["dW2_pct"] = d2
        sweep_results.append(r)

    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"{'Condition':<22} {'dW1%':>7} {'dW2%':>7} {'Retention':>11} {'Acquisition':>13} {'Overall':>9}")
    print(f"{'STATIC':<22} {'-':>7} {'-':>7} {static_r['known_acc']*100:>10.1f}% {static_r['new_acc']*100:>12.1f}% {static_r['overall_test_acc']*100:>8.1f}%")
    print(f"{'ONLINE_NAIVE':<22} {'-':>7} {'-':>7} {online_r['known_acc']*100:>10.1f}% {online_r['new_acc']*100:>12.1f}% {online_r['overall_test_acc']*100:>8.1f}%")
    for r in sweep_results:
        label = f"SLEEP(eta={r['eta']:.0e})"
        print(f"{label:<22} {r['dW1_pct']:>6.1f}% {r['dW2_pct']:>6.1f}% {r['known_acc']*100:>10.1f}% {r['new_acc']*100:>12.1f}% {r['overall_test_acc']*100:>8.1f}%")

    best = max(sweep_results, key=lambda r: r["overall_test_acc"])
    print(f"\nBest sweep point: eta={best['eta']:.0e} (overall_test_acc={best['overall_test_acc']*100:.1f}%, "
          f"vs STATIC={static_r['overall_test_acc']*100:.1f}%, ONLINE_NAIVE={online_r['overall_test_acc']*100:.1f}%)")
    print("NOTE: single seed, decay held fixed - preliminary, this just locates a sane eta scale,")
    print("      not a final robust result.")
