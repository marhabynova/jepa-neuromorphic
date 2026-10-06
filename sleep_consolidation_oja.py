"""
Redesign, not another parameter sweep (per user's push: the eta-grind was
methodical but too slow without a real idea behind it). Two principled
changes to the local update rule, both targeting the ROOT causes found in
§71a-c (magnitude instability AND no task-relevance signal), not just
re-tuning the same broken formula:

1. OJA'S RULE instead of an arbitrary fixed `decay` constant. Oja's rule
   (Oja 1982) is the classic self-stabilizing Hebbian variant: the
   "forgetting" term is proportional to the POST-synaptic activity itself
   (post^2 * W), not a hand-picked constant - it provably converges instead
   of exploding, for a principled reason, not a tuned hyperparameter. This
   removes exactly the free parameter that caused §71a's magnitude blowup.

2. ERROR-MODULATED per-sample weighting. The old design threw away
   information: the gate makes a BINARY wake/sleep decision, then every
   woken sample contributes EQUALLY to the Hebbian average, discarding how
   surprising each one actually was. Here, during the (offline, more
   compute-tolerant) night phase, we compute the TRUE per-sample
   prediction-error for the woken snapshots (cheap - we already have
   collect_surprise_dataset for this) and use it to WEIGHT each sample's
   contribution to the update - the more surprising a sample, the stronger
   its influence. This is a direct (if simplified) three-factor rule:
   pre-activity x post-activity x local-reward-magnitude, the same family
   of rule §69 found the SQS paper actually used (not plain two-factor
   Hebbian).

First pass: single seed, directional check. If this shows a real signal,
follow up with the multi-seed confirmation §70f/§71c already established
as the bar before citing anything as robust - no shortcuts on THAT part,
just skipping redundant grid search on a mechanism already shown broken.
"""
import copy
import torch

from real_data_shd import load_shd, make_loader, FLAT_DIM
from dense_jepa import DenseJEPA
from awake_gate import GateNet, split_context_target, collect_surprise_dataset, train_gate
from sleep_consolidation import (
    filter_by_classes, train_jepa_steps, evaluate_condition,
    KNOWN_CLASSES, NEW_CLASSES, JEPA_STEPS, BATCH_SIZE, DAY_STREAM_BATCHES,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
OJA_ETA = 1e-2  # Oja's rule self-stabilizes, so this can be larger than the old fixed-decay eta


def oja_modulated_update(encoder, context_batch, sample_errors, eta=OJA_ETA):
    """Error-modulated Oja's rule on the two hidden Linear layers.
    sample_errors: (batch,) true per-sample prediction-error from the
    frozen dense JEPA, used as a per-sample importance weight (normalized
    to mean 1 so overall update scale stays comparable across batches)."""
    net = encoder.net
    with torch.no_grad():
        b, t, d = context_batch.shape
        x0 = context_batch.reshape(b * t, d)
        z1 = net[0](x0)
        a1 = net[1](z1)
        z2 = net[2](a1)
        a2 = net[3](z2)

        m = (sample_errors / (sample_errors.mean() + 1e-8)).to(x0.device)  # (b,), mean=1
        m_t = m.repeat_interleave(t)  # (b*t,), matches x0's batch-major flattening

        # Layer 1: modulated Oja's rule
        post1_w = a1 * m_t.unsqueeze(1)
        hebb1 = (post1_w.t() @ x0) / x0.shape[0]
        oja1 = (post1_w * a1).sum(dim=0) / x0.shape[0]  # per-neuron self-decay coefficient
        dW1 = eta * (hebb1 - oja1.unsqueeze(1) * net[0].weight)
        net[0].weight.add_(dW1)

        # Layer 2: modulated Oja's rule
        post2_w = a2 * m_t.unsqueeze(1)
        hebb2 = (post2_w.t() @ a1) / a1.shape[0]
        oja2 = (post2_w * a2).sum(dim=0) / a1.shape[0]
        dW2 = eta * (hebb2 - oja2.unsqueeze(1) * net[2].weight)
        net[2].weight.add_(dW2)


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

    print("\n=== Train base DenseJEPA + Gate ===")
    base_model = DenseJEPA(FLAT_DIM, 128, 64).to(DEVICE)
    train_jepa_steps(base_model, known_train_loader, JEPA_STEPS)

    gate_train_contexts, gate_train_errors = collect_surprise_dataset(base_model, known_train_loader, n_batches=64)
    gate = GateNet(FLAT_DIM).to(DEVICE)
    train_gate(gate, gate_train_contexts, gate_train_errors)
    gate_thresh = torch.quantile(gate(gate_train_contexts.to(DEVICE)).detach().cpu(), 0.80).item()

    print("\n=== Day stream + gate-filtered wake decisions ===")
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

    print("\n=== Compute TRUE per-sample error for the night batch (offline, compute-tolerant) ===")
    # Need target windows too for a fair per-sample error - re-derive from original day_batches
    # by re-running the gate filter but keeping full (context,target) pairs.
    woken_pairs = []
    for x, _ in day_batches:
        x = x.to(DEVICE)
        context, target = split_context_target(x)
        with torch.no_grad():
            scores = gate(context)
        mask = scores >= gate_thresh
        if mask.any():
            woken_pairs.append((context[mask], target[mask]))
    woken_context = torch.cat([c for c, _ in woken_pairs], dim=0)
    woken_target = torch.cat([t for _, t in woken_pairs], dim=0)
    with torch.no_grad():
        ctx_repr = base_model.context_encoder(woken_context)
        pred_repr = base_model.predictor(ctx_repr)
        tgt_repr = base_model.target_encoder(woken_target)
        sample_errors = ((pred_repr - tgt_repr) ** 2).mean(dim=1).cpu()
    print(f"  per-sample error: mean={sample_errors.mean():.4f} std={sample_errors.std():.4f} "
          f"(modulation weight range: {(sample_errors/sample_errors.mean()).min():.2f}x - "
          f"{(sample_errors/sample_errors.mean()).max():.2f}x)")

    print("\n=== Condition: STATIC ===")
    static_model = copy.deepcopy(base_model)
    static_r = evaluate_condition("STATIC", static_model, mixed_probe_loader, known_eval_loader, new_eval_loader)

    print("\n=== Condition: ONLINE_NAIVE ===")
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

    print("\n=== Condition: SLEEP (error-modulated Oja's rule), eta sweep ===")
    print("  (eta=1e-2 already tried separately: ||dW2||=210% of ||W2|| - way too large,")
    print("   'self-stabilizing' only bounds the long-run equilibrium, NOT a single step's size)")
    oja_results = []
    for eta in [1e-3, 1e-4, 1e-5]:
        oja_model = copy.deepcopy(base_model)
        net = oja_model.context_encoder.net
        w1_before, w2_before = net[0].weight.detach().clone(), net[2].weight.detach().clone()
        oja_modulated_update(oja_model.context_encoder, woken_context, sample_errors, eta=eta)
        d1 = (net[0].weight.detach() - w1_before).norm().item() / w1_before.norm().item() * 100
        d2 = (net[2].weight.detach() - w2_before).norm().item() / w2_before.norm().item() * 100
        print(f"\n  --- eta={eta:.0e} --- (||dW1||={d1:.1f}%, ||dW2||={d2:.1f}%)")
        r = evaluate_condition(f"SLEEP_OJA(eta={eta:.0e})", oja_model, mixed_probe_loader, known_eval_loader, new_eval_loader)
        r["eta"] = eta
        r["d1"] = d1
        r["d2"] = d2
        oja_results.append(r)

    print(f"\n{'='*70}")
    print("SUMMARY (single seed, directional check)")
    print(f"{'='*70}")
    for r in [static_r, online_r]:
        print(f"{r['name']:<20} retention={r['known_acc']*100:5.1f}%  acquisition={r['new_acc']*100:5.1f}%  overall={r['overall_test_acc']*100:5.1f}%")
    for r in oja_results:
        print(f"{'SLEEP_OJA eta='+format(r['eta'],'.0e'):<20} dW1={r['d1']:5.1f}% dW2={r['d2']:5.1f}%  "
              f"retention={r['known_acc']*100:5.1f}%  acquisition={r['new_acc']*100:5.1f}%  overall={r['overall_test_acc']*100:5.1f}%")

    best = max(oja_results, key=lambda r: r["overall_test_acc"])
    print(f"\nBest Oja point: eta={best['eta']:.0e} overall={best['overall_test_acc']*100:.1f}% "
          f"(Static={static_r['overall_test_acc']*100:.1f}%, Online-naive={online_r['overall_test_acc']*100:.1f}%)")
    print("\nIf any point shows a real (non-decay-floor) positive delta, next step is the SAME")
    print("multi-seed confirmation bar as §70f/§71c before calling it robust - no shortcut there.")
