"""
Sleep Consolidation (RESEARCH_NOTES.md §44, now building on validated Awake
Gate §43/§70d-f). Tests the SCM (Sleep-Consolidated Memory) hypothesis from
§44a: can a LOCAL, backprop-free update rule (Hebbian strengthening +
synaptic downscaling - the two local-rule-compatible NREM mechanisms) applied
only to gate-flagged "surprising" snapshots, consolidate new knowledge
without catastrophic forgetting, as an alternative to on-chip backprop
(which §29e established is architecturally impossible on real neuromorphic
hardware)?

Design (continual-learning class-split, per §44e's explicit requirement for
a 3-way baseline comparison):
  - KNOWN classes (SHD 0-14): what the base DenseJEPA is trained on.
  - NEW classes (SHD 15-19): held back, introduced later as a "day" of new
    experience streaming in.
  - Three conditions, same starting model, same day-stream data:
    1. STATIC       - ignore the new data entirely (floor baseline).
    2. ONLINE NAIVE - standard JEPA backprop+EMA update on every new sample
                       as it streams in (ceiling baseline - this is what
                       real on-chip backprop CAN'T do, per §29e, but it's
                       the standard continual-learning comparison point).
    3. SLEEP        - gate (trained on the known-class regime, §43) decides
                       which streaming samples to buffer ("wake"); after the
                       full day, ONE local Hebbian+decay update pass
                       consolidates only the buffered snapshots. No
                       backprop anywhere in this condition.
  - Evaluated by: ONE 20-way linear probe per condition (fair, consistent
    methodology - not per-condition probes), test accuracy broken down by
    whether the true label is a KNOWN class (retention) or a NEW class
    (acquisition).

Sleep's Hebbian rule only touches the context_encoder's two hidden Linear
layers (net[0], net[2] in dense_jepa.py's Encoder) - NOT the predictor or
target_encoder, consistent with §29e ("those are training-time-only, only
the trained encoder is deployed for inference" - the linear probe already
only ever reads context_encoder output, so this is the right component to
touch).

First pass - single seed, one untuned (eta, decay) setting. Preliminary by
the same standard as Awake Gate's first run (§70d) - needs the same
multi-seed + sensitivity treatment before being cited as robust.

CORRECTED (RESEARCH_NOTES.md §73, Opus audit): evaluate_condition() used to
re-iterate a shuffle=True probe-training loader fresh on every call, so
each condition (STATIC/ONLINE_NAIVE/SLEEP) got a DIFFERENT random draw of
probe-training samples - injecting measurement noise (sd ~1.4-3.2pp per
call, 98% of it from this resampling) directly into the quantity being
compared. This made every §71b/§71c Sleep-vs-Static delta uninterpretable
(the reported effects were smaller than this noise floor). Fixed: the
probe-training set is now drawn ONCE per seed (see `draw_fixed_batches`)
and reused, unchanged, across all three conditions within that seed -
only the model differs, not the probe data.
"""
import copy
import statistics
import torch
import torch.nn as nn

from real_data_shd import load_shd, make_loader, FLAT_DIM
from dense_jepa import DenseJEPA
from awake_gate import GateNet, split_context_target, collect_surprise_dataset, train_gate, count_params

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
HIDDEN_DIM = 128
REPR_DIM = 64
NUM_CLASSES = 20
BATCH_SIZE = 32
JEPA_STEPS = 300
JEPA_LR = 3e-3

KNOWN_CLASSES = list(range(15))   # SHD 0-14
NEW_CLASSES = list(range(15, 20))  # SHD 15-19

DAY_STREAM_BATCHES = 7            # ~7*32 = 224 new-class samples seen in "a day"
HEBBIAN_ETA = 5e-3
HEBBIAN_DECAY = 1e-2              # "synaptic downscaling" term, §44a
PROBE_STEPS = 2000
PROBE_TRAIN_BATCHES = 48          # labeled samples drawn from BOTH class groups


def filter_by_classes(dataset, classes):
    classes = set(classes)
    idx = [i for i in range(len(dataset)) if dataset[i][1] in classes]
    return torch.utils.data.Subset(dataset, idx)


def hebbian_update(encoder, context_batch, eta=HEBBIAN_ETA, decay=HEBBIAN_DECAY):
    """One local (no-backprop) Hebbian-strengthening + synaptic-downscaling
    step on the encoder's two hidden Linear layers, per §44a's NREM
    description. dW = eta * (post-activation^T @ pre-activation / n) -
    decay * W; treats each (batch, timestep) pair as one Hebbian sample."""
    net = encoder.net
    with torch.no_grad():
        b, t, d = context_batch.shape
        x0 = context_batch.reshape(b * t, d)
        z1 = net[0](x0)
        a1 = net[1](z1)
        z2 = net[2](a1)
        a2 = net[3](z2)

        dW1 = (a1.t() @ x0) / x0.shape[0]
        net[0].weight.add_(eta * dW1 - decay * net[0].weight)

        dW2 = (a2.t() @ a1) / a1.shape[0]
        net[2].weight.add_(eta * dW2 - decay * net[2].weight)


def train_jepa_steps(model, loader, n_steps):
    opt = torch.optim.Adam(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()), lr=JEPA_LR
    )
    data_iter = iter(loader)
    for _ in range(n_steps):
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, _ = next(data_iter)
        x = x.to(DEVICE)
        context, target = split_context_target(x)
        loss, _, _ = model(context, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()


@torch.no_grad()
def extract_embeddings(model, loader, n_batches):
    embs, labels = [], []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x = x.to(DEVICE)
        context, _ = split_context_target(x)
        repr_out = model.context_encoder(context)
        embs.append(repr_out.cpu())
        labels.append(y)
    return torch.cat(embs, dim=0), torch.cat(labels, dim=0)


def draw_fixed_batches(loader, n_batches):
    """Materialize n_batches of (x,y) ONCE so every condition that reuses
    this list sees IDENTICAL probe-training data - the fix for the
    probe-resampling noise found in the §73 audit."""
    batches = []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        batches.append((x, y))
    return batches


@torch.no_grad()
def extract_embeddings_from_batches(model, batches):
    embs, labels = [], []
    for x, y in batches:
        x = x.to(DEVICE)
        context, _ = split_context_target(x)
        repr_out = model.context_encoder(context)
        embs.append(repr_out.cpu())
        labels.append(y)
    return torch.cat(embs, dim=0), torch.cat(labels, dim=0)


def train_probe_and_get_module(train_emb, train_lab, device, steps=PROBE_STEPS, lr=1e-2):
    """Local variant of downstream_probe.train_linear_probe that also
    returns the trained nn.Linear module, so we can evaluate it separately
    on the known-class and new-class eval subsets (the shared helper only
    returns scalar accuracy on one combined test set)."""
    in_dim = train_emb.shape[1]
    probe = nn.Linear(in_dim, NUM_CLASSES).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_emb = train_emb.to(device)
    train_lab = train_lab.to(device)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(probe(train_emb), train_lab)
        loss.backward()
        opt.step()
    return probe


def evaluate_condition(name, model, probe_train_batches, known_eval_loader, new_eval_loader):
    train_emb, train_lab = extract_embeddings_from_batches(model, probe_train_batches)
    known_emb, known_lab = extract_embeddings(model, known_eval_loader, 16)
    new_emb, new_lab = extract_embeddings(model, new_eval_loader, 16)

    mean = train_emb.mean(dim=0, keepdim=True)
    std = train_emb.std(dim=0, keepdim=True) + 1e-6
    train_std = (train_emb - mean) / std

    probe = train_probe_and_get_module(train_std, train_lab, DEVICE)

    known_std = (known_emb.to(DEVICE) - mean.to(DEVICE)) / std.to(DEVICE)
    new_std = (new_emb.to(DEVICE) - mean.to(DEVICE)) / std.to(DEVICE)
    with torch.no_grad():
        known_acc = (probe(known_std).argmax(1) == known_lab.to(DEVICE)).float().mean().item()
        new_acc = (probe(new_std).argmax(1) == new_lab.to(DEVICE)).float().mean().item()
        n_known, n_new = known_emb.shape[0], new_emb.shape[0]
        overall_acc = (known_acc * n_known + new_acc * n_new) / (n_known + n_new)

    print(f"  [{name}] overall_test_acc={overall_acc:.4f} "
          f"known_acc(retention)={known_acc:.4f} new_acc(acquisition)={new_acc:.4f}")
    return {"name": name, "overall_test_acc": overall_acc, "known_acc": known_acc, "new_acc": new_acc}


if __name__ == "__main__":
    torch.manual_seed(SEED)
    print("=== Loading & splitting SHD by class ===")
    train_set = load_shd(train=True)
    eval_set = load_shd(train=False)

    known_train = filter_by_classes(train_set, KNOWN_CLASSES)
    new_train = filter_by_classes(train_set, NEW_CLASSES)
    known_eval = filter_by_classes(eval_set, KNOWN_CLASSES)
    new_eval = filter_by_classes(eval_set, NEW_CLASSES)
    print(f"  known_train={len(known_train)} new_train={len(new_train)} "
          f"known_eval={len(known_eval)} new_eval={len(new_eval)}")

    known_train_loader = make_loader(known_train, BATCH_SIZE, shuffle=True)
    new_train_loader = make_loader(new_train, BATCH_SIZE, shuffle=True)
    known_eval_loader = make_loader(known_eval, BATCH_SIZE, shuffle=False)
    new_eval_loader = make_loader(new_eval, BATCH_SIZE, shuffle=False)
    mixed_probe_loader = make_loader(torch.utils.data.ConcatDataset([known_train, new_train]),
                                      BATCH_SIZE, shuffle=True)
    # Fixed ONCE (§73 fix) - every condition below evaluates against this
    # exact same probe-training data, not a fresh shuffle per condition.
    probe_train_batches = draw_fixed_batches(mixed_probe_loader, PROBE_TRAIN_BATCHES)

    print("\n=== Step 1: train base DenseJEPA on KNOWN classes only ===")
    base_model = DenseJEPA(FLAT_DIM, HIDDEN_DIM, REPR_DIM).to(DEVICE)
    train_jepa_steps(base_model, known_train_loader, JEPA_STEPS)

    print("\n=== Step 2: train Awake Gate on the known-class-trained model ===")
    gate_train_contexts, gate_train_errors = collect_surprise_dataset(base_model, known_train_loader, n_batches=64)
    gate = GateNet(FLAT_DIM).to(DEVICE)
    train_gate(gate, gate_train_contexts, gate_train_errors)
    gate_thresh = torch.quantile(gate(gate_train_contexts.to(DEVICE)).detach().cpu(), 0.80).item()
    print(f"  gate wake-threshold (80th pct of known-regime scores) = {gate_thresh:.4f}")

    # Side-check: does the gate, trained only on KNOWN classes, treat NEW
    # (unfamiliar) classes as more surprising on average? A real novelty
    # signal should score new-class samples higher.
    # CORRECTED (§73 audit): the KNOWN side used to redraw from
    # known_train_loader, the SAME pool the gate was fit on (64 of its
    # batches) - a fresh 7-batch draw overlapped it ~34% in expectation,
    # biasing the KNOWN baseline upward. Now uses known_eval (true holdout,
    # never seen by gate or base_model). NEW side already used new_train,
    # which is fine - the gate never saw ANY new-class data during training.
    with torch.no_grad():
        new_probe_contexts, _ = collect_surprise_dataset(base_model, new_train_loader, n_batches=7)
        known_sample_contexts, _ = collect_surprise_dataset(base_model, known_eval_loader, n_batches=7)
        new_scores = gate(new_probe_contexts.to(DEVICE)).cpu()
        known_scores = gate(known_sample_contexts.to(DEVICE)).cpu()
    print(f"  gate score on held-out KNOWN-class samples: mean={known_scores.mean():.4f}")
    print(f"  gate score on NEW (unfamiliar) class samples: mean={new_scores.mean():.4f}")
    print(f"  (if novelty signal is real, NEW should score higher than KNOWN on average)")

    print("\n=== Step 3: simulate a 'day' of new-class data streaming in ===")
    day_batches = []
    it = iter(new_train_loader)
    for _ in range(DAY_STREAM_BATCHES):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(new_train_loader)
            x, y = next(it)
        day_batches.append((x, y))
    n_day_samples = sum(x.shape[0] for x, _ in day_batches)
    print(f"  day stream: {len(day_batches)} batches, {n_day_samples} samples total")

    print("\n=== Condition 1: STATIC (ignore the day entirely) ===")
    static_model = copy.deepcopy(base_model)

    print("\n=== Condition 2: ONLINE NAIVE (full backprop+EMA on every day sample) ===")
    online_model = copy.deepcopy(base_model)
    opt = torch.optim.Adam(
        list(online_model.context_encoder.parameters()) + list(online_model.predictor.parameters()), lr=JEPA_LR
    )
    for x, _ in day_batches:
        x = x.to(DEVICE)
        context, target = split_context_target(x)
        loss, _, _ = online_model(context, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        online_model.update_target()

    print("\n=== Condition 3: SLEEP (gate-filtered snapshots, one Hebbian night pass) ===")
    sleep_model = copy.deepcopy(base_model)
    woken_contexts = []
    for x, _ in day_batches:
        x = x.to(DEVICE)
        context, _ = split_context_target(x)
        with torch.no_grad():
            scores = gate(context)
        wake_mask = scores >= gate_thresh
        if wake_mask.any():
            woken_contexts.append(context[wake_mask])
    n_woken = sum(c.shape[0] for c in woken_contexts)
    print(f"  woken (snapshotted) during the day: {n_woken} / {n_day_samples} samples "
          f"({100*n_woken/n_day_samples:.1f}% duty cycle)")
    if n_woken > 0:
        night_batch = torch.cat(woken_contexts, dim=0)
        hebbian_update(sleep_model.context_encoder, night_batch)
    else:
        print("  WARNING: gate woke zero samples - sleep update skipped, condition == static")

    print("\n=== Step 4: evaluate all 3 conditions (same probe protocol, known/new split) ===")
    results = []
    for name, model in [("STATIC", static_model), ("ONLINE_NAIVE", online_model), ("SLEEP", sleep_model)]:
        r = evaluate_condition(name, model, probe_train_batches, known_eval_loader, new_eval_loader)
        results.append(r)

    print(f"\n{'='*70}")
    print(f"SUMMARY (single seed, preliminary)")
    print(f"{'='*70}")
    print(f"{'Condition':<14} {'Retention(known)':<18} {'Acquisition(new)':<18} {'Overall':<10}")
    for r in results:
        print(f"{r['name']:<14} {r['known_acc']*100:>6.1f}%{'':<11} {r['new_acc']*100:>6.1f}%{'':<11} {r['overall_test_acc']*100:>6.1f}%")

    static_r = results[0]
    online_r = results[1]
    sleep_r = results[2]
    print(f"\nSleep vs Static - retention delta: {(sleep_r['known_acc']-static_r['known_acc'])*100:+.1f}pp, "
          f"acquisition delta: {(sleep_r['new_acc']-static_r['new_acc'])*100:+.1f}pp")
    print(f"Sleep vs Online-naive - retention delta: {(sleep_r['known_acc']-online_r['known_acc'])*100:+.1f}pp, "
          f"acquisition delta: {(sleep_r['new_acc']-online_r['new_acc'])*100:+.1f}pp")
    print("\nNOTE: single seed, one untuned (eta,decay) setting - preliminary, "
          "needs multi-seed + hyperparameter sensitivity check before citing as robust.")
