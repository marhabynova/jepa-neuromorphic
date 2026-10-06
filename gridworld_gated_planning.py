"""
Opus's recommended reuse (§81b): awake_gate.py's distillation pattern,
repurposed. The original Awake Gate distilled a JEPA's per-sample
prediction-error into a cheap network, for continual learning (shelved
after honest negatives in the Sleep Consolidation thread, §71-73).

Here it distills the DYNAMICS MODEL's per-transition prediction error
instead - directly attacks the problem §82 just measured honestly: Stage B
(pure imagined 1-step lookahead) has wild variance (detour 67.7%+/-17.3%,
worst seed 49.0%) because it has NO mechanism to detect when the dynamics
model's prediction for a specific transition is unreliable. This is the
first use of the Awake Gate pattern where its known limitation (§71 Temuan
#1: in-distribution "surprise" doesn't generalize to OOD novelty) doesn't
apply - dynamics-model error is an in-distribution quantity, not an
OOD-detection problem.

Design: HYBRID planner. For the action Stage B's value-guided lookahead
would pick, the gate predicts whether that specific transition's dynamics
prediction is likely accurate. If trustworthy -> commit to it the cheap
way (Stage B). If flagged risky -> pay for a REAL-environment check of
just that one transition before committing (Stage A, for that step only).
Reports both success rate AND how many real-environment calls were spent,
on the SAME honest held-out-pair protocol established in §82 (not the
in-distribution numbers from §78-80).
"""
import copy
import statistics

import numpy as np
import torch
import torch.nn as nn

from gridworld_obstacles import (
    GRID_SIZE, N_ACTIONS, STATE_DIM, step_env, random_free_cell, greedy_heuristic,
    encode_state_batch, DEVICE, SEED,
)
from gridworld_value_jepa_integrated import ctx, tgt, MAX_PLAN_STEPS
from gridworld_fixes import (
    FREE_CELLS, decode_accuracy, train_world_model_validated, make_pair_split,
    train_value_held_out, exact_vi_distance,
)

GATE_HIDDEN = 32
GATE_STEPS = 2000
GATE_LR = 1e-3
GATE_BATCH = 128
ERROR_THRESHOLD_PERCENTILE = 70  # flag the riskiest 30% of transitions for real-grounding


class DynamicsErrorGate(nn.Module):
    """Same spirit as awake_gate.py's GateNet, simplified to a plain MLP
    here (no spiking layer - this is about reusing the DISTILLATION
    PATTERN, not the specific neuron model). Predicts the dynamics model's
    own prediction error for a (state, action) pair, from the state's
    representation + action one-hot - cheaper than running the dynamics
    model + comparing against ground truth."""

    def __init__(self, repr_dim, action_dim, hidden_dim=GATE_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim + action_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_repr, action_onehot):
        return self.net(torch.cat([state_repr, action_onehot], dim=-1)).squeeze(-1)


def collect_dynamics_error_dataset(world_model, rng, n_samples=2000):
    """True per-transition dynamics error (||predicted - true_next_repr||^2),
    the gate's regression target - exact ground truth, same spirit as
    awake_gate.py's collect_surprise_dataset but retargeted."""
    states, actions, errors = [], [], []
    for _ in range(n_samples):
        s = random_free_cell(rng)
        a = int(rng.integers(0, N_ACTIONS))
        true_next = step_env(s[0], s[1], a)
        with torch.no_grad():
            cur_repr = ctx(world_model, *s)
            ao = nn.functional.one_hot(torch.tensor(a), N_ACTIONS).float().to(DEVICE)
            pred_repr = world_model.dynamics(cur_repr.unsqueeze(0), ao.unsqueeze(0)).squeeze(0)
            true_repr = ctx(world_model, *true_next)
            err = torch.mean((pred_repr - true_repr) ** 2).item()
        states.append(s)
        actions.append(a)
        errors.append(err)
    return states, actions, torch.tensor(errors)


def train_gate(world_model, seed):
    torch.manual_seed(seed + 2)
    rng = np.random.default_rng(seed + 2)
    states, actions, errors = collect_dynamics_error_dataset(world_model, rng, n_samples=GATE_BATCH * 8)
    gate = DynamicsErrorGate(world_model.context_encoder.net[-1].out_features, N_ACTIONS).to(DEVICE)
    opt = torch.optim.Adam(gate.parameters(), lr=GATE_LR)
    err_mean, err_std = errors.mean(), errors.std() + 1e-8

    n = len(states)
    for step in range(GATE_STEPS):
        idx = np.random.randint(0, n, size=GATE_BATCH)
        batch_states = [states[i] for i in idx]
        batch_actions = [actions[i] for i in idx]
        batch_targets = ((errors[idx] - err_mean) / err_std).to(DEVICE)
        with torch.no_grad():
            reprs = torch.stack([ctx(world_model, *s) for s in batch_states])
        aos = nn.functional.one_hot(torch.tensor(batch_actions), N_ACTIONS).float().to(DEVICE)
        pred = gate(reprs, aos)
        loss = torch.mean((pred - batch_targets) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"  [gate] step {step+1}/{GATE_STEPS} mse={loss.item():.4f}")

    # threshold on the TRAINING error distribution's own percentile
    with torch.no_grad():
        reprs_all = torch.stack([ctx(world_model, *s) for s in states])
        aos_all = nn.functional.one_hot(torch.tensor(actions), N_ACTIONS).float().to(DEVICE)
        scores_all = gate(reprs_all, aos_all).cpu()
    threshold = torch.quantile(scores_all, ERROR_THRESHOLD_PERCENTILE / 100).item()

    # Sanity check: does the gate's score actually correlate with true error?
    corr = np.corrcoef(scores_all.numpy(), errors.numpy())[0, 1]
    print(f"  [gate] score<->true_error correlation: {corr:.4f}, risk threshold={threshold:.4f}")
    return gate, threshold


def plan_hybrid(world_model, V, gate, threshold, start, goal, max_steps=MAX_PLAN_STEPS):
    x, y = start
    path = [(x, y)]
    g_repr = tgt(world_model, *goal)
    real_calls = 0
    for _ in range(max_steps):
        if (x, y) == goal:
            return path, True, real_calls
        cur_repr = ctx(world_model, x, y)

        # Stage-B-style: score all 4 candidates via IMAGINED (cheap) rollout
        best_a, best_v, pred_reprs = None, float("inf"), {}
        for a in range(N_ACTIONS):
            ao = nn.functional.one_hot(torch.tensor(a), N_ACTIONS).float().to(DEVICE)
            with torch.no_grad():
                pred_repr = world_model.dynamics(cur_repr.unsqueeze(0), ao.unsqueeze(0)).squeeze(0)
                v = V(pred_repr.unsqueeze(0), g_repr.unsqueeze(0)).item()
            pred_reprs[a] = pred_repr
            if v < best_v:
                best_v, best_a = v, a

        # Gate check on the CHOSEN action only
        ao_chosen = nn.functional.one_hot(torch.tensor(best_a), N_ACTIONS).float().to(DEVICE)
        with torch.no_grad():
            risk = gate(cur_repr.unsqueeze(0), ao_chosen.unsqueeze(0)).item()

        if risk > threshold:
            # Flagged risky: pay for a REAL-environment check of all 4
            # candidates instead (Stage-A style), just for this one step.
            real_calls += 1
            best_a, best_v = None, float("inf")
            for a in range(N_ACTIONS):
                nx, ny = step_env(x, y, a)
                next_repr = ctx(world_model, nx, ny)
                v = V(next_repr.unsqueeze(0), g_repr.unsqueeze(0)).item()
                if v < best_v:
                    best_v, best_a = v, a

        x, y = step_env(x, y, best_a)
        path.append((x, y))
    return path, (x, y) == goal, real_calls


def eval_hybrid(world_model, V, gate, threshold, pairs):
    detour, no_detour = [], []
    for s, g in pairs:
        _, greached = greedy_heuristic(s, g)
        opt = exact_vi_distance(s, g)
        (detour if not greached else no_detour).append((s, g, opt))

    def run(subset):
        succ, calls = 0, []
        for s, g, opt in subset:
            path, reached, rc = plan_hybrid(world_model, V, gate, threshold, s, g)
            calls.append(rc)
            if reached:
                succ += 1
        return (succ / len(subset) if subset else float("nan")), len(subset), (sum(calls) / len(calls) if calls else 0)

    d_rate, d_n, d_calls = run(detour)
    nd_rate, nd_n, nd_calls = run(no_detour)
    return d_rate, d_n, d_calls, nd_rate, nd_n, nd_calls


if __name__ == "__main__":
    SEEDS = [0, 1, 2]
    results = {"detour": [], "no_detour": [], "real_calls": []}
    for seed in SEEDS:
        print(f"\n{'='*20} seed={seed} {'='*20}")
        world_model, best_steps, best_acc = train_world_model_validated(seed)
        train_pairs, test_pairs = make_pair_split(seed)
        V = train_value_held_out(world_model, train_pairs, seed)
        gate, threshold = train_gate(world_model, seed)

        test_pairs_list = list(test_pairs)
        d_rate, d_n, d_calls, nd_rate, nd_n, nd_calls = eval_hybrid(
            world_model, V, gate, threshold, test_pairs_list
        )
        print(f"  Hybrid (gate-guided) on HELD-OUT pairs: detour={d_rate*100:.1f}% (n={d_n}, "
              f"avg real-env calls/episode={d_calls:.1f}), no-detour={nd_rate*100:.1f}% "
              f"(n={nd_n}, avg calls={nd_calls:.1f})")
        results["detour"].append(d_rate * 100)
        results["no_detour"].append(nd_rate * 100)
        results["real_calls"].append((d_calls + nd_calls) / 2)

    print(f"\n{'='*70}")
    print("SUMMARY - Hybrid (gate-guided Stage B/A) on held-out pairs, 3 seeds")
    print(f"{'='*70}")
    for key in ["detour", "no_detour"]:
        vals = results[key]
        mean, sd = statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)
        print(f"{key:<12} {mean:5.1f}% +/- {sd:4.1f}  (per-seed: {[f'{v:.1f}' for v in vals]}, worst={min(vals):.1f}%)")
    print(f"\nFor comparison (§82, same held-out protocol):")
    print(f"  Stage B (always imagine):  detour=67.7%+/-17.3 (worst 49.0%)  no-detour=75.6%+/-14.1 (worst 62.3%)")
    print(f"  avg real-env calls/episode spent by hybrid: {statistics.mean(results['real_calls']):.1f}")
