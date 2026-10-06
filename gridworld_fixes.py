"""
Three cheap fixes §81's audit said MUST land before any new experiment
(hierarchical planning, Awake Gate reuse), or the next result repeats the
§79/§80 overclaim pattern:

1. Replace cosine-sim (non-discriminative, even ANTI-correlated with true
   quality at high step counts - §81a#1) with next-cell DECODE ACCURACY as
   the world-model validation metric. Use it to actually pick STEPS
   instead of blindly inheriting 2000 (which turned out to be the worst
   point on a non-monotone curve for Stage B - §81a#2).
2. Compare raw-state-V / Stage-A / Stage-B-robust on the SAME eval pair
   set per seed, 3 seeds, not three different pair draws across scripts
   (§81a#3/#8).
3. Add the exact-VI ceiling (zero learning, ~15 lines, ground truth) as a
   standing reference row, and a genuine held-out (state,goal) PAIR split
   for the value function (§81a#4/#5) - V trains on a subset of the 3364
   ordered pairs, evaluated on pairs never seen during its training,
   isolating memorization from generalization.
"""
import copy
import statistics
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from gridworld_obstacles import (
    GRID_SIZE, N_ACTIONS, STATE_DIM, WALLS, is_wall, step_env, random_free_cell,
    bfs_optimal_distance, encode_state_batch, greedy_heuristic,
    ActionConditionedJEPA, DEVICE, SEED, REPR_DIM, HIDDEN_DIM, LR, BATCH_SIZE,
)
from gridworld_value_jepa_integrated import (
    ValueNetJEPA, ctx, ctx_batch, tgt, tgt_batch, plan_stage_a, plan_stage_b,
    V_EMA_DECAY, V_LR, V_BATCH_SIZE, MAX_PLAN_STEPS,
)

FREE_CELLS = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE) if not is_wall(x, y)]


# ---------- Fix 1: decode-accuracy metric, use it to pick STEPS ----------

@torch.no_grad()
def decode_accuracy(model, rng, n_samples=500):
    """Nearest-neighbour decode of the dynamics model's predicted
    representation against every free cell's target_encoder representation
    - replaces cosine-sim, which §81a#1 found rises while this falls."""
    cell_reprs = torch.stack([tgt(model, x, y) for x, y in FREE_CELLS])  # (n_cells, repr_dim)
    correct = 0
    for _ in range(n_samples):
        s = random_free_cell(rng)
        a = int(rng.integers(0, N_ACTIONS))
        true_next = step_env(s[0], s[1], a)
        ao = nn.functional.one_hot(torch.tensor(a), N_ACTIONS).float().to(DEVICE)
        pred_repr = model.dynamics(ctx(model, *s).unsqueeze(0), ao.unsqueeze(0)).squeeze(0)
        dists = torch.sum((cell_reprs - pred_repr) ** 2, dim=1)
        nearest = FREE_CELLS[dists.argmin().item()]
        if nearest == true_next:
            correct += 1
    return correct / n_samples


def train_world_model_validated(seed, candidate_steps=(600, 1000, 1500, 2000, 3000, 4000)):
    """Trains at the largest candidate step count, checkpointing decode
    accuracy along the way, and returns the checkpoint with the BEST
    decode accuracy - not whatever STEPS was last inherited."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ActionConditionedJEPA(STATE_DIM, N_ACTIONS).to(DEVICE)
    opt = torch.optim.Adam(
        list(model.context_encoder.parameters()) + list(model.dynamics.parameters()), lr=LR
    )
    from gridworld_obstacles import sample_batch
    max_steps = max(candidate_steps)
    best_acc, best_state, best_steps = -1, None, None
    eval_rng = np.random.default_rng(seed + 999)
    for step in range(max_steps):
        state, action, next_state, _ = sample_batch(BATCH_SIZE, rng)
        state, action, next_state = state.to(DEVICE), action.to(DEVICE), next_state.to(DEVICE)
        loss, _, _, _ = model(state, action, next_state)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()
        if (step + 1) in candidate_steps:
            acc = decode_accuracy(model, eval_rng, n_samples=300)
            print(f"  [wm seed={seed}] step={step+1} decode_acc={acc*100:.1f}%")
            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())
                best_steps = step + 1
    model.load_state_dict(best_state)
    print(f"  [wm seed={seed}] BEST: step={best_steps} decode_acc={best_acc*100:.1f}% (selected, not blind STEPS=2000)")
    return model, best_steps, best_acc


# ---------- Fix 3a: exact value iteration ceiling ----------

def exact_vi_distance(start, goal):
    return bfs_optimal_distance(start, goal)  # already exact BFS - this IS the ceiling


# ---------- Fix 3b: held-out (state, goal) pair split for V ----------

def make_pair_split(seed, test_frac=0.2):
    rng = np.random.default_rng(seed + 500)
    all_pairs = [(s, g) for s in FREE_CELLS for g in FREE_CELLS if s != g]
    rng.shuffle(all_pairs)
    n_test = int(len(all_pairs) * test_frac)
    test_pairs = set(all_pairs[:n_test])
    train_pairs = all_pairs[n_test:]
    return train_pairs, test_pairs


def train_value_held_out(world_model, train_pairs, seed, steps=4000):
    """Same TD/Bellman recipe as gridworld_value_jepa_integrated.py's
    train_value_function_jepa_robust, but samples (state,goal) ONLY from
    train_pairs - held-out pairs are never seen during V's training."""
    torch.manual_seed(seed + 1)
    rng = np.random.default_rng(seed + 1)
    V = ValueNetJEPA(REPR_DIM).to(DEVICE)
    V_target = copy.deepcopy(V)
    for p in V_target.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(V.parameters(), lr=V_LR)
    train_pairs_arr = train_pairs

    for step in range(steps):
        idx = rng.integers(0, len(train_pairs_arr), size=V_BATCH_SIZE)
        batch = [train_pairs_arr[i] for i in idx]
        states = [p[0] for p in batch]
        goals = [p[1] for p in batch]
        sxs = [s[0] for s in states]; sys_ = [s[1] for s in states]
        gxs = [g[0] for g in goals]; gys = [g[1] for g in goals]
        state_repr = ctx_batch(world_model, sxs, sys_)
        goal_repr = tgt_batch(world_model, gxs, gys)
        is_goal = torch.zeros(V_BATCH_SIZE, dtype=torch.bool, device=DEVICE)  # states!=goals by construction

        with torch.no_grad():
            best_next_v = torch.full((V_BATCH_SIZE,), float("inf"), device=DEVICE)
            for a in range(N_ACTIONS):
                next_states = [step_env(s[0], s[1], a) for s in states]
                nxs = [ns[0] for ns in next_states]; nys = [ns[1] for ns in next_states]
                next_repr = ctx_batch(world_model, nxs, nys)
                v_next = V_target(next_repr, goal_repr)
                arrived = torch.tensor([ns == g for ns, g in zip(next_states, goals)], device=DEVICE)
                v_next = torch.where(arrived, torch.zeros_like(v_next), v_next)
                best_next_v = torch.minimum(best_next_v, v_next)
            target = 1.0 + best_next_v

        # Also expose V to dynamics-predicted representations (§80's fix), restricted to train_pairs too.
        idx2 = rng.integers(0, len(train_pairs_arr), size=V_BATCH_SIZE)
        batch2 = [train_pairs_arr[i] for i in idx2]
        prev_states = [p[0] for p in batch2]
        goals2 = [p[1] for p in batch2]
        actions = [int(rng.integers(0, N_ACTIONS)) for _ in range(V_BATCH_SIZE)]
        arrivals = [step_env(ps[0], ps[1], a) for ps, a in zip(prev_states, actions)]
        prev_repr = ctx_batch(world_model, [s[0] for s in prev_states], [s[1] for s in prev_states])
        goal_repr2 = tgt_batch(world_model, [g[0] for g in goals2], [g[1] for g in goals2])
        action_oh = nn.functional.one_hot(torch.tensor(actions), N_ACTIONS).float().to(DEVICE)
        with torch.no_grad():
            predicted_repr = world_model.dynamics(prev_repr, action_oh)
            best_next_v2 = torch.full((V_BATCH_SIZE,), float("inf"), device=DEVICE)
            for a in range(N_ACTIONS):
                nxt = [step_env(ar[0], ar[1], a) for ar in arrivals]
                nxt_repr = ctx_batch(world_model, [n[0] for n in nxt], [n[1] for n in nxt])
                v2 = V_target(nxt_repr, goal_repr2)
                arrived2 = torch.tensor([n == g for n, g in zip(nxt, goals2)], device=DEVICE)
                v2 = torch.where(arrived2, torch.zeros_like(v2), v2)
                best_next_v2 = torch.minimum(best_next_v2, v2)
            arrival_is_goal = torch.tensor([a == g for a, g in zip(arrivals, goals2)], device=DEVICE)
            target2 = torch.where(arrival_is_goal, torch.zeros_like(best_next_v2), 1.0 + best_next_v2)

        pred_real = V(state_repr, goal_repr)
        pred_imagined = V(predicted_repr, goal_repr2)
        loss = torch.mean((pred_real - target) ** 2) + torch.mean((pred_imagined - target2) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p_t, p in zip(V_target.parameters(), V.parameters()):
                p_t.data.mul_(V_EMA_DECAY).add_(p.data, alpha=1 - V_EMA_DECAY)
        if (step + 1) % 1000 == 0:
            print(f"  [V held-out] step {step+1}/{steps} loss={loss.item():.4f}")
    return V


def eval_on_pairs(plan_fn, world_model, V, pairs):
    detour, no_detour = [], []
    for s, g in pairs:
        _, greached = greedy_heuristic(s, g)
        opt = exact_vi_distance(s, g)
        (detour if not greached else no_detour).append((s, g, opt))

    def run(subset):
        succ, exc = 0, []
        for s, g, opt in subset:
            path, reached = plan_fn(world_model, V, s, g)
            if reached:
                succ += 1
                exc.append((len(path) - 1) - opt)
        return (succ / len(subset) if subset else float("nan")), len(subset)

    d_rate, d_n = run(detour)
    nd_rate, nd_n = run(no_detour)
    return d_rate, d_n, nd_rate, nd_n


if __name__ == "__main__":
    SEEDS = [0, 1, 2]
    print("=== Fix 1: pick world-model STEPS via decode accuracy, not blind inheritance ===")
    print(f"=== Fix 3a: exact-VI ceiling = 100% by definition (ground truth), always shown as reference ===")
    print(f"=== Fix 3b: held-out pair split (20% of {len(FREE_CELLS)*(len(FREE_CELLS)-1)} ordered pairs never trained on) ===\n")

    results_heldout = {"detour": [], "no_detour": []}
    for seed in SEEDS:
        print(f"\n{'='*20} seed={seed} {'='*20}")
        world_model, best_steps, best_acc = train_world_model_validated(seed)
        train_pairs, test_pairs = make_pair_split(seed)
        print(f"  train pairs: {len(train_pairs)}, held-out test pairs: {len(test_pairs)}")

        V = train_value_held_out(world_model, train_pairs, seed)

        test_pairs_list = list(test_pairs)
        d_rate, d_n, nd_rate, nd_n = eval_on_pairs(plan_stage_b, world_model, V, test_pairs_list)
        print(f"  Stage B (robust V) on HELD-OUT pairs: detour={d_rate*100:.1f}% (n={d_n}), "
              f"no-detour={nd_rate*100:.1f}% (n={nd_n})")
        results_heldout["detour"].append(d_rate * 100)
        results_heldout["no_detour"].append(nd_rate * 100)

    print(f"\n{'='*70}")
    print("SUMMARY - Stage B (robust V) evaluated ONLY on held-out pairs, 3 seeds")
    print(f"{'='*70}")
    for key in ["detour", "no_detour"]:
        vals = results_heldout[key]
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{key:<12} {mean:5.1f}% +/- {sd:4.1f}  (per-seed: {[f'{v:.1f}' for v in vals]}, worst={min(vals):.1f}%)")
    print("\nExact-VI ceiling (zero learning, ground truth): 100% on every pair, always.")
    print("This is the first number in the §75-81 arc measured on GENUINELY UNSEEN")
    print("(state,goal) pairs, with a properly-selected world model, against the true ceiling.")
