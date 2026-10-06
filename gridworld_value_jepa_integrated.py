"""
Integrates §78's value function with §75's learned JEPA representation -
previously they were two independently-verified but separate components
(ValueNet took raw state one-hot, not context_encoder output).

Staged, not done all at once (matches this project's whole discipline -
isolate one variable per test):

STAGE A: ValueNet operates on JEPA representations (context_encoder for
"current state", target_encoder for "goal" - same context/target split
used everywhere else in this project), but the Bellman backup's next-state
still comes from the REAL environment. Isolates "does the value function
still learn correctly when given JEPA reprs instead of raw one-hot" from
"does it also survive learned-dynamics-model error" (two separate risks).

STAGE B (only if A works): planning's 1-step lookahead switches from the
real environment to the learned dynamics model's PREDICTED next
representation - true "pure imagination" planning (never touches the real
environment until the final chosen action is executed), the complete
integration this thread has been building toward since §75.
"""
import copy
import statistics
import numpy as np
import torch
import torch.nn as nn

from gridworld_obstacles import (
    GRID_SIZE, N_ACTIONS, STATE_DIM, step_env, random_free_cell, bfs_optimal_distance,
    encode_state_batch, greedy_heuristic, train_model as train_world_model,
    ActionConditionedJEPA, DEVICE, SEED, REPR_DIM,
)

HIDDEN_DIM = 64
V_EMA_DECAY = 0.99
V_STEPS = 4000
V_LR = 1e-3
V_BATCH_SIZE = 128
MAX_PLAN_STEPS = 40
N_EVAL_PAIRS = 200


class ValueNetJEPA(nn.Module):
    """Same architecture as §78's ValueNet, but repr_dim-sized inputs
    (JEPA representations) instead of state_dim one-hot."""

    def __init__(self, repr_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_repr, goal_repr):
        return self.net(torch.cat([state_repr, goal_repr], dim=-1)).squeeze(-1)


@torch.no_grad()
def ctx(model, x, y):
    return model.context_encoder(encode_state_batch(torch.tensor([x]), torch.tensor([y])).to(DEVICE)).squeeze(0)


@torch.no_grad()
def ctx_batch(model, xs, ys):
    return model.context_encoder(encode_state_batch(torch.tensor(xs), torch.tensor(ys)).to(DEVICE))


@torch.no_grad()
def tgt(model, x, y):
    return model.target_encoder(encode_state_batch(torch.tensor([x]), torch.tensor([y])).to(DEVICE)).squeeze(0)


@torch.no_grad()
def tgt_batch(model, xs, ys):
    return model.target_encoder(encode_state_batch(torch.tensor(xs), torch.tensor(ys)).to(DEVICE))


def sample_value_batch(batch_size, rng, goal_prob=0.1):
    states, goals = [], []
    for _ in range(batch_size):
        s = random_free_cell(rng)
        g = s if rng.random() < goal_prob else random_free_cell(rng)
        states.append(s)
        goals.append(g)
    return states, goals


def train_value_function_jepa(world_model, seed=SEED, steps=V_STEPS):
    """STAGE A: JEPA representations in, real environment for the backup."""
    torch.manual_seed(seed + 1)
    rng = np.random.default_rng(seed + 1)
    V = ValueNetJEPA(REPR_DIM).to(DEVICE)
    V_target = copy.deepcopy(V)
    for p in V_target.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(V.parameters(), lr=V_LR)

    for step in range(steps):
        states, goals = sample_value_batch(V_BATCH_SIZE, rng)
        sxs = [s[0] for s in states]; sys_ = [s[1] for s in states]
        gxs = [g[0] for g in goals]; gys = [g[1] for g in goals]
        state_repr = ctx_batch(world_model, sxs, sys_)
        goal_repr = tgt_batch(world_model, gxs, gys)
        is_goal = torch.tensor([s == g for s, g in zip(states, goals)], device=DEVICE)

        with torch.no_grad():
            best_next_v = torch.full((V_BATCH_SIZE,), float("inf"), device=DEVICE)
            for a in range(N_ACTIONS):
                next_states = [step_env(s[0], s[1], a) for s in states]  # REAL env (Stage A)
                nxs = [ns[0] for ns in next_states]; nys = [ns[1] for ns in next_states]
                next_repr = ctx_batch(world_model, nxs, nys)
                v_next = V_target(next_repr, goal_repr)
                best_next_v = torch.minimum(best_next_v, v_next)
            target = torch.where(is_goal, torch.zeros_like(best_next_v), 1.0 + best_next_v)

        pred = V(state_repr, goal_repr)
        loss = torch.mean((pred - target) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p_t, p in zip(V_target.parameters(), V.parameters()):
                p_t.data.mul_(V_EMA_DECAY).add_(p.data, alpha=1 - V_EMA_DECAY)
        if (step + 1) % 1000 == 0:
            print(f"  [V-JEPA] step {step+1}/{steps} loss={loss.item():.4f}")
    return V


def plan_stage_a(world_model, V, start, goal, max_steps=MAX_PLAN_STEPS):
    """1-step REAL lookahead (like §78) but scored using JEPA representations."""
    x, y = start
    path = [(x, y)]
    g_repr = tgt(world_model, *goal)
    for _ in range(max_steps):
        if (x, y) == goal:
            return path, True
        best_a, best_v = None, float("inf")
        for a in range(N_ACTIONS):
            nx, ny = step_env(x, y, a)
            next_repr = ctx(world_model, nx, ny)
            v = V(next_repr.unsqueeze(0), g_repr.unsqueeze(0)).item()
            if v < best_v:
                best_v, best_a = v, a
        x, y = step_env(x, y, best_a)
        path.append((x, y))
    return path, (x, y) == goal


def plan_stage_b(world_model, V, start, goal, max_steps=MAX_PLAN_STEPS):
    """STAGE B: 1-step IMAGINED lookahead via the learned dynamics model
    (never calls step_env until the chosen action is actually executed) -
    true pure-latent planning, value-guided."""
    x, y = start
    path = [(x, y)]
    g_repr = tgt(world_model, *goal)
    for _ in range(max_steps):
        if (x, y) == goal:
            return path, True
        cur_repr = ctx(world_model, x, y)
        best_a, best_v = None, float("inf")
        for a in range(N_ACTIONS):
            ao = nn.functional.one_hot(torch.tensor(a), N_ACTIONS).float().to(DEVICE)
            with torch.no_grad():
                pred_repr = world_model.dynamics(cur_repr.unsqueeze(0), ao.unsqueeze(0)).squeeze(0)
                v = V(pred_repr.unsqueeze(0), g_repr.unsqueeze(0)).item()
            if v < best_v:
                best_v, best_a = v, a
        x, y = step_env(x, y, best_a)  # only touches reality to actually move
        path.append((x, y))
    return path, (x, y) == goal


def eval_planner(plan_fn, world_model, V, pairs):
    succ, exc = 0, []
    for s, g, opt in pairs:
        path, reached = plan_fn(world_model, V, s, g)
        if reached:
            succ += 1
            exc.append((len(path) - 1) - opt)
    return succ, len(pairs), exc


if __name__ == "__main__":
    print("=== Training the action-conditioned JEPA world model (walled grid) ===")
    world_model, rng0 = train_world_model(steps=2000)

    print("\n=== STAGE A: value function on JEPA representations, real-env backup ===")
    V = train_value_function_jepa(world_model)

    rng = np.random.default_rng(SEED + 200)
    pairs = []
    for _ in range(N_EVAL_PAIRS):
        s = random_free_cell(rng); g = random_free_cell(rng)
        if s == g:
            continue
        pairs.append((s, g))
    detour, no_detour = [], []
    for s, g in pairs:
        _, greached = greedy_heuristic(s, g)
        (detour if not greached else no_detour).append((s, g, bfs_optimal_distance(s, g)))

    s1, n1, e1 = eval_planner(plan_stage_a, world_model, V, detour)
    s2, n2, e2 = eval_planner(plan_stage_a, world_model, V, no_detour)
    print(f"\nStage A (JEPA repr + real-env lookahead):")
    print(f"  Detour: {s1}/{n1} ({100*s1/n1 if n1 else 0:.1f}%), "
          f"exact-optimal={sum(1 for e in e1 if e==0)/len(e1)*100 if e1 else 0:.1f}%")
    print(f"  No-detour: {s2}/{n2} ({100*s2/n2 if n2 else 0:.1f}%), "
          f"exact-optimal={sum(1 for e in e2 if e==0)/len(e2)*100 if e2 else 0:.1f}%")

    print("\n=== STAGE B: pure-latent planning (dynamics model, no real env until execution) ===")
    s3, n3, e3 = eval_planner(plan_stage_b, world_model, V, detour)
    s4, n4, e4 = eval_planner(plan_stage_b, world_model, V, no_detour)
    print(f"Stage B (JEPA repr + imagined lookahead via dynamics model):")
    print(f"  Detour: {s3}/{n3} ({100*s3/n3 if n3 else 0:.1f}%), "
          f"exact-optimal={sum(1 for e in e3 if e==0)/len(e3)*100 if e3 else 0:.1f}%")
    print(f"  No-detour: {s4}/{n4} ({100*s4/n4 if n4 else 0:.1f}%), "
          f"exact-optimal={sum(1 for e in e4 if e==0)/len(e4)*100 if e4 else 0:.1f}%")

    print(f"\n{'='*70}")
    print("SUMMARY - does the integration hold up vs §78's raw-state value function?")
    print(f"{'='*70}")
    print(f"§78 (raw state, real-env):      detour=89.6%  no-detour=98.3%  (reference)")
    print(f"Stage A (JEPA repr, real-env):   detour={100*s1/n1 if n1 else 0:.1f}%  no-detour={100*s2/n2 if n2 else 0:.1f}%")
    print(f"Stage B (JEPA repr, pure latent): detour={100*s3/n3 if n3 else 0:.1f}%  no-detour={100*s4/n4 if n4 else 0:.1f}%")


def train_value_function_jepa_robust(world_model, seed=SEED, steps=V_STEPS):
    """Fixes §79 Stage B's distribution-shift gap: V was only ever trained
    with REAL-encoded representations as its own (non-target) input, never
    the dynamics model's predicted representations - exactly what Stage B
    decision-time feeds it. For every transition, train V to give the SAME
    value whether it's shown the true-encoded next-state representation OR
    the dynamics model's predicted representation of that same transition -
    both should map to the same true distance-to-go, teaching V to be
    invariant to which "flavor" of representation it's handed."""
    torch.manual_seed(seed + 1)
    rng = np.random.default_rng(seed + 1)
    V = ValueNetJEPA(REPR_DIM).to(DEVICE)
    V_target = copy.deepcopy(V)
    for p in V_target.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(V.parameters(), lr=V_LR)

    for step in range(steps):
        states, goals = sample_value_batch(V_BATCH_SIZE, rng)
        sxs = [s[0] for s in states]; sys_ = [s[1] for s in states]
        gxs = [g[0] for g in goals]; gys = [g[1] for g in goals]
        state_repr = ctx_batch(world_model, sxs, sys_)
        goal_repr = tgt_batch(world_model, gxs, gys)
        is_goal = torch.tensor([s == g for s, g in zip(states, goals)], device=DEVICE)

        # Standard Bellman target (same as Stage A) - value of THIS state.
        with torch.no_grad():
            best_next_v = torch.full((V_BATCH_SIZE,), float("inf"), device=DEVICE)
            for a in range(N_ACTIONS):
                next_states = [step_env(s[0], s[1], a) for s in states]
                nxs = [ns[0] for ns in next_states]; nys = [ns[1] for ns in next_states]
                next_repr = ctx_batch(world_model, nxs, nys)
                v_next = V_target(next_repr, goal_repr)
                best_next_v = torch.minimum(best_next_v, v_next)
            target = torch.where(is_goal, torch.zeros_like(best_next_v), 1.0 + best_next_v)

        pred_real = V(state_repr, goal_repr)
        loss_real = torch.mean((pred_real - target) ** 2)

        # NEW: also train V on the DYNAMICS MODEL's predicted representation
        # of arriving at this same state via a random action from a random
        # predecessor - same target (it's the same true position, just a
        # different representation of it).
        prev_states = [random_free_cell(rng) for _ in range(V_BATCH_SIZE)]
        actions = [int(rng.integers(0, N_ACTIONS)) for _ in range(V_BATCH_SIZE)]
        actual_arrivals = [step_env(ps[0], ps[1], a) for ps, a in zip(prev_states, actions)]
        # Use the SAME (state, goal) target pairing as above by only keeping
        # entries where the dynamics-predicted arrival corresponds to one of
        # this batch's (state) entries - simpler: just recompute a fresh
        # target for whatever position the action actually lands on.
        prev_repr = ctx_batch(world_model, [s[0] for s in prev_states], [s[1] for s in prev_states])
        action_oh = nn.functional.one_hot(torch.tensor(actions), N_ACTIONS).float().to(DEVICE)
        with torch.no_grad():
            predicted_repr = world_model.dynamics(prev_repr, action_oh)
        arrival_is_goal = torch.tensor([a == g for a, g in zip(actual_arrivals, goals)], device=DEVICE)
        with torch.no_grad():
            best_next_v2 = torch.full((V_BATCH_SIZE,), float("inf"), device=DEVICE)
            for a in range(N_ACTIONS):
                nxt = [step_env(ar[0], ar[1], a) for ar in actual_arrivals]
                nxt_repr = ctx_batch(world_model, [n[0] for n in nxt], [n[1] for n in nxt])
                v2 = V_target(nxt_repr, goal_repr)
                best_next_v2 = torch.minimum(best_next_v2, v2)
            target2 = torch.where(arrival_is_goal, torch.zeros_like(best_next_v2), 1.0 + best_next_v2)

        pred_imagined = V(predicted_repr, goal_repr)
        loss_imagined = torch.mean((pred_imagined - target2) ** 2)

        loss = loss_real + loss_imagined
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p_t, p in zip(V_target.parameters(), V.parameters()):
                p_t.data.mul_(V_EMA_DECAY).add_(p.data, alpha=1 - V_EMA_DECAY)
        if (step + 1) % 1000 == 0:
            print(f"  [V-robust] step {step+1}/{steps} loss_real={loss_real.item():.4f} "
                  f"loss_imagined={loss_imagined.item():.4f}")
    return V


if __name__ == "__main__" and "--robust" in __import__("sys").argv:
    print("=== Training the action-conditioned JEPA world model (walled grid) ===")
    world_model, _ = train_world_model(steps=2000)

    print("\n=== Training ROBUST value function (exposed to predicted reprs during training) ===")
    V_robust = train_value_function_jepa_robust(world_model)

    rng = np.random.default_rng(SEED + 200)
    pairs = []
    for _ in range(N_EVAL_PAIRS):
        s = random_free_cell(rng); g = random_free_cell(rng)
        if s == g:
            continue
        pairs.append((s, g))
    detour, no_detour = [], []
    for s, g in pairs:
        _, greached = greedy_heuristic(s, g)
        (detour if not greached else no_detour).append((s, g, bfs_optimal_distance(s, g)))

    print("\n=== Stage B re-test with the robust value function (pure-latent planning) ===")
    s3, n3, e3 = eval_planner(plan_stage_b, world_model, V_robust, detour)
    s4, n4, e4 = eval_planner(plan_stage_b, world_model, V_robust, no_detour)
    print(f"Detour: {s3}/{n3} ({100*s3/n3 if n3 else 0:.1f}%), "
          f"exact-optimal={sum(1 for e in e3 if e==0)/len(e3)*100 if e3 else 0:.1f}%")
    print(f"No-detour: {s4}/{n4} ({100*s4/n4 if n4 else 0:.1f}%), "
          f"exact-optimal={sum(1 for e in e4 if e==0)/len(e4)*100 if e4 else 0:.1f}%")

    print(f"\n{'='*60}")
    print("Before (§79 Stage B, V trained only on real encodings): detour=7.8% no-detour=55.3%")
    print(f"After (robust V, trained on both real+predicted):       detour={100*s3/n3 if n3 else 0:.1f}% no-detour={100*s4/n4 if n4 else 0:.1f}%")
