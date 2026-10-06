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
