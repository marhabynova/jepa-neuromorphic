"""
§77's root-cause diagnosis: beam search's scoring heuristic (raw MSE
distance in JEPA representation space to the goal's encoding) has ~zero
correlation with true wall-respecting path distance (r=-0.17) - for hard
detour cases it actually scores them as EASIER than simple cases. No
amount of search sophistication (beam width, depth, replanning, tabu
memory) can fix a heuristic that points in essentially random directions,
because the JEPA's training objective (predict next-state representation)
never asked embedding-space distance to track task-relevant distance.

This trains a SEPARATE value function V(state, goal) -> estimated steps-
to-go, via TD-learning / Bellman backup (the same family of technique
MuZero's value network uses, and classic Q-learning/value-iteration):

    V(s, g) = 0                                   if s == g
    V(s, g) = 1 + min_a V(next_state(s,a), g)      otherwise

Trained with a slow EMA target network for V (same stabilization pattern
as dense_jepa.py's/gridworld_jepa.py's target_encoder - already
established in this project, not a new idea here).

Scope decision, stated explicitly: the Bellman backup's next_state(s,a)
uses the REAL environment (step_env), not the learned dynamics model. This
isolates "does value learning itself work" from "does compounding
dynamics-model error corrupt it" - gridworld_jepa.py already separately
verified the dynamics model is accurate (cosine-sim 0.95), so conflating
the two questions here would make a negative result ambiguous about which
component failed.
"""
import copy
import torch
import torch.nn as nn
import numpy as np

from gridworld_obstacles import (
    GRID_SIZE, N_ACTIONS, STATE_DIM, WALLS, step_env, is_wall, random_free_cell,
    bfs_optimal_distance, encode_state_batch, greedy_heuristic, train_model as train_world_model,
    encode_pos, DEVICE, SEED,
)

HIDDEN_DIM = 64
V_EMA_DECAY = 0.99
V_STEPS = 4000
V_LR = 1e-3
V_BATCH_SIZE = 128
MAX_PLAN_STEPS = 40


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_onehot, goal_onehot):
        return self.net(torch.cat([state_onehot, goal_onehot], dim=-1)).squeeze(-1)


def sample_value_batch(batch_size, rng, goal_prob=0.1):
    """Mix in some state==goal pairs explicitly so the V=0 base case gets
    direct supervision, not just indirect pressure through bootstrapping."""
    states, goals = [], []
    for _ in range(batch_size):
        s = random_free_cell(rng)
        if rng.random() < goal_prob:
            g = s
        else:
            g = random_free_cell(rng)
        states.append(s)
        goals.append(g)
    return states, goals


def train_value_function(seed=SEED, steps=V_STEPS):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)  # offset from the world model's seed
    V = ValueNet(STATE_DIM).to(DEVICE)
    V_target = copy.deepcopy(V)
    for p in V_target.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(V.parameters(), lr=V_LR)

    for step in range(steps):
        states, goals = sample_value_batch(V_BATCH_SIZE, rng)
        state_xs = torch.tensor([s[0] for s in states])
        state_ys = torch.tensor([s[1] for s in states])
        goal_xs = torch.tensor([g[0] for g in goals])
        goal_ys = torch.tensor([g[1] for g in goals])
        state_oh = encode_state_batch(state_xs, state_ys).to(DEVICE)
        goal_oh = encode_state_batch(goal_xs, goal_ys).to(DEVICE)

        is_goal = torch.tensor([s == g for s, g in zip(states, goals)], device=DEVICE)

        # Bellman backup target: 0 if already at goal, else 1 + min_a V_target(next, goal)
        with torch.no_grad():
            best_next_v = torch.full((V_BATCH_SIZE,), float("inf"), device=DEVICE)
            for a in range(N_ACTIONS):
                next_states = [step_env(s[0], s[1], a) for s in states]
                nx = torch.tensor([ns[0] for ns in next_states])
                ny = torch.tensor([ns[1] for ns in next_states])
                next_oh = encode_state_batch(nx, ny).to(DEVICE)
                v_next = V_target(next_oh, goal_oh)
                best_next_v = torch.minimum(best_next_v, v_next)
            target = torch.where(is_goal, torch.zeros_like(best_next_v), 1.0 + best_next_v)

        pred = V(state_oh, goal_oh)
        loss = torch.mean((pred - target) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            for p_t, p in zip(V_target.parameters(), V.parameters()):
                p_t.data.mul_(V_EMA_DECAY).add_(p.data, alpha=1 - V_EMA_DECAY)

        if (step + 1) % 500 == 0:
            print(f"  step {step+1}/{steps} loss={loss.item():.4f}")

    return V


def value_greedy_plan(V, start, goal, max_steps=MAX_PLAN_STEPS):
    """1-step REAL lookahead (cheap, exact - uses step_env directly, not a
    learned dynamics model) + the trained value function for everything
    beyond that. This is the point of a good value function: it makes even
    greedy action selection far-sighted, because V already encodes the
    full future cost-to-go via the Bellman backup - no explicit multi-step
    search needed, unlike the raw-latent-MSE heuristic that required (and
    still failed with) beam search."""
    x, y = start
    path = [(x, y)]
    gx, gy = goal
    goal_oh = encode_state_batch(torch.tensor([gx]), torch.tensor([gy])).to(DEVICE)
    for _ in range(max_steps):
        if (x, y) == goal:
            return path, True
        best_a, best_v = None, float("inf")
        for a in range(N_ACTIONS):
            nx, ny = step_env(x, y, a)
            next_oh = encode_state_batch(torch.tensor([nx]), torch.tensor([ny])).to(DEVICE)
            with torch.no_grad():
                v = V(next_oh, goal_oh).item()
            if v < best_v:
                best_v, best_a = v, a
        x, y = step_env(x, y, best_a)
        path.append((x, y))
    return path, (x, y) == goal


if __name__ == "__main__":
    print("=== Training value function V(state, goal) via TD/Bellman backup ===")
    V = train_value_function()

    print("\n=== Sanity check: does V now track TRUE graph distance (not raw Manhattan)? ===")
    rng = np.random.default_rng(SEED + 2)
    pairs = []
    for _ in range(200):
        s = random_free_cell(rng)
        g = random_free_cell(rng)
        if s == g:
            continue
        pairs.append((s, g))

    v_preds, true_dists, raw_dists = [], [], []
    for s, g in pairs:
        sx, sy = s
        gx, gy = g
        s_oh = encode_state_batch(torch.tensor([sx]), torch.tensor([sy])).to(DEVICE)
        g_oh = encode_state_batch(torch.tensor([gx]), torch.tensor([gy])).to(DEVICE)
        with torch.no_grad():
            v_preds.append(V(s_oh, g_oh).item())
        true_dists.append(bfs_optimal_distance(s, g))
        raw_dists.append(abs(sx - gx) + abs(sy - gy))

    v_preds, true_dists, raw_dists = np.array(v_preds), np.array(true_dists), np.array(raw_dists)
    corr_true = np.corrcoef(v_preds, true_dists)[0, 1]
    corr_raw = np.corrcoef(v_preds, raw_dists)[0, 1]
    print(f"  V(s,g) vs TRUE graph distance correlation: {corr_true:.4f} (was -0.17 for raw latent MSE)")
    print(f"  V(s,g) vs raw Manhattan (ignores wall) correlation: {corr_raw:.4f}")
    print(f"  mean |V - true_dist|: {np.mean(np.abs(v_preds - true_dists)):.2f} steps")

    print(f"\n=== Eval: value-guided planner vs trivial greedy, detour vs no-detour ===")
    detour, no_detour = [], []
    for s, g in pairs:
        _, greached = greedy_heuristic(s, g)
        (detour if not greached else no_detour).append((s, g, bfs_optimal_distance(s, g)))

    def eval_value_planner(subset):
        succ, exc = 0, []
        for s, g, opt in subset:
            path, reached = value_greedy_plan(V, s, g)
            if reached:
                succ += 1
                exc.append((len(path) - 1) - opt)
        return succ, len(subset), exc

    s1, n1, e1 = eval_value_planner(detour)
    s2, n2, e2 = eval_value_planner(no_detour)
    print(f"\nDetour-needed cases (trivial greedy FAILS here, n={n1}): {s1}/{n1} ({100*s1/n1 if n1 else 0:.1f}%)")
    if e1:
        print(f"  mean excess: {sum(e1)/len(e1):+.2f}, exact-optimal: {sum(1 for e in e1 if e==0)/len(e1)*100:.1f}%")
    print(f"No-detour cases (n={n2}): {s2}/{n2} ({100*s2/n2 if n2 else 0:.1f}%)")
    if e2:
        print(f"  mean excess: {sum(e2)/len(e2):+.2f}, exact-optimal: {sum(1 for e in e2 if e==0)/len(e2)*100:.1f}%")

    print(f"\n{'='*60}")
    print("Does a PROPERLY-TRAINED value function fix what raw latent-MSE search couldn't?")
    print(f"{'='*60}")
    print(f"Trivial greedy:     detour={0}/{n1} (0% by construction), no-detour=~100%")
    print(f"§77 beam search:    detour=3/52 (5.8%)")
    print(f"Value-guided (new): detour={s1}/{n1} ({100*s1/n1 if n1 else 0:.1f}%)")
