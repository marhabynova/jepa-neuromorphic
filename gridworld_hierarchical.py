"""
Next item in the established priority order (§81b): hierarchical/multi-
timescale planning. Per Opus's explicit note, an 8x8 grid is too small to
show any benefit from hierarchy - primitive planning is already near-
ceiling there, so building hierarchy machinery on it would repeat §77's
mistake (building machinery for a problem that doesn't need it).

New environment: 20x20 grid, THREE staggered walls forming a serpentine
maze (not just one wall with one gap) - forces genuinely long detours
(max BFS distance far exceeds Manhattan distance for many pairs), so
there's real room for the §76a compounding-error cliff to show up, and
real room for hierarchy to help.

Design: JumpDynamics predicts the representation after persisting in one
of 4 directions for up to K primitive steps (clipped early by walls) -
an "option" in the classic hierarchical-RL sense. V_hi is a second TD/
Bellman value function over jump-transitions, with step cost = the ACTUAL
number of primitive steps taken (not always K, since walls can block
early). Two-level planning: pick the best option via V_hi, execute it for
real (K real primitive steps or until blocked), re-plan at the macro level
from the new real position - same "ground every macro-step" discipline
that worked better than pure open-loop planning in §76b/c.

Pre-registered prediction (Opus's framing, so a null is readable): the
primitive planner's success-vs-distance cliff (§76a: 100%@dist=1 ->
<15%@dist>=7) should reappear here at a similar ABSOLUTE distance: if the
hierarchical planner is doing its job, ITS cliff should appear at a much
larger distance - roughly K times further out, since it's chaining far
fewer decision points per unit of real distance.
"""
import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from gridworld_jepa import Encoder, DynamicsPredictor, cosine_sim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID_SIZE = 20
STATE_DIM = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4
N_OPTIONS = 4  # same 4 directions, but "persist for up to K steps"
K = 5
HIDDEN_DIM = 96
REPR_DIM = 48
EMA_DECAY = 0.99
LR = 3e-3
BATCH_SIZE = 128
WM_STEPS = 3000
SEED = 0

# Serpentine maze: 3 staggered walls force a long back-and-forth detour.
WALLS = set()
for y in range(0, 16):  # wall 1 at x=6, gap at y=16..19 (bottom-right open)
    WALLS.add((6, y))
for y in range(4, 20):  # wall 2 at x=12, gap at y=0..3 (top-left open)
    WALLS.add((12, y))
for y in range(0, 16):  # wall 3 at x=17, gap at y=16..19 (bottom-right open)
    WALLS.add((17, y))


def is_wall(x, y):
    return (x, y) in WALLS


def step_env(x, y, action):
    nx, ny = x, y
    if action == 0:   ny = max(0, y - 1)
    elif action == 1: ny = min(GRID_SIZE - 1, y + 1)
    elif action == 2: nx = max(0, x - 1)
    elif action == 3: nx = min(GRID_SIZE - 1, x + 1)
    if is_wall(nx, ny):
        return x, y
    return nx, ny


def jump_step_env(x, y, option, k=K, goal=None):
    """Persist in direction `option` for up to k primitive steps, stopping
    early if blocked OR if the goal is reached mid-jump (FIX: the original
    version always jumped the full k unless wall-blocked, systematically
    OVERSHOOTING nearby goals - confirmed directly: jumping 5 steps toward
    a goal 2 steps away landed 3 cells past it). Returns (final_x, final_y,
    actual_steps_taken)."""
    steps = 0
    for _ in range(k):
        if (x, y) == goal:
            break
        nx, ny = step_env(x, y, option)
        if (nx, ny) == (x, y):
            break  # blocked - stop early, don't waste "steps" bumping a wall
        x, y = nx, ny
        steps += 1
    return x, y, max(steps, 1)  # cost at least 1 even if blocked immediately


def random_free_cell(rng):
    while True:
        x, y = int(rng.integers(0, GRID_SIZE)), int(rng.integers(0, GRID_SIZE))
        if not is_wall(x, y):
            return x, y


def bfs_optimal_distance(start, goal):
    if start == goal:
        return 0
    q = deque([(start, 0)])
    seen = {start}
    while q:
        (x, y), d = q.popleft()
        for a in range(N_ACTIONS):
            nx, ny = step_env(x, y, a)
            if (nx, ny) == (x, y):
                continue
            if (nx, ny) == goal:
                return d + 1
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append(((nx, ny), d + 1))
    return None


def encode_state_batch(xs, ys):
    idx = ys * GRID_SIZE + xs
    onehot = torch.zeros(len(xs), STATE_DIM)
    onehot[torch.arange(len(xs)), idx] = 1.0
    return onehot


class ActionConditionedJEPA(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM, repr_dim=REPR_DIM, ema_decay=EMA_DECAY):
        super().__init__()
        self.context_encoder = Encoder(state_dim, hidden_dim, repr_dim)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.dynamics = DynamicsPredictor(repr_dim, action_dim, hidden_dim)
        self.ema_decay = ema_decay

    @torch.no_grad()
    def update_target(self):
        for p_t, p_c in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.mul_(self.ema_decay).add_(p_c.data, alpha=1 - self.ema_decay)

    def forward(self, state, action_onehot, next_state):
        ctx_repr = self.context_encoder(state)
        pred_repr = self.dynamics(ctx_repr, action_onehot)
        with torch.no_grad():
            target_repr = self.target_encoder(next_state)
        loss = torch.mean((pred_repr - target_repr) ** 2)
        return loss, ctx_repr, pred_repr, target_repr


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


def sample_primitive_batch(batch_size, rng):
    xs, ys, actions, nxs, nys = [], [], [], [], []
    for _ in range(batch_size):
        x, y = random_free_cell(rng)
        a = int(rng.integers(0, N_ACTIONS))
        nx, ny = step_env(x, y, a)
        xs.append(x); ys.append(y); actions.append(a); nxs.append(nx); nys.append(ny)
    state = encode_state_batch(torch.tensor(xs), torch.tensor(ys))
    next_state = encode_state_batch(torch.tensor(nxs), torch.tensor(nys))
    action_onehot = nn.functional.one_hot(torch.tensor(actions), N_ACTIONS).float()
    return state, action_onehot, next_state


def train_world_model(seed=SEED, steps=WM_STEPS):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ActionConditionedJEPA(STATE_DIM, N_ACTIONS).to(DEVICE)
    opt = torch.optim.Adam(
        list(model.context_encoder.parameters()) + list(model.dynamics.parameters()), lr=LR
    )
    for step in range(steps):
        state, action, next_state = sample_primitive_batch(BATCH_SIZE, rng)
        state, action, next_state = state.to(DEVICE), action.to(DEVICE), next_state.to(DEVICE)
        loss, _, _, _ = model(state, action, next_state)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()
        if (step + 1) % 1000 == 0:
            print(f"  [wm] step {step+1}/{steps} loss={loss.item():.6f}")
    return model


if __name__ == "__main__":
    print(f"=== {GRID_SIZE}x{GRID_SIZE} serpentine maze, {len(WALLS)} wall cells, K={K} ===")
    print("=== Sanity check: how far apart are cells now? ===")
    rng0 = np.random.default_rng(SEED)
    dists = []
    for _ in range(500):
        s = random_free_cell(rng0); g = random_free_cell(rng0)
        if s == g:
            continue
        dists.append(bfs_optimal_distance(s, g))
    print(f"  BFS distance over 500 random pairs: mean={np.mean(dists):.1f} max={max(dists)} "
          f"(vs 8x8 grid's max ~14-19)")


# ---------- Primitive value function (same TD/Bellman recipe as §78/§80) ----------

class ValueNet(nn.Module):
    def __init__(self, repr_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_repr, goal_repr):
        return self.net(torch.cat([state_repr, goal_repr], dim=-1)).squeeze(-1)


V_STEPS = 6000
V_LR = 1e-3
V_BATCH_SIZE = 128
V_EMA_DECAY = 0.99


def train_primitive_value(world_model, seed=SEED, steps=V_STEPS):
    torch.manual_seed(seed + 1)
    rng = np.random.default_rng(seed + 1)
    V = ValueNet(REPR_DIM).to(DEVICE)
    V_target = copy.deepcopy(V)
    for p in V_target.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(V.parameters(), lr=V_LR)

    for step in range(steps):
        states = [random_free_cell(rng) for _ in range(V_BATCH_SIZE)]
        goals = [s if rng.random() < 0.05 else random_free_cell(rng) for s in states]
        sxs = [s[0] for s in states]; sys_ = [s[1] for s in states]
        gxs = [g[0] for g in goals]; gys = [g[1] for g in goals]
        state_repr = ctx_batch(world_model, sxs, sys_)
        goal_repr = tgt_batch(world_model, gxs, gys)
        is_goal = torch.tensor([s == g for s, g in zip(states, goals)], device=DEVICE)

        with torch.no_grad():
            best_next_v = torch.full((V_BATCH_SIZE,), float("inf"), device=DEVICE)
            for a in range(N_ACTIONS):
                next_states = [step_env(s[0], s[1], a) for s in states]
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
        if (step + 1) % 1500 == 0:
            print(f"  [V-prim] step {step+1}/{steps} loss={loss.item():.4f}")
    return V


def flat_plan(world_model, V, start, goal, max_steps=100):
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


if __name__ == "__main__" and "--flat-baseline" in __import__("sys").argv:
    print("=== Training primitive world model + value function on the serpentine maze ===")
    world_model = train_world_model()
    V = train_primitive_value(world_model)

    print("\n=== Distance-stratified success rate (replicating §76a's cliff check) ===")
    rng = np.random.default_rng(SEED + 42)
    from collections import defaultdict
    by_dist = defaultdict(lambda: [0, 0])
    for ep in range(400):
        s = random_free_cell(rng); g = random_free_cell(rng)
        if s == g:
            continue
        d = bfs_optimal_distance(s, g)
        path, reached = flat_plan(world_model, V, s, g)
        bucket = min(d // 5 * 5, 50)  # bucket into 0-4,5-9,...,50+
        by_dist[bucket][1] += 1
        if reached:
            by_dist[bucket][0] += 1
    print("dist_bucket: success_rate (n)")
    for b in sorted(by_dist):
        s, t = by_dist[b]
        label = f"{b}-{b+4}" if b < 50 else "50+"
        print(f"  {label:>6}: {100*s/t:5.1f}% (n={t})")


# ---------- Hierarchical layer: JumpDynamics (options) + V_hi ----------

class JumpDynamics(nn.Module):
    """Same architecture family as DynamicsPredictor, but conditioned on an
    OPTION (persist in a direction for up to K steps) instead of a single
    primitive action - predicts the representation K (or fewer, if
    blocked) steps ahead."""

    def __init__(self, repr_dim, n_options, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim + n_options, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, repr_dim),
        )

    def forward(self, repr_, option_onehot):
        return self.net(torch.cat([repr_, option_onehot], dim=-1))


def sample_jump_batch(batch_size, rng):
    xs, ys, options, nxs, nys, costs = [], [], [], [], [], []
    for _ in range(batch_size):
        x, y = random_free_cell(rng)
        o = int(rng.integers(0, N_OPTIONS))
        nx, ny, cost = jump_step_env(x, y, o)
        xs.append(x); ys.append(y); options.append(o)
        nxs.append(nx); nys.append(ny); costs.append(cost)
    return xs, ys, options, nxs, nys, costs


def train_jump_dynamics(world_model, seed=SEED, steps=WM_STEPS):
    """Reuses world_model's ALREADY-TRAINED context_encoder/target_encoder
    (frozen here) - only the new JumpDynamics head is trained, against
    real K-step jump transitions."""
    torch.manual_seed(seed + 3)
    rng = np.random.default_rng(seed + 3)
    jump_dyn = JumpDynamics(REPR_DIM, N_OPTIONS, HIDDEN_DIM).to(DEVICE)
    opt = torch.optim.Adam(jump_dyn.parameters(), lr=LR)
    for step in range(steps):
        xs, ys, options, nxs, nys, costs = sample_jump_batch(BATCH_SIZE, rng)
        with torch.no_grad():
            state_repr = ctx_batch(world_model, xs, ys)
            target_repr = tgt_batch(world_model, nxs, nys)
        option_oh = nn.functional.one_hot(torch.tensor(options), N_OPTIONS).float().to(DEVICE)
        pred_repr = jump_dyn(state_repr, option_oh)
        loss = torch.mean((pred_repr - target_repr) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 1000 == 0:
            print(f"  [jump-dyn] step {step+1}/{steps} loss={loss.item():.6f}")
    return jump_dyn


def train_hi_value(world_model, seed=SEED, steps=V_STEPS):
    """TD/Bellman over JUMP transitions - step cost is the ACTUAL number
    of primitive steps the jump covered (not always K, since walls clip
    it early), not a flat 1 like the primitive value function."""
    torch.manual_seed(seed + 4)
    rng = np.random.default_rng(seed + 4)
    V_hi = ValueNet(REPR_DIM).to(DEVICE)
    V_hi_target = copy.deepcopy(V_hi)
    for p in V_hi_target.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(V_hi.parameters(), lr=V_LR)

    for step in range(steps):
        states = [random_free_cell(rng) for _ in range(V_BATCH_SIZE)]
        goals = [s if rng.random() < 0.05 else random_free_cell(rng) for s in states]
        sxs = [s[0] for s in states]; sys_ = [s[1] for s in states]
        gxs = [g[0] for g in goals]; gys = [g[1] for g in goals]
        state_repr = ctx_batch(world_model, sxs, sys_)
        goal_repr = tgt_batch(world_model, gxs, gys)
        is_goal = torch.tensor([s == g for s, g in zip(states, goals)], device=DEVICE)

        with torch.no_grad():
            best_next_v = torch.full((V_BATCH_SIZE,), float("inf"), device=DEVICE)
            for o in range(N_OPTIONS):
                jumped = [jump_step_env(s[0], s[1], o, goal=g) for s, g in zip(states, goals)]
                nxs = [j[0] for j in jumped]; nys = [j[1] for j in jumped]; jcosts = torch.tensor(
                    [float(j[2]) for j in jumped], device=DEVICE)
                next_repr = ctx_batch(world_model, nxs, nys)
                v_next = V_hi_target(next_repr, goal_repr)
                candidate = jcosts + v_next
                best_next_v = torch.minimum(best_next_v, candidate)
            target = torch.where(is_goal, torch.zeros_like(best_next_v), best_next_v)

        pred = V_hi(state_repr, goal_repr)
        loss = torch.mean((pred - target) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p_t, p in zip(V_hi_target.parameters(), V_hi.parameters()):
                p_t.data.mul_(V_EMA_DECAY).add_(p.data, alpha=1 - V_EMA_DECAY)
        if (step + 1) % 1500 == 0:
            print(f"  [V-hi] step {step+1}/{steps} loss={loss.item():.4f}")
    return V_hi


def hierarchical_plan(world_model, V_hi, start, goal, max_macro_steps=40):
    """Macro-level: pick the best OPTION via V_hi (using JumpDynamics-free
    exact jump simulation for scoring - grounded, like plan_stage_a, not
    plan_stage_b's pure imagination, since we're isolating "does chunking
    into K-step jumps help" from "does imagined multi-step lookahead
    survive" (a separate question already explored in §79/§80)."""
    x, y = start
    path = [(x, y)]
    g_repr = tgt(world_model, *goal)
    for _ in range(max_macro_steps):
        if (x, y) == goal:
            return path, True
        best_o, best_score = None, float("inf")
        for o in range(N_OPTIONS):
            nx, ny, cost = jump_step_env(x, y, o, goal=goal)
            next_repr = ctx(world_model, nx, ny)
            v = V_hi(next_repr.unsqueeze(0), g_repr.unsqueeze(0)).item()
            score = cost + v
            if score < best_score:
                best_score, best_o = score, o
        x, y, _ = jump_step_env(x, y, best_o)
        path.append((x, y))
    return path, (x, y) == goal


if __name__ == "__main__" and "--hierarchical" in __import__("sys").argv:
    print("=== Training world model + jump dynamics + V_hi on the serpentine maze ===")
    world_model = train_world_model()
    jump_dyn = train_jump_dynamics(world_model)  # trained for completeness/future pure-latent variant
    V_hi = train_hi_value(world_model)

    print("\n=== Distance-stratified success rate: HIERARCHICAL planner ===")
    rng = np.random.default_rng(SEED + 42)
    from collections import defaultdict
    by_dist = defaultdict(lambda: [0, 0])
    for ep in range(400):
        s = random_free_cell(rng); g = random_free_cell(rng)
        if s == g:
            continue
        d = bfs_optimal_distance(s, g)
        path, reached = hierarchical_plan(world_model, V_hi, s, g)
        bucket = min(d // 5 * 5, 50)
        by_dist[bucket][1] += 1
        if reached:
            by_dist[bucket][0] += 1
    print("dist_bucket: success_rate (n)")
    for b in sorted(by_dist):
        s, t = by_dist[b]
        label = f"{b}-{b+4}" if b < 50 else "50+"
        print(f"  {label:>6}: {100*s/t:5.1f}% (n={t})")

    print(f"\n{'='*60}")
    print("Flat planner (same seed, already measured): see --flat-baseline output")
    print("Prediction: hierarchical cliff should appear much further out (~K=5x) than flat's.")
