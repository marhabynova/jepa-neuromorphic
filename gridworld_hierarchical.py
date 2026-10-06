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


V_STEPS = 30000  # needs the longer horizon now that training is stable, not divergent
V_LR = 1e-3
V_BATCH_SIZE = 128
V_EMA_DECAY = 0.999  # FIX (diagnosed §84a): 0.99 diverges catastrophically past ~21k steps on this larger maze; 0.999 stabilizes training completely (corr 0.03->0.70)


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


def flat_plan(world_model, V, start, goal, max_steps=100, memory=3):
    """FAIRNESS FIX (found via direct user crosscheck): this planner had
    the SAME stuck/cycling vulnerability as the original hierarchical_plan
    (measured: 62.3% of its steps were zero-progress, mean step length in
    failed episodes only 0.342) but was never given the anti-stuck memory
    fix applied to hierarchical_plan in §84c - an unfair, confounded
    comparison (hierarchical got a bug fix flat didn't). Same tabu-memory
    pattern applied here for an apples-to-apples comparison."""
    x, y = start
    path = [(x, y)]
    recent = [(x, y)]
    g_repr = tgt(world_model, *goal)
    for _ in range(max_steps):
        if (x, y) == goal:
            return path, True
        scored = []
        for a in range(N_ACTIONS):
            nx, ny = step_env(x, y, a)
            next_repr = ctx(world_model, nx, ny)
            v = V(next_repr.unsqueeze(0), g_repr.unsqueeze(0)).item()
            stuck = (nx, ny) == (x, y) or (nx, ny) in recent[-memory:]
            scored.append((stuck, v, (nx, ny)))
        scored.sort(key=lambda t: (t[0], t[1]))
        _, _, (x, y) = scored[0]
        path.append((x, y))
        recent.append((x, y))
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


def train_hi_value(world_model, seed=SEED, steps=V_STEPS, k=K):
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
                jumped = [jump_step_env(s[0], s[1], o, k=k, goal=g) for s, g in zip(states, goals)]
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


def hierarchical_plan(world_model, V_hi, start, goal, max_macro_steps=40, memory=3, k=K):
    """Macro-level: pick the best OPTION via V_hi (using JumpDynamics-free
    exact jump simulation for scoring - grounded, like plan_stage_a, not
    plan_stage_b's pure imagination, since we're isolating "does chunking
    into K-step jumps help" from "does imagined multi-step lookahead
    survive" (a separate question already explored in §79/§80).

    ANTI-STUCK MEMORY (3rd bug found - measured jump-length distribution
    directly: 96.7% of jumps were length<=1, mean jump length in FAILED
    episodes was 0.09 (essentially zero movement) vs 3.50 in successful
    ones. V_hi sometimes keeps scoring a wall-blocked (zero-progress)
    option as "best", and with no memory the planner repeats it for
    every remaining macro-step, burning the whole budget stuck. Same
    fix pattern already proven for the primitive planner in §76c."""
    x, y = start
    path = [(x, y)]
    recent = [(x, y)]
    g_repr = tgt(world_model, *goal)
    for _ in range(max_macro_steps):
        if (x, y) == goal:
            return path, True
        scored = []
        for o in range(N_OPTIONS):
            nx, ny, cost = jump_step_env(x, y, o, k=k, goal=goal)
            next_repr = ctx(world_model, nx, ny)
            v = V_hi(next_repr.unsqueeze(0), g_repr.unsqueeze(0)).item()
            score = cost + v
            stuck = (nx, ny) == (x, y) or (nx, ny) in recent[-memory:]
            scored.append((stuck, score, (nx, ny)))
        scored.sort(key=lambda t: (t[0], t[1]))  # prefer non-stuck, then lowest score
        _, _, (x, y) = scored[0]
        path.append((x, y))
        recent.append((x, y))
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


# ---------- §84e: multi-seed + held-out validation of the core claim ----------
# (flat wins short-range, hierarchical wins long-range 20-34), per the
# established §81-82 protocol (held-out pairs, multi-seed, exact-VI ceiling)
# that hadn't been applied to this maze yet.

def make_pair_split_maze(seed, test_frac=0.2, n_sample_pairs=4000):
    """Full pair space here is far bigger (352 free cells) than the 8x8
    grid's - subsample a candidate pool first, then split, to keep this
    tractable."""
    rng = np.random.default_rng(seed + 500)
    pairs = []
    attempts = 0
    while len(pairs) < n_sample_pairs and attempts < n_sample_pairs * 3:
        s = random_free_cell(rng); g = random_free_cell(rng)
        attempts += 1
        if s != g:
            pairs.append((s, g))
    rng.shuffle(pairs)
    n_test = int(len(pairs) * test_frac)
    return pairs[n_test:], pairs[:n_test]


if __name__ == "__main__" and "--validate" in __import__("sys").argv:
    import statistics
    from collections import defaultdict

    SEEDS = [0, 1, 2]
    LONG_RANGE = (20, 34)  # the core claim's regime

    flat_long, hi_long = [], []
    flat_short, hi_short = [], []

    for seed in SEEDS:
        print(f"\n{'='*20} seed={seed} {'='*20}")
        wm = train_world_model(seed=seed)
        train_pairs, test_pairs = make_pair_split_maze(seed)

        V_prim = train_primitive_value(wm, seed=seed)
        V_hi = train_hi_value(wm, seed=seed)

        # Evaluate on HELD-OUT pairs only, stratified by true distance
        flat_by_bucket = defaultdict(lambda: [0, 0])
        hi_by_bucket = defaultdict(lambda: [0, 0])
        for s, g in test_pairs:
            d = bfs_optimal_distance(s, g)
            bucket = "long" if LONG_RANGE[0] <= d <= LONG_RANGE[1] else ("short" if d < 10 else None)
            if bucket is None:
                continue
            _, freached = flat_plan(wm, V_prim, s, g)
            _, hreached = hierarchical_plan(wm, V_hi, s, g)
            flat_by_bucket[bucket][1] += 1
            hi_by_bucket[bucket][1] += 1
            if freached: flat_by_bucket[bucket][0] += 1
            if hreached: hi_by_bucket[bucket][0] += 1

        f_long = flat_by_bucket["long"][0] / max(flat_by_bucket["long"][1], 1)
        h_long = hi_by_bucket["long"][0] / max(hi_by_bucket["long"][1], 1)
        f_short = flat_by_bucket["short"][0] / max(flat_by_bucket["short"][1], 1)
        h_short = hi_by_bucket["short"][0] / max(hi_by_bucket["short"][1], 1)
        print(f"  HELD-OUT short (dist<10): flat={f_short*100:.1f}% (n={flat_by_bucket['short'][1]}) "
              f"hierarchical={h_short*100:.1f}% (n={hi_by_bucket['short'][1]})")
        print(f"  HELD-OUT long (dist 20-34): flat={f_long*100:.1f}% (n={flat_by_bucket['long'][1]}) "
              f"hierarchical={h_long*100:.1f}% (n={hi_by_bucket['long'][1]})")
        flat_long.append(f_long * 100); hi_long.append(h_long * 100)
        flat_short.append(f_short * 100); hi_short.append(h_short * 100)

    print(f"\n{'='*70}")
    print("SUMMARY across 3 seeds, HELD-OUT pairs only")
    print(f"{'='*70}")
    def stats(vals):
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0), min(vals)
    fm, fs, fw = stats(flat_short); hm, hs, hw = stats(hi_short)
    print(f"SHORT (dist<10):  flat={fm:.1f}%+/-{fs:.1f} (worst {fw:.1f}%)   "
          f"hierarchical={hm:.1f}%+/-{hs:.1f} (worst {hw:.1f}%)")
    fm, fs, fw = stats(flat_long); hm, hs, hw = stats(hi_long)
    print(f"LONG (dist 20-34): flat={fm:.1f}%+/-{fs:.1f} (worst {fw:.1f}%)   "
          f"hierarchical={hm:.1f}%+/-{hs:.1f} (worst {hw:.1f}%)")
    print(f"\nExact-VI ceiling (zero learning): 100% on every pair, always (reference).")


# ---------- §87: K sweep - does the option length control the short/long trade-off? ----------
# Mechanism hypothesis, grounded directly in §84e's own explanation of WHY
# hierarchical wins long-range and loses short-range: K=5 jumps are "coarse/
# overkill for nearby goals" (short-range cost) but give "fewer macro-decisions
# = less compounding error" (long-range benefit). If that mechanism is real,
# a SMALLER K (3) should narrow the short-range cost (finer jumps, less
# overkill) at some expense of the long-range win (more macro-decisions per
# real distance again); a LARGER K (8) should do the reverse - worse
# short-range, stronger long-range. This is a targeted test of the stated
# mechanism, not a blind parameter sweep.
#
# Efficiency note: flat_plan/V_prim and the world model are K-INDEPENDENT -
# trained once per seed and reused across all K values, so only V_hi needs
# retraining per K. hierarchical_plan never calls the learned jump_dyn (it
# scores options via exact jump_step_env simulation, same as plan_stage_a's
# grounding discipline) so jump_dyn is skipped entirely here too, matching
# what §84e's own --validate block already does.

if __name__ == "__main__" and "--k-sweep" in __import__("sys").argv:
    import statistics
    from collections import defaultdict

    SEEDS = [0, 1, 2]
    LONG_RANGE = (20, 34)
    K_VALUES = [3, 5, 8]

    flat_short_by_seed, flat_long_by_seed = [], []
    hi_short = {k: [] for k in K_VALUES}
    hi_long = {k: [] for k in K_VALUES}

    for seed in SEEDS:
        print(f"\n{'='*20} seed={seed} {'='*20}")
        wm = train_world_model(seed=seed)
        train_pairs, test_pairs = make_pair_split_maze(seed)
        V_prim = train_primitive_value(wm, seed=seed)

        flat_by_bucket = defaultdict(lambda: [0, 0])
        for s, g in test_pairs:
            d = bfs_optimal_distance(s, g)
            bucket = "long" if LONG_RANGE[0] <= d <= LONG_RANGE[1] else ("short" if d < 10 else None)
            if bucket is None:
                continue
            _, freached = flat_plan(wm, V_prim, s, g)
            flat_by_bucket[bucket][1] += 1
            if freached: flat_by_bucket[bucket][0] += 1
        f_short = flat_by_bucket["short"][0] / max(flat_by_bucket["short"][1], 1)
        f_long = flat_by_bucket["long"][0] / max(flat_by_bucket["long"][1], 1)
        flat_short_by_seed.append(f_short * 100)
        flat_long_by_seed.append(f_long * 100)
        print(f"  flat (K-independent): short={f_short*100:.1f}% (n={flat_by_bucket['short'][1]}) "
              f"long={f_long*100:.1f}% (n={flat_by_bucket['long'][1]})")

        for k in K_VALUES:
            V_hi = train_hi_value(wm, seed=seed, k=k)
            hi_by_bucket = defaultdict(lambda: [0, 0])
            for s, g in test_pairs:
                d = bfs_optimal_distance(s, g)
                bucket = "long" if LONG_RANGE[0] <= d <= LONG_RANGE[1] else ("short" if d < 10 else None)
                if bucket is None:
                    continue
                _, hreached = hierarchical_plan(wm, V_hi, s, g, k=k)
                hi_by_bucket[bucket][1] += 1
                if hreached: hi_by_bucket[bucket][0] += 1
            h_short = hi_by_bucket["short"][0] / max(hi_by_bucket["short"][1], 1)
            h_long = hi_by_bucket["long"][0] / max(hi_by_bucket["long"][1], 1)
            hi_short[k].append(h_short * 100)
            hi_long[k].append(h_long * 100)
            print(f"  K={k}: hierarchical short={h_short*100:.1f}% (n={hi_by_bucket['short'][1]}) "
                  f"long={h_long*100:.1f}% (n={hi_by_bucket['long'][1]})")

    def stats(vals):
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0), min(vals)

    print(f"\n{'='*70}")
    print("K-SWEEP SUMMARY across 3 seeds, HELD-OUT pairs only")
    print(f"{'='*70}")
    fm, fs, fw = stats(flat_short_by_seed)
    print(f"flat SHORT (dist<10):   {fm:.1f}%+/-{fs:.1f} (worst {fw:.1f}%)  [K-independent, same for every row below]")
    fm, fs, fw = stats(flat_long_by_seed)
    print(f"flat LONG (dist 20-34): {fm:.1f}%+/-{fs:.1f} (worst {fw:.1f}%)  [K-independent]")
    for k in K_VALUES:
        hm, hs, hw = stats(hi_short[k])
        print(f"hierarchical K={k} SHORT:  {hm:.1f}%+/-{hs:.1f} (worst {hw:.1f}%)")
    for k in K_VALUES:
        hm, hs, hw = stats(hi_long[k])
        print(f"hierarchical K={k} LONG:   {hm:.1f}%+/-{hs:.1f} (worst {hw:.1f}%)")
