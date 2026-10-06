"""
Direct response to a real, correct criticism: §76's GridWorld (no
obstacles) is a problem where "always move toward the goal" is ALWAYS
optimal - verified a trivial, non-learned greedy heuristic gets 98-100%
success with 100% exactly-optimal paths, beating the latent-planning
approach (73% success, 74% optimal) outright. The elaborate JEPA + latent
beam search machinery was solving a problem that needed no machinery at
all, and losing to the one-line solution.

The actual fix is not more tuning on the same environment - it's that the
TEST was wrong. For learned world-model planning to have any reason to
exist, the environment needs cases where naive greedy movement genuinely
fails (walls that must be detoured around), so that search over a learned
model can show real value instead of losing to a trivial heuristic.

New environment: same 8x8 grid, but with a wall segment (x=4, y=0..5) with
a gap only at y=6,7 - crossing from the left half to the right half
requires detouring DOWN to the gap first, which a pure "move toward goal"
greedy policy cannot do (it will walk straight into the wall and get
stuck).
"""
import copy
import numpy as np
import torch
import torch.nn as nn

from collapse_diagnostics import diagnose
from downstream_probe import train_linear_probe
from gridworld_jepa import Encoder, DynamicsPredictor, cosine_sim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID_SIZE = 8
STATE_DIM = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4
HIDDEN_DIM = 64
REPR_DIM = 32
EMA_DECAY = 0.99
STEPS = 2000
LR = 3e-3
BATCH_SIZE = 128
SEED = 0

WALL_X = 4
WALL_Y_RANGE = range(0, 6)  # y=0..5 walled, y=6,7 is the gap
WALLS = {(WALL_X, y) for y in WALL_Y_RANGE}


def is_wall(x, y):
    return (x, y) in WALLS


def step_env(x, y, action):
    nx, ny = x, y
    if action == 0:   ny = max(0, y - 1)
    elif action == 1: ny = min(GRID_SIZE - 1, y + 1)
    elif action == 2: nx = max(0, x - 1)
    elif action == 3: nx = min(GRID_SIZE - 1, x + 1)
    if is_wall(nx, ny):
        return x, y  # blocked - stay put, same as bumping into a real wall
    return nx, ny


def random_free_cell(rng):
    while True:
        x, y = int(rng.integers(0, GRID_SIZE)), int(rng.integers(0, GRID_SIZE))
        if not is_wall(x, y):
            return x, y


def bfs_optimal_distance(start, goal):
    """Ground truth shortest path length respecting walls (BFS) - Manhattan
    distance is NOT valid here since it ignores the wall."""
    from collections import deque
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
    return None  # unreachable (shouldn't happen - grid is connected via the gap)


def encode_state_batch(xs, ys):
    idx = ys * GRID_SIZE + xs
    onehot = torch.zeros(len(xs), STATE_DIM)
    onehot[torch.arange(len(xs)), idx] = 1.0
    return onehot


def sample_batch(batch_size, rng):
    xs, ys, actions, nxs, nys = [], [], [], [], []
    for _ in range(batch_size):
        x, y = random_free_cell(rng)
        a = int(rng.integers(0, N_ACTIONS))
        nx, ny = step_env(x, y, a)
        xs.append(x); ys.append(y); actions.append(a); nxs.append(nx); nys.append(ny)
    state = encode_state_batch(torch.tensor(xs), torch.tensor(ys))
    next_state = encode_state_batch(torch.tensor(nxs), torch.tensor(nys))
    action_onehot = nn.functional.one_hot(torch.tensor(actions), N_ACTIONS).float()
    pos = torch.stack([torch.tensor(xs), torch.tensor(ys)], dim=1)
    return state, action_onehot, next_state, pos


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


def train_model(seed=SEED, steps=STEPS):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ActionConditionedJEPA(STATE_DIM, N_ACTIONS).to(DEVICE)
    opt = torch.optim.Adam(
        list(model.context_encoder.parameters()) + list(model.dynamics.parameters()), lr=LR
    )
    for step in range(steps):
        state, action, next_state, _ = sample_batch(BATCH_SIZE, rng)
        state, action, next_state = state.to(DEVICE), action.to(DEVICE), next_state.to(DEVICE)
        loss, _, _, _ = model(state, action, next_state)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()
        if (step + 1) % 400 == 0:
            print(f"  step {step+1}/{steps} loss={loss.item():.6f}")
    return model, rng


@torch.no_grad()
def encode_pos(model, encoder, x, y):
    state = encode_state_batch(torch.tensor([x]), torch.tensor([y])).to(DEVICE)
    return encoder(state).squeeze(0)


def greedy_heuristic(start, goal, max_steps=40):
    """The trivial non-learned baseline that crushed §76's obstacle-free
    result. Expected to FAIL here whenever it needs to detour around the
    wall, since it only ever moves directly toward the goal."""
    x, y = start
    path = [(x, y)]
    for _ in range(max_steps):
        if (x, y) == goal:
            return path, True
        gx, gy = goal
        if x < gx: a = 3
        elif x > gx: a = 2
        elif y < gy: a = 1
        elif y > gy: a = 0
        else: a = 0
        nx, ny = step_env(x, y, a)
        if (nx, ny) == (x, y):
            # blocked by the wall and already aligned on the other axis -
            # greedy has no way to route around, it's stuck
            break
        x, y = nx, ny
        path.append((x, y))
    return path, (x, y) == goal


@torch.no_grad()
def beam_search_plan(model, start_xy, goal_xy, beam_width=16, max_depth=20):
    goal_repr = encode_pos(model, model.target_encoder, *goal_xy)
    start_repr = encode_pos(model, model.context_encoder, *start_xy)

    def score_of(r):
        return torch.mean((r - goal_repr) ** 2).item()

    beam = [(start_repr, [], score_of(start_repr))]
    best = min(beam, key=lambda b: b[2])
    for _ in range(max_depth):
        if best[2] < 1e-4:
            break
        cands = []
        for repr_, seq, _ in beam:
            for a in range(N_ACTIONS):
                ao = nn.functional.one_hot(torch.tensor(a), N_ACTIONS).float().to(DEVICE)
                nr = model.dynamics(repr_.unsqueeze(0), ao.unsqueeze(0)).squeeze(0)
                cands.append((nr, seq + [a], score_of(nr)))
        cands.sort(key=lambda c: c[2])
        beam = cands[:beam_width]
        if beam[0][2] < best[2]:
            best = beam[0]
    return best[1]


def closed_loop_plan_antistuck(model, start_xy, goal_xy, max_real_steps=40, memory=4):
    x, y = start_xy
    path = [(x, y)]
    recent = [(x, y)]
    for _ in range(max_real_steps):
        if (x, y) == goal_xy:
            return path, True
        goal_repr = encode_pos(model, model.target_encoder, *goal_xy)
        cur_repr = encode_pos(model, model.context_encoder, x, y)
        scored = []
        for a in range(N_ACTIONS):
            nx, ny = step_env(x, y, a)
            ao = nn.functional.one_hot(torch.tensor(a), N_ACTIONS).float().to(DEVICE)
            with torch.no_grad():
                pred = model.dynamics(cur_repr.unsqueeze(0), ao.unsqueeze(0)).squeeze(0)
                score = torch.mean((pred - goal_repr) ** 2).item()
            tabu = (nx, ny) in recent[-memory:]
            scored.append((score, tabu, a, (nx, ny)))
        scored.sort(key=lambda t: (t[1], t[0]))
        _, _, _, (nx, ny) = scored[0]
        x, y = nx, ny
        path.append((x, y))
        recent.append((x, y))
    return path, (x, y) == goal_xy


if __name__ == "__main__":
    print(f"=== Wall: x={WALL_X}, y in {list(WALL_Y_RANGE)} (gap at y=6,7) ===")
    print("=== Training action-conditioned JEPA on the WALLED GridWorld ===")
    model, rng = train_model()

    print("\n=== Eval 0: sanity checks (reuse §75's protocol) ===")
    es, ea, ens, epos = sample_batch(1024, rng)
    with torch.no_grad():
        erepr = model.context_encoder(es.to(DEVICE)).cpu()
    diag = diagnose(erepr, label="Walled GridWorld context_repr")
    cell_idx = epos[:, 1] * GRID_SIZE + epos[:, 0]
    n_train = 768
    probe = train_linear_probe(erepr[:n_train], cell_idx[:n_train], erepr[n_train:], cell_idx[n_train:],
                                num_classes=GRID_SIZE * GRID_SIZE, device=DEVICE, steps=1000)
    print(f"  collapse: {'SUSPECTED' if diag['collapse_suspected'] else 'none'}, "
          f"position probe acc: {probe['test_acc']*100:.1f}%")

    print(f"\n=== Eval 1: TRIVIAL greedy heuristic (the thing that beat us last time) ===")
    N_EVAL = 100
    g_success, g_excess = 0, []
    pairs = []
    for ep in range(N_EVAL):
        s = random_free_cell(rng)
        g = random_free_cell(rng)
        if s == g:
            continue
        pairs.append((s, g))
        opt = bfs_optimal_distance(s, g)
        path, reached = greedy_heuristic(s, g)
        if reached:
            g_success += 1
            g_excess.append((len(path) - 1) - opt)
    print(f"  Greedy: {g_success}/{len(pairs)} success "
          f"({100*g_success/len(pairs):.1f}%)")
    if g_excess:
        print(f"  mean excess (successes only): {sum(g_excess)/len(g_excess):+.2f}, "
              f"exact-optimal: {sum(1 for e in g_excess if e==0)/len(g_excess)*100:.1f}%")

    print(f"\n=== Eval 2: latent closed-loop + anti-cycling planning (§76c's best variant) ===")
    p_success, p_excess = 0, []
    for s, g in pairs:
        opt = bfs_optimal_distance(s, g)
        path, reached = closed_loop_plan_antistuck(model, s, g)
        if reached:
            p_success += 1
            p_excess.append((len(path) - 1) - opt)
    print(f"  Latent planner: {p_success}/{len(pairs)} success "
          f"({100*p_success/len(pairs):.1f}%)")
    if p_excess:
        print(f"  mean excess (successes only): {sum(p_excess)/len(p_excess):+.2f}, "
              f"exact-optimal: {sum(1 for e in p_excess if e==0)/len(p_excess)*100:.1f}%")

    print(f"\n{'='*60}")
    print("SUMMARY - does latent planning earn its complexity now?")
    print(f"{'='*60}")
    print(f"Trivial greedy:  {g_success}/{len(pairs)} success, "
          f"mean excess {sum(g_excess)/len(g_excess) if g_excess else float('nan'):+.2f}")
    print(f"Latent planner:  {p_success}/{len(pairs)} success, "
          f"mean excess {sum(p_excess)/len(p_excess) if p_excess else float('nan'):+.2f}")


def closed_loop_beam_plan(model, start_xy, goal_xy, max_real_steps=40, beam_width=16, max_depth=20, memory=4):
    """The actual fix: closed_loop_plan_antistuck only did 1-step lookahead
    at decision time (greedy + tabu memory), which can NEVER discover a
    detour (needs multi-step reasoning to see "move away now pays off in
    3 steps"). This replans via the FULL multi-step beam_search_plan from
    the true position every real step - multi-step lookahead (can find
    detours) + replanning (bounds compounding error) + tabu memory (avoids
    cycling) all together."""
    x, y = start_xy
    path = [(x, y)]
    recent = [(x, y)]
    for _ in range(max_real_steps):
        if (x, y) == goal_xy:
            return path, True
        plan = beam_search_plan(model, (x, y), goal_xy, beam_width=beam_width, max_depth=max_depth)
        if not plan:
            break
        # Prefer the plan's first action, UNLESS it leads straight back to
        # a recently-visited cell (tabu) - then fall back to the best
        # non-tabu action among all 4, scored by the dynamics model's
        # single-step prediction (same anti-cycling idea as §76c).
        nx, ny = step_env(x, y, plan[0])
        if (nx, ny) in recent[-memory:]:
            goal_repr = encode_pos(model, model.target_encoder, *goal_xy)
            cur_repr = encode_pos(model, model.context_encoder, x, y)
            scored = []
            for a in range(N_ACTIONS):
                anx, any_ = step_env(x, y, a)
                ao = nn.functional.one_hot(torch.tensor(a), N_ACTIONS).float().to(DEVICE)
                with torch.no_grad():
                    pred = model.dynamics(cur_repr.unsqueeze(0), ao.unsqueeze(0)).squeeze(0)
                    score = torch.mean((pred - goal_repr) ** 2).item()
                tabu = (anx, any_) in recent[-memory:]
                scored.append((score, tabu, (anx, any_)))
            scored.sort(key=lambda t: (t[1], t[0]))
            nx, ny = scored[0][2]
        x, y = nx, ny
        path.append((x, y))
        recent.append((x, y))
    return path, (x, y) == goal_xy


if __name__ == "__main__" and "--beam-closed-loop" in __import__("sys").argv:
    print("=== Testing closed_loop_beam_plan (multi-step lookahead + replanning + tabu) ===")
    model2, rng2 = train_model(steps=2000)
    pairs2 = []
    for ep in range(150):
        s = random_free_cell(rng2); g = random_free_cell(rng2)
        if s == g: continue
        pairs2.append((s, g))

    detour2, no_detour2 = [], []
    for s, g in pairs2:
        opt = bfs_optimal_distance(s, g)
        _, greached = greedy_heuristic(s, g)
        (detour2 if not greached else no_detour2).append((s, g, opt))

    def eval_beam(subset):
        succ, exc = 0, []
        for s, g, opt in subset:
            path, reached = closed_loop_beam_plan(model2, s, g)
            if reached:
                succ += 1
                exc.append((len(path) - 1) - opt)
        return succ, len(subset), exc

    s1, n1, e1 = eval_beam(detour2)
    s2, n2, e2 = eval_beam(no_detour2)
    print(f"\nDetour-needed cases (greedy fails here, n={n1}): {s1}/{n1} ({100*s1/n1 if n1 else 0:.1f}%)")
    if e1: print(f"  mean excess: {sum(e1)/len(e1):+.2f}")
    print(f"No-detour cases (n={n2}): {s2}/{n2} ({100*s2/n2 if n2 else 0:.1f}%)")
    if e2: print(f"  mean excess: {sum(e2)/len(e2):+.2f}")
