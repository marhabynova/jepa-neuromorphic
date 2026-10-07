"""
§85d item #4b: a GENUINE 3rd hierarchy level, grounded by §90's foundation
check (the compounding-error cliff reappears for K=2-hierarchical at dist
60-72, pushed only ~1.5x further than flat_plan's own cliff at dist 40-49 -
real, measured room for a 3rd level to push further still).

NOT "bigger K" - §87a already proved that just degrades V_hi quality
without helping. This is a qualitatively different abstraction: the maze is
partitioned into coarse REGIONS, and a 3rd-level planner picks which
ADJACENT region to move toward next, hands off to the already-validated
K=2 level-1 planner to actually get there (for real, grounded - same
"replan from the real position" discipline as every other level in this
project, §76b/c's precedent), then replans at the region level from the
new real position.

DESIGN SIMPLIFICATION, stated explicitly rather than hidden: level-1/2's
V functions are trained with LIVE TD updates (re-sampling/re-executing the
cheap primitive/jump environment every gradient step, tractable because
each sample is 1 env step or 1 K-step jump). A region-transition "sample"
here means running the FULL level-1 planner (~15-20 macro-steps, each
costing 4 V_hi forward passes) - far too expensive to re-execute every
gradient step at V_STEPS=30000 scale. Instead: collect a FIXED dataset of
real, grounded region-transition rollouts ONCE (`collect_region_transitions`),
then train V_level3 via ordinary TD/Bellman replay over that fixed dataset
for many epochs. The data is still 100% real/grounded (no imagination) -
only the TRAINING LOOP's relationship to the environment changes (offline
replay vs. live online sampling), not the groundedness of the data itself.
"""
import copy
import statistics
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn

import gridworld_hierarchical as gh

DEVICE = gh.DEVICE
REGION_SIZE = 5
N_REGIONS_PER_SIDE = gh.GRID_SIZE // REGION_SIZE  # 4
MAX_L1_MACROSTEPS_PER_REGION_HOP = 20
N_COLLECT_PER_STARTCELL = 3  # how many different real start cells sampled per region, per neighbor direction
N_STARTCELLS_PER_REGION = 4


def region_of(x, y):
    return (x // REGION_SIZE, y // REGION_SIZE)


def region_cells(rx, ry):
    x0, y0 = rx * REGION_SIZE, ry * REGION_SIZE
    return [(x, y) for x in range(x0, x0 + REGION_SIZE) for y in range(y0, y0 + REGION_SIZE)
            if not gh.is_wall(x, y)]


def region_anchor(rx, ry):
    """Free cell nearest the region's geometric center - the canonical
    reference point used as this region's 'goal' for V_hi lookahead and as
    the representation fed to V_level3."""
    cx, cy = rx * REGION_SIZE + REGION_SIZE / 2, ry * REGION_SIZE + REGION_SIZE / 2
    cells = region_cells(rx, ry)
    return min(cells, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)


ALL_REGIONS = [(rx, ry) for rx in range(N_REGIONS_PER_SIDE) for ry in range(N_REGIONS_PER_SIDE)
               if region_cells(rx, ry)]
REGION_ANCHOR = {r: region_anchor(*r) for r in ALL_REGIONS}


def adjacent_regions(r):
    rx, ry = r
    cands = [(rx + 1, ry), (rx - 1, ry), (rx, ry + 1), (rx, ry - 1)]
    return [c for c in cands if c in REGION_ANCHOR]


def bfs_region_distance(r1, r2):
    """Exact region-graph distance (BFS over the ~16-node region adjacency
    graph) - NOT used for planning (that stays learned, consistent with the
    rest of this project's representation-learning narrative), only as a
    cheap sanity-check reference, same spirit as exact_vi_distance serving
    as a non-learned ceiling elsewhere."""
    if r1 == r2:
        return 0
    q = deque([(r1, 0)])
    seen = {r1}
    while q:
        cur, d = q.popleft()
        for nxt in adjacent_regions(cur):
            if nxt == r2:
                return d + 1
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, d + 1))
    return None


@torch.no_grad()
def hierarchical_plan_to_region(world_model, V_hi, start, target_region, k=2, max_macro_steps=MAX_L1_MACROSTEPS_PER_REGION_HOP, memory=3):
    """Level-1 (K=2) planner, STOPPING EARLY as soon as the current region
    equals target_region (not when a specific cell is reached) - the
    region-hop only needs to cross the border, not reach the anchor
    exactly. Uses the target region's anchor as the pseudo-goal for V_hi's
    lookahead scoring (consistent with how V_hi was trained: it estimates
    cost-to-a-specific-cell, and the anchor is the best available proxy for
    'somewhere in that region'). Returns (path, success, actual_steps)."""
    x, y = start
    path = [(x, y)]
    recent = [(x, y)]
    goal = REGION_ANCHOR[target_region]
    g_repr = gh.tgt(world_model, *goal)
    if region_of(x, y) == target_region:
        return path, True, 0
    for step in range(max_macro_steps):
        scored = []
        for o in range(gh.N_OPTIONS):
            nx, ny, cost = gh.jump_step_env(x, y, o, k=k, goal=goal)
            next_repr = gh.ctx(world_model, nx, ny)
            v = V_hi(next_repr.unsqueeze(0), g_repr.unsqueeze(0)).item()
            score = cost + v
            stuck = (nx, ny) == (x, y) or (nx, ny) in recent[-memory:]
            scored.append((stuck, score, (nx, ny)))
        scored.sort(key=lambda t: (t[0], t[1]))
        _, _, (x, y) = scored[0]
        path.append((x, y))
        recent.append((x, y))
        if region_of(x, y) == target_region:
            return path, True, step + 1
    return path, False, max_macro_steps


def collect_region_transitions(world_model, V_hi, seed):
    """ONE-TIME collection of real, grounded region-hop rollouts (not
    re-executed during V_level3 training - see module docstring). For every
    region and every adjacent region, sample several different real start
    cells within the source region and execute the level-1 planner toward
    the neighbor, recording the TRUE cost (actual level-1 macro-steps used)
    and whether it succeeded within budget."""
    rng = np.random.default_rng(seed + 10)
    dataset = []  # list of dicts: start, target_region, arrival_cell, cost, success
    for r in ALL_REGIONS:
        cells = region_cells(*r)
        for nbr in adjacent_regions(r):
            n_starts = min(N_STARTCELLS_PER_REGION, len(cells))
            start_cells = [cells[i] for i in rng.choice(len(cells), size=n_starts, replace=False)]
            for start in start_cells:
                for _ in range(N_COLLECT_PER_STARTCELL):
                    path, success, cost = hierarchical_plan_to_region(world_model, V_hi, start, nbr)
                    arrival = path[-1]
                    dataset.append({
                        "start": start, "source_region": r, "target_region": nbr,
                        "arrival": arrival, "arrival_region": region_of(*arrival),
                        "cost": cost, "success": success,
                    })
    n_success = sum(1 for d in dataset if d["success"])
    print(f"  [collect] {len(dataset)} real region-hop rollouts, {n_success}/{len(dataset)} reached target region "
          f"within {MAX_L1_MACROSTEPS_PER_REGION_HOP} macro-steps")
    return dataset


class RegionValueNet(nn.Module):
    def __init__(self, repr_dim, hidden_dim=gh.HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_repr, goal_repr):
        return self.net(torch.cat([state_repr, goal_repr], dim=-1)).squeeze(-1)


L3_TD_STEPS = 30000  # was 8000 - §91a found this undertrained per §86's exact EMA_DECAY/step-budget pairing lesson
L3_LR = 1e-3
L3_BATCH = 32
L3_EMA_DECAY = 0.999  # §86's lesson applied from the start here, not discovered the hard way again


DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def train_region_value(world_model, dataset, seed, steps=L3_TD_STEPS):
    """TD/Bellman training over the FIXED dataset. BUG FOUND AND FIXED
    DURING THIS SESSION before this version: an earlier draft regressed
    directly onto whichever single random dataset row got sampled, with NO
    min-over-available-options step - meaning different hops to different
    neighbors from the SAME source region were treated as if each were the
    only option, producing contradictory targets (loss visibly diverged,
    43->132 over 8000 steps). Fixed to match train_hi_value's actual
    pattern: for each sampled (source_region, goal_region), take the MIN
    over all adjacent regions' (real cost + bootstrapped value), same
    Bellman structure as every other level in this project - just with the
    real cost/arrival looked up from the precollected dataset instead of a
    live jump_step_env call."""
    torch.manual_seed(seed + 20)
    rng = np.random.default_rng(seed + 20)
    V3 = RegionValueNet(gh.REPR_DIM).to(DEVICE)
    V3_target = copy.deepcopy(V3)
    for p in V3_target.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(V3.parameters(), lr=L3_LR)

    with torch.no_grad():
        region_repr = {r: gh.ctx(world_model, *REGION_ANCHOR[r]) for r in ALL_REGIONS}
        region_tgt_repr = {r: gh.tgt(world_model, *REGION_ANCHOR[r]) for r in ALL_REGIONS}

    grouped = defaultdict(lambda: defaultdict(list))
    for d in dataset:
        grouped[d["source_region"]][d["target_region"]].append((d["cost"], d["arrival_region"]))

    for step in range(steps):
        source_regions = [ALL_REGIONS[i] for i in rng.integers(0, len(ALL_REGIONS), size=L3_BATCH)]
        goal_regions = [ALL_REGIONS[i] for i in rng.integers(0, len(ALL_REGIONS), size=L3_BATCH)]
        state_repr = torch.stack([region_repr[r] for r in source_regions])
        goal_repr = torch.stack([region_tgt_repr[g] for g in goal_regions])
        pred = V3(state_repr, goal_repr)

        with torch.no_grad():
            best_next_v = torch.full((L3_BATCH,), float("inf"), device=DEVICE)
            is_same = torch.tensor([r == g for r, g in zip(source_regions, goal_regions)], device=DEVICE)
            for dx, dy in DIRECTIONS:
                valid, costs, arrival_reprs, arrived_at_goal = [], [], [], []
                for b in range(L3_BATCH):
                    r = source_regions[b]
                    nbr = (r[0] + dx, r[1] + dy)
                    samples = grouped[r].get(nbr) if nbr in REGION_ANCHOR else None
                    if samples:
                        cost, arrival_region = samples[rng.integers(0, len(samples))]
                        valid.append(True)
                        costs.append(float(cost))
                        arrival_reprs.append(region_repr[arrival_region])
                        arrived_at_goal.append(arrival_region == goal_regions[b])
                    else:
                        valid.append(False)
                        costs.append(0.0)
                        arrival_reprs.append(region_repr[r])  # dummy, masked out below
                        arrived_at_goal.append(False)
                valid_t = torch.tensor(valid, device=DEVICE)
                costs_t = torch.tensor(costs, device=DEVICE)
                arrived_t = torch.tensor(arrived_at_goal, device=DEVICE)
                arrival_repr_t = torch.stack(arrival_reprs)
                v_arrival = V3_target(arrival_repr_t, goal_repr)
                v_arrival = torch.where(arrived_t, torch.zeros_like(v_arrival), v_arrival)
                candidate = costs_t + v_arrival
                candidate = torch.where(valid_t, candidate, torch.full_like(candidate, float("inf")))
                best_next_v = torch.minimum(best_next_v, candidate)
            best_next_v = torch.where(torch.isinf(best_next_v), torch.zeros_like(best_next_v), best_next_v)
            target = torch.where(is_same, torch.zeros_like(best_next_v), best_next_v)

        loss = torch.mean((pred - target) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p_t, p in zip(V3_target.parameters(), V3.parameters()):
                p_t.data.mul_(L3_EMA_DECAY).add_(p.data, alpha=1 - L3_EMA_DECAY)
        if (step + 1) % 1000 == 0:
            print(f"  [V-level3] step {step+1}/{steps} loss={loss.item():.4f}")
    return V3, region_repr, region_tgt_repr


def plan_level3(world_model, V_hi, V3, region_repr, region_tgt_repr, start, goal, max_region_hops=12, memory=3):
    """Top-level planner. Same region reached as the goal's -> delegate
    directly to level-1 (K=2), no abstraction overhead. Otherwise pick the
    best adjacent region via cost+V (same pattern as every lower level in
    this project), execute level-1 for REAL toward it, replan from the real
    resulting position."""
    x, y = start
    goal_region = region_of(*goal)
    full_path = [(x, y)]
    recent_regions = [region_of(x, y)]
    for _ in range(max_region_hops):
        cur_region = region_of(x, y)
        if cur_region == goal_region:
            sub_path, reached = gh.hierarchical_plan(world_model, V_hi, (x, y), goal, k=2)
            full_path.extend(sub_path[1:])
            return full_path, reached
        goal_r_repr = region_tgt_repr[goal_region]
        scored = []
        for nbr in adjacent_regions(cur_region):
            # Estimate hop cost via a fresh real execution - this IS the actual plan step,
            # not a training sample, so grounding it for real here is the whole point.
            sub_path, hop_success, hop_cost = hierarchical_plan_to_region(world_model, V_hi, (x, y), nbr)
            v = V3(region_repr[nbr].unsqueeze(0), goal_r_repr.unsqueeze(0)).item()
            score = hop_cost + v
            stuck = not hop_success or nbr in recent_regions[-memory:]
            scored.append((stuck, score, nbr, sub_path, hop_success))
        scored.sort(key=lambda t: (t[0], t[1]))
        _, _, chosen_nbr, sub_path, hop_success = scored[0]
        full_path.extend(sub_path[1:])
        x, y = sub_path[-1]
        recent_regions.append(region_of(x, y))
    reached_goal = (x, y) == goal
    return full_path, reached_goal


BUCKETS = [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 72)]


def bucket_of(d):
    for lo, hi in BUCKETS:
        if lo <= d <= hi:
            return (lo, hi)
    return None


def run_one_seed(seed, l3_steps=L3_TD_STEPS):
    wm = gh.train_world_model(seed=seed)
    V_hi = gh.train_hi_value(wm, seed=seed, k=2)
    V_prim = gh.train_primitive_value(wm, seed=seed)
    dataset = collect_region_transitions(wm, V_hi, seed)
    V3, region_repr, region_tgt_repr = train_region_value(wm, dataset, seed, steps=l3_steps)

    train_pairs, test_pairs = gh.make_pair_split_maze(seed, n_sample_pairs=6000)
    counts = {b: {"flat": [0, 0], "l2": [0, 0], "l3": [0, 0]} for b in BUCKETS}
    for s, g in test_pairs:
        d = gh.bfs_optimal_distance(s, g)
        if d is None:
            continue
        b = bucket_of(d)
        if b is None:
            continue
        # FAIRNESS CHECK (confirmed harmless in the single-seed pass, kept
        # here anyway so the 3-seed numbers are never exposed to the §84d-
        # style budget confound): match flat/L2's step budget to L3's own
        # worst-case ceiling (max_region_hops=12 * 20 macrosteps * k=2 = 480).
        _, freached = gh.flat_plan(wm, V_prim, s, g, max_steps=480)
        _, hreached = gh.hierarchical_plan(wm, V_hi, s, g, k=2, max_macro_steps=240)
        _, l3reached = plan_level3(wm, V_hi, V3, region_repr, region_tgt_repr, s, g)
        counts[b]["flat"][1] += 1; counts[b]["l2"][1] += 1; counts[b]["l3"][1] += 1
        if freached: counts[b]["flat"][0] += 1
        if hreached: counts[b]["l2"][0] += 1
        if l3reached: counts[b]["l3"][0] += 1
    return counts


if __name__ == "__main__" and "--3seed" not in __import__("sys").argv:
    print(f"=== {N_REGIONS_PER_SIDE}x{N_REGIONS_PER_SIDE} region partition ({len(ALL_REGIONS)} non-empty regions, "
          f"{REGION_SIZE}x{REGION_SIZE} cells each) ===")
    seed = 0
    print(f"\n{'='*20} PROOF OF CONCEPT, seed={seed} {'='*20}")
    counts = run_one_seed(seed)
    print(f"\n{'='*70}")
    print(f"PROOF-OF-CONCEPT RESULT (single seed={seed}, held-out pairs, fair budgets)")
    print(f"{'='*70}")
    for b in BUCKETS:
        n = counts[b]["flat"][1]
        if n == 0:
            continue
        f = counts[b]["flat"][0] / n * 100
        l2 = counts[b]["l2"][0] / n * 100
        l3 = counts[b]["l3"][0] / n * 100
        print(f"dist {b[0]:>2}-{b[1]:<2} (n={n:>4}): flat={f:5.1f}%  K=2-L2={l2:5.1f}%  L3={l3:5.1f}%")


if __name__ == "__main__" and "--3seed" in __import__("sys").argv:
    import statistics as st
    SEEDS = [0, 1, 2]
    all_results = {b: {"flat": [], "l2": [], "l3": []} for b in BUCKETS}
    for seed in SEEDS:
        print(f"\n{'='*20} seed={seed} {'='*20}")
        counts = run_one_seed(seed)
        for b in BUCKETS:
            n = counts[b]["flat"][1]
            if n == 0:
                continue
            all_results[b]["flat"].append(counts[b]["flat"][0] / n * 100)
            all_results[b]["l2"].append(counts[b]["l2"][0] / n * 100)
            all_results[b]["l3"].append(counts[b]["l3"][0] / n * 100)
            print(f"  dist {b[0]:>2}-{b[1]:<2} (n={n:>4}): flat={all_results[b]['flat'][-1]:5.1f}%  "
                  f"K=2-L2={all_results[b]['l2'][-1]:5.1f}%  L3={all_results[b]['l3'][-1]:5.1f}%")

    print(f"\n{'='*70}")
    print("3-SEED SUMMARY (held-out pairs, fair budgets)")
    print(f"{'='*70}")
    for b in BUCKETS:
        if not all_results[b]["flat"]:
            continue
        row = []
        for key in ["flat", "l2", "l3"]:
            vals = all_results[b][key]
            m = st.mean(vals); sd = st.stdev(vals) if len(vals) > 1 else 0.0; w = min(vals)
            row.append(f"{key}={m:5.1f}%+/-{sd:4.1f}(worst {w:.1f})")
        print(f"dist {b[0]:>2}-{b[1]:<2}: " + "  ".join(row))
