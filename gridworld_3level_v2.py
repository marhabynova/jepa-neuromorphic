"""
3-level hierarchy, v2 - rebuilt on gridworld_region_v2.py's wall-aware
(graph-Voronoi) region partition instead of gridworld_3level.py's naive
grid overlay. The v1 file accumulated real wins (§91) but also several
reactive patches (Bellman min-over-options bug, budget-fairness bug,
step-budget mismatch, direction-aware-anchor patch) on top of a partition
that static analysis later showed was structurally broken for 9/16 regions
and 18/48 "adjacent" pairs. This version fixes the ROOT CAUSE instead of
patching symptoms: regions are graph-Voronoi cells (always connected by
construction) and "adjacent" means "actually touches in the real grid"
(verified: 0/16 bisected, 0/21 misleading, all true adjacent-pair
distances are 7-12 steps - genuinely local hops).

Kept identical to v1 where nothing needs to change: the collect-once-then-
replay training pattern for V_level3 (one region-hop sample is still
expensive), the min-over-options Bellman structure, the real/grounded
execution + replan-from-real-position discipline, the fairness-matched
evaluation protocol. Changed: region identity is now an integer id (not an
(rx,ry) tuple), adjacency is graph-derived (not grid-index-derived), and
the "entry anchor" used as V_hi's pseudo-goal for a hop is the nearest
REAL BOUNDARY cell between the two specific regions (not a geometric-
center-with-direction-offset heuristic, which doesn't even make sense for
non-grid regions).
"""
import copy
import statistics
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn

import gridworld_hierarchical as gh
import gridworld_region_v2 as rv2

DEVICE = gh.DEVICE
MAX_L1_MACROSTEPS_PER_REGION_HOP = 20
N_STARTCELLS_PER_REGION = 4
N_COLLECT_PER_STARTCELL = 3

REGION_CELLS = {i: rv2.region_cells_v2(i) for i in rv2.ALL_REGIONS_V2}
ADJACENCY = {i: rv2.adjacent_regions_v2(i) for i in rv2.ALL_REGIONS_V2}


def _boundary_entry_anchor(from_region, to_region):
    """Nearest cell in to_region that is real-grid-adjacent to some cell in
    from_region (an actual shared-boundary cell), biased toward the
    from_region's anchor so the choice among several boundary candidates is
    deterministic and reasonable. Falls back to to_region's own anchor if
    the two regions don't directly touch (shouldn't happen when called only
    for genuine adjacent pairs, kept as a safety net)."""
    candidates = []
    from_cells = set(REGION_CELLS[from_region])
    for (x, y) in REGION_CELLS[to_region]:
        for a in range(4):
            nx, ny = gh.step_env(x, y, a)
            if (nx, ny) != (x, y) and (nx, ny) in from_cells:
                candidates.append((x, y))
                break
    if not candidates:
        return rv2.REGION_ANCHOR_V2[to_region]
    fx, fy = rv2.REGION_ANCHOR_V2[from_region]
    return min(candidates, key=lambda c: (c[0] - fx) ** 2 + (c[1] - fy) ** 2)


ENTRY_ANCHOR = {}
for _r in rv2.ALL_REGIONS_V2:
    for _n in ADJACENCY[_r]:
        ENTRY_ANCHOR[(_r, _n)] = _boundary_entry_anchor(_r, _n)


@torch.no_grad()
def hierarchical_plan_to_region(world_model, V_hi, start, target_region, from_region, k=2,
                                 max_macro_steps=MAX_L1_MACROSTEPS_PER_REGION_HOP, memory=3):
    """Level-1 (K=2) planner, stopping as soon as the current region equals
    target_region. Uses the real shared-boundary entry anchor as V_hi's
    pseudo-goal (see ENTRY_ANCHOR / _boundary_entry_anchor)."""
    x, y = start
    path = [(x, y)]
    recent = [(x, y)]
    goal = ENTRY_ANCHOR.get((from_region, target_region), rv2.REGION_ANCHOR_V2[target_region])
    g_repr = gh.tgt(world_model, *goal)
    if rv2.region_of_v2(x, y) == target_region:
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
        if rv2.region_of_v2(x, y) == target_region:
            return path, True, step + 1
    return path, False, max_macro_steps


def collect_region_transitions(world_model, V_hi, seed):
    rng = np.random.default_rng(seed + 10)
    dataset = []
    for r in rv2.ALL_REGIONS_V2:
        cells = REGION_CELLS[r]
        for nbr in ADJACENCY[r]:
            n_starts = min(N_STARTCELLS_PER_REGION, len(cells))
            start_cells = [cells[i] for i in rng.choice(len(cells), size=n_starts, replace=False)]
            for start in start_cells:
                for _ in range(N_COLLECT_PER_STARTCELL):
                    path, success, cost = hierarchical_plan_to_region(world_model, V_hi, start, nbr, from_region=r)
                    arrival = path[-1]
                    dataset.append({
                        "start": start, "source_region": r, "target_region": nbr,
                        "arrival": arrival, "arrival_region": rv2.region_of_v2(*arrival),
                        "cost": cost, "success": success,
                    })
    n_success = sum(1 for d in dataset if d["success"])
    print(f"  [collect v2] {len(dataset)} real region-hop rollouts, {n_success}/{len(dataset)} reached target region "
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


L3_TD_STEPS = 30000  # §91a's lesson applied from the start: match EMA_DECAY=0.999's step-budget pairing
L3_LR = 1e-3
L3_BATCH = 32
L3_EMA_DECAY = 0.999


def train_region_value(world_model, dataset, seed, steps=L3_TD_STEPS):
    """TD/Bellman over the fixed, precollected dataset - min over ALL
    actually-adjacent regions per sample (variable degree per region, not
    a fixed 4-direction slot structure like v1's grid regions had, since
    graph-Voronoi regions can have any number of real neighbors)."""
    torch.manual_seed(seed + 20)
    rng = np.random.default_rng(seed + 20)
    V3 = RegionValueNet(gh.REPR_DIM).to(DEVICE)
    V3_target = copy.deepcopy(V3)
    for p in V3_target.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(V3.parameters(), lr=L3_LR)

    with torch.no_grad():
        region_repr = {r: gh.ctx(world_model, *rv2.REGION_ANCHOR_V2[r]) for r in rv2.ALL_REGIONS_V2}
        region_tgt_repr = {r: gh.tgt(world_model, *rv2.REGION_ANCHOR_V2[r]) for r in rv2.ALL_REGIONS_V2}

    grouped = defaultdict(lambda: defaultdict(list))
    for d in dataset:
        grouped[d["source_region"]][d["target_region"]].append((d["cost"], d["arrival_region"]))

    all_regions_list = rv2.ALL_REGIONS_V2
    max_degree = max(len(ADJACENCY[r]) for r in all_regions_list)

    for step in range(steps):
        source_regions = [all_regions_list[i] for i in rng.integers(0, len(all_regions_list), size=L3_BATCH)]
        goal_regions = [all_regions_list[i] for i in rng.integers(0, len(all_regions_list), size=L3_BATCH)]
        state_repr = torch.stack([region_repr[r] for r in source_regions])
        goal_repr = torch.stack([region_tgt_repr[g] for g in goal_regions])
        pred = V3(state_repr, goal_repr)

        with torch.no_grad():
            best_next_v = torch.full((L3_BATCH,), float("inf"), device=DEVICE)
            is_same = torch.tensor([r == g for r, g in zip(source_regions, goal_regions)], device=DEVICE)
            for slot in range(max_degree):
                valid, costs, arrival_reprs, arrived_at_goal = [], [], [], []
                for b in range(L3_BATCH):
                    r = source_regions[b]
                    nbrs = ADJACENCY[r]
                    if slot >= len(nbrs):
                        valid.append(False); costs.append(0.0)
                        arrival_reprs.append(region_repr[r]); arrived_at_goal.append(False)
                        continue
                    nbr = nbrs[slot]
                    samples = grouped[r].get(nbr)
                    if not samples:
                        valid.append(False); costs.append(0.0)
                        arrival_reprs.append(region_repr[r]); arrived_at_goal.append(False)
                        continue
                    cost, arrival_region = samples[rng.integers(0, len(samples))]
                    valid.append(True)
                    costs.append(float(cost))
                    arrival_reprs.append(region_repr[arrival_region])
                    arrived_at_goal.append(arrival_region == goal_regions[b])
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
        if (step + 1) % 3000 == 0:
            print(f"  [V-level3 v2] step {step+1}/{steps} loss={loss.item():.4f}")
    return V3, region_repr, region_tgt_repr


def plan_level3(world_model, V_hi, V3, region_repr, region_tgt_repr, start, goal, max_region_hops=12, memory=3):
    x, y = start
    goal_region = rv2.region_of_v2(*goal)
    full_path = [(x, y)]
    recent_regions = [rv2.region_of_v2(x, y)]
    for _ in range(max_region_hops):
        cur_region = rv2.region_of_v2(x, y)
        if cur_region == goal_region:
            sub_path, reached = gh.hierarchical_plan(world_model, V_hi, (x, y), goal, k=2)
            full_path.extend(sub_path[1:])
            return full_path, reached
        goal_r_repr = region_tgt_repr[goal_region]
        scored = []
        for nbr in ADJACENCY[cur_region]:
            sub_path, hop_success, hop_cost = hierarchical_plan_to_region(world_model, V_hi, (x, y), nbr, from_region=cur_region)
            v = V3(region_repr[nbr].unsqueeze(0), goal_r_repr.unsqueeze(0)).item()
            score = hop_cost + v
            stuck = not hop_success or nbr in recent_regions[-memory:]
            scored.append((stuck, score, nbr, sub_path, hop_success))
        scored.sort(key=lambda t: (t[0], t[1]))
        _, _, chosen_nbr, sub_path, hop_success = scored[0]
        full_path.extend(sub_path[1:])
        x, y = sub_path[-1]
        recent_regions.append(rv2.region_of_v2(x, y))
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
        _, freached = gh.flat_plan(wm, V_prim, s, g, max_steps=480)
        _, hreached = gh.hierarchical_plan(wm, V_hi, s, g, k=2, max_macro_steps=240)
        _, l3reached = plan_level3(wm, V_hi, V3, region_repr, region_tgt_repr, s, g)
        counts[b]["flat"][1] += 1; counts[b]["l2"][1] += 1; counts[b]["l3"][1] += 1
        if freached: counts[b]["flat"][0] += 1
        if hreached: counts[b]["l2"][0] += 1
        if l3reached: counts[b]["l3"][0] += 1
    return counts


if __name__ == "__main__" and "--3seed" not in __import__("sys").argv:
    seed = 0
    print(f"\n{'='*20} PROOF OF CONCEPT (wall-aware regions), seed={seed} {'='*20}")
    counts = run_one_seed(seed)
    print(f"\n{'='*70}")
    print(f"RESULT (single seed={seed}, held-out pairs, fair budgets, wall-aware regions)")
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
    print("3-SEED SUMMARY (wall-aware regions, held-out pairs, fair budgets)")
    print(f"{'='*70}")
    for b in BUCKETS:
        if not all_results[b]["flat"]:
            continue
        row = []
        for key in ["flat", "l2", "l3"]:
            vals = all_results[b][key]
            m = statistics.mean(vals); sd = statistics.stdev(vals) if len(vals) > 1 else 0.0; w = min(vals)
            row.append(f"{key}={m:5.1f}%+/-{sd:4.1f}(worst {w:.1f})")
        print(f"dist {b[0]:>2}-{b[1]:<2}: " + "  ".join(row))
