"""
Wall-aware region partition (§91c), replacing the naive grid-overlay
partition in gridworld_3level.py that a static analysis found badly
mismatched to this maze's wall geometry: 9/16 regions internally bisected,
18/48 "adjacent" region pairs actually requiring 13-34 real steps (not
short local hops). Root cause: the old partition assigned regions by raw
(x//5, y//5) grid index, which has zero awareness of where the walls
actually are.

FIX: build regions as a graph VORONOI partition - every free cell is
assigned to whichever of N seed points is closest by REAL BFS distance
(not Euclidean distance), computed via multi-source BFS. This respects
walls by construction (a cell on the far side of a wall is far by real
distance even if geometrically close, so it naturally gets assigned to a
different seed/region rather than being grouped with cells it can't
actually reach cheaply). "Adjacent" regions are then regions that
genuinely TOUCH in the real grid (share at least one real 4-connected
edge), not regions whose arbitrary indices happen to be geometrically
next to each other.

Seeds chosen via furthest-point sampling (by real BFS distance) for even
coverage - this is also graph-aware, not geometry-aware.
"""
import sys
from collections import deque

import numpy as np

import gridworld_hierarchical as gh

N_SEEDS = 16  # same count as the old grid partition, for comparability


def all_free_cells():
    return [(x, y) for x in range(gh.GRID_SIZE) for y in range(gh.GRID_SIZE) if not gh.is_wall(x, y)]


def bfs_distances_from(start, free_set):
    """Real BFS distance from `start` to every reachable free cell."""
    dist = {start: 0}
    q = deque([start])
    while q:
        cx, cy = q.popleft()
        for a in range(4):
            nx, ny = gh.step_env(cx, cy, a)
            if (nx, ny) != (cx, cy) and (nx, ny) not in dist:
                dist[(nx, ny)] = dist[(cx, cy)] + 1
                q.append((nx, ny))
    return dist


def furthest_point_seeds(free_cells, n_seeds, start_seed):
    """Greedy furthest-point sampling by REAL BFS distance, not Euclidean -
    respects walls when choosing well-spread seeds."""
    free_set = set(free_cells)
    seeds = [start_seed]
    min_dist_to_any_seed = bfs_distances_from(start_seed, free_set)
    # fill unreachable cells (shouldn't happen on a connected maze) with 0 so they're never picked
    min_dist_to_any_seed = {c: min_dist_to_any_seed.get(c, 0) for c in free_cells}
    for _ in range(n_seeds - 1):
        next_seed = max(free_cells, key=lambda c: min_dist_to_any_seed[c])
        seeds.append(next_seed)
        new_dist = bfs_distances_from(next_seed, free_set)
        for c in free_cells:
            d = new_dist.get(c, min_dist_to_any_seed[c])
            min_dist_to_any_seed[c] = min(min_dist_to_any_seed[c], d)
    return seeds


def multi_source_bfs_partition(free_cells, seeds):
    """Assign every free cell to its nearest seed by real BFS distance
    (graph Voronoi). Ties broken by whichever seed's wavefront arrives
    first in seed-index order (stable, deterministic)."""
    free_set = set(free_cells)
    owner = {s: i for i, s in enumerate(seeds)}
    dist = {s: 0 for s in seeds}
    q = deque(seeds)
    while q:
        cx, cy = q.popleft()
        for a in range(4):
            nx, ny = gh.step_env(cx, cy, a)
            if (nx, ny) == (cx, cy) or (nx, ny) not in free_set:
                continue
            nd = dist[(cx, cy)] + 1
            if (nx, ny) not in dist or nd < dist[(nx, ny)]:
                dist[(nx, ny)] = nd
                owner[(nx, ny)] = owner[(cx, cy)]
                q.append((nx, ny))
    return owner


FREE_CELLS = all_free_cells()
SEEDS = furthest_point_seeds(FREE_CELLS, N_SEEDS, start_seed=FREE_CELLS[0])
OWNER = multi_source_bfs_partition(FREE_CELLS, SEEDS)
ALL_REGIONS_V2 = list(range(N_SEEDS))
REGION_ANCHOR_V2 = {i: SEEDS[i] for i in ALL_REGIONS_V2}


def region_of_v2(x, y):
    return OWNER[(x, y)]


def region_cells_v2(region_id):
    return [c for c, r in OWNER.items() if r == region_id]


_ADJACENCY_CACHE = None


def _build_adjacency():
    adj = {i: set() for i in ALL_REGIONS_V2}
    for (x, y), r in OWNER.items():
        for a in range(4):
            nx, ny = gh.step_env(x, y, a)
            if (nx, ny) == (x, y):
                continue
            r2 = OWNER.get((nx, ny))
            if r2 is not None and r2 != r:
                adj[r].add(r2)
    return adj


def adjacent_regions_v2(region_id):
    global _ADJACENCY_CACHE
    if _ADJACENCY_CACHE is None:
        _ADJACENCY_CACHE = _build_adjacency()
    return list(_ADJACENCY_CACHE[region_id])


if __name__ == "__main__":
    from collections import deque as _dq

    def connected_components(cells):
        cell_set = set(cells)
        seen = set()
        comps = []
        for start in cells:
            if start in seen:
                continue
            comp = []
            q = _dq([start])
            seen.add(start)
            while q:
                cx, cy = q.popleft()
                comp.append((cx, cy))
                for a in range(4):
                    nx, ny = gh.step_env(cx, cy, a)
                    if (nx, ny) != (cx, cy) and (nx, ny) in cell_set and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        q.append((nx, ny))
            comps.append(comp)
        return comps

    print(f"=== {N_SEEDS} wall-aware (graph-Voronoi) regions ===")
    sizes = [len(region_cells_v2(i)) for i in ALL_REGIONS_V2]
    print(f"Region sizes: min={min(sizes)} max={max(sizes)} mean={sum(sizes)/len(sizes):.1f} "
          f"(old grid partition: fixed 25 or fewer near walls)")

    print("\n1. INTERNAL CONNECTIVITY check:")
    n_bisected = 0
    for i in ALL_REGIONS_V2:
        cells = region_cells_v2(i)
        comps = connected_components(cells)
        if len(comps) > 1:
            n_bisected += 1
            print(f"  region {i} (seed={SEEDS[i]}): BISECTED into {len(comps)} components, "
                  f"sizes={sorted([len(c) for c in comps], reverse=True)}")
    print(f"{n_bisected}/{N_SEEDS} regions internally bisected (old partition: 9/16)")

    print("\n2. TRUE cost of each adjacent region-pair hop (anchor to anchor):")
    n_misleading = 0
    n_pairs = 0
    for i in ALL_REGIONS_V2:
        for j in adjacent_regions_v2(i):
            if j <= i:
                continue  # report each undirected pair once
            d = gh.bfs_optimal_distance(REGION_ANCHOR_V2[i], REGION_ANCHOR_V2[j])
            n_pairs += 1
            flag = d is not None and d > 12
            if flag:
                n_misleading += 1
            marker = "  <- MISLEADING" if flag else ""
            print(f"  region{i}(seed={SEEDS[i]}) <-> region{j}(seed={SEEDS[j]}): true dist={d}{marker}")
    print(f"\n{n_misleading}/{n_pairs} adjacent pairs still misleading (old partition: 18/48, but that was directed; "
          f"undirected old count was 24)")
