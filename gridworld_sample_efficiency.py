"""
§85d item #3: sample efficiency - one of the AGI puzzle pieces flagged early
in this project and explicitly left untouched until a genuine held-out
protocol existed (it didn't until §82). Question: how much training data
does V_prim actually NEED to reach near-ceiling performance on this maze -
not "what's the ceiling" (already measured in §84e/§87a), but "how cheaply
can you get there."

Design: reuse `gridworld_hierarchical.train_primitive_value()` UNCHANGED at
varying step budgets. Deterministic seeding (torch.manual_seed(seed+1),
np.random.default_rng(seed+1), no shuffling that depends on future steps)
means each call's trajectory up to step N is bit-for-bit identical
regardless of where training eventually stops - so re-running at different
`steps` values is equivalent to checkpointing one long run, just recomputed
per checkpoint. Chosen over writing a custom checkpointing loop: correctness
and reuse of an already-validated function over cleverness/new code, at the
cost of some redundant compute (acceptable at this toy scale).

World model is trained once per seed and held FIXED across all step
budgets - isolates "how much V-training is needed" from "how much
world-model training is needed" (a separate, already-answered question,
§81a#1's decode-accuracy metric).
"""
import statistics
from collections import defaultdict

import gridworld_hierarchical as gh

SEEDS = [0, 1, 2]
STEP_BUDGETS = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000]
LONG_RANGE = (20, 34)

N_FREE_CELLS = gh.GRID_SIZE * gh.GRID_SIZE - len(gh.WALLS)
N_ORDERED_PAIRS = N_FREE_CELLS * (N_FREE_CELLS - 1)


def eval_v(world_model, V, test_pairs):
    by_bucket = defaultdict(lambda: [0, 0])
    for s, g in test_pairs:
        d = gh.bfs_optimal_distance(s, g)
        bucket = "long" if LONG_RANGE[0] <= d <= LONG_RANGE[1] else ("short" if d < 10 else None)
        if bucket is None:
            continue
        _, reached = gh.flat_plan(world_model, V, s, g)
        by_bucket[bucket][1] += 1
        if reached:
            by_bucket[bucket][0] += 1
    short = by_bucket["short"][0] / max(by_bucket["short"][1], 1)
    long_ = by_bucket["long"][0] / max(by_bucket["long"][1], 1)
    return short * 100, long_ * 100


if __name__ == "__main__":
    print(f"=== Maze: {N_FREE_CELLS} free cells, {N_ORDERED_PAIRS} ordered (state,goal) pairs ===\n")

    results_short = {n: [] for n in STEP_BUDGETS}
    results_long = {n: [] for n in STEP_BUDGETS}

    for seed in SEEDS:
        print(f"\n{'='*20} seed={seed} {'='*20}")
        wm = gh.train_world_model(seed=seed)
        train_pairs, test_pairs = gh.make_pair_split_maze(seed)
        for n in STEP_BUDGETS:
            V = gh.train_primitive_value(wm, seed=seed, steps=n)
            short, long_ = eval_v(wm, V, test_pairs)
            results_short[n].append(short)
            results_long[n].append(long_)
            total_transitions = n * gh.V_BATCH_SIZE
            print(f"  steps={n:>6} ({total_transitions:>9} transitions sampled, "
                  f"{100*total_transitions/N_ORDERED_PAIRS:6.1f}% of pair-space): "
                  f"short={short:5.1f}% long={long_:5.1f}%")

    print(f"\n{'='*70}")
    print("SAMPLE-EFFICIENCY CURVE (V_prim / flat_plan), 3-seed, held-out pairs")
    print(f"{'='*70}")

    def stats(vals):
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0), min(vals)

    for n in STEP_BUDGETS:
        sm, ssd, sw = stats(results_short[n])
        lm, lsd, lw = stats(results_long[n])
        total_transitions = n * gh.V_BATCH_SIZE
        print(f"steps={n:>6} ({100*total_transitions/N_ORDERED_PAIRS:5.1f}% of pair-space): "
              f"SHORT={sm:5.1f}%+/-{ssd:4.1f} (worst {sw:4.1f})   "
              f"LONG={lm:5.1f}%+/-{lsd:4.1f} (worst {lw:4.1f})")
