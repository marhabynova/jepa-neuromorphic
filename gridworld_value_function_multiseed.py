"""
Multi-seed (3-seed) confirmation of §78's single-seed value-function result
(detour-case success 89.6%, no-detour 98.3%, both 100% optimal when
successful). Same bar this project has applied to every other promising
single-seed result before calling it robust (§70d->f, §71b->c, §76).

Scope note carried over from §78: ValueNet operates on raw state one-hot
encodings and uses the REAL environment for both TD-backup and 1-step
planning lookahead - it does not go through gridworld_jepa.py's learned
representation or dynamics model. Valid as a clean test of "value function
fixes the heuristic problem", not yet "value function on top of JEPA's
learned representation" - a distinct, still-open scope question.
"""
import statistics
import numpy as np
import torch

from gridworld_obstacles import random_free_cell, bfs_optimal_distance, greedy_heuristic
from gridworld_value_function import (
    train_value_function, value_greedy_plan, encode_state_batch, DEVICE,
)

SEEDS = [0, 1, 2]
N_EVAL_PAIRS = 200


def run_seed(seed):
    V = train_value_function(seed=seed)
    rng = np.random.default_rng(seed + 100)  # distinct from training rng's seed+1 offset

    pairs = []
    for _ in range(N_EVAL_PAIRS):
        s = random_free_cell(rng)
        g = random_free_cell(rng)
        if s == g:
            continue
        pairs.append((s, g))

    detour, no_detour = [], []
    for s, g in pairs:
        _, greached = greedy_heuristic(s, g)
        (detour if not greached else no_detour).append((s, g, bfs_optimal_distance(s, g)))

    def eval_subset(subset):
        succ, exc = 0, []
        for s, g, opt in subset:
            path, reached = value_greedy_plan(V, s, g)
            if reached:
                succ += 1
                exc.append((len(path) - 1) - opt)
        return succ, len(subset), exc

    s1, n1, e1 = eval_subset(detour)
    s2, n2, e2 = eval_subset(no_detour)
    return {
        "detour_success_rate": s1 / n1 if n1 else float("nan"),
        "detour_n": n1,
        "detour_exact_optimal": sum(1 for e in e1 if e == 0) / len(e1) if e1 else float("nan"),
        "no_detour_success_rate": s2 / n2 if n2 else float("nan"),
        "no_detour_n": n2,
        "no_detour_exact_optimal": sum(1 for e in e2 if e == 0) / len(e2) if e2 else float("nan"),
    }


if __name__ == "__main__":
    results = []
    for seed in SEEDS:
        print(f"\n{'='*20} seed={seed} {'='*20}")
        r = run_seed(seed)
        results.append(r)
        print(f"  detour: {r['detour_success_rate']*100:.1f}% success (n={r['detour_n']}), "
              f"{r['detour_exact_optimal']*100:.1f}% exact-optimal")
        print(f"  no-detour: {r['no_detour_success_rate']*100:.1f}% success (n={r['no_detour_n']}), "
              f"{r['no_detour_exact_optimal']*100:.1f}% exact-optimal")

    def summarize(key):
        vals = [r[key] * 100 for r in results]
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0), vals

    print(f"\n{'='*70}")
    print(f"SUMMARY across {len(SEEDS)} seeds (mean +/- stdev)")
    print(f"{'='*70}")
    for key, label in [("detour_success_rate", "Detour success"),
                        ("detour_exact_optimal", "Detour exact-optimal"),
                        ("no_detour_success_rate", "No-detour success"),
                        ("no_detour_exact_optimal", "No-detour exact-optimal")]:
        mean, sd, vals = summarize(key)
        worst = min(vals)
        print(f"{label:<24} {mean:5.1f}% +/- {sd:4.1f}  (per-seed: {[f'{v:.1f}' for v in vals]}, worst={worst:.1f}%)")

    print("\nWorst-seed detour success rate is the number that matters most - §77's old")
    print("heuristic got 0-5.8% on this exact subset, so even a much worse worst-case")
    print("here would still be a dramatic improvement; this checks it isn't a lucky seed.")
