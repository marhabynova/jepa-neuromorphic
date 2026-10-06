"""
Step 2 (RESEARCH_NOTES.md §75/§76): planning/search ON TOP of the
action-conditioned world model verified in gridworld_jepa.py. This is the
actual "agent-loop" piece flagged in §70g as the one mechanism this
project's own complexity-theory analysis (§12/§22/§23) found to be
necessary for crossing the TC0/single-forward-pass ceiling - a search over
multiple composed steps, not a smarter single-pass network.

Design: BEAM SEARCH entirely in latent space (no grounding in the real
environment during search - true "imagination", matching the MuZero/
H-JEPA paradigm of planning via a learned model before acting), scored by
distance to the goal's target-encoder representation. Only AFTER a full
action sequence is found does it get executed in the real environment, to
honestly check whether a plan found purely through latent rollout actually
works in reality - compounding prediction error across multiple chained
dynamics-model steps is a real risk this has to survive, not an assumption.

Baselines (mandatory, project convention): optimal path length (Manhattan
distance - exact for this obstacle-free grid, the true lower bound) and a
random-walk policy (to show the search is doing something beyond chance).
"""
import torch
import torch.nn as nn

from gridworld_jepa import (
    ActionConditionedJEPA, step_env, encode_state_batch, sample_batch,
    GRID_SIZE, N_ACTIONS, STATE_DIM, STEPS, LR, BATCH_SIZE, SEED, DEVICE,
)

BEAM_WIDTH = 16
MAX_DEPTH = 20          # generous vs. the grid's max possible Manhattan distance (14)
N_EVAL_EPISODES = 100
RANDOM_WALK_BUDGET = 200  # steps before giving up, for the random baseline


def train_model(seed=SEED, steps=STEPS):
    import numpy as np
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
    return model, rng


@torch.no_grad()
def encode_pos(model, encoder, x, y):
    state = encode_state_batch(torch.tensor([x]), torch.tensor([y])).to(DEVICE)
    return encoder(state).squeeze(0)


@torch.no_grad()
def beam_search_plan(model, start_xy, goal_xy, beam_width=BEAM_WIDTH, max_depth=MAX_DEPTH):
    """Pure latent-space planning: no real-environment grounding until the
    final chosen sequence is executed. Each beam entry is
    (latent_repr, action_sequence, score)."""
    goal_repr = encode_pos(model, model.target_encoder, *goal_xy)
    start_repr = encode_pos(model, model.context_encoder, *start_xy)

    def score_of(repr_):
        return torch.mean((repr_ - goal_repr) ** 2).item()

    beam = [(start_repr, [], score_of(start_repr))]
    best_overall = min(beam, key=lambda b: b[2])

    for depth in range(max_depth):
        if best_overall[2] < 1e-4:  # latent distance to goal is ~zero - good enough
            break
        candidates = []
        for repr_, seq, _ in beam:
            for a in range(N_ACTIONS):
                action_onehot = nn.functional.one_hot(torch.tensor(a), N_ACTIONS).float().to(DEVICE)
                next_repr = model.dynamics(repr_.unsqueeze(0), action_onehot.unsqueeze(0)).squeeze(0)
                new_seq = seq + [a]
                candidates.append((next_repr, new_seq, score_of(next_repr)))
        candidates.sort(key=lambda c: c[2])
        beam = candidates[:beam_width]
        if beam[0][2] < best_overall[2]:
            best_overall = beam[0]

    return best_overall[1]  # the action sequence


def execute_plan(start_xy, action_seq):
    """Ground-truth check: run the FOUND plan in the REAL environment (not
    the learned model) and see where it actually ends up."""
    x, y = start_xy
    path = [(x, y)]
    for a in action_seq:
        x, y = step_env(x, y, a)
        path.append((x, y))
    return path


def random_walk(start_xy, goal_xy, rng, budget=RANDOM_WALK_BUDGET):
    x, y = start_xy
    for step in range(budget):
        if (x, y) == goal_xy:
            return step, True
        a = int(rng.integers(0, N_ACTIONS))
        x, y = step_env(x, y, a)
    return budget, (x, y) == goal_xy


if __name__ == "__main__":
    print("=== Training the action-conditioned world model (same as gridworld_jepa.py) ===")
    model, rng = train_model()

    print(f"\n=== Evaluating latent beam search over {N_EVAL_EPISODES} random (start, goal) pairs ===")
    successes = 0
    excess_steps = []
    random_successes = 0
    random_excess = []

    for ep in range(N_EVAL_EPISODES):
        sx, sy = int(rng.integers(0, GRID_SIZE)), int(rng.integers(0, GRID_SIZE))
        gx, gy = int(rng.integers(0, GRID_SIZE)), int(rng.integers(0, GRID_SIZE))
        if (sx, sy) == (gx, gy):
            continue
        optimal = abs(sx - gx) + abs(sy - gy)  # Manhattan distance = true shortest path, no obstacles

        plan = beam_search_plan(model, (sx, sy), (gx, gy))
        path = execute_plan((sx, sy), plan)
        reached = (gx, gy) in path
        if reached:
            actual_len = path.index((gx, gy))
            successes += 1
            excess_steps.append(actual_len - optimal)
        # else: plan found via latent search didn't actually reach the goal in reality - a real failure

        r_len, r_reached = random_walk((sx, sy), (gx, gy), rng)
        if r_reached:
            random_successes += 1
            random_excess.append(r_len - optimal)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Latent beam search: {successes}/{N_EVAL_EPISODES} reached goal "
          f"({100*successes/N_EVAL_EPISODES:.1f}% success rate)")
    if excess_steps:
        mean_excess = sum(excess_steps) / len(excess_steps)
        print(f"  mean excess steps over optimal (successes only): {mean_excess:+.2f} "
              f"(0 = perfectly optimal path)")
        print(f"  exact-optimal fraction: {sum(1 for e in excess_steps if e == 0)/len(excess_steps)*100:.1f}%")
    print(f"\nRandom-walk baseline: {random_successes}/{N_EVAL_EPISODES} reached goal within "
          f"{RANDOM_WALK_BUDGET}-step budget ({100*random_successes/N_EVAL_EPISODES:.1f}%)")
    if random_excess:
        print(f"  mean excess steps over optimal (successes only): "
              f"{sum(random_excess)/len(random_excess):+.2f}")
    print("\nThis plan was found ENTIRELY via latent-space search (no real-environment grounding")
    print("during search) then executed once and checked against ground truth - a genuine test of")
    print("whether chained/compounded dynamics-model predictions stay accurate enough to plan on.")


def closed_loop_plan(model, start_xy, goal_xy, max_real_steps=30):
    """MPC-style: re-plan via latent beam search from the TRUE current
    position after every single real step, instead of committing to one
    long open-loop sequence. Bounds compounding dynamics-model error to
    just 1 step at a time. Added after the open-loop version's success
    rate was found to degrade sharply with path length (100% at dist=1,
    42% at dist=4, <15% at dist>=7) - confirmed as compounding prediction
    error (not a bug) by checking the degradation tracked distance, not
    noise."""
    x, y = start_xy
    path = [(x, y)]
    for _ in range(max_real_steps):
        if (x, y) == goal_xy:
            return path, True
        plan = beam_search_plan(model, (x, y), goal_xy, max_depth=MAX_DEPTH)
        if not plan:
            break
        first_action = plan[0]
        x, y = step_env(x, y, first_action)
        path.append((x, y))
    return path, (x, y) == goal_xy


if __name__ == "__main__" and "--closed-loop" in __import__("sys").argv:
    print("=== Closed-loop (replan every real step) evaluation ===")
    model2, rng2 = train_model()
    successes2 = 0
    excess2 = []
    for ep in range(N_EVAL_EPISODES):
        sx, sy = int(rng2.integers(0, GRID_SIZE)), int(rng2.integers(0, GRID_SIZE))
        gx, gy = int(rng2.integers(0, GRID_SIZE)), int(rng2.integers(0, GRID_SIZE))
        if (sx, sy) == (gx, gy):
            continue
        optimal = abs(sx - gx) + abs(sy - gy)
        path, reached = closed_loop_plan(model2, (sx, sy), (gx, gy))
        if reached:
            successes2 += 1
            excess2.append((len(path) - 1) - optimal)
    print(f"Closed-loop success: {successes2}/{N_EVAL_EPISODES} ({100*successes2/N_EVAL_EPISODES:.1f}%)")
    if excess2:
        print(f"Mean excess steps: {sum(excess2)/len(excess2):+.2f}, "
              f"exact-optimal fraction: {sum(1 for e in excess2 if e==0)/len(excess2)*100:.1f}%")


def closed_loop_plan_antistuck(model, start_xy, goal_xy, max_real_steps=30, memory=4):
    """Same as closed_loop_plan, but with a short-term "visited" memory
    (tabu list) so the agent doesn't repeat a move that leads straight back
    to a cell it was just at. Diagnosed as necessary after finding 100%
    (55/55) of closed_loop_plan's failures were cycling/oscillation at grid
    boundaries - the greedy replanner has no memory, so a locally-attractive
    but wrong move (e.g. "up" at a corner, which the real environment clamps
    to a no-op) gets repeated forever instead of being recognized as stuck."""
    x, y = start_xy
    path = [(x, y)]
    recent = [(x, y)]
    for _ in range(max_real_steps):
        if (x, y) == goal_xy:
            return path, True
        goal_repr = encode_pos(model, model.target_encoder, *goal_xy)
        cur_repr = encode_pos(model, model.context_encoder, x, y)
        scored_actions = []
        for a in range(N_ACTIONS):
            nx, ny = step_env(x, y, a)
            action_onehot = nn.functional.one_hot(torch.tensor(a), N_ACTIONS).float().to(DEVICE)
            with torch.no_grad():
                pred = model.dynamics(cur_repr.unsqueeze(0), action_onehot.unsqueeze(0)).squeeze(0)
                score = torch.mean((pred - goal_repr) ** 2).item()
            tabu = (nx, ny) in recent[-memory:]
            scored_actions.append((score, tabu, a, (nx, ny)))
        # Prefer non-tabu actions; among those, lowest score (closest to goal)
        scored_actions.sort(key=lambda t: (t[1], t[0]))  # tabu=False(0) sorts before True(1)
        _, _, best_a, (nx, ny) = scored_actions[0]
        x, y = nx, ny
        path.append((x, y))
        recent.append((x, y))
    return path, (x, y) == goal_xy


if __name__ == "__main__" and "--antistuck" in __import__("sys").argv:
    print("=== Closed-loop + anti-cycling memory evaluation ===")
    model3, rng3 = train_model()
    successes3 = 0
    excess3 = []
    for ep in range(N_EVAL_EPISODES):
        sx, sy = int(rng3.integers(0, GRID_SIZE)), int(rng3.integers(0, GRID_SIZE))
        gx, gy = int(rng3.integers(0, GRID_SIZE)), int(rng3.integers(0, GRID_SIZE))
        if (sx, sy) == (gx, gy):
            continue
        optimal = abs(sx - gx) + abs(sy - gy)
        path, reached = closed_loop_plan_antistuck(model3, (sx, sy), (gx, gy))
        if reached:
            successes3 += 1
            excess3.append((len(path) - 1) - optimal)
    print(f"Closed-loop+antistuck success: {successes3}/{N_EVAL_EPISODES} ({100*successes3/N_EVAL_EPISODES:.1f}%)")
    if excess3:
        print(f"Mean excess steps: {sum(excess3)/len(excess3):+.2f}, "
              f"exact-optimal fraction: {sum(1 for e in excess3 if e==0)/len(excess3)*100:.1f}%")
