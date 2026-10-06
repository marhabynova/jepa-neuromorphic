"""
First step toward the architectural gap flagged in RESEARCH_NOTES.md §70g/
§75: a JEPA-style world model with ZERO action-conditioning cannot support
any planning/search loop on top of it, no matter how good the search
mechanism is - you can't search over actions you can't even represent.
dense_jepa.py (and everything built on it so far: Awake Gate, Sleep
Consolidation) is pure passive-observation JEPA, no notion of agency.

Grounded in H-JEPA (LeCun et al., arXiv 2610.06805, Oct 2026) and
MuZero/UniZero-style latent dynamics models - both confirm the field's
answer to "representation + sequential composition" is action-conditioned
world models + search/planning in latent space, not something this project
needs to invent from scratch.

SHD (passive audio classification, zero actions) is structurally the wrong
testbed for this - same lesson as N-MNIST being wrong for sparsity claims
in the neuromorphic thread (§29-30). This is a fresh, minimal GridWorld
instead: the smallest environment with genuine actions, verifiable ground
truth, and zero new dependencies (built from scratch, matching this
project's whole approach to RTL/assembler/etc.).

THIS SCRIPT IS STEP 1 ONLY: verify an action-conditioned JEPA can learn an
accurate, non-collapsed, position-informative latent dynamics model.
Planning/search is explicitly NOT attempted here - that's a later step,
only once this foundation is verified (same staged discipline as
dense_jepa.py -> collapse_diagnostics -> downstream_probe before anything
was built on top of it).
"""
import copy
import torch
import torch.nn as nn

from collapse_diagnostics import diagnose
from downstream_probe import train_linear_probe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID_SIZE = 8
STATE_DIM = GRID_SIZE * GRID_SIZE  # one-hot cell encoding
N_ACTIONS = 4  # up, down, left, right
HIDDEN_DIM = 64
REPR_DIM = 32
EMA_DECAY = 0.99
STEPS = 1500
LR = 3e-3
BATCH_SIZE = 128
SEED = 0


def step_env(x, y, action):
    if action == 0:   y = max(0, y - 1)              # up
    elif action == 1: y = min(GRID_SIZE - 1, y + 1)   # down
    elif action == 2: x = max(0, x - 1)               # left
    elif action == 3: x = min(GRID_SIZE - 1, x + 1)   # right
    return x, y


def encode_state_batch(xs, ys):
    idx = ys * GRID_SIZE + xs
    onehot = torch.zeros(len(xs), STATE_DIM)
    onehot[torch.arange(len(xs)), idx] = 1.0
    return onehot


def sample_batch(batch_size, rng):
    xs = rng.integers(0, GRID_SIZE, size=batch_size)
    ys = rng.integers(0, GRID_SIZE, size=batch_size)
    actions = rng.integers(0, N_ACTIONS, size=batch_size)
    nxs, nys = [], []
    for x, y, a in zip(xs, ys, actions):
        nx, ny = step_env(int(x), int(y), int(a))
        nxs.append(nx)
        nys.append(ny)
    state = encode_state_batch(torch.tensor(xs), torch.tensor(ys))
    next_state = encode_state_batch(torch.tensor(nxs), torch.tensor(nys))
    action_onehot = nn.functional.one_hot(torch.tensor(actions), N_ACTIONS).float()
    pos = torch.stack([torch.tensor(xs), torch.tensor(ys)], dim=1)
    return state, action_onehot, next_state, pos


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DynamicsPredictor(nn.Module):
    """Takes (current representation, action) -> predicted NEXT representation.
    This is the piece dense_jepa.py's plain Predictor doesn't have - it only
    ever took the context representation alone, no action input."""

    def __init__(self, repr_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim + action_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, repr_dim),
        )

    def forward(self, repr_, action_onehot):
        return self.net(torch.cat([repr_, action_onehot], dim=-1))


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


def cosine_sim(a, b):
    return nn.functional.cosine_similarity(a, b, dim=-1)


if __name__ == "__main__":
    import numpy as np

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    model = ActionConditionedJEPA(STATE_DIM, N_ACTIONS).to(DEVICE)
    opt = torch.optim.Adam(
        list(model.context_encoder.parameters()) + list(model.dynamics.parameters()), lr=LR
    )

    print("=== Training action-conditioned JEPA on GridWorld transitions ===")
    for step in range(STEPS):
        state, action, next_state, _ = sample_batch(BATCH_SIZE, rng)
        state, action, next_state = state.to(DEVICE), action.to(DEVICE), next_state.to(DEVICE)
        loss, ctx_repr, pred_repr, target_repr = model(state, action, next_state)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()
        if (step + 1) % 300 == 0:
            print(f"  step {step+1}/{STEPS} loss={loss.item():.6f}")

    print("\n=== Eval 1: collapse diagnostics on context_encoder representations ===")
    eval_state, eval_action, eval_next_state, eval_pos = sample_batch(1024, rng)
    with torch.no_grad():
        eval_repr = model.context_encoder(eval_state.to(DEVICE)).cpu()
    diag = diagnose(eval_repr, label="GridWorld context_repr")

    print("\n=== Eval 2: linear probe - does the representation encode TRUE position? ===")
    cell_idx = eval_pos[:, 1] * GRID_SIZE + eval_pos[:, 0]  # 64-way cell classification
    n_train = 768
    train_repr, test_repr = eval_repr[:n_train], eval_repr[n_train:]
    train_lab, test_lab = cell_idx[:n_train], cell_idx[n_train:]
    probe_result = train_linear_probe(train_repr, train_lab, test_repr, test_lab,
                                       num_classes=GRID_SIZE * GRID_SIZE, device=DEVICE, steps=1000)
    print(f"  probe test_acc (64-way cell classification): {probe_result['test_acc']*100:.1f}% "
          f"(chance level: {100/64:.1f}%)")

    print("\n=== Eval 3: dynamics prediction accuracy vs trivial baseline ===")
    with torch.no_grad():
        es, ea, ens = eval_state.to(DEVICE), eval_action.to(DEVICE), eval_next_state.to(DEVICE)
        ctx = model.context_encoder(es)
        pred = model.dynamics(ctx, ea)
        true_target = model.target_encoder(ens)
        # Trivial baseline: "the action doesn't matter, next repr = current repr"
        trivial_pred = ctx

        real_cos = cosine_sim(pred, true_target).mean().item()
        trivial_cos = cosine_sim(trivial_pred, true_target).mean().item()
        real_mse = torch.mean((pred - true_target) ** 2).item()
        trivial_mse = torch.mean((trivial_pred - true_target) ** 2).item()

    print(f"  Dynamics model cosine-sim to true next-state repr: {real_cos:.4f} (MSE={real_mse:.6f})")
    print(f"  Trivial 'action doesn't matter' baseline cosine-sim: {trivial_cos:.4f} (MSE={trivial_mse:.6f})")
    print(f"  Dynamics model beats trivial baseline: {real_cos > trivial_cos and real_mse < trivial_mse}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Collapse verdict: {'COLLAPSE SUSPECTED' if diag['collapse_suspected'] else 'no obvious collapse'}")
    print(f"Position-probe accuracy: {probe_result['test_acc']*100:.1f}% (chance {100/64:.1f}%)")
    print(f"Dynamics real vs trivial cosine-sim: {real_cos:.4f} vs {trivial_cos:.4f}")
    print("\nThis is STEP 1 ONLY (verify the world model), not planning. If this checks out,")
    print("next step is a simple search/planning loop over this verified latent dynamics model.")
