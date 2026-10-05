import statistics
import torch

from data import make_sequence_batch, split_context_target
from dense_jepa import DenseJEPA
from spiking_jepa import SpikingJEPA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IN_DIM = 8
SEQ_LEN = 20
BATCH_SIZE = 64
STEPS = 500
LR = 1e-3
DIFFICULTY = "hard"
SEEDS = list(range(10))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_dense(seed):
    torch.manual_seed(seed)
    model = DenseJEPA(in_dim=IN_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(STEPS):
        batch = make_sequence_batch(BATCH_SIZE, SEQ_LEN, IN_DIM, DEVICE, difficulty=DIFFICULTY)
        context, target = split_context_target(batch)
        loss, _, _ = model(context, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()
    return loss.item(), count_params(model)


def train_spiking(seed):
    torch.manual_seed(seed)
    model = SpikingJEPA(in_dim=IN_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    spike_rate_final = None
    for _ in range(STEPS):
        batch = make_sequence_batch(BATCH_SIZE, SEQ_LEN, IN_DIM, DEVICE, difficulty=DIFFICULTY)
        context, target = split_context_target(batch)
        loss, _, _, spike_rate = model(context, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()
        spike_rate_final = spike_rate.item()
    return loss.item(), spike_rate_final, count_params(model)


def summarize(name, values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    print(f"{name}: mean={mean:.5f}  std={std:.5f}  (n={len(values)})")
    return mean, std


if __name__ == "__main__":
    print(f"device: {DEVICE} | difficulty: {DIFFICULTY} | seeds: {SEEDS}\n")

    dense_losses, dense_params = [], None
    for s in SEEDS:
        loss, params = train_dense(s)
        dense_losses.append(loss)
        dense_params = params
        print(f"[dense]   seed {s}: loss={loss:.5f}")

    print()
    spiking_losses, spiking_rates, spiking_params = [], [], None
    for s in SEEDS:
        loss, rate, params = train_spiking(s)
        spiking_losses.append(loss)
        spiking_rates.append(rate)
        spiking_params = params
        print(f"[spiking] seed {s}: loss={loss:.5f}  spike_rate={rate:.3f}")

    print(f"\n=== Summary (task difficulty = {DIFFICULTY}, {len(SEEDS)} seeds) ===")
    print(f"Dense params:   {dense_params}")
    print(f"Spiking params: {spiking_params}")
    dense_mean, dense_std = summarize("Dense loss  ", dense_losses)
    spiking_mean, spiking_std = summarize("Spiking loss", spiking_losses)
    summarize("Spiking spike_rate", spiking_rates)

    gap = spiking_mean - dense_mean
    print(f"\nLoss gap (spiking - dense): {gap:+.5f}")
    if abs(gap) < max(dense_std, spiking_std):
        print("-> Gap kecil dibanding variasi antar-seed sendiri: BELUM bisa klaim salah satu 'lebih baik'.")
    elif gap > 0:
        print("-> Spiking konsisten lebih buruk akurasinya dibanding dense pada task yang lebih sulit ini.")
    else:
        print("-> Spiking konsisten SAMA/LEBIH BAIK akurasinya, dengan compute jauh lebih sedikit (spike_rate < 1.0).")
