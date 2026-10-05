import statistics
import torch

from data import make_sequence_batch, split_context_target
from dense_jepa import DenseJEPA
from spiking_jepa import SpikingJEPA
from gru_jepa import GRUJepa

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


def train_generic(model_fn, seed, is_spiking=False):
    torch.manual_seed(seed)
    model = model_fn().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    spike_rate_final = None
    for _ in range(STEPS):
        batch = make_sequence_batch(BATCH_SIZE, SEQ_LEN, IN_DIM, DEVICE, difficulty=DIFFICULTY)
        context, target = split_context_target(batch)
        if is_spiking:
            loss, _, _, spike_rate = model(context, target)
            spike_rate_final = spike_rate.item()
        else:
            loss, _, _ = model(context, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()
    return loss.item(), spike_rate_final, count_params(model)


def summarize(name, values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    print(f"{name}: mean={mean:.5f}  std={std:.5f}  (n={len(values)})")
    return mean, std


if __name__ == "__main__":
    print(f"device: {DEVICE} | difficulty: {DIFFICULTY} | seeds: {SEEDS}\n")

    configs = [
        ("MLP dense  (no temporal integration, no sparsity)", lambda: DenseJEPA(in_dim=IN_DIM), False),
        ("GRU dense  (temporal integration, NO sparsity)   ", lambda: GRUJepa(in_dim=IN_DIM, hidden_dim=96), False),
        ("Spiking    (temporal integration, WITH sparsity) ", lambda: SpikingJEPA(in_dim=IN_DIM), True),
    ]

    results = {}
    for name, fn, is_spk in configs:
        losses, rates, params = [], [], None
        for s in SEEDS:
            loss, rate, p = train_generic(fn, s, is_spiking=is_spk)
            losses.append(loss)
            if rate is not None:
                rates.append(rate)
            params = p
        results[name] = (losses, rates, params)
        print(f"[{name}] params={params}")
        for s, l in zip(SEEDS, losses):
            extra = f"  spike_rate={rates[SEEDS.index(s)]:.3f}" if rates else ""
            print(f"  seed {s}: loss={l:.5f}{extra}")
        print()

    print("=== Summary ===")
    for name, (losses, rates, params) in results.items():
        mean, std = summarize(f"{name} loss", losses)
        if rates:
            summarize(f"{name} spike_rate", rates)

    mlp_mean = statistics.mean(results[configs[0][0]][0])
    gru_mean = statistics.mean(results[configs[1][0]][0])
    spk_mean = statistics.mean(results[configs[2][0]][0])

    print(f"\n=== Interpretasi ===")
    print(f"MLP dense loss:   {mlp_mean:.5f}  (baseline, tanpa temporal integration)")
    print(f"GRU dense loss:   {gru_mean:.5f}  (temporal integration SAJA, tanpa sparsity)")
    print(f"Spiking loss:     {spk_mean:.5f}  (temporal integration + sparsity)")

    gru_vs_mlp = mlp_mean - gru_mean
    spk_vs_gru = gru_mean - spk_mean

    if gru_vs_mlp > 0:
        print(f"\nGRU dense MENGALAHKAN MLP dense (gap={gru_vs_mlp:.5f}) -> sebagian besar")
        print("keunggulan 'spiking' kemarin kemungkinan besar dari TEMPORAL INTEGRATION,")
        print("bukan sparsity -- confound di RESEARCH_NOTES.md §5 TERKONFIRMASI.")
    else:
        print(f"\nGRU dense TIDAK mengalahkan MLP dense (gap={gru_vs_mlp:.5f}) -> temporal")
        print("integration doang BUKAN penjelasannya -- confound-nya salah, perlu dicari ulang.")

    if spk_vs_gru > 0:
        print(f"Spiking MASIH mengalahkan GRU dense (gap={spk_vs_gru:.5f}) -> ada sesuatu DI LUAR")
        print("temporal integration yang bikin spiking unggul (sparsity, atau efek lain).")
    else:
        print(f"Spiking TIDAK mengalahkan GRU dense (gap={spk_vs_gru:.5f}) -> sparsity TIDAK")
        print("menambah keunggulan apa pun di atas temporal integration biasa.")
