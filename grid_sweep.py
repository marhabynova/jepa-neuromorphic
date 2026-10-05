"""
Fase A perluasan: grid sweep buat ngecek apakah collapse SpikingJEPA (RESEARCH_NOTES.md §16)
itu bug tuning (beta/LR salah) atau beneran konsisten gagal di 12 kombinasi x 3 seed.
Simetris: Dense juga disweep 3 LR x 3 seed walau udah kerja, biar kesempatannya sama.

Laporan WAJIB semua 45 kombinasi -- bukan cherry-pick yang terbaik.
"""
import copy
import statistics
import torch

from real_data import load_nmnist, make_loader
from collapse_diagnostics import diagnose
from downstream_probe import train_linear_probe, TrivialConstantEncoder
from run_phase_a import DenseEncoder, SpikingEncoder, Predictor, split_context_target, DEVICE, BATCH_SIZE, REPR_DIM, STEPS

SPIKING_BETAS = [0.5, 0.7, 0.9, 0.95]
SPIKING_LRS = [1e-3, 3e-3, 1e-2]
DENSE_LRS = [1e-3, 3e-3, 1e-2]
SEEDS = [0, 1, 2]


def train_one(encoder_fn, is_spiking, lr, seed, train_loader, ema_decay=0.99):
    torch.manual_seed(seed)
    context_encoder = encoder_fn().to(DEVICE)
    target_encoder = copy.deepcopy(context_encoder).to(DEVICE)
    for p in target_encoder.parameters():
        p.requires_grad = False
    predictor = Predictor().to(DEVICE)
    opt = torch.optim.Adam(list(context_encoder.parameters()) + list(predictor.parameters()), lr=lr)

    data_iter = iter(train_loader)
    spike_rates = []
    for _ in range(STEPS):
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, _ = next(data_iter)
        x = x.to(DEVICE)
        context, target = split_context_target(x)
        if is_spiking:
            ctx_repr, sr = context_encoder(context)
            spike_rates.append(sr)
        else:
            ctx_repr = context_encoder(context)
        pred_repr = predictor(ctx_repr)
        with torch.no_grad():
            target_repr = target_encoder(target)[0] if is_spiking else target_encoder(target)
        loss = torch.mean((pred_repr - target_repr) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p_t, p_c in zip(target_encoder.parameters(), context_encoder.parameters()):
                p_t.data.mul_(ema_decay).add_(p_c.data, alpha=1 - ema_decay)

    avg_sr = sum(spike_rates) / len(spike_rates) if spike_rates else None
    return context_encoder, avg_sr


@torch.no_grad()
def extract_embeddings(encoder, is_spiking, loader):
    embeddings, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        repr_ = encoder(x)[0] if is_spiking else encoder(x)
        embeddings.append(repr_.cpu())
        labels.append(y)
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def run_combo(name, encoder_fn, is_spiking, lr, seed, train_loader, probe_train_loader, eval_loader, triv_acc):
    encoder, avg_sr = train_one(encoder_fn, is_spiking, lr, seed, train_loader)
    train_emb, train_lbl = extract_embeddings(encoder, is_spiking, probe_train_loader)
    eval_emb, eval_lbl = extract_embeddings(encoder, is_spiking, eval_loader)

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        diag = diagnose(train_emb, label=f"{name} beta/lr={lr} seed={seed}")
    probe = train_linear_probe(train_emb, train_lbl, eval_emb, eval_lbl, num_classes=10, device=DEVICE)

    collapsed = diag["collapse_suspected"]
    beats_trivial = probe["test_acc"] > triv_acc + 0.05
    return {
        "test_acc": probe["test_acc"],
        "collapsed": collapsed,
        "beats_trivial": beats_trivial,
        "effective_rank": diag["rank"]["effective_rank"],
        "avg_spike_rate": avg_sr,
    }


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    full = load_nmnist(train=False)
    n = len(full)
    train_set, eval_set = torch.utils.data.random_split(
        full, [int(n * 0.7), n - int(n * 0.7)], generator=torch.Generator().manual_seed(0)
    )
    train_loader = make_loader(train_set, BATCH_SIZE, shuffle=True)
    eval_loader = make_loader(eval_set, BATCH_SIZE, shuffle=False)
    probe_train_loader = make_loader(train_set, BATCH_SIZE, shuffle=False)

    print("Trivial-constant baseline (dipakai sebagai acuan semua kombinasi)...")
    trivial = TrivialConstantEncoder(REPR_DIM).to(DEVICE)
    t_train_emb, t_train_lbl = extract_embeddings(trivial, False, probe_train_loader)
    t_eval_emb, t_eval_lbl = extract_embeddings(trivial, False, eval_loader)
    triv_probe = train_linear_probe(t_train_emb, t_train_lbl, t_eval_emb, t_eval_lbl, num_classes=10, device=DEVICE)
    triv_acc = triv_probe["test_acc"]
    print(f"Trivial-constant test_acc: {triv_acc:.4f}\n")

    all_results = {"spiking": {}, "dense": {}}

    print("=== SPIKING GRID: 4 beta x 3 LR x 3 seed = 36 run ===")
    for beta in SPIKING_BETAS:
        for lr in SPIKING_LRS:
            key = (beta, lr)
            runs = []
            for seed in SEEDS:
                r = run_combo("Spiking", lambda: SpikingEncoder(beta=beta), True, lr, seed,
                              train_loader, probe_train_loader, eval_loader, triv_acc)
                runs.append(r)
            accs = [r["test_acc"] for r in runs]
            n_collapsed = sum(r["collapsed"] for r in runs)
            n_beats = sum(r["beats_trivial"] for r in runs)
            mean_acc = statistics.mean(accs)
            std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
            all_results["spiking"][key] = runs
            print(f"beta={beta} lr={lr}: acc={mean_acc:.4f}+-{std_acc:.4f} | "
                  f"collapsed={n_collapsed}/3 | beats_trivial={n_beats}/3")

    print("\n=== DENSE GRID: 3 LR x 3 seed = 9 run (simetris) ===")
    for lr in DENSE_LRS:
        key = lr
        runs = []
        for seed in SEEDS:
            r = run_combo("Dense", DenseEncoder, False, lr, seed,
                          train_loader, probe_train_loader, eval_loader, triv_acc)
            runs.append(r)
        accs = [r["test_acc"] for r in runs]
        n_collapsed = sum(r["collapsed"] for r in runs)
        n_beats = sum(r["beats_trivial"] for r in runs)
        mean_acc = statistics.mean(accs)
        std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
        all_results["dense"][key] = runs
        print(f"lr={lr}: acc={mean_acc:.4f}+-{std_acc:.4f} | collapsed={n_collapsed}/3 | beats_trivial={n_beats}/3")

    print("\n" + "=" * 70)
    print("KESIMPULAN AKHIR (semua 45 kombinasi dilaporkan, bukan cherry-pick)")
    print("=" * 70)
    spiking_reliable = [k for k, runs in all_results["spiking"].items()
                        if sum(r["beats_trivial"] for r in runs) == 3 and sum(r["collapsed"] for r in runs) == 0]
    dense_reliable = [k for k, runs in all_results["dense"].items()
                      if sum(r["beats_trivial"] for r in runs) == 3 and sum(r["collapsed"] for r in runs) == 0]

    print(f"Spiking: {len(spiking_reliable)}/12 kombinasi RELIABLE (3/3 seed lolos collapse DAN ngungguli trivial)")
    print(f"  -> kombinasi reliable: {spiking_reliable if spiking_reliable else '(tidak ada)'}")
    print(f"Dense: {len(dense_reliable)}/3 kombinasi RELIABLE")
    print(f"  -> kombinasi reliable: {dense_reliable if dense_reliable else '(tidak ada)'}")

    if len(spiking_reliable) == 0:
        print("\n-> SpikingJEPA GAGAL konsisten di SEMUA 12 kombinasi hyperparameter yang dicoba.")
        print("   Ini BUKAN cuma 1 setting yang kebetulan buruk -- konfirmasi lebih kuat dari §16.")
    else:
        print(f"\n-> SpikingJEPA BISA jalan di {len(spiking_reliable)}/12 kombinasi -- §16 perlu DIREVISI,")
        print("   collapse kemarin itu bug tuning, bukan limitasi arsitektur.")
