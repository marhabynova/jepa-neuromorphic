"""
Fase A: jawab apakah temuan "spiking menang" di eksperimen sintetis (RESEARCH_NOTES.md §5)
itu collapse artifact atau beneran, diukur di data event NYATA (N-MNIST) pakai metodologi
yang Opus audit rekomendasikan: downstream linear probe + collapse diagnostics wajib,
BUKAN raw self-supervised loss.

Hipotesis dinyatakan SEBELUM run (lihat RESEARCH_NOTES.md untuk draft plan lengkap):
- Trivial-constant encoder setara/lebih baik dari SpikingJEPA di probe -> collapse terkonfirmasi.
- Effective rank SpikingJEPA rendah (top-1 singular value > 90% variance) -> collapse terkonfirmasi.
- SpikingJEPA ngungguli DUA baseline (trivial-constant DAN dense) dengan rank sehat -> lanjut Fase B.
"""
import copy
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from real_data import load_nmnist, make_loader, FLAT_DIM, N_TIME_BINS
from collapse_diagnostics import diagnose
from downstream_probe import train_linear_probe, TrivialConstantEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPR_DIM = 64
HIDDEN_DIM = 128
STEPS = 300
LR = 1e-3
BATCH_SIZE = 64


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DenseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FLAT_DIM, HIDDEN_DIM), nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.GELU(),
            nn.Linear(HIDDEN_DIM, REPR_DIM),
        )

    def forward(self, x):  # x: (batch, T, FLAT_DIM)
        return self.net(x).mean(dim=1)


class SpikingEncoder(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(FLAT_DIM, HIDDEN_DIM)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(HIDDEN_DIM, REPR_DIM)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):  # x: (batch, T, FLAT_DIM)
        batch, T, _ = x.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_accum = torch.zeros(batch, REPR_DIM, device=x.device)
        n_spikes = x.new_zeros(())
        n_possible = x.new_zeros(())
        for t in range(T):
            cur1 = self.fc1(x[:, t, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_accum = spk3_accum + spk3
            n_spikes = n_spikes + spk1.sum() + spk2.sum() + spk3.sum()
            n_possible = n_possible + spk1.numel() + spk2.numel() + spk3.numel()
        return spk3_accum / T, (n_spikes / n_possible).item()


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(REPR_DIM, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, REPR_DIM))

    def forward(self, x):
        return self.net(x)


def split_context_target(x, context_frac=0.6):
    T = x.shape[1]
    k = int(T * context_frac)
    return x[:, :k, :], x[:, k:, :]


def train_jepa(encoder_fn, is_spiking, train_loader, ema_decay=0.99):
    context_encoder = encoder_fn().to(DEVICE)
    target_encoder = copy.deepcopy(context_encoder).to(DEVICE)
    for p in target_encoder.parameters():
        p.requires_grad = False
    predictor = Predictor().to(DEVICE)
    opt = torch.optim.Adam(list(context_encoder.parameters()) + list(predictor.parameters()), lr=LR)

    step = 0
    spike_rate_log = []
    data_iter = iter(train_loader)
    while step < STEPS:
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, _ = next(data_iter)
        x = x.to(DEVICE)
        context, target = split_context_target(x)

        if is_spiking:
            ctx_repr, spike_rate = context_encoder(context)
            spike_rate_log.append(spike_rate)
        else:
            ctx_repr = context_encoder(context)
        pred_repr = predictor(ctx_repr)
        with torch.no_grad():
            if is_spiking:
                target_repr, _ = target_encoder(target)
            else:
                target_repr = target_encoder(target)
        loss = torch.mean((pred_repr - target_repr) ** 2)

        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p_t, p_c in zip(target_encoder.parameters(), context_encoder.parameters()):
                p_t.data.mul_(ema_decay).add_(p_c.data, alpha=1 - ema_decay)
        step += 1

    avg_spike_rate = sum(spike_rate_log) / len(spike_rate_log) if spike_rate_log else None
    return context_encoder, avg_spike_rate, count_params(context_encoder)


@torch.no_grad()
def extract_embeddings(encoder, is_spiking, loader):
    embeddings, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        if is_spiking:
            repr_, _ = encoder(x)
        else:
            repr_ = encoder(x)
        embeddings.append(repr_.cpu())
        labels.append(y)
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    print("Loading N-MNIST (test split dulu, lebih kecil)...")
    full_dataset = load_nmnist(train=False)

    n = len(full_dataset)
    n_train = int(n * 0.7)
    train_set, eval_set = torch.utils.data.random_split(
        full_dataset, [n_train, n - n_train], generator=torch.Generator().manual_seed(0)
    )
    train_loader = make_loader(train_set, BATCH_SIZE, shuffle=True)
    eval_loader = make_loader(eval_set, BATCH_SIZE, shuffle=False)
    probe_train_loader = make_loader(train_set, BATCH_SIZE, shuffle=False)

    print(f"train={len(train_set)} eval={len(eval_set)}")

    results = {}

    for name, encoder_fn, is_spk in [
        ("Dense", DenseEncoder, False),
        ("Spiking", SpikingEncoder, True),
    ]:
        print(f"\n--- Training {name} JEPA self-supervised ({STEPS} steps) ---")
        encoder, spike_rate, n_params = train_jepa(encoder_fn, is_spk, train_loader)
        print(f"{name}: params={n_params}" + (f" avg_spike_rate={spike_rate:.4f}" if spike_rate else ""))

        train_emb, train_lbl = extract_embeddings(encoder, is_spk, probe_train_loader)
        eval_emb, eval_lbl = extract_embeddings(encoder, is_spk, eval_loader)

        diag = diagnose(train_emb, label=f"{name} (train embeddings)")

        probe_result = train_linear_probe(train_emb, train_lbl, eval_emb, eval_lbl, num_classes=10, device=DEVICE)
        print(f"{name} downstream probe: train_acc={probe_result['train_acc']:.4f} test_acc={probe_result['test_acc']:.4f}")

        results[name] = {"diag": diag, "probe": probe_result, "spike_rate": spike_rate, "params": n_params}

    print("\n--- Trivial-constant baseline (mandatory control) ---")
    trivial = TrivialConstantEncoder(REPR_DIM).to(DEVICE)
    train_emb_triv, train_lbl_triv = extract_embeddings(trivial, False, probe_train_loader)
    eval_emb_triv, eval_lbl_triv = extract_embeddings(trivial, False, eval_loader)
    diag_triv = diagnose(train_emb_triv, label="Trivial-constant")
    probe_triv = train_linear_probe(train_emb_triv, train_lbl_triv, eval_emb_triv, eval_lbl_triv, num_classes=10, device=DEVICE)
    print(f"Trivial-constant downstream probe: train_acc={probe_triv['train_acc']:.4f} test_acc={probe_triv['test_acc']:.4f}")
    results["Trivial"] = {"diag": diag_triv, "probe": probe_triv}

    print("\n" + "=" * 70)
    print("VERDICT (berdasarkan kriteria yang dinyatakan SEBELUM run ini)")
    print("=" * 70)
    dense_acc = results["Dense"]["probe"]["test_acc"]
    spk_acc = results["Spiking"]["probe"]["test_acc"]
    triv_acc = results["Trivial"]["probe"]["test_acc"]
    spk_collapse = results["Spiking"]["diag"]["collapse_suspected"]
    dense_collapse = results["Dense"]["diag"]["collapse_suspected"]

    print(f"Trivial-constant test_acc: {triv_acc:.4f}")
    print(f"Dense    test_acc: {dense_acc:.4f}  (collapse_suspected={dense_collapse})")
    print(f"Spiking  test_acc: {spk_acc:.4f}  (collapse_suspected={spk_collapse})")

    if spk_acc <= triv_acc + 0.05:
        print("\n-> SpikingJEPA TIDAK ngungguli trivial-constant baseline secara meaningful.")
        print("   COLLAPSE TERKONFIRMASI. Klaim '§5 spiking menang' resmi DITARIK.")
    elif spk_collapse:
        print("\n-> SpikingJEPA ngungguli trivial baseline TAPI collapse diagnostic nunjuk collapse.")
        print("   Hasil AMBIGU, butuh investigasi lebih lanjut sebelum dipercaya.")
    elif spk_acc > dense_acc and spk_acc > triv_acc:
        print("\n-> SpikingJEPA ngungguli DUA baseline dengan embedding rank sehat.")
        print("   Temuan asli MUNGKIN sebagian real. Lanjut ke Fase B (S4/Mamba control, dataset lain).")
    else:
        print("\n-> Hasil tidak mendukung klaim 'spiking menang' dalam bentuk apa pun yang jelas.")
