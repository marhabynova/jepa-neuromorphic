"""
Phase B: redesigned architecture (snn.Synaptic, §29) validated on SHD
(§29f: the right testbed, unlike N-MNIST) using the full protocol built up
through §15-28 -- collapse diagnostics + linear probe + trivial-constant
baseline as the dependent variable, NOT raw SSL loss. Also reports
per-layer spike rates so any future SynOps analysis doesn't have to assume
layer 1 is either dense or sparse -- measure it.

Dense vs Spiking JEPA, same protocol as run_phase_a.py/grid_sweep.py, ported
to SHD's dims (FLAT_DIM=700, N_TIME_BINS=20, 20 classes).
"""
import copy
import torch

from real_data_shd import load_shd, make_loader, FLAT_DIM, N_TIME_BINS
from dense_jepa import DenseJEPA
from gru_jepa import GRUJepa
from spiking_jepa_v2 import SpikingJEPA_V2
from collapse_diagnostics import diagnose
from downstream_probe import train_linear_probe, TrivialConstantEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_DIM = 128
REPR_DIM = 64
NUM_CLASSES = 20
BATCH_SIZE = 64
STEPS = 300
LR = 3e-3
SEED = 0


def split_context_target(x, context_frac=0.6):
    T = x.shape[1]
    k = int(T * context_frac)
    return x[:, :k, :], x[:, k:, :]


def train_jepa(model, is_spiking, train_loader, opt):
    data_iter = iter(train_loader)
    spike_rates, layer_rates_log = [], []
    for step in range(STEPS):
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, _ = next(data_iter)
        x = x.to(DEVICE)
        context, target = split_context_target(x)
        if is_spiking:
            loss, pred_repr, target_repr, sr, layer_rates = model(context, target)
            spike_rates.append(sr.item())
            layer_rates_log.append([r.item() for r in layer_rates])
        else:
            loss, pred_repr, target_repr = model(context, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()
        if step % 50 == 0:
            print(f"  step {step}: loss={loss.item():.6f}" + (f" spike_rate={sr.item():.4f}" if is_spiking else ""))
    avg_layer_rates = None
    if layer_rates_log:
        avg_layer_rates = [sum(r[i] for r in layer_rates_log) / len(layer_rates_log) for i in range(3)]
    return (sum(spike_rates) / len(spike_rates) if spike_rates else None), avg_layer_rates


@torch.no_grad()
def extract_embeddings(model, is_spiking, loader, n_batches):
    embs, labels = [], []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x = x.to(DEVICE)
        if is_spiking:
            repr_out, _, _ = model.context_encoder(x)
        else:
            repr_out = model.context_encoder(x)
        embs.append(repr_out.cpu())
        labels.append(y)
    return torch.cat(embs, dim=0), torch.cat(labels, dim=0)


def run_condition(name, is_spiking, model_cls=None):
    print(f"\n{'='*20} {name} {'='*20}")
    torch.manual_seed(SEED)
    train_set = load_shd(train=True)
    eval_set = load_shd(train=False)
    train_loader = make_loader(train_set, BATCH_SIZE, shuffle=True)
    eval_loader = make_loader(eval_set, BATCH_SIZE, shuffle=False)
    probe_train_loader = make_loader(train_set, BATCH_SIZE, shuffle=False)

    if is_spiking:
        model = SpikingJEPA_V2(FLAT_DIM, HIDDEN_DIM, REPR_DIM).to(DEVICE)
    else:
        model_cls = model_cls or DenseJEPA
        model = model_cls(FLAT_DIM, HIDDEN_DIM, REPR_DIM).to(DEVICE)
    opt = torch.optim.Adam(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()), lr=LR
    )

    print(f"Training {name} JEPA self-supervised ({STEPS} steps)...")
    avg_spike_rate, avg_layer_rates = train_jepa(model, is_spiking, train_loader, opt)
    if avg_spike_rate is not None:
        print(f"Avg spike rate (aggregate): {avg_spike_rate:.4f}")
        print(f"Avg per-layer spike rates: L1={avg_layer_rates[0]:.4f} L2={avg_layer_rates[1]:.4f} L3={avg_layer_rates[2]:.4f}")

    print("\nExtracting embeddings for collapse diagnostics + linear probe...")
    train_emb, train_lab = extract_embeddings(model, is_spiking, probe_train_loader, n_batches=32)
    test_emb, test_lab = extract_embeddings(model, is_spiking, eval_loader, n_batches=16)
    print(f"train_emb={train_emb.shape} test_emb={test_emb.shape}")

    diag = diagnose(train_emb, label=f"{name} JEPA, SHD, post-training")

    real_result = train_linear_probe(train_emb, train_lab, test_emb, test_lab, num_classes=NUM_CLASSES, device=DEVICE)
    trivial = TrivialConstantEncoder(REPR_DIM).to(DEVICE)
    with torch.no_grad():
        trivial_train = trivial(train_emb.to(DEVICE)).cpu()
        trivial_test = trivial(test_emb.to(DEVICE)).cpu()
    trivial_result = train_linear_probe(trivial_train, train_lab, trivial_test, test_lab, num_classes=NUM_CLASSES, device=DEVICE)

    print(f"\nReal encoder linear probe: {real_result}")
    print(f"Trivial baseline linear probe: {trivial_result}")
    majority_frac = torch.bincount(test_lab).max().item() / test_lab.shape[0]
    print(f"Majority-class floor: {majority_frac:.4f}")

    return {
        "name": name, "collapse_suspected": diag["collapse_suspected"],
        "real_test_acc": real_result["test_acc"], "trivial_test_acc": trivial_result["test_acc"],
        "majority_floor": majority_frac, "avg_spike_rate": avg_spike_rate,
        "avg_layer_rates": avg_layer_rates,
    }


if __name__ == "__main__":
    dense_result = run_condition("Dense", is_spiking=False)
    gru_result = run_condition("GRU-dense (control)", is_spiking=False, model_cls=GRUJepa)
    spiking_result = run_condition("Spiking-v2", is_spiking=True)

    print(f"\n{'='*20} SUMMARY {'='*20}")
    for r in (dense_result, gru_result, spiking_result):
        gap = r["real_test_acc"] - r["trivial_test_acc"]
        print(f"{r['name']}: collapse_suspected={r['collapse_suspected']} "
              f"real_acc={r['real_test_acc']:.4f} trivial_acc={r['trivial_test_acc']:.4f} "
              f"(gap={gap:+.4f}) majority_floor={r['majority_floor']:.4f}")
        if r["avg_spike_rate"] is not None:
            print(f"  spike_rate={r['avg_spike_rate']:.4f} per-layer={r['avg_layer_rates']}")
