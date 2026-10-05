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


def run_dense(log_every=50):
    model = DenseJEPA(in_dim=IN_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print("\n=== Dense JEPA baseline ===")
    for step in range(1, STEPS + 1):
        batch = make_sequence_batch(BATCH_SIZE, SEQ_LEN, IN_DIM, DEVICE)
        context, target = split_context_target(batch)
        loss, _, _ = model(context, target)

        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()

        if step % log_every == 0:
            print(f"step {step:4d} | loss {loss.item():.5f}")
    return loss.item()


def run_spiking(log_every=50):
    model = SpikingJEPA(in_dim=IN_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print("\n=== Spiking/event-driven JEPA ===")
    for step in range(1, STEPS + 1):
        batch = make_sequence_batch(BATCH_SIZE, SEQ_LEN, IN_DIM, DEVICE)
        context, target = split_context_target(batch)
        loss, _, _, spike_rate = model(context, target)

        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()

        if step % log_every == 0:
            print(f"step {step:4d} | loss {loss.item():.5f} | spike_rate {spike_rate.item():.3f}")
    return loss.item(), spike_rate.item()


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    dense_final_loss = run_dense()
    spiking_final_loss, spiking_final_spike_rate = run_spiking()

    print("\n=== Summary ===")
    print(f"Dense final loss:   {dense_final_loss:.5f}  (spike_rate = 1.0, semua neuron 'nyala' tiap step)")
    print(f"Spiking final loss: {spiking_final_loss:.5f}  (spike_rate = {spiking_final_spike_rate:.3f})")
    print(
        "\nBaca hasilnya: kalau spiking loss mendekati dense loss TAPI spike_rate jauh di bawah 1.0,\n"
        "itu bukti awal bahwa representasi JEPA bisa dipelajari dengan compute yang jauh lebih sedikit\n"
        "(cuma sebagian neuron yang aktif), bukan klaim kosong -- ini angka yang bisa direproduksi ulang."
    )
