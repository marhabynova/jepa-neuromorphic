"""
Second attempt per §29f after DVS128 Gesture's figshare download got blocked
by bot-protection (AWS WAF challenge, not a network issue on our end).
Spiking Heidelberg Digits (SHD, Cramer et al. 2020): spoken digit audio
(English+German, 20 classes) encoded via a cochlea model into 700 spike
channels -- genuinely needs temporal structure to classify (can't be solved
from a single static frame, unlike N-MNIST), and the raw event stream is
ALREADY much sparser than vision frame data before any sigma-delta trick,
since most of the 700 frequency channels are silent at any given instant.
Hosted at zenkelab.org, not figshare -- avoids the DVS Gesture download block.
"""
import torch
import tonic
import tonic.transforms as transforms

N_TIME_BINS = 20
SENSOR_SIZE = (700, 1, 1)  # SHD: 700 cochlea frequency channels, no spatial structure
FLAT_DIM = SENSOR_SIZE[0] * SENSOR_SIZE[1] * SENSOR_SIZE[2]
DELTA_THRESHOLD = 0.5


def load_shd(save_to="./data_real", train=False):
    frame_transform = transforms.ToFrame(sensor_size=SENSOR_SIZE, n_time_bins=N_TIME_BINS)
    dataset = tonic.datasets.SHD(save_to=save_to, train=train, transform=frame_transform)
    return dataset


def sigma_delta_encode(frames):
    delta = frames.clone()
    diff = frames[1:] - frames[:-1]
    diff[diff.abs() < DELTA_THRESHOLD] = 0.0
    delta[1:] = diff
    return delta


def collate_to_tensor(batch, use_sigma_delta=True):
    frames_list, labels = zip(*batch)
    tensors = []
    for f in frames_list:
        t = torch.from_numpy(f.astype("float32"))
        if use_sigma_delta:
            t = sigma_delta_encode(t)
        tensors.append(t.reshape(N_TIME_BINS, -1))
    x = torch.stack(tensors, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def make_loader(dataset, batch_size, shuffle, use_sigma_delta=True):
    from functools import partial
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=partial(collate_to_tensor, use_sigma_delta=use_sigma_delta),
        num_workers=0,
    )


if __name__ == "__main__":
    ds = load_shd(train=True)
    print(f"SHD train set: {len(ds)} samples, sensor_size={SENSOR_SIZE}, FLAT_DIM={FLAT_DIM}")
    labels = [ds[i][1] for i in range(min(200, len(ds)))]
    print(f"Num unique classes (first 200 samples): {len(set(labels))}")

    loader_raw = make_loader(ds, batch_size=8, shuffle=True, use_sigma_delta=False)
    x_raw, y_raw = next(iter(loader_raw))
    print(f"\nBatch shape: {x_raw.shape}")
    print(f"RAW (no sigma-delta) non-zero fraction: {(x_raw != 0).float().mean().item():.4f}")

    torch.manual_seed(0)
    loader_sd = make_loader(ds, batch_size=8, shuffle=True, use_sigma_delta=True)
    x_sd, y_sd = next(iter(loader_sd))
    print(f"SIGMA-DELTA non-zero fraction: {(x_sd != 0).float().mean().item():.4f}")
