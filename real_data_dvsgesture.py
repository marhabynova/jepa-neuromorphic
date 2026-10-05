"""
Replaces real_data.py's N-MNIST pipeline per §29f: N-MNIST is solvable from
spatial features alone (plain frame-based ANN gets 99.23%, no real temporal
structure needed) -- the wrong testbed for any sparsity/temporal claim.
DVS128 Gesture requires genuine temporal integration to classify.

Also implements the sigma-delta encoding Opus's research (§29f) identified as
the established fix for "dense input throws away sparsity" (Intel's own
SDNN/PilotNet result, not a novel idea of ours): instead of feeding the RAW
per-bin event-count frame into fc1 (what spiking_jepa.py/run_phase_a.py did --
confirmed in §27-28 to make layer 1 92% of compute and 0% sparse), feed the
frame-to-frame DELTA, thresholded to zero below a small cutoff. Most pixels
don't change between consecutive bins, so the delta is sparse -- this is what
makes layer 1's input genuinely sparse-native, not just the hidden layers.
"""
import torch
import tonic
import tonic.transforms as transforms

N_TIME_BINS = 20
DOWNSAMPLE_TO = 32  # DVS128 Gesture is natively 128x128; downsample to keep
                     # toy-scale compute comparable to the old N-MNIST setup
SENSOR_SIZE = (DOWNSAMPLE_TO, DOWNSAMPLE_TO, 2)
FLAT_DIM = SENSOR_SIZE[0] * SENSOR_SIZE[1] * SENSOR_SIZE[2]
DELTA_THRESHOLD = 0.5  # sigma-delta: changes smaller than this don't fire


def load_dvsgesture(save_to="./data_real", train=False):
    sensor_size_native = tonic.datasets.DVSGesture.sensor_size  # (128,128,2)
    frame_transform = transforms.Compose([
        transforms.Downsample(sensor_size=sensor_size_native, target_size=(DOWNSAMPLE_TO, DOWNSAMPLE_TO)),
        transforms.ToFrame(sensor_size=SENSOR_SIZE, n_time_bins=N_TIME_BINS),
    ])
    dataset = tonic.datasets.DVSGesture(save_to=save_to, train=train, transform=frame_transform)
    return dataset


def sigma_delta_encode(frames):
    """frames: (T, C, H, W) raw event-count per bin. Returns (T, C, H, W)
    sparse delta encoding: frame[0] kept as-is (first observation), frame[t>0]
    replaced by (frame[t] - frame[t-1]) thresholded to zero below
    DELTA_THRESHOLD in absolute value. This is the sigma-delta / graded-spike
    principle (§29f): only communicate CHANGES, not full values, at every step."""
    delta = frames.clone()
    diff = frames[1:] - frames[:-1]
    diff[diff.abs() < DELTA_THRESHOLD] = 0.0
    delta[1:] = diff
    return delta


def collate_to_tensor(batch):
    frames_list, labels = zip(*batch)
    tensors = []
    for f in frames_list:
        t = torch.from_numpy(f.astype("float32"))  # (T, C, H, W)
        t = sigma_delta_encode(t)
        tensors.append(t.reshape(N_TIME_BINS, -1))
    x = torch.stack(tensors, dim=0)  # (batch, T, flat_dim)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def make_loader(dataset, batch_size, shuffle):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_to_tensor,
        num_workers=0,
    )


if __name__ == "__main__":
    import torch as _torch

    ds = load_dvsgesture(train=True)
    print(f"DVS128 Gesture train set: {len(ds)} samples, {tonic.datasets.DVSGesture.sensor_size} native sensor size (downsampled to {SENSOR_SIZE})")

    def collate_raw(batch):
        frames_list, labels = zip(*batch)
        tensors = [_torch.from_numpy(f.astype("float32")).reshape(N_TIME_BINS, -1) for f in frames_list]
        return _torch.stack(tensors, dim=0), _torch.tensor(labels, dtype=_torch.long)

    _torch.manual_seed(0)
    loader_raw = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_raw, num_workers=0)
    x_raw, y_raw = next(iter(loader_raw))
    print(f"Batch shape: {x_raw.shape} (batch, T, flat_dim={FLAT_DIM})")
    print(f"Labels: {y_raw.tolist()}")
    print(f"RAW (no sigma-delta) non-zero fraction: {(x_raw != 0).float().mean().item():.4f}")

    _torch.manual_seed(0)
    loader_sd = make_loader(ds, batch_size=8, shuffle=True)
    x_sd, y_sd = next(iter(loader_sd))
    print(f"SIGMA-DELTA non-zero fraction: {(x_sd != 0).float().mean().item():.4f}")
