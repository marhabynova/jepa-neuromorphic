import torch
import tonic
import tonic.transforms as transforms

N_TIME_BINS = 20
SENSOR_SIZE = (34, 34, 2)  # N-MNIST: 34x34 pixels, 2 polarity channels
FLAT_DIM = SENSOR_SIZE[0] * SENSOR_SIZE[1] * SENSOR_SIZE[2]


def load_nmnist(save_to="./data_real", train=False):
    frame_transform = transforms.ToFrame(sensor_size=SENSOR_SIZE, n_time_bins=N_TIME_BINS)
    dataset = tonic.datasets.NMNIST(save_to=save_to, train=train, transform=frame_transform)
    return dataset


def collate_to_tensor(batch):
    """Each item is (frames, label) where frames: (T, C, H, W) numpy, variable
    event count per frame but FIXED T from ToFrame(n_time_bins=...). Stack into
    (batch, T, flat_dim) float tensor + (batch,) long label tensor."""
    frames_list, labels = zip(*batch)
    tensors = [torch.from_numpy(f.astype("float32")).reshape(N_TIME_BINS, -1) for f in frames_list]
    x = torch.stack(tensors, dim=0)  # (batch, T, flat_dim)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def make_loader(dataset, batch_size, shuffle):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_to_tensor,
        num_workers=0,
    )
