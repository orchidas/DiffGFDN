import numpy as np
import torch
from torch.utils import data


class ColorlessFDNDataset(data.Dataset):
    """
    Dataset of frequency samples (in rads) sampled at linearly spaced 
    points along the unit circle
    """

    def __init__(self, num_freq_samples: int, device: torch.device):
        """
        Args:
        num_freq_samples (int): number of frequency samples along the unit circle
        device (torch.device): the training device
        """
        angle = torch.arange(0, 1, 1 / num_freq_samples)
        mag = torch.ones(num_freq_samples)
        self.labels = torch.ones(num_freq_samples)
        self.input = torch.polar(mag, angle * np.pi)

        self.input = self.input.to(device)
        self.labels = self.labels.to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # select sample
        y = self.labels[index]
        x = self.input[index]

        return x, y


def split_dataset(dataset: data.Dataset, split: float):
    """Randomly split a dataset into non-overlapping new datasets of sizes given in 'split' argument"""
    # use split % of dataset for validation
    train_set_size = int(len(dataset) * split)
    valid_set_size = len(dataset) - train_set_size

    seed = torch.Generator(device='cpu').manual_seed(42)
    train_set, valid_set = data.random_split(dataset,
                                             [train_set_size, valid_set_size],
                                             generator=seed)

    return train_set, valid_set


def get_dataloader(dataset: data.Dataset, batch_size: int, device:'cpu', shuffle=True):
    """Create torch dataloader form given dataset"""
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator(device=device),
        drop_last=True)
    return dataloader


def get_device():
    """Output 'cuda' if gpu is available, 'cpu' otherwise"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_colorless_fdn_dataset(
    num_freq_samples: int,
    device: torch.device,
    train_valid_split_ratio: float,
    batch_size: int,
    shuffle: bool = True,
):
    """Get training and valitation dataset"""
    dataset = ColorlessFDNDataset(num_freq_samples, device)
    # split data into training and validation set
    train_set, valid_set = split_dataset(dataset, train_valid_split_ratio)

    # dataloaders
    train_loader = get_dataloader(
        train_set,
        batch_size=batch_size,
        device=device,
        shuffle=shuffle,
    )

    valid_loader = get_dataloader(
        valid_set,
        batch_size=batch_size,
        device=device,
        shuffle=shuffle,
    )
    return train_loader, valid_loader
