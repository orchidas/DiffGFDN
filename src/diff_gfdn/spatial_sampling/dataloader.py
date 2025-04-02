import math

from loguru import logger
import numpy as np
import torch
from torch.utils import data

from ..dataloader import get_dataloader, InputFeatures, RoomDataset, to_device


class SpatialSamplingDataset(data.Dataset):

    def __init__(
        self,
        device: torch.device,
        room_data: RoomDataset,
    ):
        """
        Spatial sampling dataset containing the common slope amplitudes
        for different receiver positions. During batch processing, each batch will contain all the frequency bins
        but different sets of receiver positions
        Args:
            device (str): cuda or cpu
            room_data (RoomDataset): object of the room dataset class
                        containing information about the RIRs and source and listener positions

        """
        # spatial data
        self.source_position = torch.tensor(room_data.source_position)
        # source position has to be 2D for proper length calculation
        self.source_position = self.source_position.unsqueeze(
            0) if self.source_position.dim() == 1 else self.source_position
        self.listener_positions = torch.tensor(room_data.receiver_position)
        self.norm_listener_position = torch.tensor(
            room_data.norm_receiver_position)
        self.mesh_3D = room_data.mesh_3D
        self.common_slope_amps = torch.tensor(room_data.amplitudes)
        self.device = device
        self.grid_resolution_m = room_data.grid_spacing_m
        self.room_start_coords = room_data.room_start_coord
        self.room_dims = room_data.room_dims

        # if we have multiple sources in the dataset
        if self.source_position.dim() > 1:
            # Generate all valid (idx1, idx2) pairs
            self.index_pairs = [(i, j)
                                for i in range(len(self.source_position))
                                for j in range(len(self.listener_positions))]
        # frequency-domain data
        freq_bins_rad = torch.tensor(room_data.freq_bins_rad)
        self.z_values = torch.polar(torch.ones_like(freq_bins_rad),
                                    freq_bins_rad)

    def __len__(self):
        """Get length of dataset (equal to number of receiver positions)"""
        return self.source_position.shape[0] * self.listener_positions.shape[0]

    def __getitem__(self, idx: int):
        """Get data at a particular index"""
        # Return an instance of InputFeatures

        if self.source_position.shape[0] == 1:
            input_features = InputFeatures(self.z_values,
                                           torch.squeeze(self.source_position),
                                           self.listener_positions[idx],
                                           self.norm_listener_position[idx],
                                           self.mesh_3D)
            target_labels = self.common_slope_amps[idx, :]
        else:
            idx1, idx2 = self.index_pairs[idx]
            input_features = InputFeatures(self.z_values,
                                           self.source_position[idx1],
                                           self.listener_positions[idx2],
                                           self.norm_listener_position[idx2],
                                           self.mesh_3D)
            target_labels = self.common_slope_amps[idx1, idx2, :]

        return {'input': input_features, 'target': target_labels}


def custom_collate_spatial_sampling(batch: data.Dataset):
    """
    Collate datapoints in the dataloader.
    This method is needed because Meshgrid is a custom class, and pytorch's 
    built in collate function cannot collate it
    """
    # these are independent of the source/receiver locations
    z_values = batch[0]['input'].z_values

    # mesh_3D is a Meshgrid object with attributes xmesh, ymesh, zmesh
    x_mesh = batch[0]['input'].mesh_3D.xmesh
    y_mesh = batch[0]['input'].mesh_3D.ymesh
    z_mesh = batch[0]['input'].mesh_3D.zmesh

    # this is of size (Lx * Ly * Lz) x 3
    mesh_3D_data = torch.stack((x_mesh, y_mesh, z_mesh), dim=1)

    # depends on source/receiver positions
    source_positions = [item['input'].source_position for item in batch]
    listener_positions = [item['input'].listener_position for item in batch]
    norm_listener_positions = [
        item['input'].norm_listener_position for item in batch
    ]
    target_amplitudes = [item['target'] for item in batch]

    return {
        'z_values': z_values,
        'source_position': torch.stack(source_positions),
        'listener_position': torch.stack(listener_positions),
        'norm_listener_position': torch.stack(norm_listener_positions),
        'mesh_3D': mesh_3D_data,
        'target_common_slope_amps': torch.stack(target_amplitudes),
    }


def is_multiple(value, d, tol=1e-6):
    """Check if value is an approximate multiple of d."""
    return math.isclose(value / d, round(value / d), abs_tol=tol)


def find_start_coords(num_rooms: int, dataset: data.Dataset):
    """Find the first receiver location in each room"""
    start_xcoord = -np.ones(num_rooms)
    start_ycoord = -np.ones(num_rooms)
    for k in range(num_rooms):
        for idx in range(len(dataset)):
            listener_position = dataset.listener_positions[idx]  # (x, y, z)
            x_pos, y_pos = listener_position[:2].tolist()  # Extract x, y

            # Identify which room the point belongs to
            room_start_x, room_start_y = dataset.room_start_coords[k][:2]
            room_width, room_height = dataset.room_dims[k][:2]

            if (room_start_x <= x_pos < room_start_x + room_width
                    and room_start_y <= y_pos < room_start_y + room_height):

                start_xcoord[k], start_ycoord[k] = dataset.listener_positions[
                    idx, :2]
                break

    return (start_xcoord, start_ycoord)


def split_dataset_by_resolution(
    dataset: data.Dataset,
    x_d: float,
):
    """
    Split dataset into training and validation based on x_d resolution.
    If measurements are avaialable in a uniform 2D gri, the measurements at every 
    x_d m is used for training and the rest is used for validation 

    dataset: an instance of SpatialDataset.
    x_d: Spacing between selected training points.
    shuffle (bool): whether to shuffle the indices
    """
    assert x_d >= dataset.grid_resolution_m, "The desired grid spacing must be greater '\
    'than what has been measured in the dataset"

    train_indices = []
    val_indices = []
    num_rooms = len(dataset.room_dims)  # Number of rooms
    (start_xcoord, start_ycoord) = find_start_coords(num_rooms, dataset)

    # find which points to put in the training dataset
    for idx in range(len(dataset)):
        listener_position = dataset.listener_positions[idx]  # (x, y, z)
        x_pos, y_pos = listener_position[:2].tolist()  # Extract x, y

        # Identify which room the point belongs to
        room = -1
        for k in range(num_rooms):
            room_start_x, room_start_y = dataset.room_start_coords[k][:2]
            room_width, room_height = dataset.room_dims[k][:2]

            if (room_start_x <= x_pos < room_start_x + room_width
                    and room_start_y <= y_pos < room_start_y + room_height):
                room = k
                break

        # Compute local coordinates relative to the first listener position in the room
        x_coord = x_pos - start_xcoord[room]
        y_coord = y_pos - start_ycoord[room]

        # Check if x_coord and y_coord is a multiple of x_d
        if is_multiple(x_coord, x_d) and is_multiple(y_coord, x_d):
            train_indices.append(idx)
        else:
            val_indices.append(idx)

    train_set = data.Subset(dataset, train_indices)
    val_set = data.Subset(dataset, val_indices)

    return train_set, val_set


def load_dataset(room_data: RoomDataset,
                 device: torch.device,
                 grid_resolution_m: float,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 drop_last: bool = False):
    """
    Get training and validation dataset
    Args:
        room_data (RoomDataset): object of type RoomDataset (for training with a grid of measurements) or 
        device (str): cuda (GPU) or cpu
        train_valid_split_ratio (float): ratio between training and validation set
        batch_size (int): number of samples in each batch size
        shuffle (bool): whether to randomly shuffle data during training
        drop_last (bool): whether to drop the last batch if it has less elements than batch_size
    """
    dataset = SpatialSamplingDataset(
        device,
        room_data,
    )
    dataset = to_device(dataset, device)

    # split data into training and validation set
    train_set, valid_set = split_dataset_by_resolution(dataset,
                                                       grid_resolution_m)

    logger.info(
        f'The length of training dataset is {len(train_set)} and valid ' +
        f'dataset is {len(valid_set)} for grid spacing of {grid_resolution_m}m'
    )

    # dataloaders
    train_loader = get_dataloader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        device=device,
        drop_last=drop_last,
        custom_collate_fn=custom_collate_spatial_sampling)

    if len(valid_set) > 0:
        valid_loader = get_dataloader(
            valid_set,
            batch_size=batch_size,
            shuffle=shuffle,
            device=device,
            drop_last=drop_last,
            custom_collate_fn=custom_collate_spatial_sampling)
        return train_loader, valid_loader
    else:
        return train_loader, None
