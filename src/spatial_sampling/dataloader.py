from abc import ABC
import math
import pickle
from typing import List, Optional

from loguru import logger
import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch.utils import data

from diff_gfdn.dataloader import get_dataloader, InputFeatures, Meshgrid, to_device


class SpatialRoomDataset(ABC):
    """Parent class for any room's RIR dataset measured over multiple source and receiver positions"""

    def __init__(
        self,
        num_rooms: int,
        sample_rate: float,
        source_position: NDArray,
        receiver_position: NDArray,
        rirs: NDArray,
        common_decay_times: List,
        room_dims: List,
        room_start_coord: List,
        band_centre_hz: Optional[ArrayLike] = None,
        amplitudes: Optional[NDArray] = None,
        noise_floor: Optional[NDArray] = None,
        aperture_coords: Optional[List] = None,
        sph_directions: Optional[NDArray] = None,
        ambi_order: Optional[int] = None,
        grid_spacing_m: float = 0.3,
    ):
        """
        Args:
            num_rooms (int): number of rooms in coupled space
            sample_rate (float): sample rate of dataset
            source_position (NDArray): position of sources in cartesian coordinate
            receiver_position (NDArray): position of receivers in cartesian coordinate
            rirs (NDArray): omni / directional rirs at all source and receiver positions of size
                            (num_positions, num_directions, num_time_samples)
            band_centre_hz (optinal, ArrayLike): octave band centres where common T60s are calculated
            common_decay_times (List[Union[ArrayLike, float]]): common decay times for the different rooms of 
                                                                num_freq_bands x num_rooms
            amplitudes (NDArray): the amplitudes of the common slopes model of size 
                                  (num_rec_pos x num_directions x num_rooms x num_freq_bands)
            noise_floor (NDArray): the noise floor of the common slopes model of size
                                    (num_rec_pos x num_directions x 1 x num_freq_bands)
            room_dims (List): l,w,h for each room in coupled space
            room_start_coord (List): coordinates of the room's starting vertex (first room starts at origin)
            aperture_coords (List, optional): coordinates of the apertures in the geometry
            sph_directions (NDArray, optional): if the RIRs are spatial, then the directions at which they were measured
                                                size is (2, num_directions)
            ambi_order (int. optional): order of the ambisonics SMA used for recording RIRs
            grid_spacing_m (optional, float): distance between each mic measurement in uniform grid, in m
        """
        self.sample_rate = sample_rate
        self.num_rooms = num_rooms
        self.source_position = source_position
        self.receiver_position = receiver_position
        self.rirs = rirs
        self.band_centre_hz = band_centre_hz
        self.common_decay_times = common_decay_times
        self.noise_floor = noise_floor
        self.amplitudes = amplitudes
        self.num_rec = self.receiver_position.shape[0]
        self.num_src = self.source_position.shape[
            0] if self.source_position.ndim > 1 else 1
        self.rir_length = self.rirs.shape[-1]
        self.room_dims = room_dims
        self.room_start_coord = room_start_coord
        self.aperture_coords = aperture_coords
        self._eps = 1e-12
        # create 3D mesh
        self.grid_spacing_m = grid_spacing_m
        self.sph_directions = sph_directions
        self.num_directions = None if self.sph_directions is None else self.sph_directions.shape[
            -1]
        self.ambi_order = ambi_order
        self.mesh_3D = self.get_3D_meshgrid()

    @property
    def norm_receiver_position(self):
        """
        Normalise receiver coordinates to be between 0, 1 for more
        meaningful Fourier encoding
        """
        norm_receiver_position = np.zeros_like(self.receiver_position)
        for k in range(3):
            norm_receiver_position[:, k] = (
                self.receiver_position[:, k] -
                self.receiver_position[:, k].min()) / (
                    (self.receiver_position[:, k].max() -
                     self.receiver_position[:, k].min()) + self._eps)
        return norm_receiver_position

    def update_receiver_pos(self, new_receiver_pos: NDArray):
        """Update receiver positions"""
        self.receiver_position = new_receiver_pos
        self.num_rec = self.receiver_position.shape[0]

    def update_rirs(self, new_rirs: NDArray):
        """Update room impulse responses"""
        self.rirs = new_rirs
        self.rir_length = new_rirs.shape[-1]

    def get_3D_meshgrid(self) -> Meshgrid:
        """
        Return the 3D meshgrid of the room's geometry
        Args:
            grid_spacing_m: spacing for creating the meshgrid
        Returns:
            Tuple : tuple of x, y and z meshes in 3D
        """
        Xcombined = []
        Ycombined = []
        Zcombined = []
        for nroom in range(self.num_rooms):
            num_x_points = int(self.room_dims[nroom][0] / self.grid_spacing_m)
            num_y_points = int(self.room_dims[nroom][1] / self.grid_spacing_m)
            num_z_points = int(self.room_dims[nroom][2] / self.grid_spacing_m)
            x = np.linspace(
                self.room_start_coord[nroom][0],
                self.room_start_coord[nroom][0] + self.room_dims[nroom][0],
                num_x_points)
            y = np.linspace(
                self.room_start_coord[nroom][1],
                self.room_start_coord[nroom][1] + self.room_dims[nroom][1],
                num_y_points)
            z = np.linspace(
                self.room_start_coord[nroom][2],
                self.room_start_coord[nroom][2] + self.room_dims[nroom][2],
                num_z_points)
            (xm, ym, zm) = np.meshgrid(x, y, z)
            Xcombined = np.concatenate((Xcombined, xm.flatten()))
            Ycombined = np.concatenate((Ycombined, ym.flatten()))
            Zcombined = np.concatenate((Zcombined, zm.flatten()))

        return Meshgrid(torch.from_numpy(Xcombined),
                        torch.from_numpy(Ycombined),
                        torch.from_numpy(Zcombined))


class SpatialSamplingDataset(data.Dataset):

    def __init__(
        self,
        device: torch.device,
        room_data: SpatialRoomDataset,
    ):
        """
        Spatial sampling dataset containing the common slope amplitudes
        for different receiver positions. During batch processing, each batch will contain 
        different sets of receiver positions
        Args:
            device (str): cuda or cpu
            room_data (SpatialRoomDataset): object of the room dataset class
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
        self.sph_directions = room_data.sph_directions

        # shape num_receivers, num_slopes or num_receivers, num_directions, num_slopes
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

    def __len__(self):
        """Get length of dataset (equal to number of receiver positions)"""
        return self.source_position.shape[0] * self.listener_positions.shape[0]

    def __getitem__(self, idx: int):
        """Get data at a particular index"""
        # Return an instance of InputFeatures

        if self.source_position.shape[0] == 1:
            input_features = InputFeatures(torch.squeeze(self.source_position),
                                           self.listener_positions[idx],
                                           self.norm_listener_position[idx],
                                           self.mesh_3D,
                                           sph_directions=self.sph_directions)
            target_labels = self.common_slope_amps[idx, ...]
        else:
            idx1, idx2 = self.index_pairs[idx]
            input_features = InputFeatures(self.source_position[idx1],
                                           self.listener_positions[idx2],
                                           self.norm_listener_position[idx2],
                                           self.mesh_3D,
                                           sph_directions=self.sph_directions)
            target_labels = self.common_slope_amps[idx1, idx2, ...]

        return {'input': input_features, 'target': target_labels}


def custom_collate_spatial_sampling(batch: data.Dataset):
    """
    Collate datapoints in the dataloader.
    This method is needed because Meshgrid is a custom class, and pytorch's 
    built in collate function cannot collate it
    """
    # these are independent of the source/receiver locations
    directions = batch[0]['input'].sph_directions

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
        'source_position': torch.stack(source_positions),
        'listener_position': torch.stack(listener_positions),
        'norm_listener_position': torch.stack(norm_listener_positions),
        'mesh_3D': mesh_3D_data,
        'sph_directions': directions,
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


def load_dataset(room_data: SpatialRoomDataset,
                 device: torch.device,
                 grid_resolution_m: float,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 drop_last: bool = False):
    """
    Get training and validation dataset
    Args:
        room_data (SpatialRoomDataset): object of type SpatialRoomDataset (for training with a grid of measurements) or 
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


def parse_room_data(filepath: str):
    """Read the three coupled room dataset at filepath and return a SpatialRoomDataset object"""
    assert str(filepath).endswith('.pkl'), "provide the path to the .pkl file"
    # read contents from pkl file
    try:
        logger.info('Reading pkl file ...')
        with open(filepath, 'rb') as f:
            srir_mat = pickle.load(f)
            sample_rate = srir_mat['fs']
            source_position = srir_mat['srcPos'].T
            receiver_position = srir_mat['rcvPos'].T
            srirs = np.squeeze(srir_mat['srirs']).T
            band_centre_hz = srir_mat['band_centre_hz']
            common_decay_times = srir_mat['common_decay_times']
            amplitudes = srir_mat['amplitudes'].T
            noise_floor = srir_mat['noise_floor'].T
            sph_directions = srir_mat[
                'directions'] if 'directions' in srir_mat else None
    except Exception as exc:
        raise FileNotFoundError("pickle file not read correctly") from exc

    logger.info("Done reading pkl file")
    # number of rooms in dataset
    num_rooms = 3
    # (x,y) dimensions of the 3 rooms
    room_dims = [(4.0, 8.0, 3.0), (6.0, 3.0, 3.0), (4.0, 8.0, 3.0)]
    # this denotes the 3D position of the first vertex of the floor
    room_start_coord = [(0, 0, 0), (4.0, 2.0, 0), (6.0, 5.0, 0)]
    # coordinates of the aperture
    aperture_coords = [[(4, 3), (4, 4.5)], [(8.5, 5), (10, 5)]]
    grid_spacing_m = 0.3

    return SpatialRoomDataset(
        num_rooms,
        sample_rate,
        source_position,
        receiver_position,
        srirs,
        common_decay_times,
        room_dims,
        room_start_coord,
        band_centre_hz,
        amplitudes,
        noise_floor,
        aperture_coords,
        sph_directions=sph_directions,
        ambi_order=2,
        grid_spacing_m=grid_spacing_m,
    )
