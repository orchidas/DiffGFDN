from abc import ABC
import math
import pickle
from typing import Dict, List, Optional, Tuple, Union

from loguru import logger
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import rfftfreq
from scipy.interpolate import griddata
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from diff_gfdn.dataloader import InputFeatures, split_dataset, to_device
from diff_gfdn.utils import ms_to_samps

from .config import DNNType


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
        mixing_time_ms: float = 50,
    ):
        """
        Args:
            num_rooms (int): number of rooms in coupled space
            sample_rate (float): sample rate of dataset
            source_position (NDArray): position of sources in cartesian coordinate
            receiver_position (NDArray): position of receivers in cartesian coordinate
            rirs (NDArray): omni / ambisonic /directional rirs at all source and receiver positions of size
                            (num_positions, num_ambi_channels / num_directions, num_time_samples)
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
                                                size is (2, num_directions) in radians
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
        self.grid_spacing_m = grid_spacing_m
        self.sph_directions = sph_directions
        self.num_directions = None if self.sph_directions is None else self.sph_directions.shape[
            -1]
        self.ambi_order = ambi_order
        self.mixing_time_ms = mixing_time_ms

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

    @property
    def num_freq_bins(self):
        """Number of frequency bins in the magnitude response"""
        max_rt60_samps = self.common_decay_times.max() * self.sample_rate
        return int(np.power(2, np.ceil(np.log2(max_rt60_samps))))

    @property
    def freq_bins_rad(self):
        """Frequency bins in radians"""
        return rfftfreq(self.num_freq_bins) * 2 * np.pi

    @property
    def freq_bins_hz(self):
        """Frequency bins in Hz"""
        return rfftfreq(self.num_freq_bins, d=1.0 / self.sample_rate)

    def update_receiver_pos(self, new_receiver_pos: NDArray):
        """Update receiver positions"""
        self.receiver_position = new_receiver_pos
        self.num_rec = self.receiver_position.shape[0]

    def update_rirs(self, new_rirs: NDArray):
        """Update room impulse responses"""
        self.rirs = new_rirs
        self.rir_length = new_rirs.shape[-1]

    def find_rec_idx_in_room_dataset(self, rec_pos_list: NDArray) -> List:
        """
        Return indices of the receivers in the dataset associated with 
        the list of receiver positions
        """
        # Compute Euclidean distance between each array in array_list_np and every row in matrix
        distances = np.linalg.norm(self.receiver_position[:, None, :] -
                                   rec_pos_list,
                                   axis=2)
        indices = np.argmin(distances, axis=0)
        return indices

    def early_late_split(self, win_len_ms: float = 5.0):
        """Split the RIRs into early and late response based on mixing time"""
        mixing_time_samps = ms_to_samps(self.mixing_time_ms, self.sample_rate)
        win_len_samps = ms_to_samps(win_len_ms, self.sample_rate)
        window = np.broadcast_to(np.hanning(win_len_samps),
                                 self.rirs.shape[:-1] + (win_len_samps, ))

        # create fade in and fade out windows to avoid discontinuities
        fade_in_win = window[..., :win_len_samps // 2]
        fade_out_win = window[..., win_len_samps // 2:]

        # truncate rir into early and late parts
        self.early_rirs = self.rirs[..., :mixing_time_samps]
        self.late_rirs = self.rirs[..., mixing_time_samps:]

        # apply fade-in and fade-out windows
        self.early_rirs[..., -win_len_samps // 2:] *= fade_out_win
        self.late_rirs[..., :win_len_samps // 2] *= fade_in_win


class SpatialSamplingDataset(Dataset):

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
        self.sph_directions = room_data.sph_directions

        # shape num_receivers, num_slopes or num_receivers, num_directions, num_slopes
        self.common_slope_amps = torch.tensor(room_data.amplitudes)
        self.num_rooms = room_data.num_rooms
        self.device = device
        self.grid_resolution_m = room_data.grid_spacing_m
        self.room_start_coords = room_data.room_start_coord
        self.room_dims = room_data.room_dims

        # frequency-domain data
        freq_bins_rad = torch.tensor(room_data.freq_bins_rad)
        self.z_values = torch.polar(torch.ones_like(freq_bins_rad),
                                    freq_bins_rad)

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

        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds.")

        if self.source_position.shape[0] == 1:
            input_features = InputFeatures(torch.squeeze(self.source_position),
                                           self.listener_positions[idx],
                                           self.norm_listener_position[idx],
                                           sph_directions=self.sph_directions,
                                           z_values=self.z_values)
            target_labels = self.common_slope_amps[idx, ...]
        else:
            idx1, idx2 = self.index_pairs[idx]
            input_features = InputFeatures(self.source_position[idx1],
                                           self.listener_positions[idx2],
                                           self.norm_listener_position[idx2],
                                           sph_directions=self.sph_directions,
                                           z_values=self.z_values)
            target_labels = self.common_slope_amps[idx1, idx2, ...]

        return {'input': input_features, 'target': target_labels}

    def get_binary_mask(
            self,
            mesh_2D: Union[NDArray,
                           torch.Tensor]) -> Union[NDArray, torch.Tensor]:
        """
        Return a binary mask for points in a 2D mesh that lie within the
        floor plan of the space. True if points are inside, otherwise False.
        Args:
            mesh_2D: (B x 2) / (Nx x Ny x 2) array of grid coordinates
        Returns:
            A flattened boolean array of size (B / Nx x Ny)
        """
        x_mesh = mesh_2D[..., 0]
        y_mesh = mesh_2D[..., 1]
        combined_mask = torch.tensor([], dtype=torch.bool) if isinstance(
            mesh_2D, torch.Tensor) else np.array([])
        for i in range(self.num_rooms):
            cur_mask = (x_mesh >= self.room_start_coords[i][0]) & \
                       (x_mesh <= self.room_dims[i][0] + self.room_start_coords[i][0]) & \
                       (y_mesh >= self.room_start_coords[i][1]) & \
                       (y_mesh <= self.room_dims[i][1] + self.room_start_coords[i][1])
            if isinstance(mesh_2D, torch.Tensor):
                combined_mask = cur_mask if combined_mask.numel(
                ) == 0 else torch.logical_or(combined_mask, cur_mask)
            else:
                combined_mask = cur_mask if combined_mask.size == 0 else np.logical_or(
                    combined_mask, cur_mask)

        return combined_mask


def create_2D_grid_data(
    batch: List[Dict],
    dataset_ref: Dataset = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create 2D grid data from 1D position and target labels to feed into CNN.
    The positions must be on a uniform grid
    Args:
        batch (List): the current batch of data as a list of dictionaries. This works iff
            the batch contains receiver indices that are uniformly distributed in space
        dataset_ref (Dataset): reference to the Dataset object
    Returns:
        Tuple: the 2D meshgrid of shape H,W,2 - unnormalised and normalised,
                and the labels interpolated at the grid of shape H,W,num_directions,num_groups
    """

    def create_2D_mesh(x_lin: ArrayLike, y_lin: ArrayLike) -> Tuple[NDArray]:
        """Create 2D mesh from x, y coordinates"""
        x_lin_unique = np.unique(x_lin)
        y_lin_unique = np.unique(y_lin)
        x_mesh, y_mesh = np.meshgrid(x_lin_unique, y_lin_unique)
        # print('Listener position in current batch')
        # print(x_lin_unique, y_lin_unique)

        # TO-DO discard indices that don't fall in a regular grid

        # this is of size H x W x 2
        mesh_2D_data = np.stack((x_mesh, y_mesh), axis=-1)
        return mesh_2D_data

    # create mesh for listener positions
    x_lin = np.array([item['input'].listener_position[0] for item in batch])
    y_lin = np.array([item['input'].listener_position[1] for item in batch])
    mesh_2D_data = create_2D_mesh(x_lin, y_lin)

    # create mesh for normalised listener positions
    x_lin_norm = np.array(
        [item['input'].norm_listener_position[0] for item in batch])
    y_lin_norm = np.array(
        [item['input'].norm_listener_position[1] for item in batch])
    mesh_2D_norm_data = create_2D_mesh(x_lin_norm, y_lin_norm)

    # target amplitudes, size is B x num_directions x num_groups
    target_labels = dataset_ref.common_slope_amps
    num_groups = target_labels.shape[-1]
    num_directions = target_labels.shape[1]
    H, W = mesh_2D_data.shape[:-1]

    # size is H, W, num_directions, num_groups - using scipy's interpolate because
    # torch's does not take into account the original x, y coordinates
    # note that we are using the entire set of available positions to form this 2D grid
    target_labels_2D = griddata((dataset_ref.listener_positions[:, 0],
                                 dataset_ref.listener_positions[:, 1]),
                                target_labels,
                                (mesh_2D_data[..., 0], mesh_2D_data[..., 1]),
                                method='nearest')

    # create a mask for values within the limits (so that outside the boundaries the labels are zero)
    combined_mask = dataset_ref.get_binary_mask(mesh_2D_data)
    target_labels_2D[~combined_mask, ...] = 0.0

    return torch.tensor(mesh_2D_data), torch.tensor(
        mesh_2D_norm_data), torch.tensor(target_labels_2D).view(
            H * W, num_directions, num_groups)


def custom_collate_spatial_sampling(
    batch: List[Dict],
    network_type: str,
    dataset_ref: Dataset = None,
) -> Dict:
    """
    Collate datapoints in the dataloader.
    This method is needed because Meshgrid is a custom class, and pytorch's 
    built in collate function cannot collate it
    Args:
        batch (List): the current batch of data, as a list of dicts
        network_type (str): MLP or CNN
        dataset_ref (Dataset) : reference to the dataset object
    """
    # these are independent of the source/receiver locations
    directions = batch[0]['input'].sph_directions

    # depends on source positions
    source_positions = torch.stack(
        [item['input'].source_position for item in batch])

    # depends on receiver positions
    listener_positions = torch.stack(
        [item['input'].listener_position for item in batch])
    norm_listener_positions = torch.stack(
        [item['input'].norm_listener_position for item in batch])
    target_amplitudes = torch.stack([item['target'] for item in batch])

    # these are independent of the source/receiver locations
    z_values = batch[0]['input'].z_values

    if network_type == DNNType.CNN:
        mesh_2D_data, mesh_2D_norm_data, target_amplitudes_2D = create_2D_grid_data(
            batch, dataset_ref)

        return {
            'listener_position': listener_positions,
            'norm_listener_position': norm_listener_positions,
            'source_position': source_positions,
            'mesh_2D': mesh_2D_data,
            'mesh_2D_norm': mesh_2D_norm_data,
            'sph_directions': directions,
            'z_values': z_values,
            'target_common_slope_amps': target_amplitudes_2D,
        }
    else:
        return {
            'listener_position': listener_positions,
            'norm_listener_position': norm_listener_positions,
            'source_position': source_positions,
            'sph_directions': directions,
            'z_values': z_values,
            'target_common_slope_amps': target_amplitudes,
        }


def find_start_coords(num_rooms: int, dataset: Dataset):
    """
    Find the first receiver location in each room
    Args:
        num_rooms (int): number of rooms in the space
        dataset (Dataset): contains information about room geometry 
    """
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
    dataset: Dataset,
    x_d: float,
):
    """
    Split dataset into training and validation based on x_d resolution.
    If measurements are avaialable in a uniform 2D grid, the measurements at every 
    x_d m is used for training and the rest is used for validation 

    dataset: an instance of SpatialDataset.
    x_d: Spacing between selected training points.
    """
    assert x_d >= dataset.grid_resolution_m, "The desired grid spacing must be greater '\
    'than what has been measured in the dataset"

    def is_multiple(value, d, tol=1e-6):
        """Check if value is an approximate multiple of d."""
        return math.isclose(value / d, round(value / d), abs_tol=tol)

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

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    return train_set, val_set


class SquarePatchSampler(Sampler):

    def __init__(
        self,
        coords: NDArray,
        patch_size: int,
        grid_spacing_m: float,
        parent_dataset: SpatialSamplingDataset,
        shuffle: bool = False,
        drop_incomplete: bool = False,
        step_size: int = 1,
    ):
        """
        Yield indices from square patches in a 2D grid.
        Args:
            coords (NDArray): grid of 2D coordinates of the subset
            patch_size: Size of the square patch (e.g., 3 for 3×3).
            grid_spacing: Rounding factor for position normalization.
            parent_dataset (SpatialSamplingDataset): parent dataset from which the
                    training / validation subset has been created
            shuffle: If True, shuffle the patches.
            drop_incomplete: If True, discard patches smaller than full size.
            step_size (int): determines how much the patches overlap. A step size of
                             1 gives maximum overlapping patches but increases training time.
        """
        super().__init__()
        self.patch_size = patch_size
        self.grid_spacing_m = grid_spacing_m
        self.shuffle = shuffle
        self.drop_incomplete = drop_incomplete

        # extract coordinates
        self.original_indices = list(range(len(coords)))
        self.coords = coords
        self.step_size = min(step_size, patch_size)
        self.parent_dataset = parent_dataset
        self.num_rooms = len(self.parent_dataset.room_dims)

        self.rounded_coords = self._find_rounded_coords()
        self.patches = self._find_all_patches()

    def _find_rounded_coords(self) -> torch.Tensor:
        """Find the rounded coordinates after adjusting for the origin in each room"""
        # find the start coordinates for each room
        start_xcoord, start_ycoord = find_start_coords(self.num_rooms,
                                                       self.parent_dataset)
        room = torch.zeros(len(self.coords), dtype=torch.int32)

        # find which room each point is in
        for idx in range(len(self.coords)):
            listener_position = self.coords[idx]  # (x, y, z)
            x_pos, y_pos = listener_position[:2].tolist()  # Extract x, y

            # Identify which room the point belongs to
            room[idx] = -1
            for k in range(self.num_rooms):
                room_start_x, room_start_y = self.parent_dataset.room_start_coords[
                    k][:2]
                room_width, room_height = self.parent_dataset.room_dims[k][:2]

                if (room_start_x <= x_pos < room_start_x + room_width and
                        room_start_y <= y_pos < room_start_y + room_height):
                    room[idx] = k
                    break

        # loop through each room and fill the rounded coords
        rounded_coords = torch.zeros_like(self.coords)
        for k in range(self.num_rooms):
            mask_idx = torch.argwhere(room == k).squeeze()
            adjusted_x_coord = self.coords[mask_idx, 0] - start_xcoord[k]
            adjusted_y_coord = self.coords[mask_idx, 1] - start_ycoord[k]

            cur_rounded_coords = torch.stack(
                (adjusted_x_coord, adjusted_y_coord),
                dim=1) / self.grid_spacing_m

            # adjust the current rounded coordinates to be aligned with the
            # previous rounded coordinates
            start_idx = [0, 0] if k == 0 else rounded_coords.max(dim=0)[0]
            start_idx_tensor = torch.tensor(
                start_idx, dtype=cur_rounded_coords.dtype).repeat(
                    cur_rounded_coords.shape[0], 1)
            rounded_coords[mask_idx, :] = (cur_rounded_coords +
                                           start_idx_tensor)

        return rounded_coords.round().int()

    def _find_all_patches(self) -> List[List[int]]:
        """Find all square patches of size pacth_size**2 among the given 2D coordinates"""
        x_unique = torch.unique(self.rounded_coords[:, 0])
        y_unique = torch.unique(self.rounded_coords[:, 1])
        patches = []

        # how much we step by determines the how much the patches overlap
        # and what the batch size is
        for x_start in x_unique[::self.step_size]:
            for y_start in y_unique[::self.step_size]:
                # same as meshgrid, followed by stack
                # gives all possible combinations of coordinates
                patch = torch.cartesian_prod(
                    torch.arange(x_start, x_start + self.patch_size),
                    torch.arange(y_start, y_start + self.patch_size))

                # Match the patch that actually falls in the provided grid samples
                # rounded_coords: (N, 2), patch: (P, 2) - ideal grid of square patch
                # mask gives tensor of shape (N, P, 2) - comparison of every dataset point to
                #                                         every patch location's xy coords
                # .all(-1) -> (N, P) - true if coordinates match exactly. N = len(dataset),
                #                      P=patch_size**2, N     contains many points in P, find those
                # .any(1) -> N, true if a point matches any location in patch, each variable is True/False
                #                true if the location matches that in patch, false otherwise
                mask = ((self.rounded_coords[:, None] == patch[None, :]
                         ).all(-1)).any(1)
                # extract only matching locations
                matched_idx = torch.where(mask)[0]

                if self.drop_incomplete and len(
                        matched_idx) != self.patch_size**2:
                    continue

                patch_indices = [
                    self.original_indices[i.item()] for i in matched_idx
                ]
                patches.append(patch_indices)

        return patches

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.patches)
        for patch in self.patches:
            yield from patch  # yield indices from each patch in order

    def __len__(self):
        return sum(len(p) for p in self.patches)


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    device='cpu',
    drop_last: bool = True,
    custom_collate_fn=None,
    sample_square_patches: bool = False,
    grid_spacing_m: Optional[float] = None,
) -> DataLoader:
    """
    Create torch dataloader form given dataset.
    Args:
        dataset (Dataset): SpatialRoomDataset for sampling
        batch_size (int): number of positions in each batch
        shuffle (bool): whether to shuffle the batches while training
        device (torch.device): device on which to train
        drop_last (bool): whether to drop the last few samples 
                          if it doesnt match batch size
        custom_collate_fn: custom collate function
        sample_square_patches (bool): whether to sample a square or rectangular patch of points
                                      for creating a batch. Square should be used for CNN.
    """
    if sample_square_patches:
        logger.info(
            "Sampling a square patch of 2D grid points for CNN training")
        # if training a CNN we want to sample a square patch from the
        # 2D grid of listener positions. Without this, a rectangular patch
        # is sampled

        # listener positions indices in the training / validation dataset
        subset_listener_pos_idx = dataset.indices
        # determine patch size from batch size
        patch_size = int(math.sqrt(batch_size))

        # create sampler with square patches
        sampler = SquarePatchSampler(
            dataset.dataset.listener_positions[subset_listener_pos_idx, :2],
            patch_size,
            grid_spacing_m,
            step_size=patch_size,
            parent_dataset=dataset.dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,  # << replace batch_size and shuffle
            generator=torch.Generator(device=device),
            drop_last=drop_last,  # still valid
            collate_fn=custom_collate_fn  # keep your existing collate function
        )
    else:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                generator=torch.Generator(device=device),
                                drop_last=drop_last,
                                collate_fn=custom_collate_fn)

    logger.info(f"Number of batches: {len(dataloader)}")
    return dataloader


def load_dataset(
        room_data: SpatialRoomDataset,
        device: torch.device,
        network_type: str,
        batch_size: int = 32,
        grid_resolution_m: Optional[float] = None,
        train_valid_split_ratio: Optional[float] = None,
        shuffle: bool = True,
        drop_last: bool = False) -> Tuple[DataLoader, DataLoader, Dataset]:
    """
    Get training and validation dataset
    Args:
        room_data (SpatialRoomDataset): object of type SpatialRoomDataset (for training with a grid of measurements) or 
        device (str): cuda (GPU) or cpu
        network_type (str): is it a CNN or MLP?
        batch_size (int): number of samples in each batch size
        grid_resolution_m (optional, float): what is the resolution of the uniform grid used for measurement?
        train_valid_split_ratio(optional, float): if not sampling by grid resolution, 
                                                  randomly split train and valid datset
        shuffle (bool): whether to randomly shuffle data during training
        drop_last (bool): whether to drop the last batch if it has less elements than batch_size
    Returns:
        Tuple: training dataloader, validation dataloader and dataset reference object
    """
    dataset = SpatialSamplingDataset(
        device,
        room_data,
    )
    dataset = to_device(dataset, device)
    shuffle = False if network_type == DNNType.CNN else shuffle
    sample_square_patches = network_type == DNNType.CNN

    # split data into training and validation set
    if grid_resolution_m is None:
        # randomly split data into training and validation set
        train_set, valid_set = split_dataset(dataset, train_valid_split_ratio)
        logger.info(
            'Randomly splitting training and validation sets.' +
            f'The length of training dataset is {len(train_set)} and valid ' +
            f'dataset is {len(valid_set)}')
    else:
        # split dataset uniformly by grid resolution
        train_set, valid_set = split_dataset_by_resolution(
            dataset, grid_resolution_m)
        logger.info(
            f'The length of training dataset is {len(train_set)} and valid ' +
            f'dataset is {len(valid_set)} for grid spacing of {grid_resolution_m} m'
        )

    # dataloaders
    train_loader = get_dataloader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        device=device,
        drop_last=drop_last,
        custom_collate_fn=lambda batch: custom_collate_spatial_sampling(
            batch, network_type, dataset),
        sample_square_patches=sample_square_patches,
        grid_spacing_m=grid_resolution_m)

    if len(valid_set) > 0:
        valid_loader = get_dataloader(
            valid_set,
            batch_size=batch_size,
            shuffle=shuffle,
            device=device,
            drop_last=drop_last,
            custom_collate_fn=lambda batch: custom_collate_spatial_sampling(
                batch, network_type, dataset),
            sample_square_patches=sample_square_patches,
            grid_spacing_m=grid_resolution_m)

        return train_loader, valid_loader, dataset
    else:
        return train_loader, None, dataset


def parse_three_room_data(filepath: str):
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
            amplitudes = srir_mat['amplitudes_norm'].T
            noise_floor = srir_mat['noise_floor_norm'].T
            sph_directions = np.deg2rad(
                srir_mat['directions']) if 'directions' in srir_mat else None
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
