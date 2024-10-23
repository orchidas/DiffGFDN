from abc import ABC
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
from typing import List, Optional, Tuple, Union

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import rfft, rfftfreq
import soundfile as sf
import torch
from torch.utils import data

from .config.config import DiffGFDNConfig
from .utils import ms_to_samps

# flake8: noqa: E231


@dataclass
class Meshgrid():
    xmesh: torch.tensor
    ymesh: torch.tensor
    zmesh: torch.tensor


@dataclass
class InputFeatures():
    """
    Contains input features to our Diff GFDN network. These are
    frequency bins at which the magnitude response is calculated,
    and the source and listener positions where the RIRs are measured,
    and the meshgrid of the room geometry
    Args:
        z_values: values around the unit circle (polar)
        source_position: postion of the source in cartesian coordinates
        listener_position: position of a receiver in cartesian coordinates
        mesh_3D: mesh grid of the space geometry
    """

    z_values: torch.tensor
    source_position: torch.tensor
    listener_position: torch.tensor
    mesh_3D: Meshgrid

    def __repr__(self):
        # pylint: disable=E0306
        return (
            f"InputFeatures(\n"
            f"  z_values={self.z_values.tolist()}, \n"
            f"  source_position={self.source_position.tolist()}, \n"
            f"  listener_position={self.listener_position.tolist()}, \n",
            f"  mesh_3D={self.mesh_3D.xmesh, self.mesh_3D.ymesh, self.mesh_3D.zmesh}, \n",
            ")")


@dataclass
class Target():
    # the frequency response of the target RIR, split into early and late parts
    early_rir_mag_response: torch.tensor
    late_rir_mag_response: torch.tensor
    rir_mag_response: torch.tensor


class RIRData:
    """Data for a single measured/simulated RIR"""

    def __init__(self,
                 wav_path: Path,
                 band_centre_hz: ArrayLike,
                 common_decay_times: List,
                 amplitudes: Optional[List] = None,
                 room_dims: Optional[List] = None,
                 absorption_coeffs: Optional[List] = None,
                 mixing_time_ms: float = 20.0):
        """
        Args:
            num_rooms (int): number of rooms in coupled space
            sample_rate (float): sample rate of dataset
            wav_path (Path): path to the RIR
            band_centre_hz (ArrayLike): octave band centres where common T60s are calculated
            common_decay_times (List[ArrayLike]): common decay times for the different rooms
            amplitudes (List[ArrayLike]): the amplitudes of the common slopes, unique to the receiver position,
                                          same size as common_decay times
            room_dims (optional, List): l,w,h for each room in coupled space
            absorption_coeffs (optional, List): uniform absorption coefficients for each room
            mixing_time_ms (float): time when early reflections morph into late reverb
        """

        assert str(wav_path).endswith(
            '.wav'), "provide the path to the .wav file"

        # read contents from .wav file
        try:
            (rir, sample_rate) = sf.read(str(wav_path))
        except Exception as exc:
            raise FileNotFoundError(
                f"File was not found at {str(wav_path)}") from exc

        self.rir = rir
        self.sample_rate = sample_rate
        self.common_decay_times = common_decay_times
        self.band_centre_hz = band_centre_hz
        self.amplitudes = amplitudes
        self.mixing_time_ms = mixing_time_ms
        self.room_dims = room_dims
        self.absorption_coeffs = absorption_coeffs
        self.early_late_split()

    @property
    def num_freq_bins(self):
        """Number of frequency bins in the magnitude response"""
        max_rt60_samps = self.common_decay_times.max() * self.sample_rate
        return int(np.pow(2, np.ceil(np.log2(max_rt60_samps))))

    @property
    def freq_bins_rad(self):
        """Frequency bins in radians"""
        return rfftfreq(self.num_freq_bins)

    @property
    def freq_bins_hz(self):
        """Frequency bins in Hz"""
        return rfftfreq(self.num_freq_bins, d=1.0 / self.sample_rate)

    @property
    def rir_mag_response(self):
        """Frequency response of the RIRs, time along last axis"""
        return rfft(self.rir, n=self.num_freq_bins)

    def early_late_split(self, win_len_ms: float = 5.0):
        """Split the RIRs into early and late response based on mixing time"""
        mixing_time_samps = ms_to_samps(self.mixing_time_ms, self.sample_rate)
        win_len_samps = ms_to_samps(win_len_ms, self.sample_rate)
        window = np.hanning(win_len_samps)

        # create fade in and fade out windows to avoid discontinuities
        fade_in_win = window[:win_len_samps // 2]
        fade_out_win = window[win_len_samps // 2:]

        # truncate rir into early and late parts
        self.early_rir = self.rir[:mixing_time_samps]
        self.late_rir = self.rir[mixing_time_samps:]

        # apply fade-in and fade-out windows
        self.early_rir[-win_len_samps // 2:] *= fade_out_win
        self.late_rir[:win_len_samps // 2] *= fade_in_win

        # get frequency response
        self.late_rir_mag_response = rfft(
            self.late_rir,
            n=self.num_freq_bins,
        )
        self.early_rir_mag_response = rfft(
            self.early_rir,
            n=self.num_freq_bins,
        )


class RoomDataset(ABC):
    """Parent class for any room's RIR dataset measured over multiple source and receiver positions"""

    def __init__(self,
                 num_rooms: int,
                 sample_rate: float,
                 source_position: NDArray,
                 receiver_position: NDArray,
                 rirs: NDArray,
                 band_centre_hz: ArrayLike,
                 common_decay_times: List,
                 amplitudes: NDArray,
                 room_dims: List,
                 room_start_coord: List,
                 absorption_coeffs: List,
                 mixing_time_ms: float = 20.0,
                 nfft: Optional[int] = None):
        """
        Args:
            num_rooms (int): number of rooms in coupled space
            sample_rate (float): sample rate of dataset
            source_position (NDArray): position of sources in cartesian coordinate
            receiver_position (NDArray): position of receivers in cartesian coordinate
            rirs (NDArray): omni-rirs at all source and receiver positions
            band_centre_hz (ArrayLike): octave band centres where common T60s are calculated
            common_decay_times (List[ArrayLike]): common decay times for the different rooms
            amplitudes (NDArray): the amplitudes of the common slopes of size 
                                  (num_freq_bands x  num_rooms x num_rec_pos)
            room_dims (List): l,w,h for each room in coupled space
            room_start_coord (List): coordinates of the room's starting vertex (first room starts at origin)
            absorption_coeffs (List): uniform absorption coefficients for each room
            mixing_time_ms (float): mixing time of the RIR for early-late split
            nfft (optional, int): number of frequency bins
        """
        self.sample_rate = sample_rate
        self.num_rooms = num_rooms
        self.source_position = source_position
        self.receiver_position = receiver_position
        self.rirs = rirs
        self.band_centre_hz = band_centre_hz
        self.common_decay_times = common_decay_times
        self.amplitudes = amplitudes
        self.num_rec = self.receiver_position.shape[0]
        self.num_src = self.source_position.shape[0]
        self.rir_length = self.rirs.shape[-1]
        self.absorption_coeffs = absorption_coeffs
        self.room_dims = room_dims
        self.room_start_coord = room_start_coord
        self.mixing_time_ms = mixing_time_ms
        self.nfft = nfft
        self.early_late_split()

    @property
    def num_freq_bins(self):
        """Number of frequency bins in the magnitude response"""
        if self.nfft is not None:
            return self.nfft
        else:
            max_rt60_samps = self.common_decay_times.max() * self.sample_rate
            return int(np.pow(2, np.ceil(np.log2(max_rt60_samps))))

    @property
    def freq_bins_rad(self):
        """Frequency bins in radians"""
        return rfftfreq(self.num_freq_bins)

    @property
    def freq_bins_hz(self):
        """Frequency bins in Hz"""
        return rfftfreq(self.num_freq_bins, d=1.0 / self.sample_rate)

    @property
    def rir_mag_response(self):
        """Frequency response of the RIRs, time along last axis"""
        return rfft(self.rirs, n=self.num_freq_bins, axis=-1)

    def early_late_split(self, win_len_ms: float = 5.0):
        """Split the RIRs into early and late response based on mixing time"""
        mixing_time_samps = ms_to_samps(self.mixing_time_ms, self.sample_rate)
        win_len_samps = ms_to_samps(win_len_ms, self.sample_rate)
        window = np.broadcast_to(np.hanning(win_len_samps),
                                 (self.rirs.shape[0], win_len_samps))

        # create fade in and fade out windows to avoid discontinuities
        fade_in_win = window[:, :win_len_samps // 2]
        fade_out_win = window[:, win_len_samps // 2:]

        # truncate rir into early and late parts
        self.early_rirs = self.rirs[:, :mixing_time_samps]
        self.late_rirs = self.rirs[:, mixing_time_samps:]

        # apply fade-in and fade-out windows
        self.early_rirs[:, -win_len_samps // 2:] *= fade_out_win
        self.late_rirs[:, :win_len_samps // 2] *= fade_in_win

        # get frequency response
        self.late_rir_mag_response = rfft(self.late_rirs,
                                          n=self.num_freq_bins,
                                          axis=-1)
        self.early_rir_mag_response = rfft(self.early_rirs,
                                           n=self.num_freq_bins,
                                           axis=-1)

    def get_3D_meshgrid(self, grid_spacing_m: float) -> Meshgrid:
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
            num_x_points = int(self.room_dims[nroom][0] / grid_spacing_m)
            num_y_points = int(self.room_dims[nroom][1] / grid_spacing_m)
            num_z_points = int(self.room_dims[nroom][2] / grid_spacing_m)
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

    def get_2D_meshgrid(self,
                        grid_spacing_m: float) -> Tuple[NDArray, NDArray]:
        """
        Returns the 2D meshgrid over receiver points
        """
        Xcombined = []
        Ycombined = []
        for nroom in range(self.num_rooms):
            num_x_points = int(self.room_dims[nroom][0] / grid_spacing_m)
            num_y_points = int(self.room_dims[nroom][1] / grid_spacing_m)
            x = np.linspace(
                self.room_start_coord[nroom][0],
                self.room_start_coord[nroom][0] + self.room_dims[nroom][0],
                num_x_points)
            y = np.linspace(
                self.room_start_coord[nroom][1],
                self.room_start_coord[nroom][1] + self.room_dims[nroom][1],
                num_y_points)

            (xm, ym) = np.meshgrid(x, y)
            Xcombined = np.concatenate((Xcombined, xm.flatten()))
            Ycombined = np.concatenate((Ycombined, ym.flatten()))

        return (Xcombined, Ycombined)

    def plot_3D_meshgrid(self, mesh_3D: Meshgrid):
        """Plot the 3D meshgrid to visualise the room geometry"""
        xmesh = mesh_3D.xmesh.cpu().detach().numpy()
        ymesh = mesh_3D.ymesh.cpu().detach().numpy()
        zmesh = mesh_3D.zmesh.cpu().detach().numpy()

        x_flat = xmesh.flatten()
        y_flat = ymesh.flatten()
        z_flat = zmesh.flatten()

        # Plot using scatter without any additional data for color
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the X, Y, Z points
        ax.scatter(x_flat, y_flat, z_flat, color='b', marker='.')

        # Set the limits for all axes
        ax.set_xlim(0,
                    self.room_dims[-1][0] + self.room_start_coord[-1][0] + 0.5)
        ax.set_ylim(0,
                    self.room_dims[-1][1] + self.room_start_coord[-1][1] + 0.5)
        ax.set_zlim(0, self.room_dims[-1][-1] + 0.5)

        # Set the viewing angle so the origin is in the front bottom-left corner
        # ax.view_init(elev=90, azim=-90)

        # Labels and title
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('3D mesh grid of coupled space')

        # Show the plot
        plt.show()


class ThreeRoomDataset(RoomDataset):
    """
    Parse data from the three room dataset by Gotz et al from
    'Dynamic late reverberation rendering using the common slope model'
    in Proc. of AES International Conference on Audio for Gaming, 2024.
    """

    def __init__(self, filepath: Path, config_dict: DiffGFDNConfig):
        """Read the data from the filepath"""
        num_rooms = 3
        assert str(filepath).endswith(
            '.pkl'), "provide the path to the .pkl file"
        # read contents from pkl file
        try:
            logger.info('Reading pkl file ...')
            with open(filepath, 'rb') as f:
                srir_mat = pickle.load(f)
                sample_rate = srir_mat['fs'][0][0]
                source_position = srir_mat['srcPos'].T
                receiver_position = srir_mat['rcvPos'].T
                # these are second order ambisonic signals
                # I am guessing the first channel contains the W component
                rirs = np.squeeze(srir_mat['srirs'][0, ...]).T
                band_centre_hz = srir_mat['band_centre_hz']
                common_decay_times = np.asarray(
                    np.squeeze(srir_mat['common_decay_times'], axis=1))
                amplitudes = np.asarray(srir_mat['amplitudes'])
                nfft = config_dict.trainer_config.num_freq_bins
        except Exception as exc:
            raise FileNotFoundError("pickle file not read correctly") from exc

        logger.info("Done reading pkl file")
        # uniform absorption coefficients of the three rooms
        absorption_coeffs = np.array([0.2, 0.01, 0.1])
        # (x,y) dimensions of the 3 rooms
        room_dims = [(4.0, 8.0, 3.0), (6.0, 3.0, 3.0), (4.0, 8.0, 3.0)]
        # this denotes the 3D position of the first vertex of the floor
        room_start_coord = [(0, 0, 0), (4.0, 2.0, 0), (6.0, 5.0, 0)]
        super().__init__(num_rooms,
                         sample_rate,
                         source_position,
                         receiver_position,
                         rirs,
                         band_centre_hz,
                         common_decay_times,
                         amplitudes,
                         room_dims,
                         room_start_coord,
                         absorption_coeffs,
                         nfft=nfft)

        # how far apart the receivers are placed
        mic_spacing_m = 0.3
        self.mesh_3D = super().get_3D_meshgrid(mic_spacing_m)
        if config_dict.trainer_config.save_true_irs:
            logger.info("Saving RIRs")
            self.save_omni_irs()

    def save_omni_irs(self,
                      filename_prefix: str = "ir",
                      directory: str = "../audio/true/"):
        """Save the omni RIRs for each receiver position as audio files in directory"""
        if not os.path.isdir(directory):
            os.makedirs(directory)

        for num_pos in range(self.num_rec):
            filename = (
                f'{filename_prefix}_({self.receiver_position[num_pos,0]:.2f}, '
                f'{self.receiver_position[num_pos, 1]:.2f}, {self.receiver_position[num_pos, 2]:.2f}).wav'
            )

            filepath = os.path.join(directory, filename)
            sf.write(filepath, self.rirs[num_pos, :], int(self.sample_rate))


class MultiRIRDataset(data.Dataset):

    def __init__(self,
                 device: torch.device,
                 room_data: RoomDataset,
                 new_sampling_radius: Optional[float] = None):
        """
        RIR dataset containing magnitude response of RIRs around the unit circle,
        for different receiver positions.
        During batch processing, each batch will contain all the frequency bins
        but different sets of receiver positions
        Args:
            device (str): cuda or cpu
            room_data (RoomDataset): object of the room dataset class
                        containing information about the RIRs and source and listener positions
            new_sampling_radius (float): to reduce time aliasing artifacts due to insufficient sampling
                                     in the frequency domain, sample points on a circle whose radius
                                     is larger than 1 

        """
        # spatial data
        self.source_position = torch.tensor(room_data.source_position)
        self.listener_positions = torch.tensor(room_data.receiver_position)
        self.mesh_3D = room_data.mesh_3D
        self.device = device

        # frequency-domain data
        freq_bins_rad = torch.tensor(room_data.freq_bins_rad)

        if new_sampling_radius in (1.0, None):
            # this ensures that we cover half the unit circle (other half is symmetric)
            self.z_values = torch.polar(torch.ones_like(freq_bins_rad),
                                        freq_bins_rad * 2 * np.pi)
        else:
            assert new_sampling_radius > 1.0
            logger.info(
                f"Sampling outside the unit circle at a radius {new_sampling_radius}"
            )
            # sample outside the unit circle
            self.z_values = torch.polar(
                new_sampling_radius * torch.ones_like(freq_bins_rad),
                freq_bins_rad * 2 * np.pi)

        self.rir_mag_response = torch.tensor(room_data.rir_mag_response)
        self.late_rir_mag_response = torch.tensor(
            room_data.late_rir_mag_response)
        self.early_rir_mag_response = torch.tensor(
            room_data.early_rir_mag_response)

    def __len__(self):
        """Get length of dataset (equal to number of receiver positions)"""
        return self.source_position.shape[0] * self.listener_positions.shape[0]

    def __getitem__(self, idx: int):
        """Get data at a particular index"""
        # Return an instance of InputFeatures
        input_features = InputFeatures(self.z_values, self.source_position,
                                       self.listener_positions[idx],
                                       self.mesh_3D)
        target_labels = Target(self.early_rir_mag_response[idx, :],
                               self.late_rir_mag_response[idx, :],
                               self.rir_mag_response[idx, :])
        return {'input': input_features, 'target': target_labels}


class SingleRIRDataset(data.Dataset):

    def __init__(self,
                 device: torch.device,
                 rir_data: RIRData,
                 new_sampling_radius: Optional[float] = None):
        """
        RIR dataset containing magnitude response of a single RIR around the unit circle,
        for different receiver positions.
        During batch processing, each batch will contain all the frequency bins
        but different sets of receiver positions
        Args:
            device (str): cuda or cpu
            rir_data (SingleRIRDataset): RIRData object containing a single RIR and its magnitude response
            new_sampling_radius (float): to reduce time aliasing artifacts due to insufficient sampling
                                     in the frequency domain, sample points on a circle whose radius
                                     is larger than 1 

        """
        self.device = device

        # frequency-domain data
        freq_bins_rad = torch.tensor(rir_data.freq_bins_rad)

        if new_sampling_radius in (1.0, None):
            # this ensures that we cover half the unit circle (other half is symmetric)
            self.z_values = torch.polar(torch.ones_like(freq_bins_rad),
                                        freq_bins_rad * 2 * np.pi)
        else:
            assert new_sampling_radius > 1.0
            logger.info(
                f"Sampling outside the unit circle at a radius {new_sampling_radius}"
            )
            # sample outside the unit circle
            self.z_values = torch.polar(
                new_sampling_radius * torch.ones_like(freq_bins_rad),
                freq_bins_rad * 2 * np.pi)

        self.rir_mag_response = torch.tensor(rir_data.rir_mag_response)
        self.late_rir_mag_response = torch.tensor(
            rir_data.late_rir_mag_response)
        self.early_rir_mag_response = torch.tensor(
            rir_data.early_rir_mag_response)

    def __len__(self):
        """Get length of dataset (equal to number of receiver positions)"""
        return len(self.z_values)

    def __getitem__(self, idx: int):
        """Get data at a particular index"""
        return {
            'z_values': self.z_values[idx],
            'target_rir_response': self.rir_mag_response[idx],
            'target_early_response': self.early_rir_mag_response[idx],
            'target_late_response': self.late_rir_mag_response[idx]
        }


def to_device(data_class: data.Dataset, device: torch.device):
    """Move all tensor attributes to self.device."""
    for field_name, field_value in data_class.__dict__.items():
        if isinstance(field_value, torch.Tensor):
            setattr(data_class, field_name, field_value.to(device))
        elif isinstance(field_value, Meshgrid):
            setattr(data_class, field_name, to_device(field_value, device))
        elif isinstance(field_value, np.ndarray):
            setattr(data_class, field_name,
                    torch.tensor(field_value, device=device))
    return data_class


def custom_collate(batch: data.Dataset):
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
    target_early_response = [
        item['target'].early_rir_mag_response for item in batch
    ]
    target_late_response = [
        item['target'].late_rir_mag_response for item in batch
    ]
    target_rir_response = [item['target'].rir_mag_response for item in batch]

    return {
        'z_values': z_values,
        'source_position': torch.stack(source_positions),
        'listener_position': torch.stack(listener_positions),
        'mesh_3D': mesh_3D_data,
        'target_early_response': torch.stack(target_early_response),
        'target_late_response': torch.stack(target_late_response),
        'target_rir_response': torch.stack(target_rir_response)
    }


def split_dataset(dataset: data.Dataset, split: float):
    """
    Randomly split a dataset into non-overlapping new datasets of 
    sizes given in 'split' argument
    """
    logger.info(f'Length of the dataset is {len(dataset)}')

    # use split % of dataset for validation
    train_set_size = int(len(dataset) * split)
    valid_set_size = len(dataset) - train_set_size

    train_set, valid_set = torch.utils.data.random_split(
        dataset, [train_set_size, valid_set_size])

    logger.info(
        f'The size of the training set is {len(train_set)} and the size of the validation set is {len(valid_set)}'
    )
    return train_set, valid_set


def get_dataloader(dataset: data.Dataset,
                   batch_size: int,
                   shuffle: bool = True,
                   device='cpu',
                   drop_last: bool = True,
                   collate_fn: Optional = None) -> data.DataLoader:
    """Create torch dataloader form given dataset"""
    if collate_fn is None:
        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=torch.Generator(device=device),
            drop_last=drop_last,
        )
    else:
        dataloader = data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     generator=torch.Generator(device=device),
                                     drop_last=drop_last,
                                     collate_fn=custom_collate)

    return dataloader


def get_device():
    """Output 'cuda' if gpu is available, 'cpu' otherwise"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_dataset(room_data: Union[RoomDataset, RIRData],
                 device: torch.device,
                 train_valid_split_ratio: float = 0.8,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 new_sampling_radius: Optional[float] = None):
    """
    Get training and validation dataset
    Args:
        room_data (RoomDataset/RIRData): object of type RoomDataset (for training with a grid of measurements) or 
                                         RIRData (for training with a single response)
        device (str): cuda (GPU) or cpu
        train_valid_split_ratio (float): ratio between training and validation set
        batch_size (int): number of samples in each batch size
        shuffle (bool): whether to randomly shuffle data during training
        new_sampling_radius (float): to reduce time aliasing artifacts due to insufficient sampling
                                     in the frequency domain, sample points on a circle whose radius
                                     is larger than 1 
    """
    if isinstance(room_data, RoomDataset):
        dataset = MultiRIRDataset(device,
                                  room_data,
                                  new_sampling_radius=new_sampling_radius)
        dataset = to_device(dataset, device)

        # split data into training and validation set
        train_set, valid_set = split_dataset(dataset, train_valid_split_ratio)

        # dataloaders
        train_loader = get_dataloader(train_set,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      device=device,
                                      drop_last=True,
                                      collate_fn=custom_collate)

        valid_loader = get_dataloader(valid_set,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      device=device,
                                      drop_last=True,
                                      collate_fn=custom_collate)
        return train_loader, valid_loader

    elif isinstance(room_data, RIRData):
        dataset = SingleRIRDataset(device,
                                   room_data,
                                   new_sampling_radius=new_sampling_radius)
        dataset = to_device(dataset, device)

        train_loader = get_dataloader(dataset,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      device=device,
                                      drop_last=False)
        return train_loader
