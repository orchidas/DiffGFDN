from pathlib import Path
import time
from typing import Tuple, Union

from librosa import resample
from loguru import logger
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import irfft, rfft
import sofar
import spaudiopy as spa
from tqdm import tqdm

from diff_gfdn.utils import is_unitary

# flake8: noqa:E722
# pylint: disable=E0606, E1126


def cart2sph(x: Union[NDArray, float],
             y: Union[NDArray, float],
             z: Union[NDArray, float],
             axis: int = -1,
             degrees: bool = True) -> NDArray:
    """Convert cartesian coordinates to spherical coordinates"""
    x, y, z = np.broadcast_arrays(x, y, z)
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)

    if degrees:
        azimuth = np.degrees(azimuth)
        elevation = np.degrees(elevation)

    return np.stack((azimuth, elevation, r), axis=axis)


def sph2cart(azimuth: Union[NDArray, float],
             elevation: Union[NDArray, float],
             r: Union[NDArray, float],
             axis: int = -1,
             degrees: bool = True) -> NDArray:
    """Convert spherical coordinates to cartesian coordinates"""
    azimuth, elevation, r = np.broadcast_arrays(azimuth, elevation, r)
    if degrees:
        azimuth = np.radians(azimuth)
        elevation = np.radians(elevation)
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return np.stack((x, y, z), axis=axis)


def unpack_coordinates(
        coord_matrix: NDArray,
        axis: int = -1) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Unpack a coordinate matrix into separate arrays each containing
    one of 3 dimensions
    """
    if axis != -1:
        coord_matrix = coord_matrix.T

    dim1 = coord_matrix[:, 0]
    dim2 = coord_matrix[:, 1]
    dim3 = coord_matrix[:, 2]

    return dim1, dim2, dim3


class HRIRSOFAReader:

    def __init__(self, sofa_path: Path):
        """
        Args:
            sofa_path (Path): path to the sofa file containing HRTFs
            ambi_order (int): ambisonics order for spherical harmonic conversion
        """
        try:
            self.sofa_reader = sofar.read_sofa(str(sofa_path),
                                               verify=True,
                                               verbose=True)
        except Exception as exc:
            raise FileNotFoundError(
                "SOFA file not found in specified location!") from exc

        self.fs = self.sofa_reader.Data_SamplingRate
        self.ir_data = self.sofa_reader.Data_IR
        self.num_dims = self.ir_data.ndim
        logger.info(self.sofa_reader.list_dimensions)
        logger.info(f'Shape of the IRs is {self.ir_data.shape}')

        if self.num_dims > 3:
            self.num_meas, self.num_emitter, self.num_receivers, self.ir_length = self.sofa_reader.Data_IR.shape
        else:
            self.num_meas, self.num_receivers, self.ir_length = self.sofa_reader.Data_IR.shape

        if self.sofa_reader.ListenerView.shape[0] == 1:
            self.use_source_view = True
            self.listener_view = self.get_source_view(coord_type="spherical")
        else:
            self.listener_view = self.get_listener_view(coord_type="spherical")

    def get_listener_view(self, coord_type: str = "cartesian") -> NDArray:
        """Get the listener view array. An array of vectors corresponding to the view direction of the listener.
        This can be in spherical or cartesian coordinates.

        Args:
            coord_type (str): Required coordinate system, by default "cartesian".
                Options: cartesian or spherical

        Returns:
            NDArray:
                Array of listener view vectors in HuRRAh coordinate convention. Dims: [M/I, C]
                M is number of measurements. I is 1. C is 3 (for 3D coordinates).
                Spherical coordinates have angles in degrees.

        Raises:
            ValueError: If the given coord_type is not one of the supported options.
            ValueError: If the SOFA listener view is not in degree,
                                      degree, metre units
        """
        coord_type = coord_type.lower()
        is_source_cart = self.sofa_reader.ListenerView_Type.lower(
        ) == "cartesian"

        if is_source_cart:
            list_view_cart = self.sofa_reader.ListenerView
        else:
            # check that we've got angles in degrees
            if self.sofa_reader.ListenerView_Units != "degree, degree, metre":
                raise ValueError(
                    f"Incompatible units for type of ListenerView in SOFA file. "
                    f"Type: {self.sofa_reader.ListenerView_Type}, Units: {self.sofa_reader.ListenerView_Units} "
                    "Should be: degree, degree, metre")
            list_view_sph = self.sofa_reader.ListenerView
            # if radius is set to zero in file, set to 1
            list_view_sph[list_view_sph[:, 2] == 0.0, 2] = 1.0
            az, el, r = unpack_coordinates(list_view_sph, axis=-1)

        # now convert to spherical if needed
        if coord_type == "cartesian":
            list_view_cart = sph2cart(az, el, r, axis=-1, degrees=True)
            return list_view_cart
        else:
            return np.stack((az, el, r), axis=-1)

    def get_source_view(self, coord_type: str = "cartesian") -> NDArray:
        """Get the source position array. An array of vectors corresponding to the view direction of the source.
        This can be in spherical or cartesian coordinates.

        Args:
            coord_type (str): Required coordinate system, by default "cartesian".
                Options: cartesian or spherical

        Returns:
            NDArray:
                Array of listener view vectors Dims: [M/I, C]
                M is number of measurements. I is 1. C is 3 (for 3D coordinates).
                Spherical coordinates have angles in degrees.

        Raises:
            ValueError: If the given coord_type is not one of the supported options.
            ValueError: If the SOFA listener view is not in degree,
                                      degree, metre units
        """
        coord_type = coord_type.lower()
        is_source_cart = self.sofa_reader.SourcePosition_Type.lower(
        ) == "cartesian"

        if is_source_cart:
            list_view_cart = self.sofa_reader.SourcePosition
        else:
            # check that we've got angles in degrees
            if self.sofa_reader.SourcePosition_Units != "degree, degree, metre":
                raise ValueError(
                    f"Incompatible units for type of SourcePosition in SOFA file. "
                    f"Type: {self.sofa_reader.SourcePosition_Type}, Units: {self.sofa_reader.SourcePosition_Units} "
                    "Should be: degree, degree, metre")
            list_view_sph = self.sofa_reader.SourcePosition
            # if radius is set to zero in file, set to 1
            list_view_sph[list_view_sph[:, 2] == 0.0, 2] = 1.0
            az, el, r = unpack_coordinates(list_view_sph, axis=-1)

        # now convert to spherical if needed
        if coord_type == "cartesian":
            list_view_cart = sph2cart(az, el, r, axis=-1, degrees=True)
            return list_view_cart
        else:
            return np.stack((az, el, r), axis=-1)

    def resample_hrirs(self, new_fs: float):
        """Resample the HRIRs to a new sampling rate"""
        # librosa can only handle one ear axis at a time
        left_ir_data = resample(self.ir_data[:, 0, :].copy(),
                                orig_sr=self.fs,
                                target_sr=new_fs,
                                axis=1)
        right_ir_data = resample(self.ir_data[:, 1, :].copy(),
                                 orig_sr=self.fs,
                                 target_sr=new_fs,
                                 axis=1)
        self.ir_data = np.stack((left_ir_data, right_ir_data),
                                axis=-1).transpose(0, -1, 1)

    def get_ir_corresponding_to_listener_view(
        self,
        des_listener_view: NDArray,
        axis: int = -1,
        coord_type: str = "spherical",
        degrees: bool = True,
    ) -> NDArray:
        """
        Get IR corresponding to a particular listener view
        Args:
            des_listener_view: P x 3 array of desired listener views
            axis (int): axis of coordinates
            coord_type (str): coordinate system type when specifying listener view
            degrees (bool): whether the listener view is specified in degrees
            use_source_view (bool): if the SOFA file contains only source views
        Returns:
            P x E x R x N IR corresponding to the particular listener views 
        """

        if axis != -1:
            des_listener_view = des_listener_view.T

        num_views = des_listener_view.shape[0]
        assert num_views < self.num_meas

        # euclidean distance between desired and available views
        dist = np.zeros((self.num_meas, num_views))
        des_ir_matrix = np.zeros((num_views, self.ir_data.shape[1:]),
                                 dtype=float)

        if coord_type == "spherical":
            az, el, r = unpack_coordinates(des_listener_view.copy(), axis=axis)
            des_listener_view = sph2cart(az, el, r, axis=axis, degrees=degrees)

        # find index of view that minimuses the error from the desied view
        for k in range(num_views):
            dist[:, k] = np.sqrt(
                np.sum((self.listener_view - self.des_listener_view[k, :])**2,
                       axis=axis))
            closest_idx = np.argmin(dist[:, k])
            des_ir_matrix[k, ...] = self.ir_data[closest_idx, ...]

        return des_ir_matrix

    def get_all_irs_corresponding_to_receiver(self,
                                              receiver_idx: int) -> NDArray:
        """
        Returns all HRIRs corresponding to a particular receiver. (0, 1) for binaural
        data
        Args:
            received_idx (int) : index of the receiver
        Returns:
            NDArray: array of size M x E x N containing the IRs
        """
        assert receiver_idx < self.num_receivers
        return self.sofa_reader.Data_IR[:, receiver_idx, ...]

    def get_spherical_harmonic_representation(self,
                                              ambi_order: int) -> NDArray:
        """Get the spherical harmonic representation of the HRTFs using specified ambisonics order"""

        # 1. Compute HRTFs from time-domain HRIRs
        fft_size = 2**int(np.ceil(np.log2(self.ir_length)))
        hrtfs = rfft(self.ir_data, n=fft_size,
                     axis=-1)  # shape: (num_dirs, 2, num_freq_bins)

        # 2. Get SH matrix
        incidence_az = np.deg2rad(self.listener_view[..., 0])
        # zenith angle is different from elevation angle
        incidence_zen = np.deg2rad(90 -
                                   self.listener_view[..., 1])  # zenith angle
        # (num_dirs, num_sh_channels)
        sh_matrix = spa.sph.sh_matrix(ambi_order, incidence_az, incidence_zen)
        assert is_unitary(sh_matrix)

        # 3. Least squares fit for each frequency bin
        sh_hrtfs = np.einsum('nd, drf -> nrf', sh_matrix.T, hrtfs)

        sh_ir = irfft(sh_hrtfs, n=self.ir_length, axis=-1)
        return sh_ir  # shape: (num_sh_channels, 2, num_time_samples)


class SRIRSOFAWriter:
    """Write SRIR data to a SOFA file, using convention "SingleRoomSRIR".

    This is defined online at
    https://www.sofaconventions.org/mediawiki/index.php/MultiSpeakerBRIR

    Args:
        num_receivers (int): Number microphones (measurements) distributed across the room.
        ambi_order(int): Ambisonics order
        ir_length (int): The length in samples of the IRs
        samplerate (int): The sample rate of the data, defaults to 48000
    """

    def __init__(
        self,
        num_receivers: int,
        ambi_order: int,
        ir_length: int,
        samplerate: float = 48000.0,
    ):
        # convention definition is given online
        self.conv = "SingleRoomSRIR"
        self.num_channels = (ambi_order + 1)**2
        self.num_receivers = num_receivers
        self.ir_length = ir_length
        self.dims = {
            "R": self.num_channels,
            "M": num_receivers,
            "N": ir_length,
            "C": 3,  # coordinate dimension (xyz or aed)
            "I": 1,  # for singleton dimensions
        }
        self.sofa = sofar.Sofa(self.conv)
        self.samplerate = samplerate

        # Fill in dimensions
        self.sofa.Data_SamplingRate = np.array([self.samplerate])
        # Metadata (optional)
        self.sofa.GLOBAL_ApplicationName = 'AmbisonicSRIRWriter'
        self._init_sofa()

    def _init_sofa(self):
        # other attributes that need to be saved to initialise the SOFA object
        self.sofa.ListenerPosition = np.zeros((self.dims["M"], self.dims["C"]),
                                              dtype=np.float32)
        self.sofa.ListenerPosition_Type = 'cartesian'
        self.sofa.ListenerPosition_Units = 'meter'

        # should be of shape (R, C, I)
        self.sofa.ReceiverView = np.tile(
            np.array([0, 1, 0], dtype=np.float32),
            (self.dims["R"], self.dims["I"]))[:, :, None]  # Facing +Y
        self.sofa.ReceiverUp = np.tile(
            np.array([0, 0, 1], dtype=np.float32),
            (self.dims["R"], self.dims["I"]))[:, :, None]  # Up +Z
        # shape is (R,S)
        self.sofa.ReceiverDescriptions = np.array(["SecondOrderMics"] *
                                                  self.dims["R"],
                                                  dtype='U')
        # shape is (I, R)
        self.sofa.Data_Delay = np.zeros((self.dims["I"], self.dims["R"]),
                                        dtype=np.float32)
        # shape is M,
        self.sofa.MeasurementDate = np.full(self.dims["M"],
                                            time.time(),
                                            dtype=np.float64)

    def set_source_positions(self,
                             source_position: ArrayLike,
                             coord_sys: str = 'cartesian'):
        """Set source positions"""
        if source_position.ndim > 1:
            assert source_position.shape[-1] == self.dims["C"]
        else:
            assert len(source_position) == self.dims["C"]

        # shape should be (1, 3)
        if coord_sys != 'cartesian':
            source_position = sph2cart(source_position[:, 0],
                                       source_position[:, 1],
                                       source_position[:, 2])
        else:
            source_position = source_position.reshape(1, self.dims["C"])

        # should be of shape (R, C)
        self.sofa.SourcePosition = np.tile(
            source_position, (self.dims["M"], 1)).astype(np.float32)

        # Set units
        self.sofa.SourcePosition_Type = 'cartesian'
        self.sofa.SourcePosition_Units = 'meter'

    def set_receiver_positions(self,
                               receiver_positions: NDArray,
                               coord_sys: str = 'cartesian'):
        """Set receiver positions"""
        assert receiver_positions.shape == (
            self.dims["M"],
            self.dims["C"]), "Source positions should be of size M, 3"
        if coord_sys != 'cartesian':
            receiver_positions = sph2cart(receiver_positions[:, 0],
                                          receiver_positions[:, 1],
                                          receiver_positions[:, 2])

        # should be of shape (M, C)
        self.sofa.ListenerPosition = receiver_positions.astype(np.float32)
        self.sofa.ListenerPosition_Type = 'cartesian'
        self.sofa.ListenerPosition_Units = 'meter'

        self.sofa.ReceiverPosition = np.zeros(
            (self.dims["R"], self.dims["C"], self.dims["I"]), dtype=np.float32)
        self.sofa.ReceiverPosition_Type = 'cartesian'
        self.sofa.ReceiverPosition_Units = 'meter'

    def set_ir_data(self, rir_data: NDArray):
        """
        Set the IR data for the SOFA writer, 
        dimensions should be be  num_ambi_channels x num_receivers x time_samples
        """
        assert rir_data.shape == (
            self.dims["M"], self.dims["R"], self.dims["N"]
        ), "RIRs should be of shape num_ambi_channels x num_receivers x time_samples"
        self.rir_data = rir_data
        self.sofa.Data_IR = rir_data.astype(np.float32)  # Shape: [M, R, N]

    def resample_srirs(self, new_sample_rate: float):
        """
        Resample the SRIRs to a new sample rate - this is needed when loading in REAPER
        which is at a sample rate of 48kHz by default
        """
        new_ir_length = int(
            np.round(self.ir_length * (new_sample_rate / self.samplerate)))
        resampled_rir = np.zeros(
            (self.num_receivers, self.num_channels, new_ir_length))
        for chan in range(self.num_channels):
            resampled_rir[:, chan, :] = resample(self.rir_data[:,
                                                               chan, :].copy(),
                                                 orig_sr=self.samplerate,
                                                 target_sr=new_sample_rate,
                                                 axis=-1)
        self.ir_length = new_ir_length
        self.rir_data = resampled_rir
        self.sofa.Data_IR = resampled_rir.astype(
            np.float32)  # Shape: [M, R, N]
        self.sofa.Data_SamplingRate = np.array([new_sample_rate])

    def write_to_file(self, filename: str, compression: int = 4):
        """Write the SOFa object to a file.

        Args:
            filename (str): The filename to use.
            compression (int): Amount of data compression used in the underlying HDF5 file.
            The range if 0 (no compression) to 9 (most compression). Defaults to 4.
        """
        self.sofa.verify()
        sofar.write_sofa(
            filename,
            self.sofa,
            compression=compression,
        )


def convert_srir_to_brir(srirs: NDArray, hrtf_reader: HRIRSOFAReader,
                         head_orientations: ArrayLike) -> NDArray:
    """
    Convert SRIRs to BRIRs for specific orientations
    Args:
        srirs (NDArray): SRIRs of shape num_pos x num_ambi_channels x num_time_samp
        sample_rate (float): sample rate of the SRIRs
        hrtf_reader (HRIRSOFAReader): for parsing SOFA file
        head_orientations (ArrayLike): head orientations of shape num_ori x  2
    Returns:
        BRIRs of shape num_pos x num_ori x num_time_samples x 2
    """
    ambi_order = int(np.sqrt(srirs.shape[1] - 1))
    num_receivers = srirs.shape[0]
    num_freq_bins = 2**int(np.ceil(np.log2(srirs.shape[-1])))

    # size is num_ambi_channels x num_receivers x num_time_samples
    hrir_sh = hrtf_reader.get_spherical_harmonic_representation(ambi_order)
    ambi_rtfs = rfft(srirs, num_freq_bins, axis=-1)

    # these are of shape num_ambi_channels x 2 x num_freq_samples
    ambi_hrtfs = rfft(hrir_sh, n=num_freq_bins, axis=-1)
    logger.info("Done calculating FFTs")

    num_orientations = head_orientations.shape[0]
    brirs = np.zeros((num_receivers, num_orientations, num_freq_bins, 2))

    for rec_pos_idx in tqdm(range(num_receivers)):

        # shape is num_ambi_channels x num_freqs
        cur_ambi_rtf = ambi_rtfs[rec_pos_idx, ...]

        for ori_idx in range(num_orientations):
            cur_head_orientation = head_orientations[ori_idx, :]

            #rotate the soundfield in the opposite direction - size num_freq_bins x num_ambi_channels
            cur_rotation_matrix = spa.sph.sh_rotation_matrix(
                ambi_order,
                -cur_head_orientation[0],
                -cur_head_orientation[1],
                0,
                sh_type='real')

            rotated_ambi_rtf = cur_ambi_rtf.T @ cur_rotation_matrix.T

            # get the binaural room transfer function
            cur_brtf = np.einsum('nrf, fn -> fr', np.conj(ambi_hrtfs),
                                 rotated_ambi_rtf)
            # get the BRIR
            cur_brir = irfft(cur_brtf, n=num_freq_bins, axis=0)
            brirs[rec_pos_idx, ori_idx, ...] = cur_brir

    return brirs
