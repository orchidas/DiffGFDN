from pathlib import Path
from typing import Tuple, Union

from librosa import resample
from loguru import logger
import numpy as np
from numpy.typing import ArrayLike, NDArray
import sofar
import spaudiopy as spa

# flake8: noqa:E722
# pylint: disable=E0606


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
                Array of listener view vectors in HuRRAh coordinate convention. Dims: [M/I, C]
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
        # get azimuth and elevation angles from the dataset
        incidence_az = np.deg2rad(self.listener_view[..., 0])
        # zenith angle is different from elevation angle
        incidence_zen = np.deg2rad(90 - self.listener_view[..., 1])
        # of shape (num_hrtf_directions, (N_sp + 1)**2)
        sh_matrix = spa.sph.sh_matrix(ambi_order, incidence_az, incidence_zen)
        # output is of size num_ambi_channels x num_receivers x num_time_samples
        sh_ir = np.einsum('jrt, jn -> nrt', self.ir_data, sh_matrix)
        return sh_ir
