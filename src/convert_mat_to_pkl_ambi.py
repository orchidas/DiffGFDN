import gc
import os
from pathlib import Path
import pickle
from typing import List, Union

import h5py
import joblib
from loguru import logger
import numpy as np
from numpy.typing import ArrayLike, NDArray
from slope2noise.utils import octave_filtering
import spaudiopy as sp

from convert_mat_to_pkl import calculate_cs_params_custom

# flake8: noqa:E231


def save_subband_srirs(srirs: NDArray,
                       sample_rate: float,
                       common_t60: NDArray,
                       amplitudes_norm: NDArray,
                       noise_floor_norm: NDArray,
                       centre_freqs: List,
                       source_position: Union[NDArray, ArrayLike],
                       receiver_position: NDArray,
                       sph_directions: NDArray,
                       use_amp_preserving_filterbank: bool = True):
    """
    Filter SRIRs into subbands and save the parameters
    Args: s
        srirs (NDArray): shape num_directions x num_time_samples x num_pos
        common_t60 (NDArray): shape num_freq_bands x 1 x num_slopes
        amplitudes_norm (NDArray): shape num_freq_bands x num_directions x num_slopes x num_pos
        noise_floor_norm (NDArray): shape num_freq_bands x num_directions x 1 x num_pos
    """
    num_bands = len(centre_freqs)
    num_directions, num_time_samp, num_pos = srirs.shape
    num_slopes = common_t60.shape[-1]
    filtered_srirs = np.zeros(
        (num_directions, num_time_samp, num_pos, num_bands))

    cs_save_path = Path(
        'resources/Georg_3room_FDTD/Common_Slope_Analysis_Results/')

    for band_idx, band_freq in enumerate(centre_freqs):
        logger.info(f"Processing frequency band: {band_freq} Hz")

        # Preallocate only for this band
        cur_rirs = np.zeros((num_time_samp, num_directions, num_pos))
        cur_amps = np.zeros((num_slopes, num_directions, num_pos))
        cur_amps_norm = amplitudes_norm[band_idx, ...].transpose(1, 0, -1)
        cur_noise = np.zeros((1, num_directions, num_pos))
        cur_noise_norm = noise_floor_norm[band_idx, ...].transpose(1, 0, -1)

        for j in range(num_directions):
            cs_params_pkl_path = cs_save_path / f"custom_cs_params_dir={j}.pkl"

            # this needs to run once per direction, regardless of frequency band
            if not os.path.exists(cs_params_pkl_path):
                logger.info(f"Filtering SRIRs for direction {j+1}")
                filtered_srirs[j, ...] = octave_filtering(
                    srirs[j, ...].T,
                    sample_rate,
                    centre_freqs,
                    use_amp_preserving_filterbank=use_amp_preserving_filterbank
                ).transpose(1, 0, -1)

                logger.info(f"LS amplitude calculation for direction {j+1}")
                cur_amps_ls, cur_noise_ls = calculate_cs_params_custom(
                    filtered_srirs[j, ...].transpose(1, 0, -1).copy(),
                    common_t60,
                    centre_freqs,
                    sample_rate,
                    batch_size=receiver_position.shape[-1],
                    downsample_factor=10,
                )

                # Write the data to a pickle file
                with open(cs_params_pkl_path, 'wb') as pickle_file:
                    pickle.dump(
                        {
                            'cs_t60': common_t60,
                            'cs_amps': cur_amps_ls,
                            'cs_noise_floor': cur_noise_ls,
                            'directional_filtered_rirs': filtered_srirs[j, ...]
                        }, pickle_file)
                cur_rirs[:, j, :] = filtered_srirs[..., band_idx]
                cur_amps[:, j, :, :] = cur_amps_ls[band_idx, ...]
                cur_noise[:, j, :, :] = cur_noise_ls[band_idx, ...]
            else:
                logger.info(f"Reading saved CS params for direction {j+1}")

                with open(cs_params_pkl_path, 'rb') as handle:
                    data = joblib.load(handle)  # or pickle.load()

                cur_rirs[:, j, :] = data['directional_filtered_rirs'][...,
                                                                      band_idx]
                cur_amps[:, j, :] = data['cs_amps'][band_idx, ...]
                cur_noise[:, j, :] = data['cs_noise_floor'][band_idx, ...]

                del data
                gc.collect()

        # Save one pkl file per frequency
        data_dict = {
            'fs': sample_rate,
            'srcPos': source_position,
            'rcvPos': receiver_position,
            'srirs': cur_rirs,
            'band_centre_hz': band_freq,
            'common_decay_times': common_t60[band_idx],
            'amplitudes': cur_amps,
            'amplitudes_norm': cur_amps_norm,
            'noise_floor': cur_noise,
            'noise_floor_norm': cur_noise_norm,
            'directions': sph_directions,
        }

        pickle_file_path = Path(
            f"resources/Georg_3room_FDTD/srirs_spatial_band_centre={band_freq:.0f}Hz.pkl"
        ).resolve()

        # Write the data to a pickle file
        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump(data_dict, pickle_file)

        logger.info(
            f"Done saving pickle file for centre frequency {band_freq} Hz")


def process_ambi_srirs(ambi_srirs: NDArray, ambi_order: int,
                       des_dir: NDArray) -> NDArray:
    """
    Process the SRIRs reconrded with SMA to get RIRs in different directions,
    according to  Gotz et. al, "Common slope modelling of late reverberation"
    Args:
        ambi_srirs (NDArray): ambi srirs of shape (N_sp + 1)^2 x time_samp  x num_receivers
        ambi_order (int): N_sp order of the ambisonics recordings
        des_dir (NDArray): azimuth and polar angles of desired directions
                           of size 2 x num_directions
    Returns:
        NDArray: directional RIRs of shape num_directions x time_samples x num_receivers
    """
    # get spherical harmonics matrix
    assert ambi_srirs.shape[0] == (ambi_order + 1)**2
    # output of size num_directions x (N_sp+1)^2
    sph_matrix = sp.sph.sh_matrix(ambi_order,
                                  np.deg2rad(des_dir[0, :]),
                                  np.deg2rad(des_dir[1, :]),
                                  sh_type='real')
    # butterworth weights of size (N+1)^2
    beamform_weights = sp.sph.butterworth_modal_weights(ambi_order, k=5, n_c=3)
    # beamforming matrix of size num_directions * (N+1)^2
    beamform_matrix = sp.sph.repeat_per_order(beamform_weights) * sph_matrix
    # desired SRIRs in given directions
    dir_srirs = np.einsum('jn, ntr -> jtr', beamform_matrix, ambi_srirs)
    return dir_srirs


def main():
    """Main function to save the modified ThreeRoomDataset"""
    # Converts the .mat file to pickle format which can be read much faster by Python
    # Specify output pickle path
    pickle_file_path = Path(
        "resources/Georg_3room_FDTD/srirs_spatial.pkl").resolve()
    freqs = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    use_amp_preserving_filterbank = True

    if not os.path.exists(pickle_file_path):
        logger.info("Reading mat file")

        # Load the MATLAB v7.3 .mat file using h5py
        file_path = Path("resources/Georg_3room_FDTD/srirs.mat").resolve()

        with h5py.File(file_path, 'r') as mat_file:
            # Get the dataset
            srir_mat = mat_file['srirDataset']
            sample_rate = np.squeeze(srir_mat['fs'][:])
            source_position = srir_mat['srcPos'][:]
            receiver_position = srir_mat['rcvPos'][:]
            # these are second order ambisonic signals
            srirs = srir_mat['srirs'][:]

        # load the common slopes from the other mat files
        file_path = Path(
            "resources/Georg_3room_FDTD/Common_Slope_Analysis_Results/")
        filename = 'cs_analysis_results'
        use_amp_preserving_filterbank = True

        common_t60 = []
        amplitudes_norm = []
        noise_floor_norm = []
        directions = []
        ambi_order = int(np.sqrt(srirs.shape[0]) - 1)

        for i in range(len(freqs)):
            full_path = file_path / f'{filename}_{freqs[i]}.mat'
            with h5py.File(full_path.resolve(), 'r') as mat_file:
                data = mat_file['analysisResults']
                common_t60.append(data['commonDecayTimes'][:])
                amplitudes_norm.append(data['aVals'][:])
                noise_floor_norm.append(data['nVals'][:])
                directions = data['secDirs_deg'][:]

        # get beamformed signals in different directions
        directional_rirs = process_ambi_srirs(srirs, ambi_order, directions)
        common_t60 = np.asarray(common_t60)
        amplitudes_norm = np.asarray(amplitudes_norm)
        noise_floor_norm = np.asarray(noise_floor_norm)

        # Convert the list to a NumPy array if needed
        data_dict = {
            'fs': sample_rate,
            'srcPos': source_position,
            'rcvPos': receiver_position,
            'dir_srirs': directional_rirs,
            'srirs': srirs.transpose(1, 0, -1),
            'band_centre_hz': freqs,
            'common_decay_times': common_t60,
            'amplitudes_norm': amplitudes_norm,
            'noise_floor_norm': noise_floor_norm,
            'directions': directions,
        }

        # Write the data to a pickle file
        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump(data_dict, pickle_file)

        logger.info("Saved pickle file")

    else:
        logger.info("Reading from saved pickle file")
        with open(pickle_file_path, 'rb') as handle:
            data_dict = pickle.load(handle)

        directional_rirs = data_dict['dir_srirs']
        common_t60 = data_dict['common_decay_times']
        amplitudes_norm = data_dict['amplitudes_norm']
        directions = data_dict['directions']
        noise_floor_norm = data_dict['noise_floor_norm']
        sample_rate = data_dict['fs']
        source_position = data_dict['srcPos']
        receiver_position = data_dict['rcvPos']

    save_subband_srirs(directional_rirs.copy(), sample_rate, common_t60,
                       amplitudes_norm, noise_floor_norm, freqs,
                       source_position, receiver_position, directions,
                       use_amp_preserving_filterbank)


if __name__ == '__main__':
    main()
