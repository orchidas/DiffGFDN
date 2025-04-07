import gc
import os
from pathlib import Path
import pickle
from typing import List, Tuple, Union

import h5py
import joblib
from loguru import logger
import numpy as np
from numpy.typing import ArrayLike, NDArray
from slope2noise.utils import calculate_amplitudes_least_squares, octave_filtering
import spaudiopy as sp

# flake8: noqa:E231


def save_subband_srirs(srirs: NDArray,
                       sample_rate: float,
                       common_t60: NDArray,
                       amplitudes_norm: NDArray,
                       noise_floor_norm: NDArray,
                       centre_freqs: List,
                       source_position: Union[NDArray, ArrayLike],
                       receiver_position: NDArray,
                       use_amp_preserving_filterbank: bool = True):
    """
    Filter SRIRs into subbands and save the parameters
    Args: srirs (NDArray): shape num_directions x num_time_samples x num_rirs
    """
    num_bands = len(centre_freqs)
    num_directions, num_time_samp, num_pos = srirs.shape
    filtered_srirs = np.zeros(
        (num_directions, num_time_samp, num_pos, num_bands))

    cs_save_path = Path(
        'resources/Georg_3room_FDTD/Common_Slope_Analysis_Results/')

    for band_idx, band_freq in enumerate(centre_freqs):
        logger.info(f"Processing frequency band: {band_freq} Hz")

        # Preallocate only for this band
        cur_rirs = np.zeros((num_directions, num_time_samp, num_pos))
        cur_amps = np.zeros((num_directions, *amplitudes_norm.shape[1:]))
        cur_amps_norm = amplitudes_norm[1:]
        cur_noise = np.zeros((num_directions, *noise_floor_norm.shape[1:]))
        cur_noise_norm = noise_floor_norm[1:]

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
                cur_rirs[j, ...] = filtered_srirs[..., band_idx]
                cur_amps[j, ...] = cur_amps_ls[band_idx, ...]
                cur_noise[j, ...] = cur_noise_ls[band_idx, ...]
            else:
                logger.info(f"Reading saved CS params for direction {j+1}")

                with open(cs_params_pkl_path, 'rb') as handle:
                    data = joblib.load(handle)  # or pickle.load()

                cur_rirs[j, ...] = data['directional_filtered_rirs'][...,
                                                                     band_idx]
                cur_amps[j, ...] = data['cs_amps'][band_idx, ...]
                cur_noise[j, ...] = data['cs_noise_floor'][band_idx, ...]

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
            'noise_floor_norm': cur_noise_norm
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
                                  des_dir[0, :],
                                  des_dir[1, :],
                                  sh_type='real')
    # butterworth weights
    beamform_weights = sp.sph.butterworth_modal_weights(ambi_order, k=5, n_c=3)
    # beamforming matrix of size num_directions * (N+1)^2
    beamform_matrix = sp.sph.repeat_per_order(beamform_weights) * sph_matrix
    # desired SRIRs in given directions
    dir_srirs = np.einsum('jn, ntr -> jtr', beamform_matrix, ambi_srirs)
    return dir_srirs


def calculate_cs_params_custom(
    srirs: NDArray,
    t_vals: NDArray,
    f_bands: List,
    fs: int,
    batch_size: int = 50,
) -> Tuple[NDArray, NDArray]:
    """
    Calculate custom CS parameters from the common decay times
    Args:
        srirs (NDArray): rir matrix of size n_rirs x ir_len x n_bands
        t_vals (NDArray): common decay times of size n_bands x 1 x n_slopes
        f_bands (List): list of frequencies for filtering
        fs (int): sampling frequency
        batch_size: running estimation for all RIRs at once is difficult, so split in batches
    Returns:
        Tuple[NDArray, NDArray]: Amplitudes of shape n_bands x n_slopes x n_rirs
                                 Noise of shape n_bands x 1 x n_rirs
    """
    num_rirs = srirs.shape[0]
    num_slopes = t_vals.shape[-1]
    num_batches = int(np.ceil(num_rirs / float(batch_size)))
    logger.info(f"Number of batches : {num_batches}")
    a_vals = np.zeros((len(f_bands), num_slopes, num_rirs))
    n_vals = np.zeros((len(f_bands), 1, num_rirs))

    for n in range(num_batches):
        batch_idx = np.arange(n * batch_size,
                              max(num_rirs, (n + 1) * batch_size),
                              dtype=np.int32)
        cur_srirs = srirs[batch_idx, :]
        num_rirs_per_batch = cur_srirs.shape[0]
        # convert to shape 1 x n_slopes x n_bands
        t_vals_exp = t_vals.transpose(1, -1, 0)
        # ensure t_vals is of shape n_rirs x n_slopes x n_bands
        t_vals_exp = np.repeat(t_vals_exp, num_rirs_per_batch, axis=0)

        assert t_vals_exp.shape[0] == num_rirs_per_batch and t_vals_exp.shape[
            -1] == len(f_bands)

        # of shape n_rirs x ir_len x n_bands

        # calculate amplitudes and noise floor - this is of shape nrirs x n_slopes+1 x n_bands
        est_amps = calculate_amplitudes_least_squares(t_vals_exp,
                                                      fs,
                                                      cur_srirs,
                                                      f_bands,
                                                      leave_out_ms=1000)
        a_vals[..., batch_idx] = est_amps[:, 1:, :].transpose(-1, 1, 0)
        n_vals[..., batch_idx] = np.expand_dims(est_amps[:, 0, :],
                                                axis=1).transpose(-1, 1, 0)
    return a_vals, n_vals


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
        directional_srirs = process_ambi_srirs(srirs, ambi_order, directions)
        common_t60 = np.asarray(common_t60)
        amplitudes_norm = np.asarray(amplitudes_norm)
        noise_floor_norm = np.asarray(noise_floor_norm)

        # Convert the list to a NumPy array if needed
        data_dict = {
            'fs': sample_rate,
            'srcPos': source_position,
            'rcvPos': receiver_position,
            'dir_srirs': directional_srirs,
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

        directional_srirs = data_dict['dir_srirs']
        common_t60 = data_dict['common_decay_times']
        amplitudes_norm = data_dict['amplitudes_norm']
        directions = data_dict['directions']
        noise_floor_norm = data_dict['noise_floor_norm']
        sample_rate = data_dict['fs']
        source_position = data_dict['srcPos']
        receiver_position = data_dict['rcvPos']

    save_subband_srirs(directional_srirs.copy(), sample_rate, common_t60,
                       amplitudes_norm, noise_floor_norm, freqs,
                       source_position, receiver_position,
                       use_amp_preserving_filterbank)


if __name__ == '__main__':
    main()
