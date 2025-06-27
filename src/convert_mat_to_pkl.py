from pathlib import Path
import pickle
from typing import List, Tuple, Union

import h5py
from loguru import logger
import numpy as np
from numpy.typing import ArrayLike, NDArray
from slope2noise.slope2noise.utils import calculate_amplitudes_least_squares, octave_filtering

# pylint: disable=W0621, E1120
# flake8: noqa:E231


def save_subband_rirs(rirs: NDArray, sample_rate: float, common_t60: NDArray,
                      amplitudes_norm: NDArray, noise_floor_norm: NDArray,
                      amplitudes: NDArray, noise_floor: NDArray,
                      centre_freqs: List, source_position: Union[NDArray,
                                                                 ArrayLike],
                      receiver_position: NDArray,
                      use_amp_preserving_filterbank: bool):
    """Filter RIRs into subbands and save the parameters"""

    logger.info("Saving subband RIRs after filtering")
    # filter the RIRs in octave bands
    filtered_rirs = octave_filtering(
        rirs,
        sample_rate,
        centre_freqs,
        use_amp_preserving_filterbank=use_amp_preserving_filterbank)

    num_bands = len(centre_freqs)
    for band in range(num_bands):
        cur_common_t60 = common_t60[band]
        cur_amplitudes = amplitudes[band, ...]
        cur_amplitudes_norm = amplitudes_norm[band, ...]
        cur_noise_floor = noise_floor[band, ...]
        cur_noise_floor_norm = noise_floor_norm[band, ...]
        cur_rir = filtered_rirs[..., band]

        data_dict = {
            'fs': sample_rate,
            'srcPos': source_position,
            'rcvPos': receiver_position,
            'srirs': cur_rir,
            'band_centre_hz': centre_freqs[band],
            'common_decay_times': cur_common_t60,
            'amplitudes': cur_amplitudes,
            'amplitudes_norm': cur_amplitudes_norm,
            'noise_floor': cur_noise_floor,
            'noise_floor_norm': cur_noise_floor_norm
        }
        # Specify the output pickle file path
        pickle_file_path = Path(
            f"resources/Georg_3room_FDTD/srirs_band_centre={centre_freqs[band]:.0f}Hz.pkl"
        ).resolve()

        # Write the data to a pickle file
        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump(data_dict, pickle_file)

        logger.info(
            f"Done saving pickle file for centre frequency {centre_freqs[band]} Hz"
        )


def calculate_cs_params_custom(
        srirs: NDArray,
        t_vals: NDArray,
        f_bands: List,
        fs: int,
        batch_size: int = 50,
        downsample_factor: int = 1,
        use_amp_preserving_filterbank: bool = True) -> Tuple[NDArray, NDArray]:
    """
    Calculate custom CS parameters from the common decay times
    Args:
        srirs (NDArray): rir matrix of size n_rirs x ir_len x n_bands / n_rirs x ir_len
        t_vals (NDArray): common decay times of size n_bands x 1 x n_slopes
        f_bands (List): list of frequencies for filtering
        fs (int): sampling frequency
        batch_size: running estimation for all RIRs at once is difficult, so split in batches
        downsample_factor (int): by how much to downsample the EDC when calculating amplitudes with LS
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

        if srirs.ndim == 2:
            # of shape n_rirs x ir_len x n_bands - filter if RIRs arent already in subbands
            cur_srirs_filtered = octave_filtering(
                cur_srirs,
                fs,
                f_bands,
                use_amp_preserving_filterbank=use_amp_preserving_filterbank)
            logger.info("Done with octave filtering for LS estimation")
        else:
            cur_srirs_filtered = cur_srirs

        # of shape n_rirs x ir_len x n_bands

        # calculate amplitudes and noise floor - this is of shape nrirs x n_slopes+1 x n_bands
        est_amps = calculate_amplitudes_least_squares(
            t_vals_exp,
            fs,
            cur_srirs_filtered,
            f_bands,
            leave_out_ms=1000,
            downsample_factor=downsample_factor)
        a_vals[..., batch_idx] = est_amps[:, 1:, :].transpose(-1, 1, 0)
        n_vals[..., batch_idx] = np.expand_dims(est_amps[:, 0, :],
                                                axis=1).transpose(-1, 1, 0)
    return a_vals, n_vals


def main():
    """Main function to save the modified ThreeRoomDataset"""
    # This script converts the .mat file to pickle format which can be read much faster by Python
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
        # the first channel contains the W component
        srirs = srir_mat['srirs'][0, ...][:]

    # load the common slopes from the other mat files
    file_path = Path(
        "resources/Georg_3room_FDTD/Common_Slope_Analysis_Results/")
    filename = 'cs_analysis_results_omni'
    freqs = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    use_amp_preserving_filterbank = True

    common_t60 = []
    amplitudes_norm = []
    noise_floor_norm = []

    for i in range(len(freqs)):
        full_path = file_path / f'{filename}_{freqs[i]}.mat'
        with h5py.File(full_path.resolve(), 'r') as mat_file:
            data = mat_file['analysisResults']
            common_t60.append(data['commonDecayTimes'][:])
            amplitudes_norm.append(data['aVals'][:])
            noise_floor_norm.append(data['nVals'][:])

    # get custom CS amps and noise floor
    logger.info(
        "Calculating unnormalised amplitudes and noise floor with least squares"
    )
    amps_ls, noise_ls = calculate_cs_params_custom(
        srirs.copy().T,
        np.array(common_t60),
        freqs,
        sample_rate,
        batch_size=receiver_position.shape[-1],
        use_amp_preserving_filterbank=use_amp_preserving_filterbank)

    # Convert the list to a NumPy array if needed
    data_dict = {
        'fs': sample_rate,
        'srcPos': source_position,
        'rcvPos': receiver_position,
        'srirs': srirs,
        'band_centre_hz': freqs,
        'common_decay_times': np.asarray(common_t60),
        'amplitudes_norm': np.asarray(amplitudes_norm),
        'amplitudes': amps_ls,
        'noise_floor_norm': np.asarray(noise_floor_norm),
        'noise_floor': noise_ls,
    }

    # Specify output pickle path
    pickle_file_path = Path("resources/Georg_3room_FDTD/srirs.pkl").resolve()

    # Write the data to a pickle file
    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(data_dict, pickle_file)

    logger.info("Saved pickle file")

    save_subband_rirs(srirs.copy(), sample_rate, np.asarray(common_t60),
                      np.asarray(amplitudes_norm),
                      np.asarray(noise_floor_norm), amps_ls, noise_ls, freqs,
                      source_position, receiver_position,
                      use_amp_preserving_filterbank)


if __name__ == '__main__':
    main()
