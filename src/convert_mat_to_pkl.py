from pathlib import Path
import pickle
from typing import List, Union

import h5py
from loguru import logger
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pyfar as pf
from scipy.signal import fftconvolve, sosfilt

# pylint: disable=W0621
# flake8: noqa:E231


def save_subband_rirs(rirs: NDArray,
                      sample_rate: float,
                      common_t60: NDArray,
                      amplitudes: NDArray,
                      noise_floor: NDArray,
                      centre_freqs: List,
                      source_position: Union[NDArray, ArrayLike],
                      receiver_position: NDArray,
                      use_amp_preserve_filterbank: bool = True):
    """Filter RIRs into subbands and save the parameters"""
    logger.info("Saving subband RIRs after filtering")
    if use_amp_preserve_filterbank:
        subband_filters, _ = pf.dsp.filter.reconstructing_fractional_octave_bands(
            None,
            num_fractions=1,
            frequency_range=(centre_freqs[0], centre_freqs[-1]),
            sampling_rate=sample_rate,
        )
    else:
        subband_filters, _ = pf.dsp.filter.fractional_octave_bands(
            None,
            num_fractions=1,
            frequency_range=(centre_freqs[0], centre_freqs[-1]),
            sampling_rate=sample_rate,
        )
    num_receivers = rirs.shape[0]

    num_bands = len(centre_freqs)
    for band in range(num_bands):
        cur_common_t60 = common_t60[band]
        cur_amplitudes = amplitudes[band, ...]
        cur_noise_floor = noise_floor[band, ...]
        if use_amp_preserve_filterbank:
            cur_filter = np.tile(subband_filters.coefficients[band, :],
                                 (num_receivers, 1))
            cur_rir = fftconvolve(rirs, cur_filter, axes=-1, mode='same')
        else:
            cur_filter = subband_filters.coefficients[band, ...]
            cur_rir = sosfilt(cur_filter, rirs, axis=-1)

        data_dict = {
            'fs': sample_rate,
            'srcPos': source_position,
            'rcvPos': receiver_position,
            'srirs': cur_rir,
            'band_centre_hz': centre_freqs[band],
            'common_decay_times': cur_common_t60,
            'amplitudes': cur_amplitudes,
            'noise_floor': cur_noise_floor,
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


# This script converts the .mat file to pickle format which can be read much faster by Python
logger.info("Reading mat file")

# Load the MATLAB v7.3 .mat file using h5py
file_path = Path("resources/Georg_3room_FDTD/srirs.mat").resolve()
print(file_path)

with h5py.File(file_path, 'r') as mat_file:
    # Get the dataset
    srir_mat = mat_file['srirDataset']
    sample_rate = np.squeeze(srir_mat['fs'][:])
    source_position = srir_mat['srcPos'][:]
    receiver_position = srir_mat['rcvPos'][:]
    # these are second order ambisonic signals
    # I am guessing the first channel contains the W component
    srirs = srir_mat['srirs'][0, ...][:]

# load the common slopes from the other mat files
file_path = Path("resources/Georg_3room_FDTD/Common_Slope_Analysis_Results/")
filename = 'cs_analysis_results_omni'
freqs = [63, 125, 250, 500, 1000, 2000, 4000, 8000]

common_t60 = []
amplitudes = []
noise_floor = []

for i in range(len(freqs)):
    full_path = file_path / f'{filename}_{freqs[i]}.mat'
    with h5py.File(full_path.resolve(), 'r') as mat_file:
        data = mat_file['analysisResults']
        common_t60.append(data['commonDecayTimes'][:])
        amplitudes.append(data['aVals'][:])
        noise_floor.append(data['nVals'][:])

# Convert the list to a NumPy array if needed
data_dict = {
    'fs': sample_rate,
    'srcPos': source_position,
    'rcvPos': receiver_position,
    'srirs': srirs.T,
    'band_centre_hz': freqs,
    'common_decay_times': np.asarray(common_t60),
    'amplitudes': np.asarray(amplitudes),
    'noise_floor': np.asarray(noise_floor)
}

# Specify the output pickle file path
pickle_file_path = Path("resources/Georg_3room_FDTD/srirs.pkl").resolve()

# Write the data to a pickle file
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(data_dict, pickle_file)

logger.info("Saved pickle file")

save_subband_rirs(srirs.T, sample_rate, np.asarray(common_t60),
                  np.asarray(amplitudes), np.asarray(noise_floor), freqs,
                  source_position, receiver_position)
