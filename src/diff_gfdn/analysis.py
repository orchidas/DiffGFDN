from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import fftconvolve
from slope2noise.DecayFitNet.python.toolbox.DecayFitNetToolbox import DecayFitNetToolbox
from slope2noise.DecayFitNet.python.toolbox.core import decay_model, discard_last_n_percent, PreprocessRIR
from slope2noise.DecayFitNet.python.toolbox.utils import calc_mse
from slope2noise.slope2noise.utils import octave_filtering
import torch

from .filters.geq import octave_bands
from .utils import db2lin, ms_to_samps


def get_edc_params(rir: ArrayLike,
                   n_slopes: int,
                   fs: float,
                   use_octave_freqs: bool = True):
    """
    Get decay time, amplitudes, noise level and fitted EDC
    Args:
        RIR (ArrayLike): single dimension RIR of length rir_len
        n_slopes (int): number of slopes in RIR
        fs (float): sampling frequency
        use_octave_freqs (bool): if true, then the RIR will be filtered in octave bands
    Returns:
        NDArray, NDArray, NDArray, ArrayLike, NDArray: the T60s, amplitudes and noise level for 
        n_slopes and n_subbands. The normalisation values and the fitted EDC per subband 
    """
    filter_frequencies = octave_bands() if use_octave_freqs else None
    est_params_decay_net, norm_vals, fitted_edc_subband = get_decay_fit_net_params(
        rir, filter_frequencies, n_slopes, fs)
    est_t60 = est_params_decay_net[0]
    est_amp = est_params_decay_net[1]
    est_noise_level = est_params_decay_net[2]
    return est_t60, est_amp, est_noise_level, norm_vals, fitted_edc_subband


def get_decay_fit_net_params(
        rir: NDArray,
        filter_frequencies: List,
        n_slopes: int = 1,
        sr: float = 48000,
        device='cpu',
        verbose: bool = False) -> Tuple[List, ArrayLike, ArrayLike]:
    """
    Extract EDC parameters - amplitudes, T60s, noise floor using DecayFitNet
    The RIR onset time is detected 
    Args:
        rir (array):    input room impulse response
        n_slopes (int): number of EDC slopes. 0 = number of active slopes is 
                        determined by network (between 1 and 3)
        sr (int):       sampling rate
        filter_frequencies (list):  frequency bands
        verbose (bool): whether to print out the MSE per frequency band
    Returns:
        Tuple[List, ArrayLike, ArrayLike]: the estimated EDC parameters, norm values to match initial level 
                                  (since EDC is normalised to 0 dB), and the fitted edc

    """
    # Init Preprocessing
    rir_preprocessing = PreprocessRIR(sample_rate=sr,
                                      filter_frequencies=filter_frequencies)

    # Schroeder integration, analyse_full_rir: if RIR onset should be detected, set this to False
    true_edc, _ = rir_preprocessing.schroeder(rir, analyse_full_rir=True)
    time_axis = (torch.linspace(0, true_edc.shape[2] - 1, true_edc.shape[2]) /
                 sr)

    # Permute into [n_bands, n_batches, n_samples]
    true_edc = true_edc.permute(1, 0, 2)

    # Prepare the model
    decayfitnet = DecayFitNetToolbox(n_slopes=n_slopes,
                                     sample_rate=sr,
                                     filter_frequencies=filter_frequencies)
    estimated_parameters_decayfitnet, norm_vals_decayfitnet = decayfitnet.estimate_parameters(
        rir, analyse_full_rir=True)
    # Get fitted EDC from estimated parameters
    fitted_edc_decayfitnet = decay_model(
        torch.from_numpy(estimated_parameters_decayfitnet[0]).to(device),
        torch.from_numpy(estimated_parameters_decayfitnet[1]).to(device),
        torch.from_numpy(estimated_parameters_decayfitnet[2]).to(device),
        time_axis=time_axis.to(device),
        compensate_uli=True,
        backend='torch',
        device=device)
    # Discard last 5% for MSE evaluation
    true_edc = discard_last_n_percent(true_edc, 5)
    fitted_edc_decayfitnet = discard_last_n_percent(fitted_edc_decayfitnet, 5)

    # Calculate MSE between true EDC and fitted EDC
    if verbose:
        calc_mse(true_edc, fitted_edc_decayfitnet)

    return estimated_parameters_decayfitnet, norm_vals_decayfitnet.numpy(
    ), fitted_edc_decayfitnet


def amplitudes_to_initial_level(
        decay_times: NDArray,
        amplitudes: NDArray,
        fs: float,
        ir_len: int,
        max_freq: float = 16e3,
        uses_decay_fit_net: bool = False,
        norm_vals: Optional[ArrayLike] = None) -> NDArray:
    """
    Convert decatFitNet estimation to initial level as used in FDN tone correction filter.
    Adapted from FDNToolbox by SJS.
    Args:
        decay_times (NDArray): decay times of size n_bands x n_slopes
        amplitudes (NDArray): amplitudes of size n_bands x n_slopes
        fs (float): sampling frequency
        ir_len (int): length of the RIR in samples
        max_freq (float): maximum frequency when doing octave filtering
        uses_decay_fit_net: where the decay times and amplitudes calculated using DecayFitNet?
        norm_vals (ArrayLike): normalisation values, only needs to be specified if DecayFitNet was used
    Retunrs:
        NDArray: normalised initial level of size n_bands x n_slopes
    """
    if norm_vals is None and not uses_decay_fit_net:
        norm_vals = np.ones_like(amplitudes)

    n_slopes = amplitudes.shape[-1]
    amplitudes = amplitudes * norm_vals

    # estimate energy of the octave filters
    impulse = np.zeros((ir_len))
    impulse[0] = 1
    f_bands = octave_bands(end_freq=max_freq)

    ir_octave_filter = octave_filtering(impulse, fs, f_bands, get_filter=True)
    # the input inpulse will not be used in this case actually, get_filter argument is just a quick fix
    band_energy = np.sum(ir_octave_filter**2, axis=0)
    band_energy = np.tile(band_energy[:, np.newaxis], (1, n_slopes))

    # Cumulative energy is a geometric series of the gain per sample
    slope = -60.0 / (decay_times * fs)
    gain_per_sample = db2lin(slope)
    decay_energy = 1 / (1 - gain_per_sample**2)

    # initial level
    norm_factor = ir_len if uses_decay_fit_net else 1.0
    level = np.sqrt(amplitudes / band_energy / decay_energy * norm_factor)
    # there is an offset because, the FDN is not energy normalized
    # the ir_len factor is due to the normalization in schroederInt (in DecayFitNet)

    return level


def calculate_energy_envelope(sig: ArrayLike, fs: float,
                              smooth_time_ms: float) -> ArrayLike:
    """
    Calculate the energy envelope (broadband EDC) of a RIR
    Args:
        sig (ArrayLike): 1D RIR signal
        fs (float): sampling rate
        smooth_time_ms (float): smoothing window length in ms, 
                                longer window leads to more smoothing
    """
    staps = ms_to_samps(smooth_time_ms / 2, fs)
    odd_win_len = 2 * staps - 1
    # normalised smoothing window
    bs = np.hanning(odd_win_len) / np.sum(np.hanning(odd_win_len))
    # zero-pad signal on either side
    padded_signal = np.concatenate((np.zeros(staps), sig**2, np.zeros(staps)),
                                   axis=0)
    # smooth signal by convolving with window
    smoothed_signal = fftconvolve(bs, padded_signal)
    env = np.real(np.sqrt(smoothed_signal))
    env = env[odd_win_len + np.arange(len(sig))]
    return env
