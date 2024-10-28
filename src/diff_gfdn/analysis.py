from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import fftconvolve
from slope2noise.DecayFitNet.python.toolbox.DecayFitNetToolbox import DecayFitNetToolbox
from slope2noise.DecayFitNet.python.toolbox.core import decay_model, discard_last_n_percent, PreprocessRIR
from slope2noise.DecayFitNet.python.toolbox.utils import calc_mse
import torch

from .utils import ms_to_samps


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
