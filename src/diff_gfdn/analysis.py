from typing import List, Optional, Tuple

from slope2noise.DecayFitNet.python.toolbox.DecayFitNetToolbox import DecayFitNetToolbox
from slope2noise.DecayFitNet.python.toolbox.core import decay_model, discard_last_n_percent, PreprocessRIR
from slope2noise.DecayFitNet.python.toolbox.utils import calc_mse
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from slope2noise.slope2noise.utils import calculate_amplitudes_least_squares, octave_filtering
import torch

from .filters.geq import octave_bands
from .utils import db, db2lin, ms_to_samps


def get_edc_params(rir: ArrayLike,
                   n_slopes: int,
                   fs: float,
                   f_bands: Optional[List] = None):
    """
    Get decay time, amplitudes, noise level and fitted EDC
    Args:
        RIR (ArrayLike): single dimension RIR of length rir_len
        n_slopes (int): number of slopes in RIR
        fs (float): sampling frequency
        f_bands (List): band centre frequencies for subband processing
    Returns:
        NDArray, NDArray, NDArray, ArrayLike, NDArray: the T60s, amplitudes and noise level for 
        n_slopes and n_subbands. The normalisation values and the fitted EDC per subband 
    """
    filter_frequencies = f_bands
    est_params_decay_net, norm_vals, fitted_edc_decayfitnet = get_decay_fit_net_params(
        rir, filter_frequencies, n_slopes, fs)
    est_t60 = est_params_decay_net[0]
    est_amp = est_params_decay_net[1]
    est_noise_level = est_params_decay_net[2]
    fitted_edc_subband = torch.mean(fitted_edc_decayfitnet, dim=1)
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


def get_decay_times_for_rirs(
        rir: ArrayLike,
        approx_rir: ArrayLike,
        n_slopes: int,
        fs: float,
        band_centre_hz: ArrayLike,
        plot_edc: bool = True,
        mixing_time_ms: float = 20.0,
        crop_end_ms: float = 5.0) -> Tuple[NDArray, NDArray]:
    """
    Compare and plot the T60s for two RIRs.
    Args:
        rir (ArrayLike): true (desired) RIR
        approx_rir (ArrayLike): synthesized RIR
        n_slopes (int): number of slopes in RIR
        fs (float): sampling rate
        band_centre_hz (ArrayLike): octave band frequencies in Hz
        plot_edc (bool): whether to plot the original and fitted EDC
        mixing_time_ms (float): truncate RIRs upto this time
        crop_end_ms (float): ignore the last few ms of RIRs
    Returns:
        Tuple[NDArray, NDArray]: the T60s of shape n_slopes x n_bands 
                                 for the original RIR and the approximated RIR
    """
    # crop the RIR upto the mixing time
    mixing_time_samp = ms_to_samps(mixing_time_ms, fs)
    crop_end_samp = ms_to_samps(crop_end_ms, fs)
    trunc_rir = rir[mixing_time_samp:-crop_end_samp]
    trunc_approx_rir = approx_rir[mixing_time_samp:-crop_end_samp]
    # get subband T60s
    true_t60, _, _, _, fitted_edc = get_edc_params(trunc_rir, n_slopes, fs,
                                                   band_centre_hz)
    est_t60, _, _, _, fitted_approx_edc = get_edc_params(
        trunc_approx_rir, n_slopes, fs, band_centre_hz)

    if plot_edc:
        # filter into subbands for EDF
        filtered_rir = octave_filtering(trunc_rir, fs, band_centre_hz)
        filtered_approx_rir = octave_filtering(trunc_approx_rir, fs,
                                               band_centre_hz)
        num_bands = len(band_centre_hz)
        fig, ax = plt.subplots(num_bands, 1, figsize=(6, 12))
        time = np.linspace(0, (len(trunc_rir) - 1) / fs, len(trunc_rir))

        for k in range(num_bands):

            true_edf = np.flipud(
                np.cumsum(np.flipud(filtered_rir[:, k]**2), axis=-1))
            approx_edf = np.flipud(
                np.cumsum(np.flipud(filtered_approx_rir[:, k]**2), axis=-1))

            ax[k].plot(time, db(true_edf, is_squared=True), label='True EDF')
            ax[k].plot(time[:fitted_edc.shape[-1]],
                       db(fitted_edc[k, :], is_squared=True),
                       label='DecayFitNet, True')
            ax[k].plot(time[:len(approx_edf)],
                       db(approx_edf, is_squared=True),
                       label='Synth EDF')
            ax[k].plot(time[:fitted_approx_edc.shape[-1]],
                       db(fitted_approx_edc[k, :], is_squared=True),
                       label='DecayFitNet, Approx')

            ax[k].set_title(f'{band_centre_hz[k]: .0f} Hz')

        fig.subplots_adjust(hspace=0.7)
        ax[0].legend(loc="upper right")

    return true_t60, est_t60


def get_amps_for_rir(rir: NDArray,
                     common_decay_times: NDArray,
                     band_centre_hz: ArrayLike,
                     fs: float,
                     mixing_time_ms: float = 20.0,
                     leave_out_ms: float = 10.0) -> NDArray:
    """
    Get subband amplitude estimates using least squares method
    Args:
        rir (NDArray): RIR of size n_rirs x ir_len
        common_decay_times (ArrayLike): common decay times for the decay kernels, of size n_rir x n_slopes x n_bands
        band_centre_hz (ArrayLike): filter frequencies
        fs (float): sampling rate
        mixing_time_ms (float): truncate RIR before this time
        leave_out_ms (float): leave out last few samples in calculating the energy envelope
    Returns:
        NDArray: subband amplitudes of shape n_bands x 1 x n_slopes
    """
    # crop the RIR upto the mixing time
    mixing_time_samp = ms_to_samps(mixing_time_ms, fs)
    trunc_rir = rir[mixing_time_samp:]
    if len(trunc_rir) % 2 != 0:
        trunc_rir = rir[mixing_time_samp + 1:]
    # filter and get the amplitudes for each band
    filtered_rir = octave_filtering(trunc_rir, fs, band_centre_hz)

    # estimated amplitudes with least squares
    estimated_amps = calculate_amplitudes_least_squares(
        common_decay_times,
        fs,
        filtered_rir,
        band_centre_hz,
        leave_out_ms=leave_out_ms)
    # n_bands should be the first axis
    estimated_amps = np.moveaxis(estimated_amps, -1, 0)
    return estimated_amps


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

    ir_octave_filter = octave_filtering(impulse,
                                        fs,
                                        f_bands,
                                        get_filter_ir=True)
    # the input impulse will not be used in this case actually, get_filter argument is just a quick fix
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
