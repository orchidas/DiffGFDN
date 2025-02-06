from typing import List, Tuple

import numpy as np
import torch

from .filters.geq import design_geq
from .filters.prony import interpolate_magnitude_spectrum, prony_warped, tf2minphase
from .plot import plot_t60_filter_response
from .utils import db, db2lin


def absorption_to_gain_per_sample(room_dims: Tuple, absorption_coeff: float,
                                  delay_length_samp: List[int],
                                  fs: float) -> Tuple[float, List]:
    """
    Use Sabine's equation to get T60 from absorption coefficient, then convert that to the equivalent
    gain for a delay line, given its length in samples.
    Args:
        room_dims (Tuple): room dimensions for a room as a tuple of length, width, height
        absorption_coeff (float): uniform absorption coefficient for a room
        delay_length_samp (int): length of the delay lines in samples 
        fs (float): sampling rate
    Returns:
        Tuple: RT60s and list of gain per sample (1 for each room)
    """
    volume = np.prod(room_dims)
    if len(room_dims) == 3:
        area = 2 * (room_dims[0] * room_dims[1] + room_dims[1] * room_dims[2] +
                    room_dims[2] * room_dims[0])
    else:
        area = 2 * (room_dims[0] + room_dims[1])

    # RT60 according to sabine
    rt60 = 0.161 * volume / (area * absorption_coeff)
    gain_per_sample = db2lin(-60 * np.array(delay_length_samp) / (fs * rt60))

    return (rt60, gain_per_sample)


def decay_times_to_gain_per_sample(common_decay_times: float,
                                   delay_length_samp: List[int],
                                   fs: float) -> List:
    """Convert broadband decay times to delay line gains"""
    # list should be converted to numpy array, otherwise division wont work
    gain_per_sample = db2lin(-60 * np.array(delay_length_samp) /
                             (fs * common_decay_times))
    return gain_per_sample


def decay_times_to_gain_filters_prony(band_centre_hz: List,
                                      common_decay_times: List,
                                      delay_length_samp: List[int],
                                      fs: float,
                                      filter_order: int = 8,
                                      num_freq_bins: int = 2**10,
                                      plot_response: bool = False):
    """Fit filters to the common decay times in octave bands"""
    # the T60s for each delay line need to be attenuated
    num_delay_lines = len(delay_length_samp)
    num_coeffs = np.zeros((num_delay_lines, filter_order + 1))
    den_coeffs = np.zeros_like(num_coeffs)
    delay_line_filters = np.zeros((num_delay_lines, len(band_centre_hz)))
    interp_delay_line_filter = np.zeros(
        (num_delay_lines, num_freq_bins // 2 + 1))

    for i in range(num_delay_lines):
        delay_line_filters[i, :] = db2lin(
            (-60 * (delay_length_samp[i] + filter_order)) /
            (fs * common_decay_times))

        interp_delay_line_filter[i, :], _ = interpolate_magnitude_spectrum(
            delay_line_filters[i, :],
            band_centre_hz,
            fs,
            n_fft=num_freq_bins,
            cutoff=(20, fs // 2 - 4e3),
            rolloff_dc_db=-60,
            rolloff_nyq_db=-100,
            return_one_sided=True)

        interp_min_phase_ir = tf2minphase(interp_delay_line_filter[i, :],
                                          axis=0,
                                          is_even_fft=True,
                                          is_time_domain=True)
        num_coeffs[i, :], den_coeffs[i, :] = prony_warped(
            interp_min_phase_ir, fs, filter_order, filter_order)

    if plot_response:
        plot_t60_filter_response(band_centre_hz, delay_line_filters,
                                 num_coeffs, den_coeffs, fs,
                                 interp_delay_line_filter, num_freq_bins)

    return np.stack((num_coeffs, den_coeffs), axis=-1)


def decay_times_to_gain_filters_geq(band_centre_hz: List,
                                    common_decay_times: List,
                                    delay_length_samp: List[int],
                                    fs: float,
                                    plot_response: bool = False):
    """
    Fit filters to the common decay times in octave bands using a graphic equaliser
    Ref: ACCURATE REVERBERATION TIME CONTROL IN FEEDBACK DELAY NETWORKS by Schlecht SJ and Habets EAP
    """
    shelving_crossover_hz = [
        band_centre_hz[0] / pow(2, 1 / 2), band_centre_hz[-1] * pow(2, 1 / 2)
    ]

    # the T60s for each delay line need to be attenuated
    num_delay_lines = len(delay_length_samp)
    num_coeffs = torch.zeros((len(band_centre_hz) + 3, num_delay_lines, 3))
    den_coeffs = torch.zeros_like(num_coeffs)

    target_gains_linear = torch.tensor(
        10**(-3 / fs /
             common_decay_times)).unsqueeze(-1).clone().detach()**torch.tensor(
                 delay_length_samp, dtype=torch.int32)
    # pad target gains with 0.5x of the first/last values for the shelving filters
    target_gains_linear_pad = torch.cat(
        (target_gains_linear[0:1, :] * 0.5, target_gains_linear,
         target_gains_linear[-1:, :] * 0.5),
        dim=0)

    for i in range(len(delay_length_samp)):
        b, a = design_geq(
            db(target_gains_linear_pad[:, i]),
            center_freq=torch.tensor(band_centre_hz),
            shelving_crossover=torch.tensor(shelving_crossover_hz),
            fs=torch.tensor(fs))
        num_coeffs[:, i, :], den_coeffs[:,
                                        i, :] = b.permute(1,
                                                          0), a.permute(1, 0)

    if plot_response:
        plot_t60_filter_response(band_centre_hz, target_gains_linear.T,
                                 num_coeffs.detach().numpy(),
                                 den_coeffs.detach().numpy(), fs)

    return torch.stack((num_coeffs, den_coeffs), axis=-1)
