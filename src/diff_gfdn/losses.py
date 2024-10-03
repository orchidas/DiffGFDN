from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .filters import calc_erb_filters
from .gain_filters import SOSFilter
from .utils import db


class reg_loss(nn.Module):
    """
    Penalises the rate of decay of the output filters (pole radius) to reduce time aliasing 
    caused by frequency sampling. See Lee et al, Differentiable artificial reverberation, 
    IEEE TASLP 2022
    """

    def __init__(self, num_time_samps: int, num_delay_lines: int,
                 num_biquads: int):
        """
        Args:
            num_time_samps (int): length of the IR of each output filter
            num_delay_lines (int): number of delay lines in the GFDN
            num_biquads (int): number of biquads in each output filter
        """
        super().__init__()
        self.num_delay_lines = num_delay_lines
        self.num_biquads = num_biquads
        # length of impulse response
        self.num_time_samps = num_time_samps
        self.N0 = int(np.round(num_time_samps / 8))
        self.sos_filter = SOSFilter(self.num_biquads)
        # create an impulse
        self.input_signal = torch.zeros(num_time_samps)
        self.input_signal[0] = 1.0

    def forward(self, output_biquad_cascade: List):
        """
        Apply softmax to the rate of decrease of the filter
        Args:
            output_biquad_cascade (List): B x Ndel biquad cascade filters
        """
        with torch.autograd.set_detect_anomaly(True):
            batch_size = len(output_biquad_cascade)
            gamma_list = []

            for b in range(batch_size):
                for n in range(self.num_delay_lines):
                    cur_biquad_cascade = output_biquad_cascade[b][n]

                    cur_output_signal = self.sos_filter.filter(
                        self.input_signal, cur_biquad_cascade)

                    # ratio of the late energy to the early energy
                    # if gamma is large, then IR is decaying slowly
                    # if gamma is small, then IR is decaying fast
                    gamma_list.append(
                        torch.div(
                            torch.sum(
                                torch.abs(
                                    cur_output_signal[self.num_time_samps -
                                                      self.N0:])),
                            torch.sum(torch.abs(cur_output_signal[:self.N0]))))

            # penalise long decay times more (reduce pole radii)
            # sum along delay lines
            gamma = torch.stack(gamma_list).view(batch_size,
                                                 self.num_delay_lines)
            loss = torch.div(torch.sum(gamma * torch.exp(gamma), 1),
                             torch.sum(torch.exp(gamma), 1))
            # sum along batch size
            return torch.sum(loss)


class edr_loss(nn.Module):
    """Loss function that returns the difference between the EDRs of two RIRs in dB"""

    def __init__(self,
                 sample_rate: float,
                 win_size: int = 2**9,
                 hop_size: int = 2**8,
                 use_erb_grouping: bool = False):
        """
        Args:
            sample_rate: sampling rate of the RIRs
            win_size (int): window size for the STFT (also the FFT size)
            hop_size (int): hop size for the STFT (must give COLA)
            erb_grouping (bool): whether to group in ERB bands
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.win_size = win_size
        self.hop_size = hop_size
        self.use_erb_grouping = use_erb_grouping
        if self.use_erb_grouping:
            self.erb_filters = calc_erb_filters(sample_rate,
                                                nfft=win_size,
                                                num_bands=50)

    def forward(self, target_response: torch.tensor,
                achieved_response: torch.tensor) -> torch.tensor:
        """
        Compute the EDR loss between the target response and the response of the RIR simulated by the GFDN
        Args:
            target_response : B x num_freq_pts frequency response of the target RIR
            achieved_response: B x num_freq_pts achieved response of the simulated RIR
        Returns:
            tensor: B x win_size x num_frames error between the 2 EDRs, summed over all axes to return a scalar
        """
        assert target_response.shape == achieved_response.shape
        nfft = self.win_size
        target_rir = torch.fft.irfft(target_response,
                                     target_response.shape[-1])
        achieved_rir = torch.fft.irfft(achieved_response,
                                       achieved_response.shape[-1])

        S_target, _, _ = get_stft_torch(target_rir,
                                        self.sample_rate,
                                        self.win_size,
                                        self.hop_size,
                                        nfft,
                                        erb_filters=self.erb_filters)

        S_ach, _, _ = get_stft_torch(achieved_rir,
                                     self.sample_rate,
                                     self.win_size,
                                     self.hop_size,
                                     nfft,
                                     erb_filters=self.erb_filters)

        target_edr = get_edr_from_stft(S_target)
        ach_edr = get_edr_from_stft(S_ach)
        # sum over all axes to get a scalar error
        return torch.div(torch.sum(torch.abs(target_edr - ach_edr)),
                         torch.sum(torch.abs(target_edr)))


def get_stft_torch(rir: torch.tensor,
                   sample_rate: float,
                   win_size: int,
                   hop_size: int,
                   nfft: int = 2**10,
                   window: Optional[torch.tensor] = None,
                   freq_axis: int = 1,
                   time_axis: int = -1,
                   erb_filters: Optional[torch.tensor] = None):
    """Get STFT from a time domain signal"""
    time_samps = rir.shape[time_axis]
    if time_samps % hop_size != 0:
        # zero pad the input signal
        num_extra_zeros = hop_size * int(np.ceil(
            time_samps / hop_size)) - time_samps
        # pad zeros to the right of the last dimension (time)
        rir = F.pad(input=rir,
                    pad=(0, num_extra_zeros),
                    mode='constant',
                    value=0)

    if window is None:
        window = torch.hann_window(win_size)
        assert hop_size == win_size // 2

    # complex tensor of shape B (batch size), N(num frequency samples), T(num_frames):
    S = torch.stft(rir,
                   nfft,
                   hop_length=hop_size,
                   win_length=win_size,
                   window=window,
                   center=False,
                   normalized=False,
                   onesided=True,
                   return_complex=True)

    time_frames = torch.arange(0, rir.shape[time_axis] - hop_size,
                               hop_size) / sample_rate
    freqs = torch.fft.rfftfreq(nfft, d=1.0 / sample_rate)

    assert len(freqs) == S.shape[freq_axis]
    assert len(time_frames) == S.shape[time_axis]

    if erb_filters is not None:
        if S.ndim == 2:
            S = torch.matmul(erb_filters, torch.abs(S))
        else:
            S = torch.einsum('nk, bkt -> bnt', erb_filters, torch.abs(S))

    return S, freqs, time_frames


def get_edr_from_stft(S: torch.tensor):
    """
    Get energy decay relief from the STFT
    Args:
        S(torch.tensor): complex STFT of size B x F x T
    Returns:
        real EDR of size B x F x T
    """
    if S.ndim == 3:
        batch_size, num_freqs, num_time_frames = S.shape
        edr = torch.zeros((batch_size, num_freqs, num_time_frames),
                          dtype=torch.float32)
    else:
        num_freqs, num_time_frames = S.shape
        edr = torch.zeros((num_freqs, num_time_frames), dtype=torch.float32)

    for m in range(num_time_frames):
        edr[..., m] = torch.sum(torch.abs(S[..., m:])**2, axis=-1)
    edr = db(edr, is_squared=True)
    return edr
