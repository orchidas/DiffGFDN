from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .utils import db


class edr_loss(nn.Module):
    """Loss function that returns the difference between the EDRs of two RIRs in dB"""

    def __init__(self,
                 sample_rate: float,
                 win_size: int = 2**9,
                 hop_size: int = 2**8):
        """
        Args:
            sample_rate: sampling rate of the RIRs
            win_size (int): window size for the STFT (also the FFT size)
            hop_size (int): hop size for the STFT (must give COLA)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.win_size = win_size
        self.hop_size = hop_size

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

        S_target, _, _ = get_stft_torch(target_rir, self.sample_rate,
                                        self.win_size, self.hop_size, nfft)

        S_ach, _, _ = get_stft_torch(achieved_rir, self.sample_rate,
                                     self.win_size, self.hop_size, nfft)

        target_edr = get_edr_from_stft(S_target)
        ach_edr = get_edr_from_stft(S_ach)
        # sum over all axes to get a scalar error
        return torch.div(torch.sum(torch.abs(target_edr - ach_edr)),
                         torch.sum(torch.abs(target_edr)))


def get_stft_torch(rir: torch.tensor,
                   sample_rate: float,
                   win_size: int,
                   hop_size: int,
                   window: Optional[torch.tensor] = None,
                   nfft: int = 2**10,
                   time_axis: int = -1):
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

    freqs = torch.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    time_frames = torch.arange(0, rir.shape[time_axis] - hop_size,
                               hop_size) / sample_rate
    assert len(freqs) == S.shape[0]
    assert len(time_frames) == S.shape[-1]
    return S, freqs, time_frames


def get_edr_from_stft(S: torch.zeros()):
    """Get energy decay relief from the STFT"""
    num_freqs, num_time_frames = S.shape
    edr = torch.zeros((num_freqs, num_time_frames), dtype=torch.float32)
    for m in range(num_time_frames):
        edr[:, m] = torch.sum(torch.abs(S[:, m:])**2, axis=-1)
    edr = db(edr, is_squared=True)
    return edr
