from typing import List, Optional, Tuple

import librosa
from loguru import logger
import numpy as np
import pyfar as pf
from scipy.fft import rfftfreq
from slope2noise.utils import decay_kernel
import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.functional import lfilter

from .gain_filters import BiquadCascade, SOSFilter
from .utils import db, ms_to_samps


def calc_erb_filters(
    sample_rate: float,
    nfft: int,
    num_bands: int,
    freq_lims_hz: Tuple = (63, 16e3)
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Calculate ERB filterbanks in the STFT domain
    Args:
        sample_rate (float): sampling frequency
        nfft (int): number of points in 2 sided FFT
        num_bands (int): number of ERB bands
        freq_lims_hz (tuple): frequency limits for mel filter calculation
    Returns:
        Tuple[tensor, tensor]: The mel filterbank and the corresponding mel frequencies in Hz
    """
    erb_filters = librosa.filters.mel(sr=sample_rate,
                                      n_fft=nfft,
                                      n_mels=num_bands,
                                      fmin=freq_lims_hz[0],
                                      fmax=freq_lims_hz[1])
    erb_freqs = torch.tensor(librosa.mel_frequencies(n_mels=num_bands,
                                                     fmin=freq_lims_hz[0],
                                                     fmax=freq_lims_hz[1]),
                             dtype=torch.float64)

    # Convert erb_filters to torch tensor
    erb_filters = torch.tensor(erb_filters, dtype=torch.float64)
    return erb_filters, erb_freqs


def scaled_shifted_sigmoid_inverse(x: torch.tensor, scale_factor: float,
                                   cutoff: float, top: float,
                                   bottom: float) -> torch.tensor:
    """
    Inverse of a  scaled sigmoid function that lies between bottom and top, 
    and switches from top to bottom at cutoff.
    """
    return bottom + torch.div((top - bottom),
                              (1 + torch.exp(scale_factor * (x - cutoff))))


###################################################################################


class reg_loss(nn.Module):
    """
    Penalises the rate of decay of the output filters (pole radius) to reduce time aliasing 
    caused by frequency sampling. See Lee et al, Differentiable artificial reverberation, 
    IEEE TASLP 2022
    """

    def __init__(self, num_time_samps: int, num_groups: int, num_biquads: int):
        """
        Args:
            num_time_samps (int): length of the IR of each output filter
            num_groups (int): number of groups in the GFDN
            num_biquads (int): number of biquads in each output filter
        """
        super().__init__()
        self.num_groups = num_groups
        self.num_biquads = num_biquads
        # length of impulse response
        self.num_time_samps = num_time_samps
        self.N0 = int(np.round(num_time_samps / 8))
        self.sos_filter = SOSFilter(self.num_biquads)
        # create an impulse
        self.input_signal = torch.zeros(num_time_samps)
        self.input_signal[0] = 1.0

    @staticmethod
    def is_list_of_lists(obj: List) -> bool:
        """Check if object is a list of lists"""
        if isinstance(obj, list):  # Check if it's a list
            return all(isinstance(i, list)
                       for i in obj)  # Check if all elements are lists
        return False

    def calculate_gamma(self, cur_biquad_cascade: BiquadCascade):
        """Calculate the ratio of late energy to early energy"""
        cur_output_signal = self.sos_filter.filter(self.input_signal,
                                                   cur_biquad_cascade)
        # ratio of the late energy to the early energy
        # if gamma is large, then IR is decaying slowly
        # if gamma is small, then IR is decaying fast
        return torch.div(
            torch.sum(
                torch.abs(cur_output_signal[self.num_time_samps - self.N0:])),
            torch.sum(torch.abs(cur_output_signal[:self.N0])))

    def forward(self, output_biquad_cascade: List):
        """
        Apply softmax to the rate of decrease of the filter
        Args:
            output_biquad_cascade (List): B x Ndel biquad cascade filters
        """
        with torch.autograd.set_detect_anomaly(True):

            gamma_list = []
            has_batch = self.is_list_of_lists(output_biquad_cascade)
            if has_batch:
                batch_size = len(output_biquad_cascade)

                for b in range(batch_size):
                    for n in range(self.num_groups):
                        cur_biquad_cascade = output_biquad_cascade[b][n]
                        gamma_list.append(
                            self.calculate_gamma(cur_biquad_cascade))

                # penalise long decay times more (reduce pole radii)
                # sum along delay lines
                gamma = torch.stack(gamma_list).view(batch_size,
                                                     self.num_groups)
                loss = torch.div(torch.sum(gamma * torch.exp(gamma), 1),
                                 torch.sum(torch.exp(gamma), 1))
                # sum along batch size
                return torch.sum(loss)
            else:
                for n in range(self.num_groups):
                    cur_biquad_cascade = output_biquad_cascade[n]

                    gamma_list.append(self.calculate_gamma(cur_biquad_cascade))

                # penalise long decay times more (reduce pole radii)
                # sum along delay lines
                gamma = torch.stack(gamma_list).view(self.num_groups)
                loss = torch.div(torch.sum(gamma * torch.exp(gamma)),
                                 torch.sum(torch.exp(gamma)))
                return loss


class edc_loss(nn.Module):
    """Broadband EDC loss in linear scale (to put more focus on the beginning of the RIR)"""

    def __init__(
        self,
        max_ir_len_ms: float,
        sample_rate: float,
        band_centre_hz: Optional[List] = None,
        mixing_time_ms: float = 20.0,
        use_mask: bool = False,
    ):
        """
        Args:
            max_ir_len_ms (float): maximum RIR length to take into account to ignore noise floor
                                   in EDC calculation
            sample_rate (float): sampling frequency in Hz
            band_centre_hz (list, optional): centre frequencies of filters (if calculating subband EDC)
            mixing_time_ms (float): start the EDC calculation after the mixing time
            use_mask (bool): randomly mask some of the indices to introduce stochasticity
        """
        super().__init__()
        self.max_ir_len_samps = ms_to_samps(max_ir_len_ms, sample_rate)
        self.band_centre_hz = band_centre_hz
        self.mixing_time_samps = ms_to_samps(mixing_time_ms, sample_rate)
        if band_centre_hz is not None:
            subband_filters = pf.dsp.filter.fractional_octave_bands(
                None,
                num_fractions=1,
                frequency_range=(63, 16000),
                sampling_rate=sample_rate,
            )
            self.filter_coeffs_sos = subband_filters.coefficients.transpose(
                1, -1, 0)
            self.filter_order = self.filter_coeffs_sos.shape[0]
        self.use_mask = use_mask
        if self.use_mask:
            logger.info("Using masked EDC loss")

    def schroeder_backward_integral(self,
                                    signal: torch.tensor,
                                    normalize: bool = False):
        """Schroeder backward integral to calculate energy decay curve"""
        edc = torch.flip(torch.cumsum(torch.flip(signal**2, dims=[-1]),
                                      dim=-1),
                         dims=[-1])
        if normalize:
            # Normalize to 1
            norm_vals, _ = torch.max(edc, dim=-1, keepdims=True)  # per channel
            edc = torch.div(edc, norm_vals)

        return edc

    def forward(self, target_response: torch.tensor,
                achieved_response: torch.tensor) -> torch.tensor:
        """Get the error between the EDCs of the target and achieved response"""
        max_ir_len_samps = min(self.max_ir_len_samps,
                               target_response.shape[-1])

        target_rir = torch.fft.irfft(
            target_response,
            target_response.shape[-1])[...,
                                       self.mixing_time_samps:max_ir_len_samps]
        achieved_rir = torch.fft.irfft(
            achieved_response, achieved_response.shape[-1])[
                ..., self.mixing_time_samps:max_ir_len_samps]

        if self.band_centre_hz is None:
            # broadband EDC loss
            target_edc = self.schroeder_backward_integral(target_rir)
            achieved_edc = self.schroeder_backward_integral(achieved_rir)

            # randomly mask some of the indices
            if self.use_mask:
                probs = torch.empty(target_rir.shape[-1]).uniform_(0, 1)
                masked_index = torch.argwhere(torch.bernoulli(probs))
            else:
                masked_index = torch.arange(0,
                                            target_rir.shape[-1],
                                            dtype=torch.int32)

            # according to Mezza et al
            # loss = torch.div(
            #     torch.sum(torch.pow(target_edc - achieved_edc, 2)),
            #     torch.sum(torch.pow(target_edc, 2)))

            # according to Gotz
            loss = torch.mean(
                torch.abs(
                    db(target_edc[..., masked_index], is_squared=True) -
                    db(achieved_edc[..., masked_index], is_squared=True)))
        else:
            # EDC loss in subbands
            loss = 0.0
            use_batches = True if target_rir.dim() > 1 else True

            for b_idx in range(len(self.band_centre_hz)):
                cur_filter_sos = torch.from_numpy(
                    self.filter_coeffs_sos[...,
                                           b_idx].copy()).to(torch.float32)

                target_rir_band = target_rir.clone()
                achieved_rir_band = achieved_rir.clone()

                for j in range(self.filter_order):
                    target_rir_band = lfilter(target_rir_band.to(
                        torch.float32),
                                              cur_filter_sos[j, :3],
                                              cur_filter_sos[j, 3:],
                                              batching=use_batches)
                    achieved_rir_band = lfilter(achieved_rir_band.to(
                        torch.float32),
                                                cur_filter_sos[j, :3],
                                                cur_filter_sos[j, 3:],
                                                batching=use_batches)

                target_edc_band = self.schroeder_backward_integral(
                    target_rir_band)
                achieved_edc_band = self.schroeder_backward_integral(
                    achieved_rir_band)

                if self.use_mask:
                    probs = torch.empty(target_rir_band.shape[-1]).uniform_(
                        0, 1)
                    masked_index = torch.argwhere(torch.bernoulli(probs))
                else:
                    masked_index = torch.arange(0,
                                                target_rir_band.shape[-1],
                                                dtype=torch.int32)

                loss += torch.mean(
                    torch.abs(target_edc_band - achieved_edc_band))

        return loss


class directional_edc_loss(nn.Module):
    """Mean EDC loss between true and learned directional mappings, calculated directly from the RIRs"""

    def __init__(self,
                 common_decay_times: List,
                 edc_len_ms: float,
                 fs: float,
                 mixing_time_ms: float = 20.0,
                 use_mask: bool = False):
        """
        Initialise EDC loss
        Args:
            common_decay_times (List): of size num_slopes x 1 to form the decay kernel
            edc_len_ms (float): length of the EDC in ms
            fs (float): sampling rate in Hz
            mixing_time_ms (float): mixing time in ms
            use_mask (bool): whether to randomly mask some time indices
        """
        super().__init__()
        self.mixing_time_samps = ms_to_samps(mixing_time_ms, fs)
        self.use_mask = use_mask
        self.edc_len_samps = ms_to_samps(edc_len_ms, fs)
        num_slopes = common_decay_times.shape[-1]
        self.envelopes = torch.zeros((num_slopes, self.edc_len_samps))
        time_axis = np.linspace(0, (self.edc_len_samps - 1) / fs,
                                self.edc_len_samps)

        for k in range(num_slopes):
            self.envelopes[k, :] = torch.tensor(
                decay_kernel(np.expand_dims(common_decay_times[:, k], axis=-1),
                             time_axis,
                             fs,
                             normalize_envelope=True,
                             add_noise=False)).squeeze()

    def schroeder_backward_integral(self,
                                    signal: torch.tensor,
                                    normalize: bool = False):
        """Schroeder backward integral to calculate energy decay curve"""
        edc = torch.flip(torch.cumsum(torch.flip(signal**2, dims=[-1]),
                                      dim=-1),
                         dims=[-1])
        if normalize:
            # Normalize to 1
            norm_vals, _ = torch.max(edc, dim=-1, keepdims=True)  # per channel
            edc = torch.div(edc, norm_vals)

        return edc

    def forward(self, H_pred: torch.Tensor, amps_true: torch.Tensor):
        """
        Calculate the mean EDC loss over space and time.
        The decay kernel is of shape num_slopes x time
        Args:
            H_pred (torch.Tensor): predicted transfer function of shape 
                                   batch_size x num_directions x num_freq_pts
            amps_true (torch.Tensor): true amplitudes of shape batch_size x num_directions x num_slopes
        """
        # Directional RIRs
        # desired shape is batch_size x num_directions x num_time_samples
        pred_rir = torch.fft.irfft(H_pred)[
            ...,
            self.mixing_time_samps:self.edc_len_samps + self.mixing_time_samps]

        # predicted EDC from DiffDFDN response
        edc_pred = self.schroeder_backward_integral(pred_rir)

        # true EDC from common slope amplitudes
        # sum along num slopes
        edc_true = torch.einsum('bjk, kt -> bjt', amps_true.to(torch.float32),
                                self.envelopes)

        # randomly mask some of the indices
        if self.use_mask:
            probs = torch.empty(pred_rir.shape[-1]).uniform_(0, 1)
            masked_index = torch.argwhere(torch.bernoulli(probs))
        else:
            masked_index = torch.arange(0,
                                        pred_rir.shape[-1],
                                        dtype=torch.int32)

        # according to Gotz
        loss = torch.mean(
            torch.abs(
                db(edc_true[..., masked_index], is_squared=True) -
                db(edc_pred[..., masked_index], is_squared=True)))

        return loss


#########################################################################


class edr_loss(nn.Module):
    """Loss function that returns the difference between the EDRs of two RIRs in dB"""

    def __init__(
        self,
        sample_rate: float,
        win_size: int = 2**12,
        hop_size: int = 2**11,
        reduced_pole_radius: Optional[float] = None,
        use_erb_grouping: bool = False,
        time_axis: int = -1,
        freq_axis: int = -2,
        use_weight_fn: bool = False,
    ):
        """
        Args:
            sample_rate: sampling rate of the RIRs
            win_size (int): window size for the STFT (also the FFT size)
            hop_size (int): hop size for the STFT (must give COLA)
            reduced_pole_radius (optional, float): if the sampling was done on a radius larger than the unit circle,
                                                   we need to apply an exponential envelope to the resulting IR
            use_erb_grouping (bool): whether to group EDR in ERB bands before calculating loss
            use_weight_fn (bool): whether to use frequency-dependent weighting function, 
                                  which weights the lower frequency loss more
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.win_size = win_size
        self.hop_size = hop_size
        self.use_erb_grouping = use_erb_grouping
        self.reduced_pole_radius = reduced_pole_radius
        self.time_axis = time_axis
        self.freq_axis = freq_axis
        self.use_weight_fn = use_weight_fn
        if self.use_erb_grouping:
            self.erb_filters, self.freqs_hz = calc_erb_filters(sample_rate,
                                                               nfft=win_size,
                                                               num_bands=2**6)
        else:
            self.erb_filters = None
            self.freqs_hz = rfftfreq(self.win_size, d=1.0 / self.sample_rate)

        if self.use_weight_fn:
            # apply frequency-dependent weighting with more weight on the lower frequency loss below 1kHz
            logger.info("Using frequency-dependent weighting on EDR loss")
            cutoff_freq_hz = 1e3
            scale_factor = 10**(-2.5)
            top = 2.0
            bottom = 1.0
            self.frequency_weights = scaled_shifted_sigmoid_inverse(
                torch.tensor(self.freqs_hz), scale_factor, cutoff_freq_hz,
                bottom, top)

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

        if self.reduced_pole_radius is not None:
            # undo sampling on a larger circle in the
            # z domain by multiplying with an increasing exponent in the time domain
            achieved_rir *= torch.pow(1.0 / self.reduced_pole_radius,
                                      torch.arange(0, achieved_rir.shape[-1]))

        S_target, _, _ = get_stft_torch(target_rir,
                                        self.sample_rate,
                                        self.win_size,
                                        self.hop_size,
                                        nfft,
                                        freq_axis=self.freq_axis,
                                        time_axis=self.time_axis,
                                        erb_filters=self.erb_filters)

        S_ach, _, _ = get_stft_torch(achieved_rir,
                                     self.sample_rate,
                                     self.win_size,
                                     self.hop_size,
                                     nfft,
                                     freq_axis=self.freq_axis,
                                     time_axis=self.time_axis,
                                     erb_filters=self.erb_filters)

        target_edr = get_edr_from_stft(S_target)
        ach_edr = get_edr_from_stft(S_ach)

        # according to Mezza et al, DAFx-24

        # frequency-based loss, of size (B, N (num_freq_samples))
        # sum over time axis
        freq_loss = torch.abs(target_edr - ach_edr)

        freq_loss = torch.sum(freq_loss, dim=self.time_axis)
        if self.use_weight_fn:
            freq_loss *= self.frequency_weights
        # logger.info(f'Frequencies: {self.freqs_hz}, \nLoss value: {freq_loss}')

        # sum over frequency and time axes and normalise to get a scalar error
        if target_edr.ndim == 3:
            loss_per_item = torch.div(
                torch.sum(freq_loss, dim=-1),
                torch.sum(torch.abs(target_edr),
                          dim=[self.time_axis, self.freq_axis]))
            # additionally sum over all items in batch
            return torch.sum(loss_per_item)
        else:
            return torch.div(torch.sum(freq_loss),
                             torch.sum(torch.abs(target_edr)))


##################################################################################


def get_stft_torch(rir: torch.tensor,
                   sample_rate: float,
                   win_size: int,
                   hop_size: int,
                   nfft: int = 2**10,
                   window: Optional[torch.tensor] = None,
                   freq_axis: int = -2,
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

    # apply ERB filters to the STFT
    if erb_filters is not None:
        if S.ndim == 2:
            S = torch.einsum('nk, kt -> nt',
                             erb_filters.to(torch.abs(S).dtype), torch.abs(S))
        else:
            S = torch.einsum('nk, bkt -> bnt',
                             erb_filters.to(torch.abs(S).dtype), torch.abs(S))

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
