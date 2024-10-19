from typing import List, Union

import torch

from ..utils import hertz2rad
from .utils import RegularGridInterpolator

# pylint: disable=C0301


def biquad_freqz(b: torch.Tensor, a: torch.Tensor, nfft: int):
    """
    Convert a biquad filter representation to its frequency response.
    Shape of :math:`b` and :math:`a` is (3, n_sections)

    **Args**:
        - b (torch.Tensor): Coefficients of the numerator polynomial of the biquad filter.
        - a (torch.Tensor): Coefficients of the denominator polynomial of the biquad filter.
        - nfft (int): The number of points to evaluate the transfer function.

    **Returns**:
        - torch.Tensor: Frequency response of the biquad.
    """
    if len(b.shape) < 2:
        b = b.unsqueeze(-1)
    if len(a.shape) < 2:
        a = a.unsqueeze(-1)
    B = torch.fft.rfft(b, nfft, dim=0)
    A = torch.fft.rfft(a, nfft, dim=0)
    H = torch.prod(B, dim=1) / torch.prod(A, dim=1)
    return H


def sosfreqz(sos: torch.Tensor, nfft: int = 512):
    """
    Compute the complex frequency response via FFT of cascade of biquads

        **Args**:
            - sos (torch.Tensor): Second order filter sections with shape (n_sections, 6)
            - nfft (int): FFT size. Default: 512

        **Returns**:
            - H (torch.Tensor): Overall complex frequency response with shape (bs, n_bins)
    """
    _, n_coeffs = sos.size()
    assert n_coeffs == 6  # must be second order

    B = torch.fft.rfft(sos[:, :3], nfft, dim=-1)
    A = torch.fft.rfft(sos[:, 3:], nfft, dim=-1)
    H = torch.prod(B, dim=0) / (torch.prod(A, dim=0))
    return H


def lowpass_filter(fc: float = 500.0,
                   gain: float = 0.0,
                   fs: int = 48000,
                   device=None) -> tuple:
    r"""
    Lowpass filter coefficients. It uses the `RBJ cookbook formulas <https://webaudio.github.io/Audio-EQ-Cookbook/Audio-EQ-Cookbook.txt>`_ to map 
    the cutoff frequency and gain to the filter coefficients to the to the :math:`\mathbf{b}` and :math:`\mathbf{a}` biquad coefficients.
    The transfer function of the filter is given by

    .. math::
        H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}

    for

    .. math::
        b_0 = \frac{1 - \cos(\omega_c)}{2},\;\; b_1 = 1 - \cos(\omega_c),\;\; b_2 = \frac{1 - \cos(\omega_c)}{2}

    .. math::
        a_0 = 1 + \alpha,\;\; a_1 = -2 \cos(\omega_c),\;\; a_2 = 1 - \alpha

    where :math:`\omega_c = 2\pi f_c / f_s`, :math:`\alpha = \sin(\omega_c)/2 \cdot \sqrt{2}` and :math:`\cos(\omega_c)` is the cosine of the cutoff frequency.
    The gain is applied to the filter coefficients as :math:`b = 10^{g_{\textrm{dB}}/20} b`.

    **Args**:
        - fc (float): The cutoff frequency of the filter in Hz. Default: 500 Hz.
        - gain (float): The gain of the filter in dB. Default: 0 dB.
        - fs (int): The sampling frequency of the signal in Hz. Default: 48000 Hz.
        - device (torch.device, optional): The device of constructed tensors. Default: None.

    **Returns**:
        - b (ndarray): The numerator coefficients of the filter transfer function.
        - a (ndarray): The denominator coefficients of the filter transfer function.
    """
    omegaC = hertz2rad(fc, fs).to(device=device)
    two = torch.tensor(2, device=device)
    alpha = torch.sin(omegaC) / 2 * torch.sqrt(two)
    cosOC = torch.cos(omegaC)

    a = torch.ones(3, *omegaC.shape, device=device)
    b = torch.ones(3, *omegaC.shape, device=device)

    b[0] = (1 - cosOC) / 2
    b[1] = 1 - cosOC
    b[2] = (1 - cosOC) / 2
    a[0] = 1 + alpha
    a[1] = -2 * cosOC
    a[2] = 1 - alpha

    return 10**(gain / 20) * b, a


def highpass_filter(fc: float = 10000.0,
                    gain: float = 0.0,
                    fs: int = 48000,
                    device=None) -> tuple:
    r"""
    Highpass filter coefficients. It uses the `RBJ cookbook formulas <https://webaudio.github.io/Audio-EQ-Cookbook/Audio-EQ-Cookbook.txt>`_ to map 
    the cutoff frequency and gain to the filter coefficients to the to the :math:`\mathbf{b}` and :math:`\mathbf{a}` biquad coefficients.

    .. math::
        H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}

    for

    .. math::
        b_0 = \frac{1 + \cos(\omega_c)}{2},\;\; b_1 = - 1 - \cos(\omega_c),\;\; b_2 = \frac{1 + \cos(\omega_c)}{2}

    .. math::
        a_0 = 1 + \alpha,\;\; a_1 = -2 \cos(\omega_c),\;\; a_2 = 1 - \alpha

    where :math:`\omega_c = 2\pi f_c / f_s`, :math:`\alpha = \sin(\omega_c)/2 \cdot \sqrt{2}` and :math:`\cos(\omega_c)` is the cosine of the cutoff frequency.
    The gain is applied to the filter coefficients as :math:`b = 10^{g_{\textrm{dB}}/20} b`.

        **Args**:
            - fc (float, optional): The cutoff frequency of the filter in Hz. Default: 10000 Hz.
            - gain (float, optional): The gain of the filter in dB. Default: 0 dB.
            - fs (int, optional): The sampling frequency of the signal in Hz. Default: 48000 Hz.
            - device (torch.device, optional): The device of constructed tensors. Default: None.

        **Returns**:
            - b (ndarray): The numerator coefficients of the filter transfer function.
            - a (ndarray): The denominator coefficients of the filter transfer function.
    """
    omegaC = hertz2rad(fc, fs)
    two = torch.tensor(2, device=device)
    alpha = torch.sin(omegaC) / 2 * torch.sqrt(two)
    cosOC = torch.cos(omegaC)

    a = torch.ones(3, *omegaC.shape, device=device)
    b = torch.ones(3, *omegaC.shape, device=device)

    b[0] = (1 + cosOC) / 2
    b[1] = -(1 + cosOC)
    b[2] = (1 + cosOC) / 2
    a[0] = 1 + alpha
    a[1] = -2 * cosOC
    a[2] = 1 - alpha

    return 10**(gain / 20) * b, a


def bandpass_filter(fc1: torch.Tensor,
                    fc2: torch.Tensor,
                    gain: float = 0.0,
                    fs: int = 48000,
                    device=None) -> tuple:
    r"""
    Bandpass filter coefficients. It uses the `RBJ cookbook formulas <https://webaudio.github.io/Audio-EQ-Cookbook/Audio-EQ-Cookbook.txt>`_ to map 
    the cutoff frequencies and gain to the filter coefficients to the to the :math:`\mathbf{b}` and :math:`\mathbf{a}` biquad coefficients.

    .. math::
        H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}

    for

    .. math::
        b_0 = \alpha,\;\; b_1 = 0,\;\; b_2 = - \alpha

    .. math::
        a_0 = 1 + \alpha,\;\; a_1 = -2 \cos(\omega_c),\;\; a_2 = 1 - \alpha

    where 

    .. math::
        \omega_c = \frac{2\pi f_{c1} + 2\pi f_{c2}}{2 f_s}`,

    .. math::
        \text{ BW } = \log_2\left(\frac{f_{c2}}{f_{c1}}\right), 

    .. math::
        \alpha = \sin(\omega_c) \sinh\left(\frac{\log(2)}{2} \text{ BW } \frac{\omega_c}{\sin(\omega_c)}\right)

    The gain is applied to the filter coefficients as :math:`b = 10^{g_{\textrm{dB}}/20} b`.

        **Args**:
            - fc1 (float): The left cutoff frequency of the filter in Hz. 
            - fc2 (float): The right cutoff frequency of the filter in Hz. 
            - gain (float, optional): The gain of the filter in dB. Default: 0 dB.
            - fs (int, optional): The sampling frequency of the signal in Hz. Default: 48000 Hz.
            - device (torch.device, optional): The device of constructed tensors. Default: None.

        **Returns**:
            - b (ndarray): The numerator coefficients of the filter transfer function.
            - a (ndarray): The denominator coefficients of the filter transfer function.
    """
    omegaC = (hertz2rad(fc1, fs) + hertz2rad(fc2, fs)) / 2
    BW = torch.log2(fc2 / fc1)
    two = torch.tensor(2, device=device)
    alpha = torch.sin(omegaC) * torch.sinh(
        torch.log(two) / two * BW * (omegaC / torch.sin(omegaC)))

    cosOC = torch.cos(omegaC)

    a = torch.ones(3, *omegaC.shape, device=device)
    b = torch.ones(3, *omegaC.shape, device=device)

    b[0] = alpha
    b[1] = 0
    b[2] = -alpha
    a[0] = 1 + alpha
    a[1] = -2 * cosOC
    a[2] = 1 - alpha

    return 10**(gain / 20) * b, a


def shelving_filter(fc: torch.Tensor,
                    gain: torch.Tensor,
                    filt_type: str = 'low',
                    fs: int = 48000,
                    device=None):
    r"""
    Shelving filter coefficents. 
    Outputs the cutoff frequencies and gain to the filter coefficients to the to the :math:`\mathbf{b}` and :math:`\mathbf{a}` biquad coefficients.
    .. math::
        H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}

    for low shelving filter:
    .. math::
        b_0 = g^{1/2} ( g^{1/2} \tau^2 + \sqrt{2} \tau g^{1/4} + 1 ),\;\; b_1 = g^{1/2} (2 g^{1/2} \tau^2 - 2 ),\;\; b_2 = g^{1/2} ( g^{1/2} \tau^2 - \sqrt{2} \tau g^{1/4} + 1 )

        a_0 = g^{1/2} + \sqrt{2} \tau g^{1/4} + \tau^2,\;\; a_1 = 2 \tau^{2} - 2 g^{1/2} ,\;\; a_2 = g^{1/2} - \sqrt{2} \tau g^{1/4} + \tau^2

    for high shelving filter:
    .. math::
        b_0 = g ( g^{1/2} + \sqrt{2} \tau g^{1/4} + \tau^2 ),\;\; a_1 = g ( 2 \tau^{2} - 2 g^{1/2} ),\;\; a_2 = g (g^{1/2} - \sqrt{2} \tau g^{1/4} + \tau^2)

        a_0 = g^{1/2} ( g^{1/2} \tau^2 + \sqrt{2} \tau g^{1/4} + 1 ),\;\; a_1 = g^{1/2} (2 g^{1/2} \tau^2 - 2 ),\;\; a_2 = g^{1/2} ( g^{1/2} \tau^2 - \sqrt{2} \tau g^{1/4} + 1 )

    where :math:`\tau = \tan(2 \pi f_c/ (2 f_s))`, :math:`f_c`is the cutoff frequency, :math:`f_s`is the sampling frequency, and :math:`g`is the linear gain.

        **Args**:
            - fc (torch.Tensor): The cutoff frequency of the filter in Hz.
            - gain (torch.Tensor): The linear gain of the filter.
            - filt_type (str, optional): The type of shelving filter. Can be 'low' or 'high'. Default: 'low'.
            - fs (int, optional): The sampling frequency of the signal in Hz.
            - device (torch.device, optional): The device of constructed tensors. Default: None.

        **Returns**:
            - b (torch.Tensor): The numerator coefficients of the filter transfer function.
            - a (torch.Tensor): The denominator coefficients of the filter transfer function.
    """
    b = torch.ones(3, device=device)
    a = torch.ones(3, device=device)

    omegaC = hertz2rad(fc, fs)
    t = torch.tan(omegaC / 2)
    t2 = t**2
    g2 = gain**0.5
    g4 = gain**0.25

    two = torch.tensor(2, device=device)
    b[0] = g2 * t2 + torch.sqrt(two) * t * g4 + 1
    b[1] = 2 * g2 * t2 - 2
    b[2] = g2 * t2 - torch.sqrt(two) * t * g4 + 1

    a[0] = g2 + torch.sqrt(two) * t * g4 + t2
    a[1] = 2 * t2 - 2 * g2
    a[2] = g2 - torch.sqrt(two) * t * g4 + t2

    b = g2 * b

    if filt_type == 'high':
        tmp = torch.clone(b)
        b = a * gain
        a = tmp

    return b, a


def peak_filter(fc: torch.Tensor,
                gain: torch.Tensor,
                Q: torch.Tensor,
                fs: int = 48000,
                device=None) -> tuple:
    r"""
    Peak filter coefficients.
    Outputs the cutoff frequencies and gain to the filter coefficients to the to the :math:`\mathbf{b}` and :math:`\mathbf{a}` biquad coefficients.
    .. math::
        H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}

    for peak filter:
    .. math::
        b_0 = \sqrt{g} + g \tau,\;\; b_1 = -2 \sqrt{g} \cos(\omega_c),\;\; b_2 = \sqrt{g} - g \tau

        a_0 = \sqrt{g} + \tau,\;\; a_1 = -2 \sqrt{g} \cos(\omega_c),\;\; a_2 = \sqrt{g} - \tau

    where :math:`\tau = \tan(\text{BW}/2)`, :math:`BW = \omega_c / Q`, :math:`\omega_c = 2\pi f_c / f_s`, :math:`g`is the linear gain, and :math:`Q` is the quality factor.

        **Args**:
            - fc (torch.Tensor): The cutoff frequency of the filter in Hz.
            - gain (torch.Tensor): The linear gain of the filter.
            - Q (torch.Tensor): The quality factor of the filter.
            - fs (int, optional): The sampling frequency of the signal in Hz. Default: 48000.
            - device (torch.device, optional): The device of constructed tensors. Default: None.

        **Returns**:
            - b (torch.Tensor): The numerator coefficients of the filter transfer function.
            - a (torch.Tensor): The denominator coefficients of the filter transfer function
    """
    b = torch.ones(3, device=device)
    a = torch.ones(3, device=device)

    omegaC = hertz2rad(fc, fs)
    bandWidth = omegaC / Q
    t = torch.tan(bandWidth / 2)

    b[0] = torch.sqrt(gain) + gain * t
    b[1] = -2 * torch.sqrt(gain) * torch.cos(omegaC)
    b[2] = torch.sqrt(gain) - gain * t

    a[0] = torch.sqrt(gain) + t
    a[1] = -2 * torch.sqrt(gain) * torch.cos(omegaC)
    a[2] = torch.sqrt(gain) - t

    return b, a


def probe_sos(sos: torch.Tensor,
              control_freqs: Union[List, torch.Tensor],
              nfft: int,
              fs: float,
              device=None):
    r"""Probe the frequency / magnitude response of a cascaded SOS filter at the points
    specified by the control frequencies.

        **Args**:
            - sos (torch.Tensor): Cascaded second-order sections (SOS) filter coefficients.
            - control_freqs (list or torch.Tensor): Frequencies at which to probe the filter response.
            - nfft (int): Length of the FFT used for frequency analysis.
            - fs (float): Sampling frequency in Hz.

        **Returns**:
            tuple: A tuple containing the following:
                - G (torch.Tensor): Magnitude response of the filter at the control frequencies.
                - H (torch.Tensor): Frequency response of the filter.
                - W (torch.Tensor): Phase response of the filter.
    """
    n_freqs = sos.shape[-1]

    H = torch.zeros((nfft // 2 + 1, n_freqs),
                    dtype=torch.cdouble,
                    device=device)
    W = torch.zeros((nfft // 2 + 1, n_freqs), device=device)
    G = torch.zeros((len(control_freqs), n_freqs), device=device)

    for band in range(n_freqs):
        sos[:, band] = sos[:, band] / sos[3, band]

        B = torch.fft.rfft(sos[:3, band], nfft, dim=0)
        A = torch.fft.rfft(sos[3:, band], nfft, dim=0)
        h = B / (A + torch.tensor(1e-10, device=device))
        f = torch.fft.rfftfreq(nfft, 1 / fs)
        interp = RegularGridInterpolator([f], 20 * torch.log10(torch.abs(h)))
        g = interp([control_freqs])

        G[:, band] = g
        H[:, band] = h
        W[:, band] = 2 * torch.pi * f / fs

    return G, H, W
