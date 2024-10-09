from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy.fft import rfftfreq
from scipy.signal import freqz

from .losses import get_edr_from_stft, get_stft_torch
from .utils import db


def plot_t60_filter_response(
        freqs: List,
        desired_filter_mag: NDArray,
        num_coeffs: NDArray,
        den_coeffs: NDArray,
        sample_rate: float,
        interp_delay_line_filter: Optional[NDArray] = None,
        num_freq_bins: int = 2**12):
    """Plot the fitted T60 filters to see how well they match to the specified spectrum"""
    num_delay_lines = desired_filter_mag.shape[0]
    total_response = np.zeros((num_delay_lines, num_freq_bins), dtype=complex)
    freq_axis_one_sided = rfftfreq(num_freq_bins, d=1.0 / sample_rate)

    for k in range(num_delay_lines):
        freq_axis, total_response[k, :] = freqz(num_coeffs[k, :],
                                                den_coeffs[k, :],
                                                worN=num_freq_bins,
                                                fs=sample_rate)

    plt.figure()
    line0 = plt.semilogx(freqs, db(desired_filter_mag.T), marker="o")
    line1 = plt.semilogx(freq_axis_one_sided,
                         db(np.abs(interp_delay_line_filter.T)),
                         linestyle='--')
    line2 = plt.semilogx(freq_axis, db(np.abs(total_response.T)))
    plt.legend([line0[0], line1[0], line2[0]],
               ["Measured", "Interpolated", "Warped prony fit"])
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.tight_layout()


def plot_polynomial_matrix_impulse_response(
    PolyMat: NDArray,
    fs: float,
    plot_db: bool = False,
    title: Optional[str] = None,
    rt60: Optional[float] = None,
):
    """Plot the impulse response of a polynomial matrix. T
    Args:
        PolyMat: the matrix to plot of size NxNxir_len
        fs: sampling frequency
        title (Optional): title of the plot
        plot_db (bool): whether to plot the amplitude in dB
    """
    N, _, ir_len = PolyMat.shape
    time_axis = np.arange(0, ir_len / fs, 1.0 / fs)
    fig, ax = plt.subplots(N, N, figsize=[8, 5])
    for i in range(N):
        for j in range(N):
            if plot_db:
                ax[i, j].plot(time_axis, db(PolyMat[i, j, :]))
            else:
                ax[i, j].plot(time_axis, PolyMat[i, j, :])
            if rt60 is not None:
                ax[i, j].set_xlim([0, rt60 + 0.1])

    ax[1, 0].set_xlabel('Time (seconds)')
    if plot_db:
        ax[0, 0].set_ylabel('Amplitude (dB)')
    else:
        ax[0, 0].set_ylabel('Amplitude')
    if title is not None:
        fig.suptitle(title)
    plt.show()


def get_polynomial_matrix_response(A: NDArray, K: int) -> NDArray:
    """
    Given A 3d polynomial matrix of size NxNXp, 
    returns the magnitude response for K bins in the frequency domain
    Args:
        A (NxNxp): 3D polynomial matrix
        K (int) : number of bins in the frequency domain for response evaluation
    Returns:
        NxNxK : complex matrix with frequency response
        Kx1 : frequency bins over which response is evaluated
    """
    k = np.arange(0, K)
    # z^{-1} over semi-unit circle
    z = np.exp(-np.pi * 1j * k / K)

    N = A.shape[0]
    order = A.shape[-1]
    Y = np.zeros((N, N, K), dtype=complex)

    for i in range(order):
        Arep = np.repeat(A[..., i][..., np.newaxis], K, axis=-1)
        Y += Arep * np.power(z, i)

    return Y, k / K


def plot_polynomial_matrix_magnitude_response(PolyMat: NDArray,
                                              sample_rate: float,
                                              num_bins: int,
                                              title: Optional[str] = None):
    """
    Plot the frequency response of the given polynomial matrix, and the frequency
    axis over which it is computed.
    Args:
        PolyMat (NDArray): 3D polynomial matrix in the time domain with the last axis denoting polynomial order
        sample_rate (float): sampling frequency of the polynomial matrix
        num_bins (int) : number of bins in the frequency domain for response evaluation
        title (optional, str): title of the figure
    """
    N = PolyMat.shape[0]
    PolyMat_response, freq_axis_rad = get_polynomial_matrix_response(
        PolyMat, num_bins)
    freq_axis_hz = (sample_rate / 2.0) * freq_axis_rad
    fig, ax = plt.subplots(N, N, figsize=[8, 5])
    for i in range(N):
        for j in range(N):
            ax[i, j].semilogx(freq_axis_hz,
                              db(np.abs(PolyMat_response[i, j, :])))

    ax[-1, 0].set_xlabel('Frequency (Hz)')
    ax[0, 0].set_ylabel('Magnitude (dB)')
    if title is not None:
        fig.suptitle(title)
    plt.show()


def plot_spectrogram(S: torch.tensor,
                     freqs: ArrayLike,
                     time_frames: ArrayLike,
                     title: Optional[str] = None,
                     save_path: Optional[str] = None):
    """
    Plot spectrogram from STFT data
    Args:
        S (torch.tensor): STFT data (2D) tensor, of size (N x T)
        freqs (ArrayLike): frequency bins in Hz of length N
        time_frames (ArrayLike): time indices in s of length T
        title (optional, str): title of plot
        save_path (optional, str): path used for saving file
    """
    plt.figure()
    plt.imshow(db(np.abs(S)).cpu().detach().numpy(),
               aspect='auto',
               origin='lower',
               extent=[
                   time_frames.min(),
                   time_frames.max(),
                   freqs.min(),
                   freqs.max()
               ])
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    cbar = plt.colorbar()
    cbar.set_label('dB')
    if title is not None:
        plt.title(title)
    plt.savefig(save_path)
    plt.show()


def plot_edr(
    h: torch.tensor,
    fs: float,
    win_size: int = 2**9,
    hop_size: int = 2**8,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Tuple[torch.tensor, ArrayLike, ArrayLike]:
    """
    Plot EDR of the RIR h
    Args:
        h (torch.tensor): time domain RIR
        fs (float): sampling frequency
        win_size (int): length of window used for STFT
        hop_size (int): hop size used for STFT
        title (optional, str): title of plot
        save_path (optional, str): path where to save plot
    """
    S, freqs, time_frames = get_stft_torch(h,
                                           fs,
                                           win_size=win_size,
                                           hop_size=hop_size,
                                           nfft=win_size,
                                           freq_axis=0)
    edr = get_edr_from_stft(S)
    plot_spectrogram(edr, freqs, time_frames, title, save_path=save_path)
    return edr
