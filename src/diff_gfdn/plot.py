from pathlib import Path
from typing import List, Optional, Tuple

from IPython import display
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import rfftfreq
from scipy.signal import freqz, sosfreqz
from slope2noise.slope2noise.utils import octave_filtering
import torch

from .losses import get_edr_from_stft, get_stft_torch
from .utils import db, ms_to_samps


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
        if num_coeffs.ndim == 2:
            freq_axis, total_response[k, :] = freqz(num_coeffs[k, :],
                                                    den_coeffs[k, :],
                                                    worN=num_freq_bins,
                                                    fs=sample_rate)
        else:
            sos_coeffs = np.concatenate(
                (num_coeffs[:, k, :], den_coeffs[:, k, :]), axis=-1)

            for i in range(sos_coeffs.shape[0]):
                sos_coeffs[i, :] /= sos_coeffs[i, 3]

            freq_axis, total_response[k, :] = sosfreqz(sos_coeffs,
                                                       worN=num_freq_bins,
                                                       fs=sample_rate)

    plt.figure()
    line0 = plt.semilogx(freqs, db(desired_filter_mag.T), marker="o")
    line1 = plt.semilogx(freq_axis, db(np.abs(total_response.T)))

    if interp_delay_line_filter is not None:
        line2 = plt.semilogx(freq_axis_one_sided,
                             db(np.abs(interp_delay_line_filter.T)),
                             linestyle='--')
        plt.legend([line0[0], line1[0], line2[0]],
                   ["Measured", "Warped prony fit", "Interpolated"])
    else:
        plt.legend([line0[0], line1[0]], ["Measured", "GEQ fit"])
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.ylim([-30, 5])
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


def plot_polynomial_matrix_magnitude_response(
    PolyMat: NDArray,
    sample_rate: float,
    num_bins: int,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
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
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_spectrogram(S: torch.tensor,
                     freqs: ArrayLike,
                     time_frames: ArrayLike,
                     title: Optional[str] = None,
                     save_path: Optional[str] = None,
                     log_freq_axis: bool = False):
    """
    Plot spectrogram from STFT data
    Args:
        S (torch.tensor): STFT data (2D) tensor, of size (N x T)
        freqs (ArrayLike): frequency bins in Hz of length N
        time_frames (ArrayLike): time indices in s of length T
        title (optional, str): title of plot
        save_path (optional, str): path used for saving file
        log_freq_axis (bool) : whether the frequency axis should be in log scale

    """
    plt.figure()
    plt.imshow(db(np.abs(S)).cpu().detach().numpy(),
               aspect='auto',
               origin='lower',
               extent=[
                   time_frames.min(),
                   time_frames.max(),
                   freqs.min(),
                   freqs.max(),
               ],
               vmin=-35,
               vmax=35)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim([20, max(freqs)])

    if log_freq_axis:
        plt.yscale('log')
    cbar = plt.colorbar()
    cbar.set_label('dB')
    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_edr(
    h: torch.tensor,
    fs: float,
    win_size: int = 2**9,
    hop_size: int = 2**8,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    log_freq_axis: bool = False,
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
        log_freq_axis (bool) : whether the frequency axis should be in log scale
    """
    S, freqs, time_frames = get_stft_torch(h,
                                           fs,
                                           win_size=win_size,
                                           hop_size=hop_size,
                                           nfft=win_size,
                                           freq_axis=0)
    edr = get_edr_from_stft(S)
    plot_spectrogram(edr,
                     freqs,
                     time_frames,
                     title,
                     save_path=save_path,
                     log_freq_axis=log_freq_axis)
    return edr


def animate_coupled_feedback_matrix(
        coupled_feedback_matrix: List[NDArray],
        coupling_matrix: Optional[List[NDArray]] = None,
        save_path: Optional[str] = None):
    """Animate a list of feedback matrices (as functions of epoch number)"""

    def init_plots():
        # Initialize the figure and the first matrix display
        if coupling_matrix is None:
            fig, ax = plt.subplots()
            mat_plot = ax.matshow(coupled_feedback_matrix[0],
                                  cmap='viridis')  # Initial display
            fig.colorbar(mat_plot, ax=ax)
            ax.set_title('Coupled feedback matrix')
            return fig, ax, mat_plot

        else:
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
            mat_plot = ax[0].matshow(coupled_feedback_matrix[0],
                                     cmap='viridis')  # Initial display
            coupled_mat_plot = ax[1].matshow(coupling_matrix[0],
                                             cmap='viridis',
                                             vmin=0,
                                             vmax=1)
            # add colorbar
            fig.colorbar(mat_plot, ax=ax[0])
            fig.colorbar(coupled_mat_plot, ax=ax[1])

            # add titles
            ax[0].set_title('Coupled feedback matrix')
            ax[1].set_title('Coupling matrix')
            return fig, ax, (mat_plot, coupled_mat_plot)

    # Update function for animation
    def update(frame: int):
        if coupling_matrix is None:
            mat_plot.set_array(
                coupled_feedback_matrix[frame])  # Update matrix data
            return [mat_plot]
        else:
            mat_plot.set_array(
                coupled_feedback_matrix[frame])  # Update the first matrix
            # pylint: disable=E0606
            coupled_mat_plot.set_array(
                coupling_matrix[frame])  # Update the second matrix
            return [mat_plot, coupled_mat_plot]

    # Create animation
    fig, _, mat_plots = init_plots()
    if not isinstance(mat_plots, tuple):
        mat_plot = mat_plots
    else:
        mat_plot, coupled_mat_plot = mat_plots
    ani = animation.FuncAnimation(fig,
                                  update,
                                  frames=len(coupled_feedback_matrix),
                                  interval=500,
                                  blit=True)

    # Show the animation
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9,
                        bottom=0.1)  # Fine-tune margins
    if save_path is not None:
        ani.save(save_path, writer="pillow", fps=2, dpi=100)
    plt.show()


def plot_subband_edc(h_true: ArrayLike,
                     h_approx: List[ArrayLike],
                     fs: float,
                     band_centre_hz: ArrayLike,
                     pos_to_investigate: List,
                     mixing_time_ms: float = 20.0,
                     crop_end_ms: float = 5.0,
                     save_path: Optional[str] = None):
    """
    Plot true and synthesised EDC curves for each frequency band, as a function of epoch number
    Args:
        h_true (ArrayLike): true (desired) RIR at a particular position
        h_approx (List[ArrayLike]): synthesized RIRs, one for each epoch
        fs (float): sampling rate
        band_centre_hz (ArrayLike): centre frequencies of the octave filters
        pos_to_investigate (List): cartesian coordinate of position where RIR was measured
        mixing_time_ms (float): truncate RIR before this time
        crop_end_ms (float): truncate last few samples of RIR
        save_path (optional, str): where to save figure
    """
    mixing_time_samp = ms_to_samps(mixing_time_ms, fs)
    crop_end_samp = ms_to_samps(crop_end_ms, fs)

    trunc_true_ir = h_true[mixing_time_samp:-crop_end_samp]
    filtered_true_ir = octave_filtering(trunc_true_ir, fs, band_centre_hz)
    time = np.linspace(0, (len(trunc_true_ir) - 1) / fs, len(trunc_true_ir))

    num_bands = len(band_centre_hz)
    time = np.linspace(0, (len(trunc_true_ir) - 1) / fs, len(trunc_true_ir))
    fig, ax = plt.subplots(num_bands, 1, figsize=(6, 12))
    fig.subplots_adjust(hspace=0.7)
    leg = []

    num_epochs = len(h_approx)
    for epoch in range(num_epochs):
        approx_ir = h_approx[epoch]
        trunc_approx_ir = approx_ir[mixing_time_samp:mixing_time_samp +
                                    len(trunc_true_ir)]
        filtered_approx_ir = octave_filtering(trunc_approx_ir, fs,
                                              band_centre_hz)
        leg.append(f'Epoch = {epoch}')

        for k in range(num_bands):
            if epoch == 0:
                true_edf = np.flipud(
                    np.cumsum(np.flipud(filtered_true_ir[:, k]**2), axis=-1))
                ax[k].plot(time,
                           db(true_edf, is_squared=True),
                           label='True EDF')

            synth_edf = np.flipud(
                np.cumsum(np.flipud(filtered_approx_ir[:, k]**2), axis=-1))
            ax[k].plot(time,
                       db(synth_edf, is_squared=True),
                       label=f'Epoch={epoch}')
            ax[k].set_title(f'{band_centre_hz[k]: .0f} Hz')
            ax[k].set_ylim([-100, 0])

        display.display(fig)  # Display the updated figure
        display.clear_output(
            wait=True)  # Clear the previous output to keep updates in place

    # Collect handles and labels from all axes
    handles, labels = [], []
    for handle, label in zip(*ax[-1].get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)
    fig.legend(handles,
               labels,
               loc="upper right",
               ncol=1,
               frameon=False,
               bbox_to_anchor=(1.2, 0.5))
    fig.suptitle(
        'Truncated EDF at position ' +
        f'{pos_to_investigate[0]: .2f}, {pos_to_investigate[1]: .2f}, {pos_to_investigate[2]: .2f} m'
    )
    if save_path is not None:
        fig.savefig(Path(save_path).resolve())
    plt.show()
