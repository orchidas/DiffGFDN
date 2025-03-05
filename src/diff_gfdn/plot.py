from pathlib import Path
from typing import List, Optional, Tuple, Union

from IPython import display
from loguru import logger
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import rfftfreq
from scipy.signal import freqz, sos2zpk, sosfreqz
from scipy.spatial.distance import cdist
from slope2noise.rooms import RoomGeometry
from slope2noise.utils import calculate_amplitudes_least_squares, octave_filtering, schroeder_backward_int
import torch
from torch import nn
from tqdm import tqdm

from .analysis import get_amps_for_rir
from .config.config import DiffGFDNConfig
from .dataloader import RoomDataset
from .filters.geq import eq_freqs
from .losses import get_edr_from_stft, get_stft_torch
from .utils import db, db2lin, ms_to_samps, spectral_flatness

# flake8: noqa:E231


def plot_t60_filter_response(
        freqs: List,
        desired_filter_mag: NDArray,
        num_coeffs: NDArray,
        den_coeffs: NDArray,
        sample_rate: float,
        interp_delay_line_filter: Optional[NDArray] = None,
        num_freq_bins: int = 2**12,
        save_path: Optional[str] = None):
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

    fig = plt.figure()
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
    plt.ylim([-15, 0])
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(
            Path(f'{save_path}_absorption_filter_response.png').resolve())


def plot_magnitude_response(
    room_data: RoomDataset,
    config_dict: DiffGFDNConfig,
    model: nn.Module,
    save_path: Optional[str] = None,
):
    """Plot the magnitude response of each FDN"""

    trainer_config = config_dict.trainer_config
    freq_bins_rad = torch.tensor(room_data.freq_bins_rad)
    freq_bins_hz = room_data.freq_bins_hz
    z_values = torch.polar(torch.ones_like(freq_bins_rad),
                           freq_bins_rad * 2 * np.pi)

    max_epochs = trainer_config.max_epochs
    checkpoint_dir = Path(trainer_config.train_dir + 'checkpoints/').resolve()
    init_checkpoint = torch.load(f'{checkpoint_dir}/model_e-1.pt',
                                 weights_only=True,
                                 map_location=torch.device('cpu'))
    model.load_state_dict(init_checkpoint, strict=False)
    model.eval()
    H_sub_fdn_init, _ = model.sub_fdn_output(z_values)

    final_checkpoint = torch.load(f'{checkpoint_dir}/model_e{max_epochs-1}.pt',
                                  weights_only=True,
                                  map_location=torch.device('cpu'))
    # Load the trained model state
    model.load_state_dict(final_checkpoint, strict=False)
    model.eval()
    H_sub_fdn_final, _ = model.sub_fdn_output(z_values)

    # Create subplots
    fig, axes = plt.subplots(room_data.num_rooms,
                             1,
                             figsize=(8, 10),
                             sharex=True)

    for i in range(room_data.num_rooms):
        axes[i].semilogx(freq_bins_hz,
                         db(H_sub_fdn_init[:, i].detach().numpy()),
                         label="Initial",
                         linestyle="--")
        axes[i].semilogx(
            freq_bins_hz,
            db(H_sub_fdn_final[:, i].detach().numpy()),
            label="Final",
            linestyle="-",
            alpha=0.8,
        )

        axes[i].set_ylabel("Magnitude (dB)")
        axes[i].set_xlabel('Frequencies (Hz)')
        axes[i].set_title(f"FDN {i+1}")
        axes[i].grid(True)
        axes[i].legend()
        logger.info(
            f'Init FDN spectral flatness is {spectral_flatness(db(H_sub_fdn_init[:, i].detach().numpy())):.3f}'
        )
        logger.info(
            f'Final FDN spectral flatness is {spectral_flatness(db(H_sub_fdn_final[:, i].detach().numpy())):.3f}'
        )

    if save_path is not None:
        fig.savefig(save_path)

    plt.show()
    return axes


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
        S (torch.tensor): STFT data (2D) tensor in dB, of size (N x T)
        freqs (ArrayLike): frequency bins in Hz of length N
        time_frames (ArrayLike): time indices in s of length T
        title (optional, str): title of plot
        save_path (optional, str): path used for saving file
        log_freq_axis (bool) : whether the frequency axis should be in log scale

    """
    plt.figure()
    plt.imshow(
        S.cpu().detach().numpy(),
        aspect='auto',
        origin='lower',
        extent=[
            time_frames.min(),
            time_frames.max(),
            freqs.min(),
            freqs.max(),
        ],
        vmin=-60,
        vmax=10,
    )
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
    filtered_true_ir = octave_filtering(trunc_true_ir,
                                        fs,
                                        band_centre_hz,
                                        compensate_filter_energy=True,
                                        use_pyfar_filterbank=True)
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
        filtered_approx_ir = octave_filtering(trunc_approx_ir,
                                              fs,
                                              band_centre_hz,
                                              compensate_filter_energy=True,
                                              use_pyfar_filterbank=True)
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
            ax[k].set_ylim([-80, 0])
            ax[k].set_xlim([0, 2.0])

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
        fig.savefig(Path(save_path).resolve(), bbox_inches="tight")
    plt.show()


def plot_subband_amplitudes(h_true: Union[ArrayLike, torch.Tensor],
                            h_approx: Union[ArrayLike, torch.Tensor],
                            sample_rate: float,
                            num_groups: int,
                            amplitudes: NDArray,
                            common_decay_times: NDArray,
                            band_centre_hz: ArrayLike,
                            mixing_time_ms: float = 20.0,
                            crop_end_ms: float = 5.0,
                            save_path: Optional[str] = None):
    """
    Plot the true and estimated amplitudes in subbands using least squares and common decay times
    Args:
        h_true (ArrayLike): true (desired) RIR
        h_approx (ArrayLike): approximate synthesized RIR
        sample_rate (float): sampling rate
        num_groups (int): number of groups in GFDN
        amplitudes (NDArray): theoretical amplitudes at position, of size n_bands x n_groups
        common_decay_times (NDArray): common decay times, of size n_bands x n_groups
        band_centre_hz (ArrayLike): octave band frequencies
        save_path (str, optional): path to save file
    """
    # get the actual RIR levels
    expanded_amplitudes = np.moveaxis(amplitudes, -1, 1)

    # get the estimated levels of the original RIR
    og_estimated_amps = get_amps_for_rir(h_true,
                                         common_decay_times.T,
                                         band_centre_hz,
                                         sample_rate,
                                         mixing_time_ms=mixing_time_ms,
                                         leave_out_ms=crop_end_ms)

    # get the estimated levels of the approx RIR
    estimated_amps = get_amps_for_rir(h_approx, common_decay_times.T,
                                      band_centre_hz, sample_rate)

    fig, ax = plt.subplots(num_groups, figsize=(6, 3 * num_groups))
    for n in range(num_groups):
        cur_ax = ax if num_groups == 1 else ax[n]
        cur_ax.semilogx(band_centre_hz,
                        db(np.squeeze(expanded_amplitudes[..., n]),
                           is_squared=True),
                        marker='o',
                        label='Actual amplitudes (theoretical)')
        cur_ax.semilogx(band_centre_hz,
                        db(np.squeeze(og_estimated_amps[..., n]),
                           is_squared=True),
                        marker='d',
                        label='Actual amplitudes, est. with LS')
        cur_ax.semilogx(band_centre_hz,
                        db(np.squeeze(estimated_amps[..., n]),
                           is_squared=True),
                        marker='x',
                        label='Estimated amplitudes')

        cur_ax.set_xlabel('Frequency Hz')
        cur_ax.set_ylabel('Magnitude dB')
        cur_ax.set_ylim([-80, 10])
        cur_ax.set_title(f'Group {n + 1}')
        cur_ax.grid()
    cur_ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.9))
    fig.subplots_adjust(hspace=0.5)
    if save_path is not None:
        fig.savefig(Path(save_path).resolve())


def order_position_matrices(pos1: NDArray, pos2: NDArray) -> ArrayLike:
    """Arrange 3D coordinates in pos1 and pos2 so that they are in the same order"""
    # Step 1: Compute pairwise distances
    distances = cdist(pos1, pos2)

    # Step 2: Find the closest matches for each point in array1
    matching_indices = np.argmin(distances, axis=1)

    return matching_indices


def find_correct_axis(desired_dim: int, array_shape: Tuple):
    """Find the axis having the desired dimension from array_shape"""
    # Identify the receiver axis based on matching dimension
    desired_axis = None
    for i, dim in enumerate(array_shape):
        if dim == desired_dim:
            desired_axis = i
            break

    return desired_axis


def plot_edc_error_in_space(
    room_data: RoomDataset,
    all_rirs: Union[NDArray, List],
    all_pos: Union[NDArray, List],
    freq_to_plot: Optional[float] = 1000.0,
    scatter: bool = False,
    save_path: Optional[str] = None,
    pos_sorted: bool = False,
    norm_edc: bool = False,
):
    """
    Plot the EDC matching error in dB as a function of spatial location
    Args:
        room_data (RoomDataset): object containing information of room geometry, decay times and amplitudes
        all_rirs (List): list of RIRs at all positions synthesized by the GFDN
        all_pos (List): list of positions at which the RIRs are synthesized
        freq_to_plot (optional, float): which frequency to plot the amplitudes at
        scatter (bool): whether to make a scatter plot (discrete), or a surface plot (continuous)
        save_path (optional, str): path to save the file
        mixing_time_ms (float): truncate RIR before this time
        pos_sorted (bool): whether the positions are sorted in all_{src,rec}_pos
        norm_edc (bool): whether to normalise the EDC before calculating the error
    """

    def get_edc_error(original_rirs: NDArray,
                      original_points: NDArray,
                      estimated_rirs: NDArray,
                      est_points: NDArray,
                      norm_flag: bool = False):
        """Get MSE error between the EDC mismatch"""

        if not pos_sorted:
            ordered_pos_idx = order_position_matrices(original_points,
                                                      est_points)
        else:
            ordered_pos_idx = np.arange(0, len(est_points), dtype=np.int32)
        est_rirs_ordered = estimated_rirs[ordered_pos_idx, ...]

        original_edc = schroeder_backward_int(original_rirs,
                                              time_axis=-2,
                                              normalize=norm_flag,
                                              discard_last_zeros=False)
        est_edc = schroeder_backward_int(est_rirs_ordered,
                                         time_axis=-2,
                                         normalize=norm_flag,
                                         discard_last_zeros=False)
        error_db = np.mean(np.abs(
            db(original_edc, is_squared=True) - db(est_edc, is_squared=True)),
                           axis=-2)
        error_mse = np.linalg.norm(error_db, axis=0) / original_points.shape[0]
        return error_db, error_mse

    num_rooms = room_data.num_rooms
    room_dims = room_data.room_dims
    start_coordinates = room_data.room_start_coord
    aperture_coordinates = room_data.aperture_coords
    t_vals = room_data.common_decay_times.T
    room = RoomGeometry(room_data.sample_rate,
                        num_rooms,
                        np.array(room_dims),
                        np.array(start_coordinates),
                        aperture_coords=aperture_coordinates)
    rec_points = np.array(room_data.receiver_position)
    src_pos = np.array(room_data.source_position)
    if src_pos.ndim == 1:
        src_pos = src_pos[np.newaxis, :]
    is_in_subbands = t_vals.shape[-1] > 1
    original_rirs = room_data.rirs

    for src_idx in tqdm(range(len(src_pos))):
        cur_src_pos = np.squeeze(src_pos[src_idx])
        cur_est_rirs = np.squeeze(
            all_rirs[[src_idx],
                     ...]) if len(src_pos) > 1 else np.asarray(all_rirs)

        cur_original_rirs = np.squeeze(
            original_rirs[src_idx, ...]) if len(src_pos) > 1 else original_rirs

        rir_len_samps = min(cur_original_rirs.shape[-1],
                            cur_est_rirs.shape[-1])

        cur_est_rirs = cur_est_rirs[..., :rir_len_samps]
        cur_original_rirs = cur_original_rirs[..., :rir_len_samps]

        # do subband filtering
        if is_in_subbands and freq_to_plot is not None:
            cur_original_rirs_filtered = octave_filtering(
                cur_original_rirs,
                room_data.sample_rate,
                room_data.band_centre_hz,
                compensate_filter_energy=True,
                use_pyfar_filterbank=True)
            cur_est_rirs_filtered = octave_filtering(
                cur_est_rirs,
                room_data.sample_rate,
                room_data.band_centre_hz,
                compensate_filter_energy=True,
                use_pyfar_filterbank=True)
            save_name = f'{save_path}_{freq_to_plot / 1000: .0f}kHz\
            _src=({cur_src_pos[0]:.2f}, {cur_src_pos[1]:.2f}, {cur_src_pos[2]:.2f})'

        else:
            cur_original_rirs_filtered = cur_original_rirs[..., np.newaxis]
            cur_est_rirs_filtered = cur_est_rirs[..., np.newaxis]

            save_name = f'{save_path}_src=({cur_src_pos[0]:.2f}, {cur_src_pos[1]:.2f}, {cur_src_pos[2]:.2f})'

        est_rec_pos = np.asarray(all_pos)
        # get error metrics
        error_func, error_mse = get_edc_error(
            cur_original_rirs_filtered.copy(), rec_points,
            cur_est_rirs_filtered.copy(), est_rec_pos, norm_edc)
        if is_in_subbands and freq_to_plot is not None:
            idx = np.argwhere(
                np.array(room_data.band_centre_hz) == freq_to_plot)[0][0]
            for k in range(len(room_data.band_centre_hz)):
                logger.info(
                    f'The RMSE in matching EDC at frequency {room_data.band_centre_hz[k]: .0f}Hz'
                    f' is {error_mse[k]: .3f} dB')
                var_to_plot = db2lin(error_func[..., idx])
        else:
            logger.info(f'The RMSE in matching EDC is {error_mse} dB')
            var_to_plot = db2lin(error_func)

        # plot the error in amplitude matching
        room.plot_edc_error_at_receiver_points(
            rec_points,
            cur_src_pos,
            var_to_plot,
            scatter_plot=scatter,
            cur_freq_hz=freq_to_plot,
            save_path=Path(f'{save_name}_edc_error_in_space.png').resolve()
            if save_path is not None else None)


def plot_edr_error_in_space(
    room_data: RoomDataset,
    all_rirs: Union[NDArray, List],
    all_pos: Union[NDArray, List],
    scatter: bool = False,
    save_path: Optional[str] = None,
    pos_sorted: bool = False,
):
    """
    Plot the EDR matching error in dB as a function of spatial location
    Args:
        room_data (RoomDataset): object containing information of room geometry, decay times and amplitudes
        all_rirs (List): list of RIRs at all positions synthesized by the GFDN
        all_pos (List): list of positions at which the RIRs are synthesized
        freq_to_plot (optional, float): which frequency to plot the amplitudes at
        scatter (bool): whether to make a scatter plot (discrete), or a surface plot (continuous)
        save_path (optional, str): path to save the file
        mixing_time_ms (float): truncate RIR before this time
        pos_sorted (bool): whether the positions are sorted in all_{src,rec}_pos
    """

    def get_edr_error(
        original_rirs: NDArray,
        original_points: NDArray,
        estimated_rirs: NDArray,
        est_points: NDArray,
        sample_rate: float,
        win_size: int = 2**12,
        hop_size: int = 2**11,
    ):
        """Get MSE error between the EDC mismatch"""

        if not pos_sorted:
            ordered_pos_idx = order_position_matrices(original_points,
                                                      est_points)
        else:
            ordered_pos_idx = np.arange(0, len(est_points), dtype=np.int32)
        est_rirs_ordered = estimated_rirs[ordered_pos_idx, ...]

        S_orig, _, _ = get_stft_torch(
            torch.tensor(original_rirs),
            sample_rate,
            win_size=win_size,
            hop_size=hop_size,
            nfft=win_size,
        )
        original_edr = get_edr_from_stft(S_orig).cpu().detach().numpy()

        S_est, _, _ = get_stft_torch(
            torch.tensor(est_rirs_ordered),
            sample_rate,
            win_size=win_size,
            hop_size=hop_size,
            nfft=win_size,
        )
        est_edr = get_edr_from_stft(S_est).cpu().detach().numpy()

        # take mean error along time and frequency axies - first axis contains location
        error_db = np.abs(original_edr - est_edr).mean(axis=(1, 2))
        error_mse = np.linalg.norm(error_db) / original_points.shape[0]
        return error_db, error_mse

    num_rooms = room_data.num_rooms
    room_dims = room_data.room_dims
    start_coordinates = room_data.room_start_coord
    aperture_coordinates = room_data.aperture_coords
    room = RoomGeometry(room_data.sample_rate,
                        num_rooms,
                        np.array(room_dims),
                        np.array(start_coordinates),
                        aperture_coords=aperture_coordinates)
    rec_points = np.array(room_data.receiver_position)
    src_pos = np.array(room_data.source_position)
    if src_pos.ndim == 1:
        src_pos = src_pos[np.newaxis, :]
    original_rirs = room_data.rirs

    for src_idx in tqdm(range(len(src_pos))):
        cur_src_pos = np.squeeze(src_pos[src_idx])
        cur_est_rirs = np.squeeze(
            all_rirs[[src_idx],
                     ...]) if len(src_pos) > 1 else np.asarray(all_rirs)

        cur_original_rirs = np.squeeze(
            original_rirs[src_idx, ...]) if len(src_pos) > 1 else original_rirs

        rir_len_samps = min(cur_original_rirs.shape[-1],
                            cur_est_rirs.shape[-1])

        cur_est_rirs = cur_est_rirs[..., :rir_len_samps]
        cur_original_rirs = cur_original_rirs[..., :rir_len_samps]

        est_rec_pos = np.asarray(all_pos)
        # get error metrics
        error_func, error_mse = get_edr_error(cur_original_rirs.copy(),
                                              rec_points, cur_est_rirs.copy(),
                                              est_rec_pos,
                                              room_data.sample_rate)

        logger.info(f'The RMSE in matching EDR is {error_mse} dB')
        var_to_plot = db2lin(error_func)
        save_name = f'{save_path}_src=({cur_src_pos[0]:.2f}, {cur_src_pos[1]:.2f}, {cur_src_pos[2]:.2f})'

        # plot the error in amplitude matching
        room.plot_edc_error_at_receiver_points(
            rec_points,
            cur_src_pos,
            var_to_plot[..., np.newaxis],
            scatter_plot=scatter,
            cur_freq_hz=None,
            save_path=Path(f'{save_name}_edr_error_in_space.png').resolve()
            if save_path is not None else None)


def plot_amps_in_space(room_data: RoomDataset,
                       all_rirs: Union[NDArray, List],
                       all_rec_pos: Union[NDArray, List],
                       freq_to_plot: Optional[float] = 1000.0,
                       scatter: bool = False,
                       save_path: Optional[str] = None,
                       pos_sorted: bool = False,
                       plot_original_amps: bool = True,
                       plot_amp_error: bool = True):
    """
    Plot the amplitudes as a function of spatial location at frequency 'freq_to_plot' Hz
    Args:
        room_data (RoomDataset): object containing information of room geometry, decay times and amplitudes
        all_rirs (List): list of RIRs at all positions synthesized by the GFDN
        all_rec_pos (List): list of receiver positions at which the RIRs are synthesized
        freq_to_plot (optional, float): which frequency to plot the amplitudes at
        scatter (bool): whether to make a scatter plot (discrete), or a surface plot (continuous)
        save_path (optional, str): path to save the file
        pos_sorted (bool): whether the positions are sorted in all_{src,rec}_pos
        plot_original_amps (bool): whether to plot the true amplitudes as a function of space
        plot_amp_error (bool): whether to plot the amplitude matching error
    """

    def get_amplitude_error(original_amps: NDArray, original_points: NDArray,
                            estimated_amps: NDArray, est_points: NDArray):
        """Get MSE error between the amplitude mismatch"""
        if not pos_sorted:
            ordered_pos_idx = order_position_matrices(original_points,
                                                      est_points)
        else:
            ordered_pos_idx = np.arange(0, len(est_points), dtype=np.int32)

        est_amps_ordered = estimated_amps[ordered_pos_idx, ...]
        error_db = np.abs(
            db(original_amps, is_squared=True) -
            db(est_amps_ordered, is_squared=True))
        error_mse = np.linalg.norm(error_db, axis=(0, 1)) / np.sqrt(
            original_points.shape[0])
        return error_db, error_mse

    num_rooms = room_data.num_rooms
    room_dims = room_data.room_dims
    start_coordinates = room_data.room_start_coord
    aperture_coordinates = room_data.aperture_coords
    original_amps = room_data.amplitudes.copy()
    t_vals = room_data.common_decay_times.T
    room = RoomGeometry(room_data.sample_rate,
                        num_rooms,
                        np.array(room_dims),
                        np.array(start_coordinates),
                        aperture_coords=aperture_coordinates)
    rec_points = np.array(room_data.receiver_position)
    src_pos = np.array(room_data.source_position)
    if src_pos.ndim == 1:
        src_pos = src_pos[np.newaxis, :]
    is_in_subbands = t_vals.shape[-1] > 1
    est_amps = np.zeros_like(original_amps)

    for src_idx in tqdm(range(len(src_pos))):
        cur_src_pos = np.squeeze(src_pos[src_idx])

        est_rirs = np.squeeze(
            all_rirs[src_idx,
                     ...]) if len(src_pos) > 1 else np.asarray(all_rirs)

        num_est_rirs = est_rirs.shape[0]
        num_og_rirs = rec_points.shape[0]
        cur_original_amps = original_amps[src_idx, ...] if len(
            src_pos) > 1 else original_amps

        # do subband filtering
        if is_in_subbands:
            est_rirs_filtered = octave_filtering(est_rirs,
                                                 room_data.sample_rate,
                                                 room_data.band_centre_hz,
                                                 compensate_filter_energy=True,
                                                 use_pyfar_filterbank=True)
            t_vals_expanded = np.tile(
                np.squeeze(t_vals)[np.newaxis, ...], (num_est_rirs, 1, 1))
            band_centre_hz = room_data.band_centre_hz
            save_name = f'{save_path}_{freq_to_plot / 1000: .0f}kHz_'+ \
            f'src=({cur_src_pos[0]:.2f}, {cur_src_pos[1]:.2f}, {cur_src_pos[2]:.2f})'

            num_bands = len(room_data.band_centre_hz)

        else:
            est_rirs_filtered = est_rirs[..., np.newaxis]
            t_vals_expanded = np.tile(np.squeeze(t_vals),
                                      (num_est_rirs, 1))[..., np.newaxis]
            band_centre_hz = None
            save_name = f'{save_path}_src=({cur_src_pos[0]:.2f}, {cur_src_pos[1]:.2f}, {cur_src_pos[2]:.2f})'

        est_rec_pos = np.asarray(all_rec_pos)
        # these are of shape num_rec x num_slope x num_fbands

        cur_est_amps = calculate_amplitudes_least_squares(
            t_vals_expanded,
            room_data.sample_rate,
            est_rirs_filtered,
            band_centre_hz,
            use_non_linear_ls=True)

        # ignore the noise term
        cur_est_amps = cur_est_amps[:, 1:, :]

        # if amplitudes are specified in subbands
        if is_in_subbands:

            # amplitudes should be of shape num_og_rirs x num_slope x num_fbands
            if cur_original_amps.shape != (num_og_rirs, num_rooms, num_bands):
                original_shape = cur_original_amps.shape
                rec_axis = find_correct_axis(num_og_rirs, original_shape)
                slope_axis = find_correct_axis(num_rooms, original_shape)
                freq_axis = find_correct_axis(num_bands, original_shape)

                cur_original_amps = np.transpose(
                    cur_original_amps, (rec_axis, slope_axis, freq_axis))

            idx = np.argwhere(
                np.array(room_data.band_centre_hz) == freq_to_plot)[0][0]
            amps_mid_band = cur_original_amps[..., idx].T
            cur_est_amps_mid_band = cur_est_amps[..., idx].T
        # if amplitudes are broadband
        else:
            amps_mid_band = cur_original_amps.T
            cur_est_amps = np.squeeze(cur_est_amps)
            cur_est_amps_mid_band = cur_est_amps.T

        # save the amplitudes for the source position in a larger matrix
        if len(src_pos) > 1:
            est_amps[src_idx, ...] = cur_est_amps
        else:
            est_amps = cur_est_amps

        if plot_original_amps:
            room.plot_amps_at_receiver_points(
                rec_points,
                cur_src_pos,
                amps_mid_band,
                scatter_plot=scatter,
                save_path=Path(f'{save_name}_actual_amplitudes_in_space.png'
                               ).resolve() if save_path is not None else None)

        room.plot_amps_at_receiver_points(
            est_rec_pos,
            cur_src_pos,
            cur_est_amps_mid_band,
            scatter_plot=scatter,
            save_path=Path(f'{save_name}_learnt_amplitudes_in_space.png'
                           ).resolve() if save_path is not None else None)

        if plot_amp_error:

            # get error metrics
            error_func, error_mse = get_amplitude_error(
                cur_original_amps, rec_points, cur_est_amps, est_rec_pos)
            if is_in_subbands:
                idx = np.argwhere(
                    np.array(room_data.band_centre_hz) == freq_to_plot)[0][0]
                for k in range(len(room_data.band_centre_hz)):
                    logger.info(
                        f'The RMSE in matching amplitudes at frequency {room_data.band_centre_hz[k]: .0f}Hz'
                        f' is {error_mse[k]: .3f} dB')
                    var_to_plot = db2lin(error_func[..., idx].T)
            else:
                logger.info(
                    f'The RMSE in matching amplitudes is {error_mse} dB')
                var_to_plot = db2lin(error_func.T)

            # plot the error in amplitude matching
            room.plot_amps_at_receiver_points(
                rec_points,
                cur_src_pos,
                var_to_plot,
                scatter_plot=scatter,
                save_path=Path(f'{save_name}_amplitude_error_in_space.png'
                               ).resolve() if save_path is not None else None)

    return est_amps


def plot_learned_svf_response(
        num_groups: int,
        fs: float,
        output_biquad_coeffs: Union[List[List], List[NDArray]],
        pos_to_investigate: List,
        verbose: bool = False,
        svf_params: Optional[Union[List[List], List[NDArray]]] = None,
        save_path: Optional[str] = None):
    """
    Plot the magnitude response and the pole-zero plot of the learned SVF
    at position pos_to_investigate.
    Args:
        num_groups (int): number of groups in the GFDN
        fs (float): sampling frequency
        output_biquad_coeffs: output filter biquad coefficients, list containing num_group filters. 
                              Can also be a list of lists containing filters for each epoch
        pos_to_investigate (List): cartesian coordinates of the position under invesigation
        verbose (bool): whether to print theoretical SVF poles
        svf_params (optional): list containing learned SVF params for each group. 
                               Can also be a list of lists containing params for each epoch.
        save_path (str): where to save the figure
    """
    if verbose:
        centre_freq, shelving_crossover = eq_freqs()
        svf_freqs = torch.pi * torch.cat(
            (torch.tensor([shelving_crossover[0]]), centre_freq,
             torch.tensor([shelving_crossover[-1]]))) / fs
        svf_freqs = svf_freqs.numpy()

    fig, ax = plt.subplots(num_groups, 1)
    fig2, ax2 = plt.subplots(num_groups,
                             1,
                             subplot_kw={'projection': 'polar'},
                             figsize=(6, 8))

    # are the output_biquad_coeffs also a function of epoch number?
    is_list_of_lists = all(
        isinstance(item, list) for item in output_biquad_coeffs)
    num_epochs = len(output_biquad_coeffs) if is_list_of_lists else 1

    # loop over epochs
    for i in range(0, num_epochs):

        if is_list_of_lists:
            opt_output_biquad_coeffs = output_biquad_coeffs[i]

            if svf_params is not None:
                opt_svf_params = svf_params[i]
                opt_svf_params = opt_svf_params[
                    np.newaxis, ...] if num_groups == 1 else opt_svf_params
        else:
            opt_output_biquad_coeffs = output_biquad_coeffs
            opt_svf_params = svf_params

        # loop over groups
        for n in range(num_groups):
            cur_biquad_coeffs = opt_output_biquad_coeffs[n]
            num_biquads = cur_biquad_coeffs.shape[0]

            # ensure a0 = 1 (needed by scipy)
            for k in range(num_biquads):
                cur_biquad_coeffs[k, :] /= cur_biquad_coeffs[k, 3]

            freqs, filt_response = sosfreqz(cur_biquad_coeffs,
                                            worN=2**9,
                                            fs=fs)
            ax[n].semilogx(freqs,
                           db(filt_response),
                           label=f'Group {n}, epoch {i}')

            # also plot the poles and zeros
            zeros, poles, _ = sos2zpk(cur_biquad_coeffs)
            ax2[n].plot(np.angle(zeros),
                        np.abs(zeros),
                        'o',
                        label=f'Group {n}, epoch {i}')
            ax2[n].plot(np.angle(poles),
                        np.abs(poles),
                        'x',
                        label=f'Group {n}, epoch {i}')

            if verbose:

                # print the theoretical poles and zeros
                cur_svf_params = opt_svf_params[n, ...]
                svf_res = cur_svf_params[:, 0]
                svf_gain = cur_svf_params[:, 1]
                pole_radius = np.sqrt(
                    (1 - svf_freqs**2)**2 + 4 * (svf_freqs**2) *
                    (1 - svf_res**2)) / (svf_freqs**2 + 1 +
                                         2 * svf_freqs * svf_res)
                pole_freqs = np.atan2(2 * svf_freqs * np.sqrt(1 - svf_res**2),
                                      (1 - svf_freqs**2))

                print(f'Pole frequencies (exp): {pole_freqs / np.pi * fs / 2}')
                print(
                    f'Pole frequencies (est): {np.angle(poles[np.angle(poles) > 0]) / np.pi * fs / 2}'
                )
                print(f'Pole radius (exp): {pole_radius}')
                print(f'Pole radius (est): {np.abs(poles)}')

                print(f'SVF gain: {db2lin(svf_gain)}')
                print(f'SVF Q factor: {svf_res}')

    # set axis labels
    ax[0].legend(loc='upper right', bbox_to_anchor=(1.5, 1.0))
    for n in range(num_groups):
        ax[n].set_xlabel('Frequency (Hz)')
        ax[n].set_ylabel('Magnitude (dB)')
        ax[n].set_title(
            f'Output filter for group {n+1} at position {pos_to_investigate}')
        ax[n].grid(True)

        ax2[n].set_rmax(1)
        ax2[n].set_rticks([0.25, 0.5, 1])  # Less radial ticks
        ax2[n].set_rlabel_position(
            -22.5)  # Move radial labels away from plotted line
        ax2[n].grid(True)

    fig.subplots_adjust(hspace=0.3 * num_groups)
    fig2.subplots_adjust(hspace=0.5)

    if save_path is not None:
        fig.savefig(Path(f'{save_path}_output_filter_response.png').resolve())
        fig2.savefig(Path(f'{save_path}_output_filter_pz_plot.png').resolve())
