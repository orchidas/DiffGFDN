import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pyfar as pf
from scipy.fft import rfft, rfftfreq
from scipy.signal import fftconvolve
from slope2noise.generate import shaped_wgn
from slope2noise.utils import schroeder_backward_int
import soundfile as sf
import spaudiopy as sp

from diff_gfdn.config.config import DiffGFDNConfig
from diff_gfdn.dataloader import ThreeRoomDataset
from diff_gfdn.utils import db, db2lin, ms_to_samps
from spatial_sampling.config import BeamformerType
from spatial_sampling.dataloader import SpatialThreeRoomDataset


# flake8: noqa:E231
def almost_equal(a: NDArray, b: NDArray, eps: float = 1e-8) -> bool:
    """Check if the elements in two arrays are almost equal"""
    return np.all(np.abs(a - b) < eps)


def get_pyfar_octave_filterbank(fs: float):
    """Get the octave filterbank using the pyfar library"""
    subband_filters, freqs = pf.dsp.filter.reconstructing_fractional_octave_bands(
        None,
        num_fractions=1,
        frequency_range=(63, 8000),
        sampling_rate=fs,
    )
    return subband_filters, freqs


def filter_signal_octave_bands(signal: ArrayLike,
                               subband_filters,
                               freqs: List,
                               mode: str = 'full'):
    """Filter signal in full-octave bands"""
    num_bands = len(freqs)
    if mode == 'full':
        cur_sig_filtered = np.zeros(
            (len(signal) + subband_filters.coefficients.shape[-1] - 1,
             num_bands))
    else:
        cur_sig_filtered = np.zeros((len(signal), num_bands))
    for k in range(num_bands):
        cur_sig_filtered[:,
                         k] = fftconvolve(signal,
                                          subband_filters.coefficients[k, :],
                                          mode=mode)
    return cur_sig_filtered


def test_pyfar_filterbank():
    """Assure that pyfar's fractional octave reconstructing filterbank works correctly"""
    fs = 44100
    x = pf.signals.impulse(2**12)
    subband_filters, freqs = get_pyfar_octave_filterbank(fs)

    ir_filtered = filter_signal_octave_bands(np.squeeze(x.time),
                                             subband_filters, freqs)
    ir_response = rfft(ir_filtered, n=2**12, axis=0)
    fft_freqs = rfftfreq(n=2**12, d=1.0 / fs)

    plt.figure()
    plt.semilogx(fft_freqs, db(ir_response))
    plt.semilogx(fft_freqs, db(np.sum(ir_response, axis=-1)))

    assert np.allclose(np.sum(np.abs(ir_response), axis=-1),
                       np.ones(ir_response.shape[0]))


def test_pyfar_filterbank_white_noise():
    """Test pyfar filterbank on the same white noise, vs incoherent white noise"""

    def get_multichannel_noise(
            sig_len_samp: int,
            num_bands: int,
            noise_rms_db: float,
            sparsity_thres_db: Optional[float] = None) -> NDArray:
        """Add Gaussian noise of desired RMS value to input signal"""
        noise = np.random.randn(
            sig_len_samp,
            num_bands)  # Generate standard Gaussian noise (mean=0, std=1)
        des_rms = db2lin(noise_rms_db)
        current_rms = np.sqrt(np.mean(noise**2, axis=0))  # Compute current RMS
        scaled_noise = noise * (des_rms / current_rms)  # Scale to desired RMS
        if sparsity_thres_db is not None:
            indices = np.abs(scaled_noise[:, 0]) < db2lin(sparsity_thres_db)
            scaled_noise[indices, :] = 0.0
        return scaled_noise

    fs = 44100
    ir_len_ms = 1000
    ir_len_samps = ms_to_samps(ir_len_ms, fs)
    fft_size = 2**np.ceil(np.log2(ir_len_samps)).astype(int)

    # generate the same noise sequence and filter it
    rms_noise_db = -20
    sparsity_thres_db = -10
    noise = get_multichannel_noise(ir_len_samps, 1, rms_noise_db,
                                   sparsity_thres_db)
    subband_filters, freqs = get_pyfar_octave_filterbank(fs)
    noise_filtered = filter_signal_octave_bands(np.squeeze(noise),
                                                subband_filters, freqs)
    noise_filtered_spectrum = rfft(noise_filtered, n=fft_size, axis=0)

    # generate a sequence of independent noise
    num_bands = len(freqs)
    ind_noise = get_multichannel_noise(ir_len_samps, num_bands, rms_noise_db,
                                       sparsity_thres_db)
    ind_noise_filtered = np.zeros(
        (ir_len_samps + subband_filters.coefficients.shape[-1] - 1, num_bands))
    for b_idx in range(num_bands):
        ind_noise_filtered[:, b_idx] = fftconvolve(
            ind_noise[:, b_idx],
            subband_filters.coefficients[b_idx, :],
            mode='full')
    ind_noise_filtered_spectrum = rfft(ind_noise_filtered, n=fft_size, axis=0)

    # plot the results
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'hspace': 0.5})
    freq_bins_hz = rfftfreq(n=fft_size, d=1.0 / fs)
    ax[0].semilogx(
        freq_bins_hz,
        db(noise_filtered_spectrum),
        label=[f"{np.round(freqs[k])} Hz" for k in range(len(freqs))],
        linestyle="-",
        alpha=0.8,
    )

    ax[0].semilogx(freq_bins_hz,
                   db(np.sum(noise_filtered_spectrum, axis=-1)),
                   label='summed',
                   linestyle='--')

    ax[1].semilogx(
        freq_bins_hz,
        db(ind_noise_filtered_spectrum),
        label=[f"{np.round(freqs[k])} Hz" for k in range(len(freqs))],
        linestyle="-",
        alpha=0.8,
    )

    ax[1].semilogx(freq_bins_hz,
                   db(np.sum(ind_noise_filtered_spectrum, axis=-1)),
                   label='summed',
                   linestyle='--')

    for i in range(2):
        ax[i].set_ylabel("Magnitude (dB)")
        ax[i].set_xlabel('Frequencies (Hz)')
        ax[i].grid(True)
        ax[i].set_xlim([20, 20000])
        ax[i].set_ylim([-60, 30])

    ax[0].legend(bbox_to_anchor=(1, 1))
    ax[0].set_title("Coherent sparse white noise filtered in subbands")
    ax[1].set_title("Incoherent sparse white noise filtered in subbands")

    fig.savefig(Path(
        'figures/test_plots/test_pyfar_filterbank_sparse_white_noise_mag_spectrum.png'
    ).resolve(),
                bbox_inches='tight')


def test_pyfar_filterbank_rir_data():
    """Test the reconstructing filterbank on RIR data"""
    config_dict = DiffGFDNConfig()
    room_dataset = ThreeRoomDataset(Path(config_dict.room_dataset_path),
                                    config_dict)

    use_fixed_pos = True
    if use_fixed_pos:
        pos_to_investigate = [0.20, 2.90, 1.50]
        rec_idx = np.argwhere(
            np.all(np.round(room_dataset.receiver_position,
                            2) == pos_to_investigate,
                   axis=1))[0]
    else:
        # pick a random receiver position
        rec_idx = np.random.randint(0,
                                    high=room_dataset.num_rec,
                                    size=1,
                                    dtype=int)
    pos_to_investigate = np.round(
        np.squeeze(room_dataset.receiver_position[rec_idx, :]), 2)

    rir = np.squeeze(room_dataset.rirs[rec_idx, :])

    subband_filters, freqs = get_pyfar_octave_filterbank(
        config_dict.sample_rate)
    rir_filtered = filter_signal_octave_bands(rir,
                                              subband_filters,
                                              freqs,
                                              mode='same')
    # assert np.allclose(rir, np.sum(rir_filtered, axis=-1))

    rir_response = rfft(rir, n=room_dataset.num_freq_bins)
    rir_filtered_response = rfft(rir_filtered,
                                 n=room_dataset.num_freq_bins,
                                 axis=0)

    assert almost_equal(db(rir_response),
                        db(np.sum(rir_filtered_response, axis=-1)),
                        eps=1e-5)

    # write to audio file
    desired_filename = f'synth_ir_({pos_to_investigate[0]:.2f}, {pos_to_investigate[1]:.2f}, ' \
    f'{pos_to_investigate[2]:.2f}).wav'
    write_path = Path(f'audio/true/{desired_filename}').resolve()
    sf.write(write_path, np.sum(rir_filtered, axis=-1),
             int(config_dict.sample_rate))


def test_pyfar_edc_broadband_wn_rir():
    """
    Test the reconstructing filterbank on a broadband RIR constructed with shaped white noise
    and observe the shape of the EDC before and after filtering
    """
    # Generate broadband noise with 3 slopes
    config_dict = DiffGFDNConfig()
    room_dataset = ThreeRoomDataset(Path(config_dict.room_dataset_path),
                                    config_dict)
    n_bands = len(room_dataset.band_centre_hz)
    pos_to_investigate = [9.30, 6.60, 1.50]
    rec_pos_idx = np.argwhere(
        np.all(np.round(room_dataset.receiver_position,
                        2) == pos_to_investigate,
               axis=1))[0]
    fs = room_dataset.sample_rate
    ir_len = room_dataset.rir_length
    rir_ref = room_dataset.rirs[rec_pos_idx, :ir_len].squeeze()

    t_vals = room_dataset.common_decay_times.transpose(1, -1, 0)
    a_vals = room_dataset.amplitudes[rec_pos_idx]
    n_vals = room_dataset.noise_floor[rec_pos_idx].squeeze(axis=1)

    # filter in octave bands
    _, wgn_rir = shaped_wgn(t_vals,
                            a_vals,
                            fs,
                            ir_len,
                            f_bands=room_dataset.band_centre_hz,
                            n_vals=n_vals)
    wgn_rir = np.squeeze(wgn_rir)
    subband_filters, freqs = get_pyfar_octave_filterbank(fs)
    wgn_rir_filtered = filter_signal_octave_bands(wgn_rir,
                                                  subband_filters,
                                                  freqs,
                                                  mode='same')
    n_bands = len(freqs)
    edc_ref = schroeder_backward_int(rir_ref, normalize=False)
    edc_syn = schroeder_backward_int(wgn_rir, normalize=False)
    edc_filtered = np.zeros((n_bands, ir_len))
    time_axis = np.linspace(0, (ir_len - 1) / fs, ir_len)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(time_axis, db(edc_ref, is_squared=True), label='Reference EDC')
    ax.plot(time_axis,
            db(edc_syn, is_squared=True),
            label='Broadband synth EDC')

    for k in range(n_bands):
        edc_filtered[k, :] = schroeder_backward_int(wgn_rir_filtered[:, k],
                                                    normalize=False)
        ax.plot(time_axis,
                db(edc_filtered[k, :], is_squared=True),
                label=f'Filtered EDC, fc={np.round(freqs[k], 2)}Hz')

    ax.legend()
    fig.savefig(
        Path('figures/test_plots/test_pyfar_filterbank_edc_white_noise.png').
        resolve())


def test_pyfar_edc_broadband_gfdn_rir():
    """
    Test the EDC of a broadband RIR constructed with DiffGFDN
    and compare it to the reference broadband EDC
    """
    audio_path = Path('audio/').resolve()
    config_dict = DiffGFDNConfig()
    room_dataset = ThreeRoomDataset(Path(config_dict.room_dataset_path),
                                    config_dict)
    freq_bands = room_dataset.band_centre_hz
    n_bands = len(room_dataset.band_centre_hz)
    fs = room_dataset.sample_rate
    ir_len = room_dataset.rir_length

    pos_to_investigate = [9.30, 6.60, 1.50]
    rec_pos_idx = np.argwhere(
        np.all(np.round(room_dataset.receiver_position,
                        2) == pos_to_investigate,
               axis=1))[0]
    rir_ref = room_dataset.rirs[rec_pos_idx, :].squeeze()
    desired_pos = f'({pos_to_investigate[0]:.2f}, {pos_to_investigate[1]:.2f}, {pos_to_investigate[2]:.2f}).wav'

    subband_filters, _ = get_pyfar_octave_filterbank(fs)
    gfdn_rir_filtered = np.zeros(
        (ir_len + subband_filters.coefficients.shape[-1] - 1, n_bands))

    for b_idx in range(len(freq_bands)):
        try:
            rir_path = os.path.join(
                audio_path,
                f'grid_rir_treble_band_centre={freq_bands[b_idx]}Hz_colorless_loss_diff_delays/ir_{desired_pos}'
            )
            # this RIR has not been filtered into subbands, so it needs to be
            gfdn_rir, fs = sf.read(rir_path)

        except sf.LibsndfileError:
            rir_path = os.path.join(
                audio_path,
                f'grid_rir_treble_band_centre={freq_bands[b_idx]}Hz_colorless_loss_diff_delays/valid_ir_{desired_pos}'
            )
            # this RIR has not been filtered into subbands, so it needs to be
            gfdn_rir, fs = sf.read(rir_path)

        gfdn_rir_filtered[..., b_idx] = fftconvolve(
            gfdn_rir[:ir_len, 0],
            subband_filters.coefficients[b_idx, :],
            mode='full')

    gfdn_rir_broadband = np.sum(gfdn_rir_filtered[:ir_len, :], axis=-1)
    # normalize is set to True here because when I saved the audio files
    # I normalised them, so without normalisation the EDCs of the reference
    # and synth RIRs won't match
    edc_syn = schroeder_backward_int(
        gfdn_rir_broadband,
        normalize=True,
    )
    edc_ref = schroeder_backward_int(rir_ref, normalize=True)
    edc_filtered = np.zeros((n_bands, ir_len))
    time_axis = np.linspace(0, (ir_len - 1) / fs, ir_len)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(time_axis, db(edc_ref, is_squared=True), label='Reference EDC')
    ax.plot(time_axis,
            db(edc_syn, is_squared=True),
            label='Synth broadband EDC')

    for k in range(n_bands):
        edc_filtered[k, :] = schroeder_backward_int(
            gfdn_rir_filtered[:ir_len, k],
            normalize=False,
        )
        ax.plot(time_axis,
                db(edc_filtered[k, :], is_squared=True),
                label=f'Filtered EDC, fc={np.round(freq_bands[k], 2)}Hz')

    ax.legend()
    fig.savefig(
        Path('figures/test_plots/test_pyfar_filterbank_edc_diff_gfdn.png').
        resolve())


###############################################################################


def get_beamformer_weights(ambi_order: int,
                           beamformer_type: BeamformerType) -> NDArray:
    # get the modal beamformer weights
    if beamformer_type == BeamformerType.MAX_DI:
        modal_weights = sp.sph.cardioid_modal_weights(ambi_order)
    elif beamformer_type == BeamformerType.MAX_RE:
        modal_weights = sp.sph.maxre_modal_weights(ambi_order)
    elif beamformer_type == BeamformerType.BUTTER:
        modal_weights = sp.sph.butterworth_modal_weights(ambi_order,
                                                         k=5,
                                                         n_c=3)
    else:
        raise NameError("Other types of beamformers not available")
    return modal_weights


def spherical_analysis_filterbank(ambi_order: int,
                                  des_dir: NDArray,
                                  srir: NDArray,
                                  beamformer_type: BeamformerType,
                                  mode: str = 'energy') -> NDArray:
    """Convert SRIR to DRIR using spherical analysis filterbank"""
    modal_weights = get_beamformer_weights(ambi_order, beamformer_type)

    # output of size num_directions x (N_sp+1)^2
    sph_an_matrix, _ = sp.sph.design_sph_filterbank(ambi_order,
                                                    des_dir[0, :],
                                                    np.pi / 2 - des_dir[1, :],
                                                    modal_weights,
                                                    mode=mode,
                                                    sh_type='real')
    # directional RIR in desired directions
    dir_rir = np.einsum('jn, nt -> jt', sph_an_matrix, srir)
    return dir_rir, sph_an_matrix


def spherical_synthesis_filterbank(ambi_order: int,
                                   des_dir: NDArray,
                                   drir: NDArray,
                                   beamformer_type: BeamformerType,
                                   mode: str = 'energy') -> NDArray:
    """Convert DRIR to ambi_order SRIR using spherical synthesis filterbank"""
    # size is num_ambi_channels x num_directions
    modal_weights = get_beamformer_weights(ambi_order, beamformer_type)

    _, sph_syn_matrix = sp.sph.design_sph_filterbank(ambi_order,
                                                     des_dir[0, :],
                                                     np.pi / 2 - des_dir[1, :],
                                                     modal_weights,
                                                     mode=mode,
                                                     sh_type='real')

    srir = np.einsum('nj, jt -> nt', sph_syn_matrix, drir)
    return srir, sph_syn_matrix


def spatial_bandlimiting(ambi_order: int, des_dir: NDArray, drir: NDArray,
                         beamformer_type: BeamformerType):
    """Ensure spatial band limitation of directional RIRs - see Holdt et al"""
    modal_weights = get_beamformer_weights(ambi_order, beamformer_type)

    sh_matrix = sp.sph.sh_matrix(ambi_order, des_dir[0, :],
                                 np.pi / 2 - des_dir[1, :])
    # size is num_directions x num_directions
    spatial_cov_matrix = sh_matrix @ np.diag(
        sp.sph.repeat_per_order(modal_weights)) @ sh_matrix.T

    # sum over any one dimension since the matrix is symmetric
    norm_factor = spatial_cov_matrix / np.sum(
        spatial_cov_matrix, axis=1, keepdims=True)
    bandlimited_drir = np.einsum('jj, jt -> jt', norm_factor, drir)
    return bandlimited_drir


def test_spherical_t_design_grid(ambi_order: NDArray, des_dir: NDArray):
    """Test whether des_dir directions belong to a spherical t-design grid of order ambi_order"""
    N = ambi_order
    J = des_dir.shape[1]
    Y = sp.sph.sh_matrix(N, des_dir[0, :], np.pi / 2 - des_dir[1, :],
                         'real')  # shape (J, (N+1)^2)

    # Compute approximate integral of Y^T Y over sphere
    G = (4 * np.pi / J) * (Y.T @ Y)

    # Compare to identity
    err = np.linalg.norm(G - np.eye(
        (N + 1)**2), ord='fro') / np.linalg.norm(np.eye((N + 1)**2), ord='fro')
    assert err < 1e-10


def test_spherical_filterbank():
    """Test the spherical filterbank analysis -> synthesis pipeline with a reference SRIR"""

    room_data_pkl_path = Path(
        'resources/Georg_3room_FDTD/srirs_spatial.pkl').resolve()
    room_dataset = SpatialThreeRoomDataset(room_data_pkl_path)
    pos_to_investigate = [9.30, 6.60, 1.50]
    rec_pos_idx = np.argwhere(
        np.all(np.round(room_dataset.receiver_position,
                        2) == pos_to_investigate,
               axis=1))[0]
    ir_len = room_dataset.rir_length
    des_dir = room_dataset.sph_directions
    ambi_order = room_dataset.ambi_order
    num_channels = (ambi_order + 1)**2

    test_spherical_t_design_grid(ambi_order, des_dir)

    srir_ref = room_dataset.rirs[rec_pos_idx, :, :ir_len].squeeze()
    drir, sph_an_matrix = spherical_analysis_filterbank(ambi_order,
                                                        des_dir,
                                                        srir_ref,
                                                        BeamformerType.MAX_DI,
                                                        mode='perfect')
    srir_recon, sph_syn_matrix = spherical_synthesis_filterbank(
        ambi_order, des_dir, drir, BeamformerType.MAX_DI, mode='perfect')

    fig, ax = plt.subplots(num_channels, 1, figsize=(8, 10),
                           sharey=True)  # rows, cols
    for j in range(num_channels):
        ax[j].plot(srir_recon[j, :])
        ax[j].plot(srir_ref[j, :])
        ax[j].set_title(f'Channel = {j+1}')

    ax[-1].set_xlabel('Time (samples)')
    fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')
    # increase space between subplots
    fig.subplots_adjust(hspace=1.0)  # increase vertical spacing
    fig.savefig(
        Path('figures/test_plots/test_spherical_filterbank.png').resolve())

    assert almost_equal(sph_syn_matrix @ sph_an_matrix,
                        np.eye(num_channels),
                        eps=1e-6)
    assert np.allclose(srir_ref, srir_recon, atol=1e-8)


def test_wn_recons_with_spherical_filterbank():
    """Shape white noise with common slope amplitudes to predict SRIR and 
    plot EDC mismatch errors with reference SRIR"""

    room_data_pkl_path = Path(
        'resources/Georg_3room_FDTD/srirs_spatial.pkl').resolve()
    room_dataset = SpatialThreeRoomDataset(room_data_pkl_path)
    pos_to_investigate = [9.30, 6.60, 1.50]
    rec_pos_idx = np.argwhere(
        np.all(np.round(room_dataset.receiver_position,
                        2) == pos_to_investigate,
               axis=1))[0]

    ir_len = room_dataset.rir_length
    fs = room_dataset.sample_rate
    des_dir = room_dataset.sph_directions
    ambi_order = room_dataset.ambi_order
    num_channels = (ambi_order + 1)**2
    num_directions = room_dataset.num_directions
    srir_ref = room_dataset.rirs[rec_pos_idx, :, :ir_len].squeeze()

    # get CS model parameters
    t_vals_exp = np.repeat(room_dataset.common_decay_times,
                           num_directions,
                           axis=1).transpose(1, -1, 0)
    a_vals = room_dataset.amplitudes[rec_pos_idx].squeeze().transpose(1, 0, -1)

    # get shaped white noise srir
    _, wgn_drir = shaped_wgn(
        t_vals_exp,
        a_vals,
        fs,
        ir_len,
        f_bands=room_dataset.band_centre_hz,
    )

    # convert to ambisonics signal
    bandlimit_wgn_drir = spatial_bandlimiting(ambi_order, des_dir, wgn_drir,
                                              BeamformerType.MAX_DI)
    wgn_srir, _ = spherical_synthesis_filterbank(ambi_order, des_dir,
                                                 bandlimit_wgn_drir,
                                                 BeamformerType.MAX_DI)

    # get EDC of reference and synthesised tails and plot them
    edc_ref = schroeder_backward_int(srir_ref, normalize=True)
    edc_syn = schroeder_backward_int(wgn_srir, normalize=True)

    fig, ax = plt.subplots(num_channels, 1, figsize=(8, 10),
                           sharey=True)  # rows, cols
    for j in range(num_channels):
        ax[j].plot(db(edc_ref[j, :], is_squared=True))
        ax[j].plot(db(edc_syn[j, :], is_squared=True))
        ax[j].set_title(f'Channel = {j+1}')

    ax[-1].set_xlabel('Time (samples)')
    fig.text(0.04, 0.5, 'EDC (dB)', va='center', rotation='vertical')
    # increase space between subplots
    fig.subplots_adjust(hspace=1.0)  # increase vertical spacing
    fig.savefig(
        Path(
            'figures/test_plots/test_wn_recons_spherical_filterbank_edc_spatial_bandlimit.png'
        ).resolve())


if __name__ == '__main__':
    # test_pyfar_filterbank_white_noise()
    # test_spherical_filterbank()
    # test_pyfar_edc_broadband_wn_rir()
    # test_pyfar_edc_broadband_gfdn_rir()
    test_wn_recons_with_spherical_filterbank()
