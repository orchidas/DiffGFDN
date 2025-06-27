import subprocess
from typing import List, Optional, Tuple

from loguru import logger
from matplotlib import animation, patches
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pyloudnorm as pyln
from scipy.fft import irfft, rfft
from scipy.signal import fftconvolve
from slope2noise.slope2noise.rooms import RoomGeometry
import spaudiopy as spa

from diff_gfdn.dataloader import RoomDataset
from diff_gfdn.utils import ms_to_samps
from sofa_parser import HRIRSOFAReader
from spatial_sampling.dataloader import SpatialRoomDataset

# pylint: disable=R1707, E1126, E0203
# flake8: noqa:E231


class dynamic_rendering_moving_receiver:

    def __init__(
        self,
        room_dataset: RoomDataset,
        rec_pos_list: NDArray,
        stimulus: ArrayLike,
        update_ms: float = 100,
    ):
        """
        Create sound examples of rendered stimuli (and optionally, animation) of a receiver moving through a room.
        Args:
            room_dataset: RoomDataset object containing the RIRs and the source-receiver configuration
            rec_pos_list: List of receiver positions to navigate through, of size num_pos, 3
            update_ms: How often is the receiver position updated
            stimulus: dry source, mono
        """
        self.room = room_dataset
        self.sample_rate = self.room.sample_rate
        self.rec_pos_list = rec_pos_list
        self.num_pos = self.rec_pos_list.shape[0]
        self.update_ms = update_ms
        self.update_len_samp = ms_to_samps(update_ms, self.sample_rate)
        self.stimulus = stimulus
        self.room_geometry = RoomGeometry(
            self.room.sample_rate,
            self.room.num_rooms,
            np.array(self.room.room_dims),
            np.array(self.room.room_start_coord),
            aperture_coords=self.room.aperture_coords)
        self.extended_stimulus = self.create_extended_stimulus()

    @property
    def hop_size(self) -> int:
        """Hop size for each update"""
        return self.update_len_samp

    @staticmethod
    def get_fade_windows(win_len_samps: int,
                         fade_out: bool = False,
                         uncorr_fade: bool = False) -> ArrayLike:
        """
        Linear fade-in and fade-out windows. Can be correlated or uncorrelated
        """
        n = np.linspace(start=-1, stop=1, num=win_len_samps)
        fade = 0.5 * (1 + (1 - 2 * int(fade_out)) * n)
        # fade: NDArray = 0.5 - 0.5 * np.cos(np.pi * (n + int(fade_out)))
        return np.sqrt(fade) if uncorr_fade else fade

    @property
    def total_sim_len(self) -> int:
        """Length of the total simulation"""
        return self.num_pos * self.hop_size

    @property
    def rec_idxs(self) -> List:
        """Indices of the receivers in the dataset associated with the list of receiver positions"""
        return self.room.find_rec_idx_in_room_dataset(self.rec_pos_list)

    @property
    def rirs(self) -> NDArray:
        """Create list of RIRs associated with the moving listeners"""
        return self.room.rirs[self.rec_idxs, :]

    @property
    def late_rirs(self) -> NDArray:
        """Create a list of late RIRs associated with the moving listeners"""
        return self.room.late_rirs[self.rec_idxs, :]

    def create_extended_stimulus(self) -> NDArray:
        """Repeat the stimulus to match the total simulation length"""
        cur_stim_len = len(self.stimulus)
        num_rep = int(np.ceil(self.total_sim_len / cur_stim_len))
        stimulus_extended = np.zeros(self.total_sim_len, dtype=np.float32)
        for rep in range(num_rep):
            b_idx = np.arange(rep * cur_stim_len,
                              min((rep + 1) * cur_stim_len,
                                  self.total_sim_len),
                              dtype=np.int32)
            stimulus_extended[b_idx] = self.stimulus[:len(b_idx)]

        return stimulus_extended

    def filter_overlap_add(self,
                           use_whole_rir: bool = False,
                           alpha: float = 0.5,
                           fade_len_ms: float = 50.0) -> NDArray:
        """Filter and cross-fade the stimulus with the RIRs associated with the moving listener."""
        output_signal = np.zeros_like(self.extended_stimulus)
        fade_len = ms_to_samps(fade_len_ms, self.sample_rate)
        fade_out = self.get_fade_windows(
            fade_len,
            fade_out=True,
        )
        fade_in = self.get_fade_windows(
            fade_len,
            fade_out=False,
        )
        prev_tail = np.zeros(fade_len)

        for k in range(self.num_pos):
            b_idx = np.arange(k * self.hop_size,
                              min((k + 1) * self.hop_size, self.total_sim_len),
                              dtype=np.int32)
            if use_whole_rir:
                cur_filter = self.rirs[k, :]
            else:
                cur_filter = self.late_rirs[k, :]

            # interpolate between current and previous RIR
            if hasattr(self, 'prev_filter'):
                cur_filter = alpha * cur_filter + (1 -
                                                   alpha) * self.prev_filter
            self.prev_filter = cur_filter

            cur_stimulus = self.extended_stimulus[b_idx]
            cur_filtered_signal = fftconvolve(cur_stimulus,
                                              cur_filter,
                                              mode='full')
            start_idx = k * self.hop_size
            end_idx = min(start_idx + len(cur_filtered_signal),
                          len(output_signal))
            cur_trunc_filtered_signal = cur_filtered_signal[:end_idx -
                                                            start_idx]
            # Crossfade overlapping samples between previous tail and current head
            if k > 0:
                overlap_len = min(fade_len, len(cur_trunc_filtered_signal))
                crossfaded = prev_tail[:overlap_len] * fade_out[:overlap_len] + \
                             cur_trunc_filtered_signal[:overlap_len] * fade_in[:overlap_len]

                output_signal[start_idx:start_idx + overlap_len] += crossfaded

                output_signal[
                    start_idx +
                    overlap_len:end_idx] += cur_trunc_filtered_signal[
                        overlap_len:]
            else:
                output_signal[start_idx:end_idx] += cur_trunc_filtered_signal

            # Save tail for next iteration
            if len(cur_trunc_filtered_signal) >= fade_len:
                prev_tail[:fade_len] = cur_trunc_filtered_signal[-fade_len:]
            else:
                prev_tail[:len(cur_trunc_filtered_signal
                               )] = cur_trunc_filtered_signal

        return output_signal

    def animate_moving_listener(self,
                                save_path: Optional[str] = None,
                                yaw_angles: Optional[ArrayLike] = None):
        """Animate the listener moving through the room"""
        fig, ax = plt.subplots(figsize=(6, 6))
        # draw the boundaries in the room
        ax = self.room_geometry.draw_boundaries(ax)

        num_frames = self.num_pos  # Total frames
        x_vals = self.rec_pos_list[:, 0]  # X-coordinates
        y_vals = self.rec_pos_list[:, 1]  # Y-coordinates

        src_x = self.room.source_position[:, 0].squeeze()
        src_y = self.room.source_position[:, 1].squeeze()

        # Create a small moving circle
        circle = patches.Circle((x_vals[0], y_vals[0]),
                                radius=0.15,
                                color='red',
                                animated=True)
        ax.add_patch(circle)

        # Create cross at source (two perpendicular lines)
        ax.plot([src_x - 0.25, src_y + 0.25], [src_x, src_y], 'r-',
                lw=2)  # Horizontal line
        ax.plot([src_x, src_y], [src_x - 0.25, src_y + 0.25], 'r-',
                lw=2)  # Vertical line

        # Create an initial arrow to represent head rotation yaw(we’ll update it each frame)
        if yaw_angles is not None:
            plot_yaw_angles = True
            arrow_len = 0.4
            initial_yaw = yaw_angles[0]
            dx = arrow_len * np.cos(initial_yaw)
            dy = arrow_len * np.sin(initial_yaw)
            arrow = ax.arrow(x_vals[0],
                             y_vals[0],
                             dx,
                             dy,
                             head_width=0.1,
                             head_length=0.1,
                             fc='blue',
                             ec='blue')
        else:
            plot_yaw_angles = False

        def update(frame):
            """Update function for animation"""

            pos = (x_vals[frame], y_vals[frame])

            circle.set_center(pos)  # Move the circle

            if plot_yaw_angles:
                nonlocal arrow  # allow us to modify the arrow reference

                # remove the previous arrow
                arrow.remove()

                # Compute new arrow direction
                yaw = yaw_angles[frame]
                dx = arrow_len * np.cos(yaw)
                dy = arrow_len * np.sin(yaw)

                arrow = ax.arrow(pos[0],
                                 pos[1],
                                 dx,
                                 dy,
                                 head_width=0.1,
                                 head_length=0.1,
                                 fc='blue',
                                 ec='blue')
                return circle, arrow
            else:
                return circle,

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=num_frames,
            interval=self.update_ms,
            blit=True,
        )
        if save_path is not None:
            ani.save(f"{save_path}_moving_listener.mp4",
                     writer=animation.FFMpegWriter(fps=1000 // self.update_ms))

        plt.show()

    @staticmethod
    def normalise_loudness(output_signal: NDArray,
                           sample_rate: float,
                           db_lufs: float = -18.0):
        """Normalise the output signal to _ dB LUFS"""
        # measure the loudness first
        meter = pyln.Meter(sample_rate)
        # create BS.1770 meter
        loudness = meter.integrated_loudness(output_signal)

        # loudness normalize audio
        loudness_normalized_audio = pyln.normalize.loudness(
            output_signal, loudness, db_lufs)
        return loudness_normalized_audio

    @staticmethod
    def combine_animation_and_sound(mp4_file_path: str, wav_file_path: str,
                                    save_path: str):
        """Combine animation with sound to generate video, we use ffmpeg for this"""
        try:
            subprocess.call(['rm', f'{save_path}_avcombined.mp4'])

            subprocess.call([
                'ffmpeg',
                '-i',
                f'{mp4_file_path}',
                '-i',
                f'{wav_file_path}',
                '-c:v',
                'copy',
                '-c:a',
                'aac',
                f'{save_path}_avcombined.mp4',
            ])
        except subprocess.CalledProcessError as e:
            logger.error(e)
        logger.info("Done saving video file")


class binaural_dynamic_rendering(dynamic_rendering_moving_receiver):
    """
    Class for binaural dynamic rendering with moving listener and rotating head.
    Directional, position-dependent common slope amplitudes learned by the DNN
    """

    def __init__(self,
                 room_dataset: SpatialRoomDataset,
                 rec_pos_list: NDArray,
                 orientation_list: NDArray,
                 stimulus: ArrayLike,
                 hrtf_reader: HRIRSOFAReader,
                 update_ms: float = 100,
                 use_whole_rir: bool = False):
        """
        Create sound examples of rendered stimuli (and optionally, animation) of a receiver moving through a room.
        Args:
            room_dataset: SpatialRoomDataset object containing the SRIRs and the source-receiver configuration
            rec_pos_list: List of receiver positions to navigate through, of size num_pos, 3
            orientation_list: List of head orientations to navigate through, of size num_pos, 2, az, el in radians
            stimulus: dry source, mono
            hrtf_reader(HRIRSOFAReader): reader object for the HRTF dataset
            update_ms: How often is the receiver position updated
            use_whole_rir (bool): whether to use the whole RIR or the late part only for rendering?

        """
        super().__init__(room_dataset, rec_pos_list, stimulus, update_ms)
        # split RIRs in room dataset into early and late parts
        self.use_whole_rir = use_whole_rir
        # list of head orientations
        self.orientation_list = orientation_list
        # negate the elevation angle to represent pitch
        self.orientation_list[:, -1] = -self.orientation_list[:, -1]
        self.num_out_channels = 2
        assert self.orientation_list.shape[0] == self.rec_pos_list.shape[
            0], "Number of orientations must match number of listener positions"
        # initialise HRTF reader
        self.hrtf_reader = hrtf_reader
        # if there is a sampling rate mismatch
        if self.hrtf_reader.fs != room_dataset.sample_rate:
            logger.info(
                f"Resampling HRTFs to {room_dataset.sample_rate:.0f} Hz")
            self.hrtf_reader.resample_hrirs(room_dataset.sample_rate)
        # size is num_ambi_channels x num_receivers x num_time_samples
        self.hrir_sh = self.hrtf_reader.get_spherical_harmonic_representation(
            self.room.ambi_order)

        self.initialise_filters_in_frequency_domain()

    def initialise_filters_in_frequency_domain(self,
                                               use_whole_rir: bool = False):
        """Get the FFTs of the HRIRs and the ambi RIRs"""
        self.num_freq_bins = 2**int(np.ceil(np.log2(self.room.rir_length)))

        # these are of shape num_pos x num_ambi_channels x num_freq_samples
        if use_whole_rir:
            self.ambi_rtfs = rfft(self.room.rirs,
                                  n=self.num_freq_bins,
                                  axis=-1)
        else:
            self.room.early_late_split(win_len_ms=10)
            self.ambi_rtfs = rfft(self.room.late_rirs,
                                  n=self.num_freq_bins,
                                  axis=-1)

        # these are of shape num_ambi_channels x 2 x num_freq_samples
        self.ambi_hrtfs = rfft(self.hrir_sh, n=self.num_freq_bins, axis=-1)
        logger.info("Done calculating FFTs")

    def get_binaural_rir(self,
                         cur_head_orientation: Tuple,
                         rec_pos_idx: int,
                         alpha: float = 0.5) -> NDArray:
        """
        For a particular head orientation and receiver position, calculate the binaural RIR
        Args:
            cur_head_orientation (tuple): tuple containing the yaw, pitch, 
                                          roll values of head orientation in radians

            rec_pos_idx (int): current receiver index
            alpha (float): weighting factor between current and previous head orientation
        Returns:
            NDArray: rir_length x 2 binaural RIR in time domain
        """
        # shape is num_ambi_channels x num_freqs
        cur_ambi_rtf = self.ambi_rtfs[rec_pos_idx, ...]

        #rotate the soundfield in the opposite direction - size num_freq_bins x num_ambi_channels
        cur_rotation_matrix = spa.sph.sh_rotation_matrix(
            self.room.ambi_order,
            -cur_head_orientation[0],
            -cur_head_orientation[1],
            0,
            sh_type='real')
        # interpolate rotation matrices to avoid ITD discontinuities
        if hasattr(self, 'prev_rotation_matrix'):
            weighted_rotation_matrix = alpha * cur_rotation_matrix + (
                1 - alpha) * self.prev_rotation_matrix
        else:
            weighted_rotation_matrix = cur_rotation_matrix

        if hasattr(self, 'prev_ambi_rtf'):
            weighted_ambi_rtf = alpha * cur_ambi_rtf + (
                1 - alpha) * self.prev_ambi_rtf
        else:
            weighted_ambi_rtf = cur_ambi_rtf

        rotated_ambi_rtf = weighted_ambi_rtf.T @ weighted_rotation_matrix.T

        # get the binaural room transfer function
        cur_brtf = np.einsum('nrf, fn -> fr', np.conj(self.ambi_hrtfs),
                             rotated_ambi_rtf)
        # get the BRIR
        cur_brir = irfft(cur_brtf, n=self.num_freq_bins, axis=0)
        self.prev_rotation_matrix = cur_rotation_matrix.copy()
        self.prev_ambi_rtf = cur_ambi_rtf.copy()
        return cur_brir

    def binaural_filter_overlap_add(self):
        """Filter and cross-fade the stimulus with the RIRs associated with the moving listener."""
        output_signal = np.zeros(
            (len(self.extended_stimulus), self.num_out_channels))
        fade_len_ms = self.update_ms
        fade_len = ms_to_samps(fade_len_ms, self.sample_rate)
        fade_out = self.get_fade_windows(fade_len,
                                         fade_out=True,
                                         uncorr_fade=True)
        fade_in = self.get_fade_windows(fade_len,
                                        fade_out=False,
                                        uncorr_fade=True)
        prev_tail = np.zeros((fade_len, self.num_out_channels))

        for k in range(self.num_pos):
            b_idx = np.arange(k * self.hop_size,
                              min((k + 1) * self.hop_size, self.total_sim_len),
                              dtype=np.int32)

            cur_stimulus = self.extended_stimulus[b_idx]
            cur_head_orientation = self.orientation_list[k]

            cur_brir = self.get_binaural_rir(cur_head_orientation, k)
            cur_filter = cur_brir

            start_idx = k * self.hop_size

            # loop over both ears
            for j in range(self.num_out_channels):
                cur_filtered_signal = fftconvolve(cur_stimulus,
                                                  cur_filter[..., j],
                                                  mode='full')

                end_idx = min(start_idx + len(cur_filtered_signal),
                              output_signal.shape[0])

                cur_trunc_filtered_signal = cur_filtered_signal[:end_idx -
                                                                start_idx]
                # Crossfade overlapping samples between previous tail and current head
                if k > 0:
                    overlap_len = min(fade_len, len(cur_trunc_filtered_signal))
                    crossfaded = prev_tail[:overlap_len, j] * fade_out[:overlap_len] + \
                                 cur_trunc_filtered_signal[:overlap_len] * fade_in[:overlap_len]

                    output_signal[start_idx:start_idx + overlap_len,
                                  j] += crossfaded

                    output_signal[start_idx + overlap_len:end_idx,
                                  j] += cur_trunc_filtered_signal[overlap_len:]
                else:
                    output_signal[start_idx:end_idx,
                                  j] += cur_trunc_filtered_signal

                # Save tail for next iteration
                if len(cur_trunc_filtered_signal) >= fade_len:
                    prev_tail[:fade_len,
                              j] = cur_trunc_filtered_signal[-fade_len:]
                else:
                    prev_tail[:len(cur_trunc_filtered_signal),
                              j] = cur_trunc_filtered_signal

        return output_signal

    def animate_moving_listener(self, save_path: Optional[str] = None):
        """Animate the moving listener and their head orientation"""
        super().animate_moving_listener(save_path, self.orientation_list[:, 0])
