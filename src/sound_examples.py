import subprocess
from typing import List, Optional

from loguru import logger
from matplotlib import animation, patches
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pyloudnorm as pyln
from scipy.signal import fftconvolve
from slope2noise.rooms import RoomGeometry

from diff_gfdn.dataloader import RoomDataset
from diff_gfdn.utils import ms_to_samps

# pylint: disable=R1707


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
    def win_size(self) -> int:
        """Window size for each update"""
        assert self.update_len_samp % 2 == 0
        return 2 * self.update_len_samp

    @property
    def hop_size(self) -> int:
        """Hop size for each update"""
        return self.update_len_samp

    @property
    def window(self) -> ArrayLike:
        """Window for fade in and fade out"""
        return np.hanning(self.win_size)

    @property
    def total_sim_len(self) -> int:
        """Length of the total simulation"""
        return self.num_pos * self.hop_size

    @property
    def rec_idxs(self) -> List:
        """Indices of the receivers in the dataset associated with the list of receiver positions"""
        # Compute Euclidean distance between each array in array_list_np and every row in matrix

        distances = np.linalg.norm(self.room.receiver_position[:, None, :] -
                                   self.rec_pos_list,
                                   axis=2)
        indices = np.argmin(distances, axis=0)
        return indices

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

    def filter_overlap_add(self, use_whole_rir: bool = False) -> NDArray:
        """Filter and cross-fade the stimulus with the RIRs associated with the moving listener."""
        output_signal = np.zeros_like(self.extended_stimulus)

        for k in range(self.num_pos):
            b_idx = np.arange(k * self.hop_size,
                              min(k * self.hop_size + self.win_size,
                                  self.total_sim_len),
                              dtype=np.int32)
            if use_whole_rir:
                cur_filter = self.rirs[k, :]
            else:
                cur_filter = self.late_rirs[k, :]

            cur_stimulus = self.extended_stimulus[b_idx]
            cur_filtered_signal = fftconvolve(cur_stimulus,
                                              cur_filter,
                                              mode='full')
            cur_win_size = min(self.win_size, len(cur_filtered_signal),
                               len(b_idx))
            cur_filtered_signal_trunc = cur_filtered_signal[:cur_win_size]
            output_signal[
                b_idx] += self.window[:cur_win_size] * cur_filtered_signal_trunc

        return output_signal

    def animate_moving_listener(self, save_path: Optional[str] = None):
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

        def update(frame):
            """Update function for animation"""
            circle.set_center(
                (x_vals[frame], y_vals[frame]))  # Move the circle
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
