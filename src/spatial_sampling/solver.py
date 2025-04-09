from copy import deepcopy
import os
from pathlib import Path
from typing import Tuple, Union

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from slope2noise.rooms import RoomGeometry
from slope2noise.utils import decay_kernel
import spaudiopy as spa
import torch

from diff_gfdn.plot import order_position_matrices
from diff_gfdn.save_results import save_loss
from diff_gfdn.utils import db, db2lin, ms_to_samps, samps_to_ms

from .config import SpatialSamplingConfig
from .dataloader import load_dataset, parse_room_data, SpatialRoomDataset
from .model import Directional_Beamforming_Weights_from_MLP, Omni_Amplitudes_from_MLP
from .trainer import SpatialSamplingTrainer

# pylint: disable=E0606
# flake8: noqa:E231


class make_plots:
    """Class for making plots"""

    def __init__(self, room_data: SpatialRoomDataset,
                 config_dict: SpatialSamplingConfig,
                 model: Union[Omni_Amplitudes_from_MLP,
                              Directional_Beamforming_Weights_from_MLP]):
        """
        Initialise parameters for the class
        Args:
        room_data (SpatialRoomDataset): object of SpatialRoomDataset dataclass
        config_dict (SpatialSamplingConfig): config file, read as dictionary
        model (Omni_Amplitudes_from_MLP): the NN model to be tested
        """
        self.room_data = deepcopy(room_data)
        self.model = deepcopy(model)
        self.config_dict = deepcopy(config_dict)
        self.room = RoomGeometry(room_data.sample_rate,
                                 room_data.num_rooms,
                                 np.array(room_data.room_dims),
                                 np.array(room_data.room_start_coord),
                                 aperture_coords=room_data.aperture_coords)

        # prepare the training and validation data
        self.train_dataset, _ = load_dataset(
            room_data,
            config_dict.device,
            grid_resolution_m=room_data.grid_spacing_m,
            batch_size=config_dict.batch_size,
            shuffle=False,
        )

        # get the reference output
        self.src_pos = np.array(self.room_data.source_position).squeeze()
        self.true_points = self.room_data.receiver_position
        self.true_amps = self.room_data.amplitudes
        self._init_decay_kernel()

    def _init_decay_kernel(self):
        """Initialise the decay kernels for calculating EDC errors"""
        num_slopes = self.room_data.num_rooms
        edc_len_samps = ms_to_samps(2000, self.room_data.sample_rate)
        self.envelopes = np.zeros((num_slopes, edc_len_samps))
        time_axis = np.linspace(0, (edc_len_samps - 1) /
                                self.room_data.sample_rate, edc_len_samps)

        for k in range(num_slopes):
            self.envelopes[k, :] = decay_kernel(np.expand_dims(
                self.room_data.common_decay_times[:, k], axis=-1),
                                                time_axis,
                                                self.room_data.sample_rate,
                                                normalize_envelope=True,
                                                add_noise=False).squeeze()

    def get_model_output(self, num_epochs: int) -> Tuple[NDArray, NDArray]:
        """
        Get the estimated common slope amplitudes.
        Returns the positions and the amplitudes at those positions
        """
        # load the trained weights for the particular epoch
        checkpoint = torch.load(Path(
            f'{self.config_dict.train_dir}/checkpoints/model_e{num_epochs - 1}.pt'
        ).resolve(),
                                weights_only=True,
                                map_location=torch.device('cpu'))
        # Load the trained model state
        self.model.load_state_dict(checkpoint, strict=False)

        # run the model in eval mode
        self.model.eval()

        est_pos = np.empty((0, 3))
        est_amps = np.empty((0, self.room_data.num_rooms)) if isinstance(
            self.model, Omni_Amplitudes_from_MLP) else np.empty(
                (0, self.room_data.num_directions, self.room_data.num_rooms))
        with torch.no_grad():
            for data in self.train_dataset:
                position = data['listener_position']
                model_output = self.model(data)
                print(
                    'Standard deviation across spatial position of beamformer weights:'
                    +
                    f' {np.round(np.std(model_output.detach().numpy(), axis=0), 3)}'
                )
                if isinstance(self.model,
                              Directional_Beamforming_Weights_from_MLP):
                    cur_est_amps = self.model.get_directional_amplitudes(
                        data['sph_directions'])
                else:
                    cur_est_amps = model_output.copy()
                est_pos = np.vstack((est_pos, position))
                est_amps = np.vstack((est_amps, cur_est_amps))
        return est_pos, est_amps

    def plot_beamformer_output(self, est_amps: NDArray, filename: str):
        """
        Plot beamformer output as function of elevation and azimuth angles
        filename (str): filename for saving
        """
        # Create grid of elevation and azimuth angles
        num_azi = 20
        num_el = 20
        azimuths = np.degrees(np.linspace(0, 2 * np.pi, num_azi))
        elevations = np.degrees(np.linspace(-np.pi / 2, np.pi / 2, num_el))
        polars = 90 - elevations

        azimuth_grid, polar_grid = np.meshgrid(np.deg2rad(azimuths),
                                               np.deg2rad(polars))
        elevation_grid = np.pi / 2 - polar_grid
        x = np.cos(elevation_grid) * np.sin(azimuth_grid)
        y = np.cos(elevation_grid) * np.cos(azimuth_grid)
        z = np.sin(elevation_grid)

        # Plotting beamforming weights as a spherical surface
        fig, ax = plt.subplots(self.room_data.num_rooms,
                               1,
                               subplot_kw={'projection': '3d'},
                               figsize=(6, 3 * self.room_data.num_rooms))

        # spherical harmonic interpolation
        sph_matrix = spa.sph.sh_matrix(self.room_data.ambi_order,
                                       self.room_data.sph_directions[0, :],
                                       self.room_data.sph_directions[1, :],
                                       sh_type='real')

        sph_matrix_dense = spa.sph.sh_matrix(self.room_data.ambi_order,
                                             np.degrees(azimuth_grid).ravel(),
                                             np.degrees(polar_grid).ravel(),
                                             sh_type='real')

        weights = np.einsum('bjk, jn -> bkn', est_amps, sph_matrix)
        amps_interp = np.einsum('bkn, nd -> bdk', weights, sph_matrix_dense.T)
        num_row, num_col = azimuth_grid.shape

        for k in range(self.room_data.num_rooms):
            amps_mean_interp = amps_interp[0, :, k].reshape(num_row, num_col)

            # Plot the ellipsoid surface with beamforming weights as color values
            surf = ax[k].plot_surface(
                x,
                y,
                z,
                facecolors=plt.cm.viridis(amps_mean_interp /
                                          amps_mean_interp.max()),
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=False,
                alpha=0.5,
            )

            # Add a colorbar
            fig.colorbar(surf, ax=ax[k], shrink=0.5, aspect=5)
            ax[k].set_xlabel('X')
            ax[k].set_ylabel('Y')
            ax[k].set_zlabel('Z)')
            ax[k].set_title(f'Group = {k+1}')

        fig.subplots_adjust(hspace=0.3)
        fig.savefig(Path(f'{self.config_dict.train_dir}/{filename}').resolve())

    def plot_amplitudes_in_space(self, grid_resolution_m: float,
                                 est_amps: NDArray, est_points: NDArray):
        """Plot the true and learned amplitudes as a function of space"""
        logger.info("Making amplitude plots")
        db_limits = np.zeros((2, self.room_data.num_rooms))

        if self.true_amps.ndim == 2:
            db_limits[0, :] = np.min(db(self.true_amps, is_squared=True),
                                     axis=-1)
            db_limits[1, :] = np.max(db(self.true_amps, is_squared=True),
                                     axis=-1)
            # the amplitudes are omni directional
            self.room.plot_amps_at_receiver_points(
                self.true_points,
                self.src_pos,
                self.true_amps.T,
                scatter_plot=False,
                save_path=Path(
                    f'{self.config_dict.train_dir}/actual_amplitudes_in_space.png'
                ).resolve(),
                title='Common slopes',
                db_limits=db_limits)

            self.room.plot_amps_at_receiver_points(
                est_points,
                self.src_pos,
                est_amps.T,
                scatter_plot=False,
                save_path=Path(
                    f'{self.config_dict.train_dir}/learnt_amplitudes_in_space_'
                    + f'grid_resolution_m={np.round(grid_resolution_m, 3)}.png'
                ).resolve(),
                title=
                f'Training grid resolution={np.round(grid_resolution_m, 3)}m',
                db_limits=db_limits)

        else:
            # the amplitudes are direction dependent
            for j in range(self.room_data.num_directions):
                print(
                    f'Actual amplitudes mean : {np.round(self.true_amps[:, j, :].mean(axis=0), 3)},'
                    +
                    f'Est amplitudes mean: {np.round(est_amps[:, j, :].mean(axis=0), 3)} for direction {j}'
                )
                print(
                    f'Actual amplitudes STD: {np.round(self.true_amps[:, j, :].std(axis=0),3)},'
                    +
                    f'est amplitudes STD: {np.round(est_amps[:, j, :].std(axis=0), 3)} for direction {j}'
                )

                db_limits[0, :] = np.min(db(self.true_amps[:, j, :],
                                            is_squared=True),
                                         axis=0)
                db_limits[1, :] = np.max(db(self.true_amps[:, j, :],
                                            is_squared=True),
                                         axis=0)
                dir_string = (
                    f'az = {self.room_data.sph_directions[0, j]:.2f} deg, ' +
                    f' pol = {self.room_data.sph_directions[1, j]:.2f} deg')

                directory = Path(
                    f'{self.config_dict.train_dir}/direction={j+1}').resolve()
                if not os.path.exists(directory):
                    os.makedirs(directory)

                self.room.plot_amps_at_receiver_points(
                    self.true_points,
                    self.src_pos,
                    self.true_amps[:, j, :].T,
                    scatter_plot=False,
                    save_path=f'{directory}/actual_amplitudes_in_space.png',
                    title='Common slopes, ' + dir_string,
                    db_limits=db_limits)

                self.room.plot_amps_at_receiver_points(
                    est_points,
                    self.src_pos,
                    est_amps[:, j, :].T,
                    scatter_plot=False,
                    save_path=f'{directory}/learnt_amplitudes_in_space_' +
                    f'grid_resolution_m={np.round(grid_resolution_m, 3)}.png',
                    title=
                    f'Training grid resolution={np.round(grid_resolution_m, 3)}m, '
                    + dir_string,
                    db_limits=db_limits)

    def plot_edc_error_in_space(self, grid_resolution_m: float,
                                est_amps: NDArray, est_points: NDArray):
        """Plot the error between the CS EDC and MLP EDC in space"""
        logger.info("Making EDC error plots")

        # order the position indices in the estimated data according to the
        # reference dataset
        ordered_pos_idx = order_position_matrices(self.true_points, est_points)

        if self.true_amps.ndim == 2:
            original_edc = db(np.sum(np.einsum('bk, kt -> bkt', self.true_amps,
                                               self.envelopes),
                                     axis=1),
                              is_squared=True)
            est_edc = db(np.sum(np.einsum('bk, kt -> bkt', est_amps,
                                          self.envelopes),
                                axis=1),
                         is_squared=True)

            error_db = np.mean(np.abs(original_edc -
                                      est_edc[ordered_pos_idx, ...]),
                               axis=-1)
            self.room.plot_edc_error_at_receiver_points(
                self.true_points,
                self.src_pos,
                db2lin(error_db),
                scatter_plot=False,
                cur_freq_hz=None,
                save_path=Path(
                    f'{self.config_dict.train_dir}/edc_error_in_space_' +
                    f'grid_resolution_m={np.round(grid_resolution_m, 3)}.png').
                resolve(),
                title=
                f'Training grid resolution={np.round(grid_resolution_m, 3)}m')
        else:
            original_edc = db(np.einsum('bjk, kt -> bjt', self.true_amps,
                                        self.envelopes),
                              is_squared=True)
            est_edc = db(np.einsum('bjk, kt -> bjt', est_amps, self.envelopes),
                         is_squared=True)

            error_db = np.mean(np.abs(original_edc -
                                      est_edc[ordered_pos_idx, ...]),
                               axis=-1)

            for j in range(self.room_data.num_directions):
                self.room.plot_edc_error_at_receiver_points(
                    self.true_points,
                    self.src_pos,
                    db2lin(error_db[:, j]),
                    scatter_plot=False,
                    cur_freq_hz=None,
                    save_path=Path(
                        f'{self.config_dict.train_dir}/direction={j+1}/edc_error_in_space_'
                        +
                        f'grid_resolution_m={np.round(grid_resolution_m, 3)}.png'
                    ).resolve(),
                    title=
                    f'Training grid resolution={np.round(grid_resolution_m, 3)}m, '
                    + f'az = {self.room_data.sph_directions[0, j]:.2f} deg,' +
                    f' pol = {self.room_data.sph_directions[1, j]:.2f} deg')


############################################################################


def run_training_spatial_sampling(config_dict: SpatialSamplingConfig):
    """
    Run the training to test for spatial sampling resolution
    Returns:
        A list of ColorlessFDNResults dataclass, each for one FDN in the GFDN
    """
    logger.info("Training the MLP to learn spatial mappings")
    if "3room_FDTD" in config_dict.room_dataset_path:
        # read the coupled room dataset
        room_data = parse_room_data(
            Path(config_dict.room_dataset_path).resolve())
    else:
        logger.error("Currently only the three room dataset is supported")

    # are we learning OMNI amplitudes or directional amplitudes?
    if room_data.sph_directions is None:
        model = Omni_Amplitudes_from_MLP(
            room_data.num_rooms,
            config_dict.num_fourier_features,
            config_dict.num_hidden_layers,
            config_dict.num_neurons_per_layer,
            config_dict.encoding_type,
            device=config_dict.device,
            gain_limits=(db2lin(-100), db2lin(0)),
        )

    else:
        model = Directional_Beamforming_Weights_from_MLP(
            room_data.num_rooms,
            room_data.ambi_order,
            config_dict.num_fourier_features,
            config_dict.num_hidden_layers,
            config_dict.num_neurons_per_layer,
            config_dict.encoding_type,
            device=config_dict.device,
        )

    # set default device
    torch.set_default_device(config_dict.device)
    # move model to device (cuda or cpu)
    model = model.to(config_dict.device)
    # plot object
    plot_obj = make_plots(room_data, config_dict, model)

    plot_obj.plot_beamformer_output(room_data.amplitudes,
                                    filename='true_directional_amplitudes.png')

    # at least one mic in each room
    assert config_dict.num_grid_spacing * room_data.grid_spacing_m <= np.min(
        np.asarray(room_data.room_dims)[:, :2]
    ), "Reduce number of grid spacing points to have at least one mic in each room"

    grid_resolution_m = np.arange(config_dict.num_grid_spacing, 0,
                                  -1) * room_data.grid_spacing_m

    # dictionary contains training loss for each grid_resolution of size num_epochs
    trainer_loss = {}
    valid_loss = {}
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))

    for k in range(config_dict.num_grid_spacing):
        logger.info(
            f"Training MLP for grid resolution = {np.round(grid_resolution_m[k], 1)} m"
        )

        # go back to training mode from evaluation mode
        model.train()

        # prepare the training and validation data for DiffGFDN
        train_dataset, valid_dataset = load_dataset(
            room_data,
            config_dict.device,
            np.round(grid_resolution_m[k], 1),
            config_dict.batch_size,
        )

        # create the trainer object
        trainer = SpatialSamplingTrainer(
            model,
            config_dict,
            common_decay_times=room_data.common_decay_times,
            receiver_positions=room_data.receiver_position,
            sampling_rate=room_data.sample_rate,
            ir_len_ms=samps_to_ms(room_data.rir_length, room_data.sample_rate),
        )

        # train the network
        trainer.train(train_dataset, valid_dataset)
        # save train loss evolution
        save_loss(trainer.train_loss,
                  config_dict.train_dir +
                  f"grid_resolution={np.round(grid_resolution_m[k], 3)}",
                  save_plot=True,
                  filename='training_loss_vs_epoch')

        # save the validation loss
        save_loss(
            trainer.valid_loss,
            config_dict.train_dir +
            f"grid_resolution={np.round(grid_resolution_m[k], 3)}",
            save_plot=True,
            filename='valid_loss_vs_position',
            xaxis_label='Position #',
        )

        trainer_loss[grid_resolution_m[k]] = trainer.train_loss
        valid_loss[grid_resolution_m[k]] = trainer.valid_loss

        # plot the loss as a function of the grid_resolution
        ax[0].semilogy(
            np.arange(len(trainer_loss[grid_resolution_m[k]])),
            trainer_loss[grid_resolution_m[k]],
            label=f'Grid resolution = {np.round(grid_resolution_m[k], 3)}m')
        ax[1].semilogy(
            np.arange(len(trainer_loss[grid_resolution_m[k]])),
            valid_loss[grid_resolution_m[k]],
            label=f'Grid resolution = {np.round(grid_resolution_m[k], 3)}m')

        del trainer

        # get the model output
        est_points, est_amps = plot_obj.get_model_output(
            len(trainer_loss[grid_resolution_m[k]]))

        # make plots
        if isinstance(model, Directional_Beamforming_Weights_from_MLP):
            plot_obj.plot_beamformer_output(
                est_amps,
                filename='learned_directional_amplitudes ' +
                f'grid_resolution_m={np.round(grid_resolution_m[k], 3)}.png')

        plot_obj.plot_amplitudes_in_space(
            grid_resolution_m[k],
            est_amps,
            est_points,
        )

        plot_obj.plot_edc_error_in_space(
            grid_resolution_m[k],
            est_amps,
            est_points,
        )

    ax[0].set_xlabel('Epoch #')
    ax[0].set_ylabel('Training loss (log)')
    ax[1].set_xlabel('Epoch #')
    ax[1].set_ylabel('Validation loss (log)')
    ax[1].legend(loc='best', bbox_to_anchor=(1.1, 0.5))
    fig.savefig(os.path.join(config_dict.train_dir,
                             'loss_vs_grid_resolution.png'),
                bbox_inches="tight")
