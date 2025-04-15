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

from diff_gfdn.dnn import OneHotEncoding
from diff_gfdn.plot import order_position_matrices
from diff_gfdn.save_results import save_loss
from diff_gfdn.utils import db, db2lin, ms_to_samps, samps_to_ms

from .config import DNNType, SpatialSamplingConfig
from .dataloader import load_dataset, parse_room_data, SpatialRoomDataset
from .model import (
    Directional_Beamforming_Weights,
    Directional_Beamforming_Weights_from_CNN,
    Directional_Beamforming_Weights_from_MLP,
    Omni_Amplitudes_from_MLP,
)
from .trainer import SpatialSamplingTrainer

# pylint: disable=E0606, E0601
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
        self.train_dataset, _, _ = load_dataset(
            room_data,
            config_dict.device,
            grid_resolution_m=room_data.grid_spacing_m,
            network_type=config_dict.network_type,
            batch_size=config_dict.batch_size,
            shuffle=False,
        )

        # get the reference output
        self.src_pos = np.array(self.room_data.source_position).squeeze()
        self.true_points = self.room_data.receiver_position
        self.true_amps = self.room_data.amplitudes
        self.one_hot_encoder = OneHotEncoding()
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

    def get_model_output(self, num_epochs: int,
                         grid_spacing_m: float) -> Tuple[NDArray, NDArray]:
        """
        Get the estimated common slope amplitudes.
        Returns the positions and the amplitudes at those positions
        """
        # load the trained weights for the particular epoch
        checkpoint = torch.load(Path(
            f'{self.config_dict.train_dir}/checkpoints/grid_resolution={grid_spacing_m:.1f}/model_e{num_epochs - 1}.pt'
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

                if isinstance(self.model, Directional_Beamforming_Weights):
                    cur_est_amps = self.model.get_directional_amplitudes()
                    if isinstance(self.model,
                                  Directional_Beamforming_Weights_from_CNN):
                        # shape H, W, 2
                        cur_est_mesh = data['mesh_2D']
                        # find points in the meshgrid closest to current receiver points
                        # shape B
                        _, _, closest_points_idx = self.one_hot_encoder(
                            cur_est_mesh, position)
                        # check if this works correctly
                        assert np.allclose(
                            cur_est_mesh.reshape(-1, 2)[closest_points_idx, :],
                            position[:, :2])
                        # sample the closest points from the amplitudes
                        cur_est_amps = cur_est_amps[closest_points_idx, ...]

                else:
                    cur_est_amps = model_output
                est_pos = np.vstack((est_pos, position))
                est_amps = np.vstack((est_amps, cur_est_amps))

        return est_pos, est_amps

    def plot_beamformer_output(self,
                               est_amps: NDArray,
                               filename: str,
                               contour_plot: bool = True):
        """
        Plot beamformer output as function of elevation and azimuth angles
        est_amps (NDArray): amplitudes estimated by the DNN
        filename (str): filename for saving
        contour_plot (bool): whether to plot spherical or contour plot
        """
        # Create grid of elevation and azimuth angles
        num_azi = 10
        num_el = 10
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
        fig, ax = plt.subplots(
            self.room_data.num_rooms,
            1,
            # subplot_kw={'projection': '3d'},
            figsize=(6, 3 * self.room_data.num_rooms))

        # spherical harmonic interpolation
        sph_matrix_orig = spa.sph.sh_matrix(
            self.room_data.ambi_order,
            self.room_data.sph_directions[0, :],
            self.room_data.sph_directions[1, :],
            sh_type='real')

        sph_matrix_dense = spa.sph.sh_matrix(self.room_data.ambi_order,
                                             np.degrees(azimuth_grid).ravel(),
                                             np.degrees(polar_grid).ravel(),
                                             sh_type='real')

        # project on original spherical harmonic matrix
        weights = np.einsum('bjk, jn -> bkn', est_amps, sph_matrix_orig)
        # retrieve the amplitudes by projecting on denser spherical grid
        amps_interp = np.einsum('bkn, nd -> bdk', weights, sph_matrix_dense.T)
        num_row, num_col = azimuth_grid.shape

        for k in range(self.room_data.num_rooms):
            amps_mean_interp = np.mean(amps_interp[..., k],
                                       axis=0).reshape(num_row, num_col)

            # Plot the ellipsoid surface with beamforming weights as color values
            if contour_plot:
                surf = ax[k].contourf(np.degrees(azimuth_grid),
                                      np.degrees(polar_grid),
                                      db(amps_mean_interp, is_squared=True),
                                      cmap='plasma')
                ax[k].set_xlabel('Azimuth angles')
                ax[k].set_ylabel('Polar angles')
            else:
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
                ax[k].set_xlabel('X')
                ax[k].set_ylabel('Y')
                ax[k].set_zlabel('Z)')

            # Add a colorbar
            cbar = fig.colorbar(surf, ax=ax[k], shrink=0.8, aspect=5)
            cbar.set_label('dB')
            ax[k].set_title(f'Group = {k+1}')

        fig.subplots_adjust(hspace=0.4)
        fig.savefig(Path(f'{self.config_dict.train_dir}/{filename}').resolve())

    def plot_amplitudes_in_space(self,
                                 grid_resolution_m: float,
                                 est_amps: NDArray,
                                 est_points: NDArray,
                                 verbose: bool = False):
        """
        Plot the true and learned amplitudes as a function of space
        Args:
            grid_resolution_m (float): resolution of the uniform grid used for training
            est_amps (NDArray): estimated omni (num_pos, num_groups) / 
                                directional amplitudes from NN (num_pos, num_directions, num_groups)
            est_points (NDArray): receiver positions at which the amplitudes were estimatied
            verbose (bool): whether to print out mean and std of amplitudes
        """
        logger.info("Making amplitude plots")
        db_limits = np.zeros((2, self.room_data.num_rooms))

        if self.true_amps.ndim == 2:
            db_limits[0, :] = np.min(db(self.true_amps, is_squared=True),
                                     axis=0)
            db_limits[1, :] = np.max(db(self.true_amps, is_squared=True),
                                     axis=0)
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
                if verbose:
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
        """
        Plot the error between the CS EDC and MLP EDC in space
        Args:
            grid_resolution_m (float): resolution of the uniform grid used for training
            est_amps (NDArray): estimated omni (num_pos, num_groups) / 
                                directional amplitudes from NN (num_pos, num_directions, num_groups)
            est_points (NDArray): receiver positions at which the amplitudes were estimatied
        """
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
                scatter_plot=True,
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
                    scatter_plot=True,
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

    config_dict = config_dict.model_copy(
        update={'use_directional_rirs': room_data.sph_directions is not None})

    # are we learning OMNI amplitudes or directional amplitudes?
    if config_dict.use_directional_rirs:
        if config_dict.network_type == DNNType.MLP:
            logger.info("Using MLP for training")
            model = Directional_Beamforming_Weights_from_MLP(
                room_data.num_rooms,
                room_data.ambi_order,
                config_dict.dnn_config.num_fourier_features,
                config_dict.dnn_config.mlp_config.num_hidden_layers,
                config_dict.dnn_config.mlp_config.num_neurons_per_layer,
                desired_directions=room_data.sph_directions,
                beamformer_type=config_dict.dnn_config.beamformer_type,
                device=config_dict.device,
            )
        elif config_dict.network_type == DNNType.CNN:
            logger.info("Using CNN for training")
            model = Directional_Beamforming_Weights_from_CNN(
                room_data.num_rooms,
                room_data.ambi_order,
                config_dict.dnn_config.num_fourier_features,
                config_dict.dnn_config.cnn_config.num_hidden_channels,
                config_dict.dnn_config.cnn_config.num_layers,
                config_dict.dnn_config.cnn_config.kernel_size,
                desired_directions=room_data.sph_directions,
                beamformer_type=config_dict.dnn_config.beamformer_type,
                device=config_dict.device)
    else:
        model = Omni_Amplitudes_from_MLP(
            room_data.num_rooms,
            config_dict.dnn_config.num_fourier_features,
            config_dict.dnn_config.mlp_config.num_hidden_layers,
            config_dict.dnn_config.mlp_config.num_neurons_per_layer,
            device=config_dict.device,
            gain_limits=(db2lin(-100), db2lin(0)),
        )

    # set default device
    torch.set_default_device(config_dict.device)
    # move model to device (cuda or cpu)
    model = model.to(config_dict.device)
    # plot object
    plot_obj = make_plots(room_data, config_dict, model)

    if isinstance(model, Directional_Beamforming_Weights_from_MLP):
        plot_obj.plot_beamformer_output(
            room_data.amplitudes, filename='true_directional_amplitudes.png')

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
            f"Training DNN for grid resolution = {np.round(grid_resolution_m[k], 1)} m"
        )

        # go back to training mode from evaluation mode
        model.train()

        # prepare the training and validation data for DiffGFDN
        train_dataset, valid_dataset, dataset_ref = load_dataset(
            room_data,
            config_dict.device,
            grid_resolution_m=np.round(grid_resolution_m[k], 1),
            network_type=config_dict.network_type,
            batch_size=config_dict.batch_size)

        # create the trainer object
        trainer = SpatialSamplingTrainer(
            model,
            config_dict,
            grid_spacing_m=grid_resolution_m[k],
            common_decay_times=room_data.common_decay_times,
            # receiver_positions=room_data.receiver_position,
            sampling_rate=room_data.sample_rate,
            ir_len_ms=samps_to_ms(room_data.rir_length, room_data.sample_rate),
            dataset_ref=dataset_ref,
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
            len(trainer_loss[grid_resolution_m[k]]), grid_resolution_m[k])

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
