import os
from pathlib import Path
import pickle

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from slope2noise.rooms import RoomGeometry
import torch

from ..dataloader import RoomDataset
from ..gain_filters import Gains_from_MLP
from ..save_results import save_loss
from ..utils import db2lin, samps_to_ms
from .config import SpatialSamplingConfig
from .dataloader import load_dataset
from .trainer import SpatialSamplingTrainer

# pylint: disable=E0606


def parse_room_data(filepath: str):
    """Read the dataset at filepath and return a RoomDataset object"""
    assert str(filepath).endswith('.pkl'), "provide the path to the .pkl file"
    # read contents from pkl file
    try:
        logger.info('Reading pkl file ...')
        with open(filepath, 'rb') as f:
            srir_mat = pickle.load(f)
            sample_rate = srir_mat['fs']
            source_position = srir_mat['srcPos'].T
            receiver_position = srir_mat['rcvPos'].T
            rirs = np.squeeze(srir_mat['srirs'])
            band_centre_hz = srir_mat['band_centre_hz']
            common_decay_times = srir_mat['common_decay_times']
            amplitudes = srir_mat['amplitudes'].T
            amplitudes_norm = srir_mat['amplitudes_norm'].T
            noise_floor = srir_mat['noise_floor'].T
            noise_floor_norm = srir_mat['noise_floor_norm'].T
    except Exception as exc:
        raise FileNotFoundError("pickle file not read correctly") from exc

    logger.info("Done reading pkl file")
    # number of rooms in dataset
    num_rooms = 3
    # uniform absorption coefficients of the three rooms
    absorption_coeffs = np.array([0.2, 0.01, 0.1])
    # (x,y) dimensions of the 3 rooms
    room_dims = [(4.0, 8.0, 3.0), (6.0, 3.0, 3.0), (4.0, 8.0, 3.0)]
    # this denotes the 3D position of the first vertex of the floor
    room_start_coord = [(0, 0, 0), (4.0, 2.0, 0), (6.0, 5.0, 0)]
    # coordinates of the aperture
    aperture_coords = [[(4, 3), (4, 4.5)], [(8.5, 5), (10, 5)]]
    grid_spacing_m = 0.3

    return RoomDataset(
        num_rooms,
        sample_rate,
        source_position,
        receiver_position,
        rirs,
        common_decay_times,
        room_dims,
        room_start_coord,
        band_centre_hz,
        amplitudes,
        amplitudes_norm,
        noise_floor,
        noise_floor_norm,
        absorption_coeffs,
        aperture_coords,
        grid_spacing_m=grid_spacing_m,
    )


def plot_amplitudes_in_space(room_data: RoomDataset,
                             config_dict: SpatialSamplingConfig,
                             model: SpatialSamplingConfig, num_epochs: int,
                             grid_resolution_m: float):
    """Plot the true and learned amplitudes as a function of space"""
    logger.info("Making amplitude plots")

    room = RoomGeometry(room_data.sample_rate,
                        room_data.num_rooms,
                        np.array(room_data.room_dims),
                        np.array(room_data.room_start_coord),
                        aperture_coords=room_data.aperture_coords)
    src_pos = np.array(room_data.source_position).squeeze()
    rec_points = room_data.receiver_position
    true_amps = room_data.amplitudes

    room.plot_amps_at_receiver_points(
        rec_points,
        src_pos,
        true_amps.T,
        scatter_plot=False,
        save_path=Path(
            f'{config_dict.train_dir}/actual_amplitudes_in_space.png').resolve(
            ),
        title='Common slopes')

    # prepare the training and validation data
    train_dataset, _ = load_dataset(
        room_data,
        config_dict.device,
        grid_resolution_m=room_data.grid_spacing_m,
        batch_size=config_dict.batch_size,
        shuffle=False,
    )

    # load the trained weights for the particular epoch
    checkpoint = torch.load(
        Path(f'{config_dict.train_dir}/checkpoints/model_e{num_epochs - 1}.pt'
             ).resolve(),
        weights_only=True,
        map_location=torch.device('cpu'))
    # Load the trained model state
    model.load_state_dict(checkpoint, strict=False)

    # run the model in eval mode
    model.eval()

    all_pos = np.empty((0, 3))
    all_amps = np.empty((0, room_data.num_rooms))
    with torch.no_grad():
        for data in train_dataset:
            position = data['listener_position']
            est_amps = model(data)[..., 0]
            all_pos = np.vstack((all_pos, position))
            all_amps = np.vstack((all_amps, est_amps))

    room.plot_amps_at_receiver_points(
        all_pos,
        src_pos,
        all_amps.T,
        scatter_plot=False,
        save_path=Path(
            f'{config_dict.train_dir}/learnt_amplitudes_in_space_grid_resolution_m={np.round(grid_resolution_m, 3)}.png'
        ).resolve(),
        title=f'Training grid resolution = {np.round(grid_resolution_m, 3)}m')


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

    model = Gains_from_MLP(
        room_data.num_rooms,
        1,
        config_dict.num_fourier_features,
        config_dict.num_hidden_layers,
        config_dict.num_neurons_per_layer,
        config_dict.encoding_type,
        position_type="output_gains",
        device=config_dict.device,
        gain_limits=(db2lin(-100), db2lin(0)),
    )

    # set default device
    torch.set_default_device(config_dict.device)
    # move model to device (cuda or cpu)
    model = model.to(config_dict.device)

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
        plot_amplitudes_in_space(room_data, config_dict, model,
                                 len(trainer_loss[grid_resolution_m[k]]),
                                 grid_resolution_m[k])

    ax[0].set_xlabel('Epoch #')
    ax[0].set_ylabel('Training loss (log)')
    ax[1].set_xlabel('Epoch #')
    ax[1].set_ylabel('Validation loss (log)')
    ax[1].legend(loc='best', bbox_to_anchor=(1.1, 0.5))
    fig.savefig(os.path.join(config_dict.train_dir,
                             'loss_vs_grid_resolution.png'),
                bbox_inches="tight")
