import os
from pathlib import Path
import pickle
import re
from typing import List, Optional, Tuple

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import optuna
from slope2noise.dataclass import Slope2NoiseUnpickler
import torch

from spatial_sampling.config import DNNType
from spatial_sampling.dataloader import load_dataset as load_dataset_spatial
from spatial_sampling.dataloader import parse_three_room_data, SpatialRoomDataset

from .colorless_fdn.dataloader import load_colorless_fdn_dataset
from .colorless_fdn.model import ColorlessFDN
from .colorless_fdn.trainer import ColorlessFDNTrainer
from .colorless_fdn.utils import ColorlessFDNResults
from .config.config import DiffGFDNConfig
from .dataloader import load_dataset, RIRData, RoomDataset, ThreeRoomDataset
from .hypertuning import mlp_hyperparameter_tuning, MLPTuningConfig
from .model import (
    DiffDirectionalFDNVarReceiverPos,
    DiffGFDNSinglePos,
    DiffGFDNVarReceiverPos,
    DiffGFDNVarSourceReceiverPos,
)
from .save_results import save_colorless_fdn_parameters, save_diff_gfdn_parameters, save_loss
from .trainer import DirectionalFDNVarReceiverPosTrainer, SinglePosTrainer, VarReceiverPosTrainer
from .utils import db

# pylint: disable=W0718

####################################################################
# For training with artificial dataset


def convert_common_slopes_rir_to_room_dataset(
        data_path: str, num_freq_bins: Optional[int] = None) -> RoomDataset:
    """Convert the dataclass CommonSlopesRIR to RoomDataset"""
    data_path = Path(data_path).resolve()
    with open(data_path, 'rb') as f:
        # rir_data = pickle.load(f)
        rir_data = Slope2NoiseUnpickler(f).load()

    num_rooms = rir_data.n_slopes
    room_dims = rir_data.room_dims
    room_start_coords = rir_data.room_start_coords
    try:
        aperture_coords = rir_data.aperture_coords
    except Exception as e:
        logger.warning(e)
        aperture_coords = None

    source_locs = rir_data.source_locs
    receiver_locs = rir_data.receiver_locs
    common_decay_times = np.array(rir_data.t_vals).T
    band_centre_hz = np.array(rir_data.f_bands)
    amplitudes = rir_data.a_vals
    rirs = rir_data.rir
    sample_rate = rir_data.sample_rate

    room_data = RoomDataset(
        num_rooms,
        sample_rate,
        source_locs,
        receiver_locs,
        rirs,
        common_decay_times=common_decay_times,
        room_dims=room_dims,
        room_start_coord=room_start_coords,
        amplitudes=amplitudes,
        band_centre_hz=band_centre_hz,
        aperture_coords=aperture_coords,
        nfft=num_freq_bins,
    )

    return room_data


def data_parser_var_receiver_pos(
        config_dict: DiffGFDNConfig,
        num_freq_bins: Optional[int] = None) -> RoomDataset:
    """
    Parse the training data for training over a grid of receiver positions (could belong to different rooms)
    Args:
        config_dict (DiffGFDNConfig): config dictionary
        num_freq_bins (int): number of frequency bins to train on
    Returns:
        RoomDataset: object of data type RoomDataset with all the room information
    """
    if "3room_FDTD" in config_dict.room_dataset_path:
        # read the coupled room dataset
        room_data = ThreeRoomDataset(Path(
            config_dict.room_dataset_path).resolve(),
                                     config_dict=config_dict)
    else:
        room_data = convert_common_slopes_rir_to_room_dataset(
            config_dict.room_dataset_path, num_freq_bins)

    return room_data


def data_parser_anisotropic_decay_var_receiver_pos(
        config_dict: DiffGFDNConfig) -> SpatialRoomDataset:
    """
    Parse the training data for training over a grid of receiver positions (could belong to different rooms)
    Args:
        config_dict (DiffGFDNConfig): config dictionary
    Returns:
        SpatialRoomDataset: object of data type SpatialRoomDataset with all the room information
    """
    if "3room_FDTD" in config_dict.room_dataset_path:
        # read the coupled room dataset
        spatial_room_data = parse_three_room_data(
            config_dict.room_dataset_path)
        return spatial_room_data
    else:
        logger.error("Other datasets not implemented yet!")


def data_parser_single_receiver_pos(
    config_dict: DiffGFDNConfig,
    num_freq_bins: Optional[int] = None,
    debug: bool = False,
) -> Tuple[RIRData, RoomDataset, str]:
    """
    Parse the training data for a single receiver position
    Args:
        config_dict (DiffGFDNConfig): config dictionary
        num_freq_bins (int): number of frequency bins to train on
        debug (bool): plots the EDC of the desired RIR, this has caused issues previously
    Returns:
        RIRData, RoomDataset, str: single position dataset, room dataset and the name of the ir
    """
    if "3room_FDTD" in config_dict.room_dataset_path:
        # read the coupled room dataset
        room_data = ThreeRoomDataset(
            Path(config_dict.room_dataset_path).resolve(), config_dict)

    # from the synthetic dataset created in slope2rir
    else:
        room_data = convert_common_slopes_rir_to_room_dataset(
            config_dict.room_dataset_path, num_freq_bins)

    # create a dataset for a single measured IR in the room dataset
    ir_path = Path(config_dict.ir_path).resolve()
    match = re.search(r'ir_\([^)]+\)', config_dict.ir_path)
    ir_name = match.group()

    # find receiver position from string
    match = re.search(r'ir_\(([^,]+), ([^,]+), ([^,]+)\)', ir_name)
    # Convert the extracted values to floats
    x, y, z = map(float, match.groups())
    rec_pos = np.array([x, y, z])

    # find amplitudes corresponding to the receiver position
    rec_pos_idx = np.where(
        np.all(np.round(room_data.receiver_position, 2) == rec_pos,
               axis=1))[0][0]
    amplitudes = np.squeeze(room_data.amplitudes[rec_pos_idx, :])

    # if the reference IR does not exist, create it from the dataset
    if not os.path.isfile(str(ir_path)):
        logger.warning("RIR does not exist in path, creating it...")
        room_data.save_individual_irs(directory=Path(ir_path).parent.resolve())

    rir_data = RIRData(rir=np.squeeze(room_data.rirs[rec_pos_idx, :]),
                       sample_rate=room_data.sample_rate,
                       common_decay_times=room_data.common_decay_times,
                       band_centre_hz=room_data.band_centre_hz,
                       amplitudes=amplitudes,
                       nfft=num_freq_bins)

    # for debugging
    if debug:
        true_rir = rir_data.rir
        true_rir_room_data = np.squeeze(room_data.rirs[rec_pos_idx, :])
        true_edf = np.flipud(np.cumsum(np.flipud(true_rir**2), axis=-1))
        true_edf_room_data = np.flipud(
            np.cumsum(np.flipud(true_rir_room_data**2), axis=-1))

        time = np.linspace(0, (len(true_rir) - 1) / rir_data.sample_rate,
                           len(true_rir))

        plt.figure()
        plt.plot(time, db(true_edf, is_squared=True))
        plt.plot(time, db(true_edf_room_data, is_squared=True))
        plt.plot(np.zeros(len(amplitudes)), db(amplitudes, is_squared=True),
                 'kx')
        plt.xlabel('Time (s)')
        plt.ylabel('Magnitude (dB)')
        plt.show()

    return rir_data, room_data, ir_name


#############################################################################
# For training the model


def run_training_colorless_fdn(
        config_dict: DiffGFDNConfig,
        num_freq_bins: int) -> List[ColorlessFDNResults]:
    """
    Run the training for a colorless prototype
    Returns:
        A list of ColorlessFDNResults dataclass, each for one FDN in the GFDN
    """
    logger.info("Training a colorless prototype")
    trainer_config = config_dict.trainer_config

    # prepare the training and validation data for DiffGFDN
    train_dataset, valid_dataset = load_colorless_fdn_dataset(
        num_freq_bins,
        trainer_config.device,
        config_dict.colorless_fdn_config.train_valid_split,
        config_dict.colorless_fdn_config.batch_size,
    )

    params_opt = []
    num_delay_lines_per_group = int(config_dict.num_delay_lines /
                                    config_dict.num_groups)
    for i in range(config_dict.num_groups):

        model = ColorlessFDN(
            config_dict.sample_rate,
            config_dict.delay_length_samps[i *
                                           num_delay_lines_per_group:(i + 1) *
                                           num_delay_lines_per_group],
            trainer_config.device)

        # set default device
        torch.set_default_device(trainer_config.device)
        # move model to device (cuda or cpu)
        model = model.to(trainer_config.device)
        # create the trainer object
        trainer = ColorlessFDNTrainer(model, trainer_config,
                                      config_dict.colorless_fdn_config)

        # save initial parameters and ir
        save_colorless_fdn_parameters(
            trainer.net, trainer_config.train_dir + "colorless-fdn/",
            f'parameters_init_group={i + 1}.pkl')

        # train the network
        trainer.train(train_dataset, valid_dataset)
        # save final trained parameters
        params_opt.append(
            save_colorless_fdn_parameters(
                trainer.net, trainer_config.train_dir + "colorless-fdn/",
                f'parameters_opt_group={i + 1}.pkl'))

        # save loss evolution
        save_loss(trainer.train_loss,
                  trainer_config.train_dir + "colorless-fdn/",
                  save_plot=True,
                  filename=f'training_loss_vs_epoch_group={i + 1}')

    return params_opt


################################################################################


def run_training_var_receiver_pos(config_dict: DiffGFDNConfig):
    """
    Run the training for the differentiable GFDN for a grid of different receiver positions, and save
    its parameters
    Args:
        config_dict (DiffGFDNTrainConfig): configuration parameters for training
    """
    # get the data
    room_data = data_parser_var_receiver_pos(
        config_dict, num_freq_bins=config_dict.trainer_config.num_freq_bins)

    if room_data.num_src == 1:
        logger.info("Training over a grid of listener positions")
    else:
        logger.info("Training over a grid of source and listener positions")

    # add number of groups to the config dictionary
    config_dict = config_dict.model_copy(
        update={"num_groups": room_data.num_rooms})
    assert config_dict.num_delay_lines % config_dict.num_groups == 0, "Delay lines must be \
    divisible by number of groups in network"

    if config_dict.sample_rate != room_data.sample_rate:
        logger.warning("Config sample rate does not match data, alterning it")
        config_dict.sample_rate = room_data.sample_rate

    # get the training config
    trainer_config = config_dict.trainer_config
    # update num_freq_bins in pydantic class
    trainer_config = trainer_config.model_copy(
        update={"num_freq_bins": room_data.num_freq_bins})
    # also update the calculation of reduced_pole_radius
    trainer_config = trainer_config.calculate_reduced_pole_radius(
        trainer_config)

    if config_dict.colorless_fdn_config.use_colorless_prototype and trainer_config.use_colorless_loss:
        raise ValueError(
            "Cannot use optimised colorless FDN parameters and colorless FDN loss together"
        )

    # are we using a colorless FDN to get the feedback matrix?
    if config_dict.colorless_fdn_config.use_colorless_prototype:
        colorless_fdn_params = run_training_colorless_fdn(
            config_dict, room_data.num_freq_bins)
    else:
        colorless_fdn_params = None

    # prepare the training and validation data for DiffGFDN
    train_dataset, valid_dataset = load_dataset(
        room_data,
        trainer_config.device,
        trainer_config.train_valid_split,
        trainer_config.batch_size,
        new_sampling_radius=1.0 / trainer_config.reduced_pole_radius,
    )

    # are we tuning hyperparameters?
    if config_dict.output_filter_config.mlp_tuning_config is not None:
        logger.debug("Tuning MLP hyperparameters")

        study = optuna.create_study(direction="minimize")
        # to enable passing other parameters to this function
        mlp_tuning_with_params = MLPTuningConfig(config_dict, room_data,
                                                 train_dataset, valid_dataset,
                                                 colorless_fdn_params)
        study.optimize(
            lambda trial: mlp_hyperparameter_tuning(trial,
                                                    mlp_tuning_with_params),
            n_trials=config_dict.output_filter_config.mlp_tuning_config.
            num_trials)  # Set number of trials as needed

        logger.debug(f"Best hyperparameters: {study.best_params}")
        logger.debug(f"Best validation loss: {study.best_value}")
        with open(os.path.join(trainer_config.train_dir, "MLP_tuning.pkl"),
                  "wb") as f:
            pickle.dump(
                {
                    'best_hyperparameters': study.best_params,
                    'best_valid_loss': study.best_value
                }, f)
    # or are we running the network with fixed hyperparameters?
    else:
        if room_data.num_src == 1:
            # initialise the model
            model = DiffGFDNVarReceiverPos(
                room_data.sample_rate,
                room_data.num_rooms,
                config_dict.delay_length_samps,
                trainer_config.device,
                config_dict.feedback_loop_config,
                config_dict.output_filter_config,
                config_dict.decay_filter_config.use_absorption_filters,
                common_decay_times=room_data.common_decay_times
                if config_dict.decay_filter_config.initialise_with_opt_values
                else None,
                learn_common_decay_times=config_dict.decay_filter_config.
                learn_common_decay_times,
                band_centre_hz=room_data.band_centre_hz,
                colorless_fdn_params=colorless_fdn_params,
                use_colorless_loss=trainer_config.use_colorless_loss,
            )
        else:
            model = DiffGFDNVarSourceReceiverPos(
                room_data.sample_rate,
                room_data.num_rooms,
                config_dict.delay_length_samps,
                trainer_config.device,
                config_dict.feedback_loop_config,
                config_dict.output_filter_config,
                config_dict.input_filter_config,
                config_dict.decay_filter_config.use_absorption_filters,
                common_decay_times=room_data.common_decay_times
                if config_dict.decay_filter_config.initialise_with_opt_values
                else None,
                learn_common_decay_times=config_dict.decay_filter_config.
                learn_common_decay_times,
                band_centre_hz=room_data.band_centre_hz,
                colorless_fdn_params=colorless_fdn_params,
                use_colorless_loss=trainer_config.use_colorless_loss,
            )
        # set default device
        torch.set_default_device(trainer_config.device)
        # move model to device (cuda or cpu)
        model = model.to(trainer_config.device)
        # create the trainer object
        trainer = VarReceiverPosTrainer(model, trainer_config)

        # save initial parameters and ir
        save_diff_gfdn_parameters(trainer.net, trainer_config.train_dir,
                                  'parameters_init.mat')

        # train the network
        trainer.train(train_dataset)
        # save final trained parameters
        save_diff_gfdn_parameters(trainer.net, trainer_config.train_dir,
                                  'parameters_opt.mat')
        # save loss evolution
        save_loss(trainer.train_loss,
                  trainer_config.train_dir,
                  save_plot=True,
                  filename='training_loss_vs_epoch',
                  individual_losses=trainer.individual_train_loss)

        # test the network with the validation set
        trainer.validate(valid_dataset)
        # save the validation loss
        save_loss(trainer.valid_loss,
                  trainer_config.train_dir,
                  save_plot=True,
                  filename='test_loss_vs_position',
                  xaxis_label='Position #',
                  individual_losses=trainer.individual_valid_loss)


#######################################################################################


def run_training_single_pos(config_dict: DiffGFDNConfig):
    """
    Run the training for the differentiable GFDN for a single RIR measurement, and save its parameters
    Args:
        config_dict (DiffGFDNConfig): configuration parameters for training
    """
    logger.info("Training for a single RIR measurement")

    # get the data
    rir_data, room_data, ir_name = data_parser_single_receiver_pos(
        config_dict, num_freq_bins=config_dict.trainer_config.num_freq_bins)

    # add number of groups to the config dictionary
    config_dict = config_dict.copy(update={"num_groups": room_data.num_rooms})
    assert config_dict.num_delay_lines % config_dict.num_groups == 0, "Delay lines must be \
    divisible by number of groups in network"

    if config_dict.sample_rate != room_data.sample_rate:
        logger.warn("Config sample rate does not match data, alterning it")
        config_dict.sample_rate = room_data.sample_rate

    # get the training config
    trainer_config = config_dict.trainer_config
    # update num_freq_bins in pydantic class
    trainer_config = trainer_config.model_copy(
        update={"num_freq_bins": rir_data.num_freq_bins})

    # prepare the training and validation data for DiffGFDN
    if trainer_config.train_valid_split < 1.0:
        logger.warning('There can be no data in the validation set!')
    if trainer_config.batch_size != rir_data.num_freq_bins:
        logger.warning(
            "Cannot train in batches here. Training on the full unit circle")
        trainer_config = trainer_config.copy(
            update={"batch_size": rir_data.num_freq_bins})
    if config_dict.colorless_fdn_config.use_colorless_prototype and trainer_config.use_colorless_loss:
        raise ValueError(
            "Cannot use optimised colorless FDN parameters and colorless FDN loss together"
        )

    # whether to use a colorless FDN to get input-output gains and feedback matrix
    if config_dict.colorless_fdn_config.use_colorless_prototype:
        colorless_fdn_params = run_training_colorless_fdn(
            config_dict, room_data.num_freq_bins)
    else:
        colorless_fdn_params = None

    train_dataset = load_dataset(rir_data,
                                 trainer_config.device,
                                 trainer_config.train_valid_split,
                                 trainer_config.batch_size,
                                 shuffle=False,
                                 new_sampling_radius=1.0 /
                                 trainer_config.reduced_pole_radius)

    # initialise the model
    model = DiffGFDNSinglePos(
        room_data.sample_rate,
        room_data.num_rooms,
        config_dict.delay_length_samps,
        trainer_config.device,
        config_dict.feedback_loop_config,
        config_dict.output_filter_config,
        config_dict.decay_filter_config.use_absorption_filters,
        common_decay_times=room_data.common_decay_times if
        config_dict.decay_filter_config.initialise_with_opt_values else None,
        learn_common_decay_times=config_dict.decay_filter_config.
        learn_common_decay_times,
        band_centre_hz=room_data.band_centre_hz,
        colorless_fdn_params=colorless_fdn_params,
        use_colorless_loss=trainer_config.use_colorless_loss,
    )
    # set default device
    torch.set_default_device(trainer_config.device)
    # move model to device (cuda or cpu)
    model = model.to(trainer_config.device)
    # create the trainer object
    trainer = SinglePosTrainer(model, trainer_config, filename=ir_name)

    # save initial parameters and ir
    save_diff_gfdn_parameters(trainer.net, trainer_config.train_dir,
                              'parameters_init.mat')

    # train the network
    trainer.train(train_dataset)
    # save final trained parameters
    save_diff_gfdn_parameters(trainer.net, trainer_config.train_dir,
                              'parameters_opt.mat')
    # save loss evolution
    save_loss(trainer.train_loss,
              trainer_config.train_dir,
              save_plot=True,
              filename='training_loss_vs_epoch',
              individual_losses=trainer.individual_train_loss)


######################################################################################


def run_training_anisotropic_decay_var_receiver_pos(
        config_dict: DiffGFDNConfig):
    """
    Run the training for the differentiable directional FDN for a grid of different receiver positions, and save
    its parameters
    Args:
        config_dict (DiffGFDNConfig): configuration parameters
    """
    # get the data
    spatial_room_data = data_parser_anisotropic_decay_var_receiver_pos(
        config_dict)

    # add number of groups to the config dictionary
    config_dict = config_dict.model_copy(
        update={"num_groups": spatial_room_data.num_rooms})

    # update ambisonics order
    config_dict = config_dict.model_copy(
        update={"ambi_order": spatial_room_data.ambi_order})

    # update number of delay lines per group
    config_dict = config_dict.model_copy(
        update={
            "num_delay_lines":
            spatial_room_data.num_directions * config_dict.num_groups
        })
    assert config_dict.num_delay_lines % config_dict.num_groups == 0, "Delay lines must be \
    divisible by number of groups in network"

    if config_dict.sample_rate != spatial_room_data.sample_rate:
        logger.warning("Config sample rate does not match data, alterning it")
        config_dict.sample_rate = spatial_room_data.sample_rate

    # get the training config
    trainer_config = config_dict.trainer_config
    # update num_freq_bins in pydantic class
    trainer_config = trainer_config.model_copy(
        update={"num_freq_bins": spatial_room_data.num_freq_bins})

    if config_dict.colorless_fdn_config.use_colorless_prototype and trainer_config.use_colorless_loss:
        raise ValueError(
            "Cannot use optimised colorless FDN parameters and colorless FDN loss together"
        )

    # are we using a colorless FDN to get the feedback matrix?
    if config_dict.colorless_fdn_config.use_colorless_prototype:
        colorless_fdn_params = run_training_colorless_fdn(
            config_dict, spatial_room_data.num_freq_bins)
    else:
        colorless_fdn_params = None

    # prepare the training and validation data for DiffGFDN
    train_dataset, valid_dataset, _ = load_dataset_spatial(
        spatial_room_data,
        trainer_config.device,
        network_type=DNNType.MLP,
        batch_size=trainer_config.batch_size,
        grid_resolution_m=trainer_config.grid_resolution_m,
        train_valid_split_ratio=trainer_config.train_valid_split,
    )

    # initialise the model
    model = DiffDirectionalFDNVarReceiverPos(
        spatial_room_data.sample_rate,
        spatial_room_data.num_rooms,
        config_dict.delay_length_samps,
        trainer_config.device,
        config_dict.feedback_loop_config,
        config_dict.output_filter_config,
        ambi_order=config_dict.ambi_order,
        desired_directions=spatial_room_data.sph_directions,
        common_decay_times=spatial_room_data.common_decay_times if
        config_dict.decay_filter_config.initialise_with_opt_values else None,
        band_centre_hz=spatial_room_data.band_centre_hz,
        colorless_fdn_params=colorless_fdn_params,
        use_colorless_loss=trainer_config.use_colorless_loss,
    )

    # set default device
    torch.set_default_device(trainer_config.device)
    # move model to device (cuda or cpu)
    model = model.to(trainer_config.device)
    # create the trainer object
    trainer = DirectionalFDNVarReceiverPosTrainer(model, trainer_config)

    # save initial parameters and ir
    save_diff_gfdn_parameters(trainer.net, trainer_config.train_dir,
                              'parameters_init.mat')

    # train the network
    trainer.train(train_dataset)
    # save final trained parameters
    save_diff_gfdn_parameters(trainer.net, trainer_config.train_dir,
                              'parameters_opt.mat')
    # save loss evolution
    save_loss(trainer.train_loss,
              trainer_config.train_dir,
              save_plot=True,
              filename='training_loss_vs_epoch',
              individual_losses=trainer.individual_train_loss)

    # test the network with the validation set
    trainer.validate(valid_dataset)
    # save the validation loss
    save_loss(trainer.valid_loss,
              trainer_config.train_dir,
              save_plot=True,
              filename='test_loss_vs_position',
              xaxis_label='Position #',
              individual_losses=trainer.individual_valid_loss)
