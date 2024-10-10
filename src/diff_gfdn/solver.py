import os
import re
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from scipy.io import savemat

from .config.config import DiffGFDNConfig
from .dataloader import RIRData, ThreeRoomDataset, load_dataset
from .model import DiffGFDN, DiffGFDNSinglePos, DiffGFDNVarReceiverPos
from .trainer import SinglePosTrainer, VarReceiverPosTrainer


def save_parameters(net: DiffGFDN, dir_path: str, filename: str):
    """
    Save parameters of DiffGFDN() net to .mat file 
    Args    net (nn.Module): trained FDN() network
            dir_path (string): path to output firectory
            filename (string): name of the file 
    Output  param (dictionary of tensors): FDN() net parameters
            param_np (dictionary of numpy arrays): DiffGFDN() net parameters
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    param = gfdn2dir(net)
    param_np = {}
    for name, value in param.items():
        try:
            param_np[name] = value.squeeze().cpu().numpy()
        except AttributeError:
            param_np[name] = value
    # save parameters in numpy format
    savemat(os.path.join(dir_path, filename), param_np)

    return param, param_np


def gfdn2dir(net: DiffGFDN):
    """
    Save learnable parameters to a dictionary  
    Args    net (nn.Module): trained FDN() network
    Output  d (dictionary of tensors): FDN() net parameters 
    """
    d = {}  # empty dictionary
    # from parameter dictionary of model
    d = net.get_param_dict()

    # MLP learned weights and biases
    for name, param in net.named_parameters():
        if param.requires_grad and name not in d.keys():
            d[name] = param.data

    return d


def save_loss(train_loss: List,
              output_dir: str,
              save_plot=True,
              filename: str = '',
              xaxis_label: Optional[str] = "epoch #"):
    """
    Save training and validation loss values in .mat format
    Args    train_loss (list): training loss values at each epoch
            output_dir (string): path to output directory
            save_plot (bool): if True saves the plot of the losses in .pdf format
            filename (string): additional string to add before .pdf and .mat
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    losses = {}
    losses['train'] = train_loss
    n_epochs = len(train_loss)

    if save_plot:
        plt.plot(range(1, n_epochs + 1), train_loss)
        plt.xlabel(xaxis_label)
        plt.ylabel('loss')
        plt.savefig(os.path.join(output_dir, filename + '.pdf'))
    savemat(os.path.join(output_dir, 'losses_' + filename + '.mat'), losses)


def run_training_var_receiver_pos(config_dict: DiffGFDNConfig):
    """
    Run the training for the differentiable GFDN for a grid of different receiver positions, and save
    its parameters
    Args:
        config_dict (DiffGFDNTrainConfig): configuration parameters for training
    """
    logger.info("Training over a grid of listener positions")

    # read the coupled room dataset
    room_data = ThreeRoomDataset(Path(config_dict.room_dataset_path).resolve())

    # add number of groups to the config dictionary
    config_dict = config_dict.copy(update={"num_groups": room_data.num_rooms})
    assert config_dict.num_delay_lines % config_dict.num_groups == 0, "Delay lines must be \
    divisible by number of groups in network"

    if config_dict.sample_rate != room_data.sample_rate:
        logger.warn("Config sample rate does not match data, alterning it")
        config_dict.sample_rate = room_data.sample_rate

    # get the training config
    trainer_config = config_dict.trainer_config

    # prepare the training and validation data for DiffGFDN
    train_dataset, valid_dataset = load_dataset(
        room_data,
        trainer_config.device,
        trainer_config.train_valid_split,
        trainer_config.batch_size,
        new_sampling_radius=trainer_config.new_sampling_radius)

    # initialise the model
    model = DiffGFDNVarReceiverPos(
        room_data.sample_rate,
        room_data.num_rooms,
        config_dict.delay_length_samps,
        trainer_config.device,
        config_dict.feedback_loop_config,
        config_dict.output_filter_config,
        config_dict.use_absorption_filters,
        room_dims=room_data.room_dims,
        absorption_coeffs=room_data.absorption_coeffs,
        common_decay_times=room_data.common_decay_times,
        band_centre_hz=room_data.band_centre_hz,
    )
    # set default device
    torch.set_default_device(trainer_config.device)
    # move model to device (cuda or cpu)
    model = model.to(trainer_config.device)
    # create the trainer object
    trainer = VarReceiverPosTrainer(model, trainer_config)

    # save initial parameters and ir
    save_parameters(trainer.net, trainer_config.train_dir,
                    'parameters_init.mat')

    # train the network
    trainer.train(train_dataset)
    # save final trained parameters
    save_parameters(trainer.net, trainer_config.train_dir,
                    'parameters_opt.mat')
    # save loss evolution
    save_loss(trainer.train_loss,
              trainer_config.train_dir,
              save_plot=True,
              filename='training_loss_vs_epoch')

    # test the network with the validation set
    trainer.validate(valid_dataset)
    # save the validation loss
    save_loss(trainer.valid_loss,
              trainer_config.train_dir,
              save_plot=True,
              filename='test_loss_vs_position',
              xaxis_label='Position #')


def run_training_single_pos(config_dict: DiffGFDNConfig):
    """
    Run the training for the differentiable GFDN for a single RIR measurement, and save its parameters
    Args:
        config_dict (DiffGFDNTrainConfig): configuration parameters for training
    """
    logger.info("Training for a single RIR measurement")

    # read the coupled room dataset
    room_data = ThreeRoomDataset(Path(config_dict.room_dataset_path).resolve())

    # add number of groups to the config dictionary
    config_dict = config_dict.copy(update={"num_groups": room_data.num_rooms})
    assert config_dict.num_delay_lines % config_dict.num_groups == 0, "Delay lines must be \
    divisible by number of groups in network"

    if config_dict.sample_rate != room_data.sample_rate:
        logger.warn("Config sample rate does not match data, alterning it")
        config_dict.sample_rate = room_data.sample_rate

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
        np.all(room_data.receiver_position == rec_pos, axis=1))[0]
    amplitudes = room_data.amplitudes[..., rec_pos_idx]

    rir_data = RIRData(ir_path,
                       room_data.band_centre_hz,
                       room_data.common_decay_times,
                       amplitudes=amplitudes)

    # get the training config
    trainer_config = config_dict.trainer_config

    # prepare the training and validation data for DiffGFDN
    if trainer_config.train_valid_split < 1.0:
        logger.warning('There can be no data in the validation set!')
    if trainer_config.batch_size != rir_data.num_freq_bins:
        logger.warning(
            "Cannot train in batches here. Training on the full unit circle")
        trainer_config = trainer_config.copy(
            update={"batch_size": rir_data.num_freq_bins})

    train_dataset = load_dataset(
        rir_data,
        trainer_config.device,
        trainer_config.train_valid_split,
        trainer_config.batch_size,
        shuffle=False,
        new_sampling_radius=trainer_config.new_sampling_radius)

    # initialise the model
    model = DiffGFDNSinglePos(
        room_data.sample_rate,
        room_data.num_rooms,
        config_dict.delay_length_samps,
        trainer_config.device,
        config_dict.feedback_loop_config,
        config_dict.output_filter_config,
        config_dict.use_absorption_filters,
        room_dims=room_data.room_dims,
        absorption_coeffs=room_data.absorption_coeffs,
        common_decay_times=room_data.common_decay_times,
        band_centre_hz=room_data.band_centre_hz,
    )
    # set default device
    torch.set_default_device(trainer_config.device)
    # move model to device (cuda or cpu)
    model = model.to(trainer_config.device)
    # create the trainer object
    trainer = SinglePosTrainer(model, trainer_config, filename=ir_name)

    # save initial parameters and ir
    save_parameters(trainer.net, trainer_config.train_dir,
                    'parameters_init.mat')

    # train the network
    trainer.train(train_dataset)
    # save final trained parameters
    save_parameters(trainer.net, trainer_config.train_dir,
                    'parameters_opt.mat')
    # save loss evolution
    save_loss(trainer.train_loss,
              trainer_config.train_dir,
              save_plot=True,
              filename='training_loss_vs_epoch')
