import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import torch
from loguru import logger
from scipy.io import savemat

from .config.config import DiffGFDNConfig
from .dataloader import ThreeRoomDataset, load_dataset
from .model import DiffGFDN


def save_parameters(net: DiffGFDN, dir_path: str, filename: str):
    """
    save parameters of DiffGFDN() net to .mat file 
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
    save learnable parameters to a dictionary  
    Args    net (nn.Module): trained FDN() network
    Output  d (dictionary of tensors): FDN() net parameters 
    """
    d = {}  # enpty dictionary
    for name, param in net.named_parameters():
        if param.requires_grad:
            d[name] = param.data
    d['gain_per_sample'] = net.absorption_coeffs
    d['delays'] = net.delays
    return d


def save_loss(train_loss: List,
              output_dir: str,
              save_plot=True,
              filename: str = '',
              xaxis_label: Optional[str] = "epoch #"):
    '''
    save training and validation loss values in .mat format
    Args    train_loss (list): training loss values at each epoch
            output_dir (string): path to output directory
            save_plot (bool): if True saves the plot of the losses in .pdf format
            filename (string): additional string to add before .pdf and .mat
    '''
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
    savemat(os.path.join(output_dir, 'losses' + filename + '.mat'), losses)


def run_training(config_dict: DiffGFDNConfig):
    """
    Run the training for the differentiable GFDN, and save
    its parameters
    Args:
        config_dict (DiffGFDNTrainConfig): configuration parameters for training
    """
    # read the coupled room dataset
    room_data = ThreeRoomDataset(Path(config_dict.room_dataset_path).resolve())

    # add number of groups to the config dictionary
    config_dict = config_dict.copy(update={"num_groups": room_data.num_rooms})
    assert config_dict.num_delay_lines % config_dict.num_groups == 0, "Delay lines must be divisible by number of groups in network"

    if config_dict.sample_rate != room_data.sample_rate:
        logger.warn("Config sample rate does not match data, alterning it")
        config_dict.sample_rate = sample_rate

    # get the training config
    trainer_config = config_dict.trainer_config

    # prepare the training and validation data for DiffGFDN
    train_dataset, valid_dataset = load_dataset(
        room_data, trainer_config.device, trainer_config.train_valid_split,
        trainer_config.batch_size)

    # initialise the model
    model = DiffGFDN(room_data.sample_rate, room_data.num_rooms,
                     config_dict.delay_length_samps,
                     room_data.absorption_coeffs, room_data.room_dims,
                     trainer_config.device, feedback_loop_config,
                     output_filter_config)

    # create the trainer object
    trainer = Trainer(model, trainer_config)

    # save initial parameters and ir
    save_parameters(trainer.net, trainer_config.train_dir,
                    'parameters_init.mat')
    trainer.save_ir(trainer_config.ir_dir, filename='ir_init.wav')

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
