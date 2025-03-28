from typing import Optional

import torch
from torch.utils.data import DataLoader

from .colorless_fdn.utils import ColorlessFDNResults
from .config.config import DiffGFDNConfig
from .dataloader import RoomDataset
from .model import DiffGFDNVarReceiverPos, DiffGFDNVarSourceReceiverPos
from .save_results import save_loss
from .trainer import VarReceiverPosTrainer


class MLPTuningConfig:
    """Class for specifying input arguments to the hyperparameter tuning method"""

    def __init__(self,
                 config_dict: DiffGFDNConfig,
                 room_data: RoomDataset,
                 train_dataset: DataLoader,
                 valid_dataset: DataLoader,
                 colorless_fdn_params: Optional[ColorlessFDNResults] = None):
        """
        Args:
            config_dict (DiffGFDNConfig): config file for the network
            room_data (RoomDataset): object containing information about the room's geometry
            train_dataset (Dataloader): dataset containing training samples
            valid_dataset (Dataloader): dataset containing validation samples
            colorless_fdn_params (ColorlessFDNResults): if using a colorless FDN, then the optimised parameters
        """
        self.config_dict = config_dict
        self.room_data = room_data
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.colorless_fdn_params = colorless_fdn_params


def mlp_hyperparameter_tuning(trial, hyp_config: MLPTuningConfig):
    """Do hyperparameter tuning for the MLP"""
    # Define hyperparameters to tune
    output_filter_config = hyp_config.config_dict.output_filter_config
    mlp_tuning_config = output_filter_config.mlp_tuning_config
    trainer_config = hyp_config.config_dict.trainer_config
    output_filter_config = output_filter_config.model_copy(
        update={
            'num_hidden_layers':
            trial.suggest_int('num_hidden_layers', mlp_tuning_config.
                              min_layers, mlp_tuning_config.max_layers),
            'num_neurons_per_layer':
            trial.suggest_int('num_neurons_per_layer',
                              mlp_tuning_config.min_neurons,
                              mlp_tuning_config.max_neurons,
                              step=mlp_tuning_config.step_size)
        })

    # initialise the model
    if hyp_config.room_data.num_src == 1:
        # initialise the model
        model = model = DiffGFDNVarReceiverPos(
            hyp_config.room_data.sample_rate,
            hyp_config.room_data.num_rooms,
            hyp_config.config_dict.delay_length_samps,
            trainer_config.device,
            hyp_config.config_dict.feedback_loop_config,
            output_filter_config,
            hyp_config.config_dict.decay_filter_config.use_absorption_filters,
            common_decay_times=hyp_config.room_data.common_decay_times
            if hyp_config.config_dict.decay_filter_config.
            initialise_with_opt_values else None,
            learn_common_decay_times=hyp_config.config_dict.
            decay_filter_config.learn_common_decay_times,
            band_centre_hz=hyp_config.room_data.band_centre_hz,
            colorless_fdn_params=hyp_config.colorless_fdn_params,
            use_colorless_loss=trainer_config.use_colorless_loss)

    else:
        model = DiffGFDNVarSourceReceiverPos(
            hyp_config.room_data.sample_rate,
            hyp_config.room_data.num_rooms,
            hyp_config.config_dict.delay_length_samps,
            trainer_config.device,
            hyp_config.config_dict.feedback_loop_config,
            hyp_config.config_dict.output_filter_config,
            hyp_config.config_dict.input_filter_config,
            hyp_config.config_dict.decay_filter_config.use_absorption_filters,
            common_decay_times=hyp_config.room_data.common_decay_times
            if hyp_config.config_dict.decay_filter_config.
            initialise_with_opt_values else None,
            learn_common_decay_times=hyp_config.config_dict.
            decay_filter_config.learn_common_decay_times,
            band_centre_hz=hyp_config.room_data.band_centre_hz,
            colorless_fdn_params=hyp_config.colorless_fdn_params,
            use_colorless_loss=trainer_config.use_colorless_loss,
        )

    # set default device
    torch.set_default_device(trainer_config.device)
    # move model to device (cuda or cpu)
    model = model.to(trainer_config.device)
    # create the trainer object
    trainer = VarReceiverPosTrainer(model, trainer_config)
    # train the network
    trainer.train(hyp_config.train_dataset)

    # test the network with the validation set
    trainer.validate(hyp_config.valid_dataset)
    # save the validation loss
    save_loss(trainer.valid_loss,
              trainer_config.train_dir,
              save_plot=True,
              filename='test_loss_vs_position',
              xaxis_label='Position #')
    return sum(trainer.valid_loss)
