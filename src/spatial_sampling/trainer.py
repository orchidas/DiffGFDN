import os
import time
from typing import List, Optional, Union

from loguru import logger
import numpy as np
from slope2noise.utils import decay_kernel
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from diff_gfdn.utils import db, get_str_results, ms_to_samps

from .config import SpatialSamplingConfig
from .model import Directional_Beamforming_Weights_from_MLP, Omni_Amplitudes_from_MLP

# pylint: disable=W0632


class spatial_mse_loss(nn.Module):
    """Means absolute error in dB between the true and predicted spatial embeddings"""

    def forward(self, y_pred: torch.tensor, y_true: torch.tensor):
        """
        Args:
            y_pred (torch.tensor): output of the model, array of batch_size x num_slopes
            y_true (torch.tensor): expected output, array of size batch_size x num_slopes
        """
        loss = torch.mean(torch.abs(db(y_pred) - db(y_true)), dim=0)
        return torch.sum(loss)


class spatial_edc_loss(nn.Module):
    """Mean EDC loss between true and learned spatial mappings"""

    def __init__(self, common_decay_times: List, edc_len_ms: float, fs: float):
        """
        Initialise EDC loss
        Args:
            common_decay_times (List): of size num_slopes x 1 to form the decay kernel
            edc_len_ms (float): length of the EDC in ms
            fs (float): sampling rate in Hz
        """
        super().__init__()
        edc_len_samps = ms_to_samps(edc_len_ms, fs)
        num_slopes = common_decay_times.shape[-1]
        self.envelopes = torch.zeros((num_slopes, edc_len_samps))

        time_axis = np.linspace(0, (edc_len_samps - 1) / fs, edc_len_samps)

        for k in range(num_slopes):
            self.envelopes[k, :] = torch.tensor(
                decay_kernel(np.expand_dims(common_decay_times[:, k], axis=-1),
                             time_axis,
                             fs,
                             normalize_envelope=True,
                             add_noise=False)).squeeze()

    def forward(self, amps_pred: torch.Tensor, amps_true: torch.Tensor):
        """
        Calculate the mean EDC loss over space and time.
        The inputs are of shape batch size x num_slopes, 
        the decay kernel is of shape num_slopes x time
        """
        # Omni RIRs only
        if amps_true.ndim == 2:
            # desired shape is batch_size x num_slopes x num_time_samples
            # sum along time samples
            edc_true = db(torch.einsum('bk, kt -> bkt', amps_true,
                                       self.envelopes),
                          is_squared=True)
            edc_pred = db(torch.einsum('bk, kt -> bkt', amps_pred,
                                       self.envelopes),
                          is_squared=True)
            edc_loss = torch.sum(
                torch.mean(torch.abs(edc_true - edc_pred), dim=(0, -1)))
        # Directional RIRs
        else:
            # desired shape is batch_size x num_directions x num_slopes x num_time_samples
            # sum along num slopes
            edc_true = db(torch.sum(torch.einsum('bjk, kt -> bjkt', amps_true,
                                                 self.envelopes),
                                    axis=-2),
                          is_squared=True)
            edc_pred = db(torch.sum(torch.einsum('bjk, kt -> bjkt', amps_pred,
                                                 self.envelopes),
                                    axis=-2),
                          is_squared=True)
            edc_loss = torch.mean(torch.abs(edc_true - edc_pred),
                                  dim=(0, 1, -1))

        return edc_loss


class SpatialSamplingTrainer:
    """Class for training an MLP that learns the spatial mapping of the common slope amplitudes"""

    def __init__(
        self,
        net: Union[Omni_Amplitudes_from_MLP,
                   Directional_Beamforming_Weights_from_MLP],
        trainer_config: SpatialSamplingConfig,
        common_decay_times: Optional[List] = None,
        sampling_rate: float = 44100,
        ir_len_ms: float = 2000,
    ):
        """
        Args:
            net (Gains_from_MLP): the MLP module (pre-initialised)
            trainer_config (SpatialSamplingConfig): config containing training params
        """
        self.net = net
        self.device = trainer_config.device
        self.max_epochs = trainer_config.max_epochs
        self.patience = 5
        self.early_stop = 0
        self.train_dir = trainer_config.train_dir

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=trainer_config.lr)
        if common_decay_times is None:
            logger.info("Using spatial MSE loss")
            self.criterion = [spatial_mse_loss()]
        else:
            logger.info("Using spatial EDC loss")
            self.criterion = [
                spatial_edc_loss(common_decay_times, ir_len_ms, sampling_rate)
            ]

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=10,
                                                         gamma=0.1)

    def to_device(self):
        """Convert losses to run on the correct device"""
        for i in range(len(self.criterion)):
            self.criterion[i] = self.criterion[i].to(self.device)

    def train(self, train_dataset: DataLoader, valid_dataset: DataLoader):
        """Training and validation method"""
        self.train_loss, self.valid_loss = [], []

        st = time.time()  # start time
        for epoch in trange(self.max_epochs, desc='Training'):
            st_epoch = time.time()

            # training
            epoch_loss = 0
            for data in train_dataset:
                epoch_loss += self.train_step(data)
            self.scheduler.step()
            self.train_loss.append(epoch_loss / len(train_dataset))

            # validation
            if valid_dataset is not None:
                epoch_loss = 0
                for data in valid_dataset:
                    epoch_loss += self.valid_step(data)
                self.valid_loss.append(epoch_loss / len(valid_dataset))
            else:
                self.valid_loss.append(0.0)
            et_epoch = time.time()

            self.print_results(epoch, et_epoch - st_epoch)
            self.save_model(epoch)

            # early stopping
            if epoch >= 1:
                if abs(self.valid_loss[-2] - self.valid_loss[-1]) <= 1e-4:
                    self.early_stop += 1
                else:
                    self.early_stop = 0
            if self.early_stop == self.patience:
                break

        et = time.time()  # end time
        print('Training time: {:.3f}s'.format(et - st))

    def train_step(self, data):
        """Train each batch"""
        # batch processing
        self.optimizer.zero_grad()
        gains = self.net(data)
        # if batch has a single data point
        gains = gains.unsqueeze(0) if gains.ndim == 1 else gains

        if isinstance(self.net, Omni_Amplitudes_from_MLP):
            loss = self.criterion[0](gains, data['target_common_slope_amps'])
        else:
            loss = self.criterion[0](self.net.get_directional_amplitudes(
                data['sph_directions']), data['target_common_slope_amps'])

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def valid_step(self, data):
        """Validate each batch"""
        # batch processing
        self.optimizer.zero_grad()
        gains = self.net(data)
        # if batch has a single data point
        gains = gains.unsqueeze(0) if gains.ndim == 1 else gains
        if isinstance(self.net, Omni_Amplitudes_from_MLP):
            loss = self.criterion[0](gains, data['target_common_slope_amps'])
        else:
            loss = self.criterion[0](self.net.get_directional_amplitudes(
                data['sph_directions']), data['target_common_slope_amps'])
        return loss.item()

    def print_results(self, e: int, e_time):
        """Print training results"""
        print(get_str_results(epoch=e, train_loss=self.train_loss,
                              time=e_time))

    def save_model(self, e: int):
        """Save the model parameters at each epoch"""
        dir_path = os.path.join(self.train_dir, 'checkpoints')
        # create checkpoint folder
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # save model
        torch.save(self.net.state_dict(),
                   os.path.join(dir_path, 'model_e' + str(e) + '.pt'))
