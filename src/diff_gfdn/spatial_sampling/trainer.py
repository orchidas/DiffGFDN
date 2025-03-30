import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from ..gain_filters import Gains_from_MLP
from ..utils import db, get_str_results
from .config import SpatialSamplingConfig

# pylint: disable=W0632


class spatial_mse_loss(nn.Module):
    """Means absolute error in dB between the true and predicted spatial embeddings"""

    def forward(self, y_pred: torch.tensor, y_true: torch.tensor):
        """
        Args:
            y_pred (torch.tensor): output of the model, array of batch_size x num_slopes
            y_true (torch.tensor): expected output, array of size batch_size x num_slopes
        """
        loss = torch.mean(torch.abs(db(y_pred) - db(y_true)),
                          dim=0) / torch.mean(torch.abs(db(y_true)), dim=0)
        return torch.sum(loss)


class SpatialSamplingTrainer:
    """Class for training an MLP that learns the spatial mapping of the common slope amplitudes"""

    def __init__(
        self,
        net: Gains_from_MLP,
        trainer_config: SpatialSamplingConfig,
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
        self.criterion = [spatial_mse_loss()]

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
            epoch_loss = 0
            for data in valid_dataset:
                epoch_loss += self.valid_step(data)
            self.valid_loss.append(epoch_loss / len(valid_dataset))
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
        loss = self.criterion[0](gains[..., 0].squeeze(),
                                 data['target_common_slope_amps'])

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def valid_step(self, data):
        """Validate each batch"""
        # batch processing
        self.optimizer.zero_grad()
        gains = self.net(data)
        loss = self.criterion[0](gains[..., 0].squeeze(),
                                 data['target_common_slope_amps'])
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
