import os
import time

import torch
from torch.utils.data import DataLoader
import torchaudio
from tqdm import trange

from ..config.config import ColorlessFDNConfig, TrainerConfig
from ..utils import get_frequency_samples, get_response, get_str_results
from .losses import amse_loss, sparsity_loss
from .model import ColorlessFDN


class ColorlessFDNTrainer:
    """Class for training a colorless FDN"""

    def __init__(self, net: ColorlessFDN, trainer_config: TrainerConfig,
                 colorless_fdn_config: ColorlessFDNConfig):
        """
        Args:
            net (ColorlessFDN): the colorless FDN module (pre-initialised)
            trainer_config (TrainerConfig): config containing training params
        """
        self.net = net
        self.device = trainer_config.device
        self.max_epochs = colorless_fdn_config.max_epochs
        self.patience = 5
        self.early_stop = 0
        self.alpha = colorless_fdn_config.alpha
        self.train_dir = trainer_config.train_dir + "colorless-fdn/"

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=colorless_fdn_config.lr)
        self.criterion = [amse_loss(), sparsity_loss()]
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=10,
                                                         gamma=0.1)
        self.z = get_frequency_samples(int(self.net.sample_rate * 2),
                                       device=self.device)
        self.z_batch = get_frequency_samples(colorless_fdn_config.batch_size,
                                             device=self.device)
        self.normalize()

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
        inputs, labels = data
        self.optimizer.zero_grad()
        H = self.net(inputs)
        loss = self.criterion[0](H, labels) + self.alpha * self.criterion[1](
            self.net.feedback_loop.ortho_param(
                self.net.feedback_loop.random_feedback_matrix))

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def valid_step(self, data):
        """Validate each batch"""
        # batch processing
        inputs, labels = data
        self.optimizer.zero_grad()
        H = self.net(inputs)
        loss = self.criterion[0](H, labels) + self.alpha * self.criterion[1](
            self.net.feedback_loop.ortho_param(
                self.net.feedback_loop.random_feedback_matrix))
        return loss.item()

    @torch.no_grad()
    def normalize(self):
        """Average enery normalization"""
        H, _ = get_response(self.z_batch, self.net)
        energyH = torch.sum(torch.pow(torch.abs(H), 2)) / torch.tensor(
            H.size(0))

        # apply energy normalization on input and output gains only
        for name, prm in self.net.named_parameters():
            if name in ('input_gains', 'output_gains'):
                prm.data.copy_(torch.div(prm.data, torch.pow(energyH, 1 / 4)))

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

    @torch.no_grad()
    def save_ir(self,
                directory: str,
                filename='colorless_fdn_ir.wav',
                norm=False):
        """Save impulse response"""
        _, h = get_response(self.z, self.net)
        if norm:
            h = torch.div(h, torch.max(torch.abs(h)))
        filepath = os.path.join(directory, filename)
        torchaudio.save(filepath,
                        torch.stack((h, h), dim=1).cpu(),
                        self.net.sample_rate,
                        bits_per_sample=32,
                        channels_first=False)
