import os
import time
from pathlib import Path
from typing import Dict

import torch
import torchaudio
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import trange

from .config.config import TrainerConfig
from .losses import edr_loss
from .model import DiffGFDN
from .utils import get_response, get_str_results


# flake8: noqa: E231
class Trainer:

    def __init__(self, net: DiffGFDN, trainer_config: TrainerConfig):
        """Class to train the DiffGFDN"""
        self.net = net
        self.device = trainer_config.device
        self.max_epochs = trainer_config.max_epochs
        self.patience = 5
        self.early_stop = 0
        self.train_dir = Path(trainer_config.train_dir).resolve()
        self.ir_dir = Path(trainer_config.ir_dir).resolve()

        self.optimizer = torch.optim.Adam(net.parameters(),
                                          lr=trainer_config.lr)
        self.criterion = edr_loss(self.net.sample_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=10,
                                                         gamma=0.1)

    def to_device(self):
        """Return the device to train on - CPU or GPU"""
        for i in range(len(self.criterion)):
            self.criterion[i] = self.criterion[i].to(self.device)

    def train(self, train_dataset: DataLoader):
        """Train the network"""
        self.train_loss = []

        st = time.time()  # start time
        for epoch in trange(self.max_epochs, desc='Training'):
            logger.info(f'Epoch #{epoch}')
            st_epoch = time.time()

            # training
            epoch_loss = 0
            for data in train_dataset:
                epoch_loss += self.train_step(data)
            self.scheduler.step()
            self.train_loss.append(epoch_loss / len(train_dataset))
            et_epoch = time.time()
            self.save_model(epoch)
            self.print_results(epoch, et_epoch - st_epoch)

            # early stopping
            if epoch >= 1:
                if abs(self.train_loss[-2] - self.train_loss[-1]) <= 0.0001:
                    self.early_stop += 1
                else:
                    self.early_stop = 0
            if self.early_stop == self.patience:
                break

        et = time.time()  # end time
        logger.info('Training time: {:.3f}s'.format(et - st))

        # save the trained IRs
        logger.info("Saving the trained IRs...")
        for data in train_dataset:
            position = data['listener_position']
            self.save_ir(data, directory=self.ir_dir, pos_list=position)

    def train_step(self, data):
        """Single step of training"""
        self.optimizer.zero_grad()
        H = self.net(data)
        loss = self.criterion(data['target_rir_response'], H)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def validate(self, valid_dataset: DataLoader):
        """Validate the training with unseen data and save the resulting IRs"""
        total_loss = 0
        self.valid_loss = []
        for data in valid_dataset:
            position = data['listener_position']
            logger.info("Running the network for new batch of positiions")
            H = self.save_ir(data, directory=self.ir_dir, pos_list=position)
            loss = self.criterion(data['target_rir_response'], H)
            cur_loss = loss.item()
            total_loss += cur_loss
            self.valid_loss.append(cur_loss)
            logger.info(
                f"The validation loss for the current position is {cur_loss:.4f}"
            )

        net_valid_loss = total_loss / len(valid_dataset)
        logger.info(f"The net validation loss is {net_valid_loss:.4f}")

    def print_results(self, e: int, e_time):
        """Print results of training"""
        print(get_str_results(epoch=e, train_loss=self.train_loss,
                              time=e_time))

    def save_model(self, e: int):
        """Save the model at epoch number e"""
        dir_path = os.path.join(self.train_dir, 'checkpoints')
        # create checkpoint folder
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # save model
        torch.save(self.net.state_dict(),
                   os.path.join(dir_path, 'model_e' + str(e) + '.pt'))

    @torch.no_grad()
    def save_ir(self,
                input_features: Dict,
                directory: str,
                pos_list: torch.tensor,
                filename_prefix: str = "ir",
                norm=True) -> torch.tensor:
        """
        Save the impulse response generated from the model
        Args:
            input_features (Dict): dictionary of input features
            directory (str): where to save the audio
            pos_list (torch.tensor): B x 3 position coordinates
            norm (bool): whether to normalise the RIR
        Returns:
            torch.tensor - the frequency response at the given input features
        """
        H, h = get_response(input_features, self.net)
        if norm:
            h = torch.div(h, torch.max(torch.abs(h)))

        for num_pos in range(pos_list.shape[0]):
            filename = (
                f'{filename_prefix}_({pos_list[num_pos,0]:.2f}, '
                f'{pos_list[num_pos, 1]:.2f}, {pos_list[num_pos, 2]:.2f}).wav')

            filepath = os.path.join(directory, filename)
            torchaudio.save(filepath,
                            h[num_pos, :].cpu(),
                            int(self.net.sample_rate),
                            bits_per_sample=32,
                            channels_first=False)
        return H
