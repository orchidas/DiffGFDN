import os
from pathlib import Path
import time
from typing import List, Optional, Union

from loguru import logger
from numpy.typing import NDArray
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from diff_gfdn.utils import get_str_results

from .config import DNNType, SpatialSamplingConfig
from .losses import spatial_edc_loss, spatial_mse_loss, spatial_smoothness_loss
from .model import Directional_Beamforming_Weights, Omni_Amplitudes_from_MLP

# pylint: disable=W0632
# flake8: noqa:E231


class SpatialSamplingTrainer:
    """Class for training an MLP that learns the spatial mapping of the common slope amplitudes"""

    def __init__(
        self,
        net: Union[Omni_Amplitudes_from_MLP, Directional_Beamforming_Weights],
        trainer_config: SpatialSamplingConfig,
        grid_spacing_m: float,
        sampling_rate: float = 44100,
        ir_len_ms: float = 2000,
        dataset_ref: Dataset = None,
        common_decay_times: Optional[List] = None,
        receiver_positions: Optional[NDArray] = None,
    ):
        """
        Args:
            net (Gains_from_MLP): the MLP module (pre-initialised)
            trainer_config (SpatialSamplingConfig): config containing training params
            grid_spacing_m (float): the grid spacing in m
            sampling_rate (float): sampling rate in Hz
            ir_len_ms (float): length of the RIR in ms
            dataset_ref (Dataset): reference to the dataset object
            common_decay_tomes (Optional, list): if using EDC loss, then the common decay times
            receiver_positions (Optiona, list): if using spatial smoothness loss, 
                                                then the list of receiver positions of shape num_receiversx3
        """
        self.net = net
        self.network_type = trainer_config.network_type
        self.device = trainer_config.device
        self.grid_spacing_m = grid_spacing_m
        self.max_epochs = trainer_config.max_epochs
        self.patience = 5
        self.early_stop = 0
        self.train_dir = trainer_config.train_dir
        self.dataset_ref = dataset_ref

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=trainer_config.lr)
        if common_decay_times is None:
            logger.info("Using spatial MSE loss")
            self.criterion = [spatial_mse_loss()]
            self.mse_loss_weight = 1.0
        else:
            logger.info("Using spatial EDC loss")
            self.criterion = [
                spatial_edc_loss(common_decay_times, ir_len_ms, sampling_rate)
            ]
            self.edc_loss_weight = 1.0
            self.num_slopes = common_decay_times.shape[-1]

        if receiver_positions is not None and isinstance(
                self.net, Directional_Beamforming_Weights):
            logger.info("Adding spatial smoothness loss")
            self.criterion.append(
                spatial_smoothness_loss(
                    torch.tensor(
                        receiver_positions,
                        device=self.device,
                    ),
                    savepath=Path(self.train_dir +
                                  "kernel_weights.png").resolve()))
            self.spatial_smoothness_weight = 1e-1

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=20,
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
            spatial_loss = 0
            for data in train_dataset:

                if len(self.criterion
                       ) == 1 or self.network_type == DNNType.CNN:
                    cur_epoch_loss = self.train_step(data)
                else:
                    cur_epoch_loss, cur_spatial_loss = self.train_step(data)
                    spatial_loss += cur_spatial_loss

                epoch_loss += cur_epoch_loss

            print(
                f"Spatial smoothness loss at epoch {epoch} is {spatial_loss / len(train_dataset):.3f}"
            )

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
        mlp_output = self.net(data)
        # if batch has a single data point
        mlp_output = mlp_output.unsqueeze(
            0) if mlp_output.ndim == 1 else mlp_output

        if isinstance(self.net, Omni_Amplitudes_from_MLP):
            loss = self.criterion[0](mlp_output,
                                     data['target_common_slope_amps'])
        else:
            loss = 0.0
            # convert MLP output to directional output by multipying
            # with SH matrix
            est_dir_output = self.net.get_directional_amplitudes()
            target_dir_output = data['target_common_slope_amps'].float()

            if self.network_type == DNNType.CNN:
                # make sure no region outside the boundary is chosen for EDC loss calculation
                binary_floor_mask = self.dataset_ref.get_binary_mask(
                    data['mesh_2D'])

                # this chooses a subset of the output that lies within the boundaries
                masked_est_output = est_dir_output[binary_floor_mask.view(-1)]
                masked_target_output = target_dir_output[
                    binary_floor_mask.view(-1)]

                loss += self.edc_loss_weight * self.criterion[0](
                    masked_est_output, masked_target_output)

            elif self.network_type == DNNType.MLP:
                if len(self.criterion) > 1:
                    spatial_loss = self.spatial_smoothness_weight * self.criterion[
                        1](data['listener_position'], est_dir_output)
                    loss += spatial_loss
                loss += self.edc_loss_weight * self.criterion[0](
                    est_dir_output, target_dir_output)

        loss.backward()
        self.optimizer.step()
        return loss.item() if len(
            self.criterion) == 1 or self.network_type == DNNType.CNN else (
                loss.item(), spatial_loss.item())

    def valid_step(self, data):
        """Validate each batch"""
        # batch processing
        self.optimizer.zero_grad()
        mlp_output = self.net(data)
        # if batch has a single data point
        mlp_output = mlp_output.unsqueeze(
            0) if mlp_output.ndim == 1 else mlp_output
        if isinstance(self.net, Omni_Amplitudes_from_MLP):
            loss = self.criterion[0](mlp_output,
                                     data['target_common_slope_amps'])
        else:
            loss = 0.0
            # convert MLP output to directional output by multipying
            # with SH matrix
            est_dir_output = self.net.get_directional_amplitudes()
            target_dir_output = data['target_common_slope_amps'].float()

            if self.network_type == DNNType.CNN:
                # make sure no region outside the boundary is chosen for loss calculation
                # size H, W
                binary_floor_mask = self.dataset_ref.get_binary_mask(
                    data['mesh_2D'])

                # this chooses a subset of the output that lies within the boundaries
                masked_est_output = est_dir_output[binary_floor_mask.view(-1)]
                masked_target_output = target_dir_output[
                    binary_floor_mask.view(-1)]

                loss += self.edc_loss_weight * self.criterion[0](
                    masked_est_output, masked_target_output)

            elif self.network_type == DNNType.MLP:
                if len(self.criterion) > 1:
                    loss += self.spatial_smoothness_weight * self.criterion[1](
                        data['listener_position'], est_dir_output)

                loss += self.edc_loss_weight * self.criterion[0](
                    est_dir_output, target_dir_output)

        return loss.item()

    def print_results(self, e: int, e_time):
        """Print training results"""
        print(get_str_results(epoch=e, train_loss=self.train_loss,
                              time=e_time))

    def save_model(self, e: int):
        """Save the model parameters at each epoch"""
        dir_path = os.path.join(
            self.train_dir,
            f'checkpoints/grid_resolution={self.grid_spacing_m:.1f}/')
        # create checkpoint folder
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # save model
        torch.save(self.net.state_dict(),
                   os.path.join(dir_path, 'model_e' + str(e) + '.pt'))
