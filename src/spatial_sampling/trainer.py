import os
from pathlib import Path
import time
from typing import List, Optional, Union

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from slope2noise.utils import decay_kernel
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from diff_gfdn.utils import db, get_str_results, ms_to_samps

from .config import SpatialSamplingConfig
from .model import Directional_Beamforming_Weights_from_MLP, Omni_Amplitudes_from_MLP

# pylint: disable=W0632
# flake8: noqa:E231


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


class spatial_smoothness_loss(nn.Module):
    """
    Spatial smoothness loss when computing beamforming weights for finding directional
    amplitudes. Without this loss the amplitudes vary sharply across space
    """

    def __init__(self,
                 all_receiver_pos: torch.Tensor,
                 savepath: Optional[str] = None):
        """
        Pre-compute kernel matrix based on euclidean distance between each position pair in the dataset
        This will be used while computing the spatial variation loss.
        Args:
            all_receiver_pos (torch.Tensor): num_receiver_pos x 3 tensor of cartesian coordinates
            sigma (float): std of the Gaussian kernel. A smaller sigma ensures more localised loss
        """
        super().__init__()
        # pairwise euclidean distance between all points
        pairwise_distance = torch.cdist(all_receiver_pos, all_receiver_pos)
        self.all_receiver_pos = all_receiver_pos
        sigma = 1.0 / np.sqrt(2)
        # RBF kernel
        self.kernel_weights = torch.exp(-pairwise_distance**2 / (2 * sigma**2))
        # normalise weights along rows - rows sum to 1
        self.kernel_weights /= (self.kernel_weights.sum(dim=1, keepdim=True) +
                                1e-10)
        self.plot_kernel_matrix(savepath)

    def plot_kernel_matrix(self, savepath: Optional[str] = None):
        """Plot the kernel matrix"""
        plt.figure(figsize=(6, 5))
        plt.imshow(self.kernel_weights.cpu().detach().numpy(), cmap='plasma')
        plt.colorbar(label="Affinity")
        plt.title("Full Kernel Matrix")
        plt.xlabel("Position index")
        plt.ylabel("Position index")
        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)

    def find_position_idx(self, cur_positions: torch.Tensor) -> torch.Tensor:
        """Find the indices for all positions in cur_positions in all_receiver_pos"""
        # Step 1: Compare each cur_positions[i] to every row in all_receiver_pos
        matches = (self.all_receiver_pos[None, :, :] == cur_positions[:,
                                                                      None, :]
                   )  # shape (B, M, 3)

        # Step 2: Check all elements match along last dim (3D match)
        match_rows = matches.all(dim=-1)  # shape (B, M)

        # Step 3: For each cur_positions[i], find the matching index in all_receiver_pos
        # shape (B, 2), each row is [i (in cur_positions), j (in all_receiver_pos)]
        indices = match_rows.nonzero(as_tuple=False)
        # We want the indices in all_receiver_pos corresponding to rows in cur_positions
        # shape (B,), indices in cur_positions
        pos_idxs = indices[:, 1]
        return pos_idxs

    def forward(self, cur_positions: torch.Tensor, cur_weights: torch.Tensor):
        """
        Encourage he deviation between the weights of neighbouring positions
        cur_positions (torch.Tensor): batch_size x 3 matrix of cartesian coordinates
        cur_weights (torch.Tensor): batch_size x (N_sp+1)**2 x num_slopes output of MLP
        """
        # order the position indices in the estimated data according to the
        # reference dataset
        pos_idx = self.find_position_idx(cur_positions)
        kernel_weights_batch = self.kernel_weights[pos_idx][:,
                                                            pos_idx]  # (B, B)

        # Compute the pairwise differences in the weights for each group across all positions

        # Shape: (num_groups, batch_size, num_dir), pairwise difference in weights
        cur_weights_mod = cur_weights.permute(1, 0, -1)
        # this has the shape num_groups, batch_size, batch_size
        weight_diff = torch.cdist(cur_weights_mod, cur_weights_mod)
        # Compute the smoothness loss: sum of squared differences weighted by the kernel values
        smoothness_loss = torch.einsum('kbp, bp -> k', weight_diff,
                                       kernel_weights_batch.float())

        # Sum the losses across all groups. The negative sign encourages spatial variation
        return -smoothness_loss.sum()


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
            edc_true = db(torch.einsum('bjk, kt -> bjt', amps_true,
                                       self.envelopes),
                          is_squared=True)
            edc_pred = db(torch.einsum('bjk, kt -> bjt', amps_pred,
                                       self.envelopes),
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
        sampling_rate: float = 44100,
        ir_len_ms: float = 2000,
        common_decay_times: Optional[List] = None,
        receiver_positions: Optional[NDArray] = None,
    ):
        """
        Args:
            net (Gains_from_MLP): the MLP module (pre-initialised)
            trainer_config (SpatialSamplingConfig): config containing training params
            sampling_rate (float): sampling rate in Hz
            ir_len_ms (float): length of the RIR in ms
            common_decay_tomes (Optional, list): if using EDC loss, then the common decay times
            receiver_positions (Optiona, list): if using spatial smoothness loss, 
                                                then the list of receiver positions of shape num_receiversx3
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
            self.mse_loss_weight = 1.0
        else:
            logger.info("Using spatial EDC loss")
            self.criterion = [
                spatial_edc_loss(common_decay_times, ir_len_ms, sampling_rate)
            ]
            self.edc_loss_weight = 1.0

        if receiver_positions is not None and isinstance(
                self.net, Directional_Beamforming_Weights_from_MLP):
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
            spatial_loss = 0
            for data in train_dataset:
                # epoch_loss += self.train_step(data)
                cur_epoch_loss, cur_spatial_loss = self.train_step(data)
                epoch_loss += cur_epoch_loss
                spatial_loss += cur_spatial_loss

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
            directional_output = self.net.get_directional_amplitudes(
                data['sph_directions'])

            if len(self.criterion) > 1:
                spatial_loss = self.spatial_smoothness_weight * self.criterion[
                    1](data['listener_position'], directional_output)
                loss += spatial_loss

            loss += self.edc_loss_weight * self.criterion[0](
                directional_output, data['target_common_slope_amps'].float())

        loss.backward()
        self.optimizer.step()
        # return loss.item()
        return loss.item(), spatial_loss.item()

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
            directional_output = self.net.get_directional_amplitudes(
                data['sph_directions'])
            if len(self.criterion) > 1:
                loss += self.spatial_smoothness_weight * self.criterion[1](
                    data['listener_position'], directional_output)
            loss += self.edc_loss_weight * self.criterion[0](
                directional_output, data['target_common_slope_amps'].float())

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
