from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from slope2noise.utils import decay_kernel
import torch
from torch import nn

from diff_gfdn.utils import db, ms_to_samps


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
        The decay kernel is of shape num_slopes x time
        Args:
            amps_pred (torch.Tensor): predicted amplitudes of shape 
                                      batch_size x num_slopes / batch_size x num_directions x num_slopes
            amps_true (torch.Tensor): true amplitudes of same shape
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
            edc_loss = torch.abs(edc_true - edc_pred).mean()

        return edc_loss
