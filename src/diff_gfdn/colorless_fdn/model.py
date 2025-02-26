from typing import Dict, List, Tuple

import torch
from torch import nn

from ..absorption_filters import decay_times_to_gain_per_sample
from ..config.config import CouplingMatrixType
from ..feedback_loop import FeedbackLoop
from ..utils import to_complex


class ColorlessFDN(nn.Module):
    """
    Colorless FDN module that finds the optimum feedback matrix 
    and input/output gains for given delay line lengths
    """

    def __init__(
        self,
        sample_rate: int,
        delays: List[int],
        device: torch.device,
        nominal_t60: float = 10.0,
    ):
        """
        Args:
        sample_rate (float): sampling frequency
        delays (list): delay line lengths in samples
        nominal_t60 (float): shared broadband T60 amongst all delay lines
        device (torch.device): training device
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device
        self.delays = torch.tensor(delays,
                                   dtype=torch.float32,
                                   device=self.device)
        self.delays = self.delays.to(self.device)
        self.num_delay_lines = len(delays)

        # learnable input and output gains
        self.input_gains = nn.Parameter(
            (2 * torch.randn(self.num_delay_lines, 1) - 1) /
            self.num_delay_lines)

        self.output_gains = nn.Parameter(
            (2 * torch.randn(self.num_delay_lines, 1) - 1) /
            self.num_delay_lines)

        self.gain_per_sample = torch.tensor(decay_times_to_gain_per_sample(
            nominal_t60, delays, self.sample_rate),
                                            device=self.device)

        self.feedback_loop = FeedbackLoop(
            num_groups=1,
            num_delay_lines_per_group=self.num_delay_lines,
            delays=self.delays,
            gains=self.gain_per_sample,
            use_absorption_filters=False,
            coupling_matrix_type=CouplingMatrixType.RANDOM)

    def forward(self, z: torch.tensor) -> Tuple:
        """
        Calculate the FDN's frequency response for points on the unit circle
        Returns:
            H: array of size num_freq_bins , outputcol FDN
            H_per_del: num_delay_lines x num_freq_bins, 
                          output of each delay line, weighted by c_i
        """
        num_freq_pts = len(z)

        C = to_complex(
            self.output_gains.expand(self.num_delay_lines, num_freq_pts))

        # this is also of size Ndel x 1
        B = to_complex(self.input_gains)

        # get the output of the feedback loop, this is of size num_freq_points x Ndel x Ndel
        P = self.feedback_loop(z)

        # C.T @ P of size Ndel x num_freq_pts
        Htemp = torch.einsum('kn, knm -> km', C.permute(-1, 0), P)
        # C.T @ P @ B
        H = torch.einsum('ik, kj -> ij', Htemp, B).squeeze()

        # output of each delay line scaled by c_i
        H_tmp = torch.einsum('kn, knm -> knm', C.permute(1, 0),
                             P).permute(1, -1, 0)
        H_per_del = torch.einsum('nmk, mk -> nk', H_tmp, B)

        return (H, H_per_del)

    @torch.no_grad()
    def get_param_dict(self) -> Dict:
        """Return the parameters as a dict"""
        param_np = {}
        param_np['delays'] = self.delays.squeeze().cpu().numpy()
        param_np['gains_per_sample'] = self.gain_per_sample.squeeze().cpu(
        ).numpy()
        param_np['input_gains'] = self.input_gains.squeeze().cpu().numpy()
        param_np['output_gains'] = self.output_gains.squeeze().cpu().numpy()
        # any unitary matrix without any specific coupling structure
        param_np[
            'feedback_matrix'] = self.feedback_loop.random_feedback_matrix.squeeze(
            ).cpu().numpy()
        return param_np
