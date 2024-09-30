from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from .config.config import FeedbackLoopConfig, OutputFilterConfig
from .feedback_loop import FeedbackLoop
from .gain_filters import SVF_from_MLP
from .utils import absorption_to_gain_per_sample, to_complex


class DiffGFDN(nn.Module):

    def __init__(self, sample_rate: int, num_groups: int, delays: List[int],
                 absorption_coeffs: List[float], room_dims: List[Tuple],
                 device: torch.device,
                 feedback_loop_config: FeedbackLoopConfig,
                 output_filter_config: OutputFilterConfig):
        """
        Differentiable GFDN module.
        Args:
            sample_rate (int): sampling rate of the FDN
            num_groups (int): number of rooms in coupled space
            delays (list): list of delay line lengths (integer)
            absorption_coeffs (list): uniform absorption coefficients (one for each room)
            room_dims (list): dimensions of each room as a tuple
            device: GPU or CPU for training
            feedback_loop_config (FeedbackLoopConfig): config file for training the feedback loop
            output_filter_config (OutputFilterConfig): config file for training the output SVF filters

        """
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device
        # input parameters
        self.num_groups = num_groups
        assert len(absorption_coeffs) == self.num_groups
        self.absorption_coeffs = absorption_coeffs
        self.delays = torch.tensor(delays).squeeze()
        self.num_delay_lines = len(delays)
        self.num_delay_lines_per_group = int(self.num_delay_lines /
                                             self.num_groups)
        self.delays_by_group = np.array([
            self.delays[i:i + self.num_delay_lines_per_group] for i in range(
                0, self.num_delay_lines, self.num_delay_lines_per_group)
        ])
        self.gain_per_sample = torch.flatten(
            torch.from_numpy(
                np.array([
                    absorption_to_gain_per_sample(room_dims[i],
                                                  absorption_coeffs[i],
                                                  self.delays_by_group[i],
                                                  self.sample_rate)[1]
                    for i in range(self.num_groups)
                ])))

        # here are the different operating blocks
        self.input_gains = nn.Parameter(
            torch.randn(self.num_delay_lines, 1) / self.num_delay_lines)
        self.feedback_loop = FeedbackLoop(
            self.num_groups, self.num_delay_lines_per_group, self.delays,
            self.absorption_coeffs, feedback_loop_config.coupling_matrix_type,
            feedback_loop_config.coupling_matrix_order)
        self.output_filters = SVF_from_MLP(
            output_filter_config.num_biquads_svf, self.num_delay_lines,
            output_filter_config.num_fourier_features,
            output_filter_config.num_hidden_layers,
            output_filter_config.num_neurons_per_layer)

    def forward(self, x: Dict) -> torch.tensor:
        """
        Compute H(z) = c(z)^H (D - A(z)Gamma(z))^{-1} b(z) + d(z)
        Args:
            x(dict) : input feature dict
        """
        z = x['z_values']
        num_freq_pts = len(z)
        # this is of size Ndel x num_freq_points
        C = self.output_filters(x)
        # this is also of size Ndel x num_freq_points
        B = to_complex(
            self.input_gains.expand(self.num_delay_lines, num_freq_pts))
        # get the output of the feedback loop
        P = self.feedback_loop(z)
        # C.T @ P
        Htemp = torch.einsum('kn, knm -> km', C.T, P)
        # C.T @ P @ B + d(z)
        direct_filter = x['target_early_response']
        H = torch.diagonal(torch.mm(Htemp, B)) + direct_filter
        return H

    def get_parameters(self) -> Tuple:
        """Return the parameters as a tuple"""
        delays = self.delays
        gain_per_sample = self.gain_per_sample
        b = self.input_gains
        (M, Phi, _, _,
         coupled_feedback_matrix) = self.feedback_loop.get_parameters()
        (svf_params, biquad_coeffs) = self.output_filters.get_parameters()
        return (delays, gain_per_sample, b, M, Phi, coupled_feedback_matrix,
                svf_params, biquad_coeffs)

    @torch.no_grad()
    def get_param_dict(self) -> Dict:
        """Return the parameters as a dict"""
        param_np = {}
        param_np['delays'] = self.delays.squeeze().cpu().numpy()
        param_np['gains_per_sample'] = self.gain_per_sample.squeeze().cpu(
        ).numpy()
        param_np['input_gains'] = self.input_gains.squeeze().cpu().numpy()
        param_np['coupling_matrix'] = self.feedback_loop.phi.squeeze().cpu(
        ).numpy()
        param_np['individual_mixing_matrix'] = self.feedback_loop.M.squeeze(
        ).cpu().numpy()
        param_np[
            'coupled_feedback_matrix'] = self.feedback_loop.coupled_feedback_matrix.squeeze(
            ).cpu().numpy()
        param_np['output_svf_params'] = self.svf_params.squeeze().cpu().numpy()
        param_np['output_biquad_coeffs'] = torch.cat(
            (self.output_filters.biquad_cascade.num_coeffs,
             self.output_filters.biquad_cascade.den_coeffs),
            dim=-1).squeeze().cpu().numpy()
        return param_np
