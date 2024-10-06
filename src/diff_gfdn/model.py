from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch import nn

from .config.config import FeedbackLoopConfig, OutputFilterConfig
from .feedback_loop import FeedbackLoop
from .filters import decay_times_to_gain_filters
from .gain_filters import SVF_from_MLP
from .utils import absorption_to_gain_per_sample, to_complex

# pylint: disable=W0718


class DiffGFDN(nn.Module):

    def __init__(self,
                 sample_rate: int,
                 num_groups: int,
                 delays: List[int],
                 room_dims: List[Tuple],
                 device: torch.device,
                 feedback_loop_config: FeedbackLoopConfig,
                 output_filter_config: OutputFilterConfig,
                 use_absorption_filters: bool,
                 absorption_coeffs: Optional[List] = None,
                 common_decay_times: Optional[List] = None,
                 band_centre_hz: Optional[List] = None):
        """
        Differentiable GFDN module.
        Args:
            sample_rate (int): sampling rate of the FDN
            num_groups (int): number of rooms in coupled space
            delays (list): list of delay line lengths (integer)
            room_dims (list): dimensions of each room as a tuple
            device: GPU or CPU for training
            feedback_loop_config (FeedbackLoopConfig): config file for training the feedback loop
            output_filter_config (OutputFilterConfig): config file for training the output SVF filters
            use_absorption_filters (bool): whether to use scalar absorption gains or filters
            absorption_coeffs (optional, list): uniform absorption coefficients (one for each room)
            common_decay_times (optional, list): list of common decay times (one for each room)
            band_centre_hz (optional, list): frequencies where common decay times are measured
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device
        # input parameters
        self.num_groups = num_groups
        self.absorption_coeffs = absorption_coeffs
        self.delays = torch.tensor(delays, dtype=torch.int32)
        self.num_delay_lines = len(delays)
        self.num_delay_lines_per_group = int(self.num_delay_lines /
                                             self.num_groups)
        self.use_absorption_filters = use_absorption_filters
        self.delays_by_group = [
            self.delays[i:i + self.num_delay_lines_per_group] for i in range(
                0, self.num_delay_lines, self.num_delay_lines_per_group)
        ]
        if self.use_absorption_filters:
            # this will be of size (num_groups, num_del_per_group, numerator (filter_order), denominator(filter_order))
            self.gain_per_sample = torch.tensor([
                    decay_times_to_gain_filters(band_centre_hz,
                                                common_decay_times[:, i],
                                                self.delays_by_group[i],
                                                self.sample_rate)
                    for i in range(self.num_groups)
                ], device=self.device)
            self.filter_order = self.gain_per_sample.shape[-2]
            self.gain_per_sample = self.gain_per_sample.view(
                self.num_delay_lines, self.filter_order, 2)

        else:
            self.gain_per_sample = torch.flatten(
                torch.tensor([
                        absorption_to_gain_per_sample(room_dims[i],
                                                      absorption_coeffs[i],
                                                      self.delays_by_group[i],
                                                      self.sample_rate)[1]
                        for i in range(self.num_groups)
                    ], device=self.device))

        logger.info(f'Gains for delay lines are {self.gain_per_sample}')

        # here are the different operating blocks
        self.delays = self.delays.to(self.device)
        self.input_gains = nn.Parameter(
            torch.randn(self.num_delay_lines, 1) / self.num_delay_lines)
        self.feedback_loop = FeedbackLoop(
            self.num_groups, self.num_delay_lines_per_group, self.delays,
            self.gain_per_sample, self.use_absorption_filters,
            feedback_loop_config.coupling_matrix_type,
            feedback_loop_config.pu_matrix_order)
        self.output_filters = SVF_from_MLP(
            output_filter_config.num_biquads_svf, self.num_delay_lines,
            output_filter_config.num_fourier_features,
            output_filter_config.num_hidden_layers,
            output_filter_config.num_neurons_per_layer,
            output_filter_config.encoding_type,
            output_filter_config.apply_pooling)

    def forward(self, x: Dict) -> torch.tensor:
        """
        Compute H(z) = c(z)^T (D - A(z)Gamma(z))^{-1} b(z) + d(z)
        Args:
            x(dict) : input feature dict
        """
        z = x['z_values']

        num_freq_pts = len(z)
        # this is of size B x Ndel x num_freq_points
        C = self.output_filters(x)
        self.batch_size = C.shape[0]
        # this is also of size B x Ndel x num_freq_points
        B = to_complex(
            self.input_gains.expand(self.batch_size, self.num_delay_lines,
                                    num_freq_pts))
        # get the output of the feedback loop, this is of size num_freq_points x Ndel x Ndel
        P = self.feedback_loop(z)
        # C.T @ P of size B x Ndel x num_freq_pts
        Htemp = torch.einsum('knb, knm -> knb', C.permute(-1, 1, 0),
                             P).permute(-1, 1, 0)
        # C.T @ P @ B + d(z)
        direct_filter = x['target_early_response']
        H = torch.einsum('bmk, bmk -> bk', Htemp, B) + direct_filter
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
        param_np['individual_mixing_matrix'] = self.feedback_loop.M.squeeze(
        ).cpu().numpy()
        try:
            param_np['coupling_matrix'] = self.feedback_loop.nd_rotation(
                self.feedback_loop.alpha).squeeze().cpu().numpy()
            param_np[
                'coupled_feedback_matrix'] = self.feedback_loop.get_coupled_feedback_matrix(
                ).squeeze().cpu().numpy()
        except Exception:
            logger.warning('Parameter not initialised yet in FeedbackLoop!')
        try:
            param_np[
                'output_svf_params'] = self.output_filters.svf_params.squeeze(
                ).cpu().numpy()
            param_np['output_biquad_coeffs'] = [[
                torch.cat(
                    (self.output_filters.biquad_cascade[b][n].num_coeffs,
                     self.output_filters.biquad_cascade[b][n].den_coeffs),
                    dim=-1).squeeze().cpu().numpy()
                for n in range(self.num_delay_lines)
            ] for b in range(self.batch_size)]
        except Exception as e:
            logger.warning(e)

        return param_np
