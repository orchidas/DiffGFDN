from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from loguru import logger
import numpy as np
from scipy.signal import butter
import torch
from torch import nn

from .absorption_filters import decay_times_to_gain_filters_geq, decay_times_to_gain_per_sample
from .colorless_fdn.utils import ColorlessFDNResults
from .config.config import CouplingMatrixType, FeedbackLoopConfig, OutputFilterConfig
from .feedback_loop import FeedbackLoop
from .filters.geq import eq_freqs
from .gain_filters import BiquadCascade, Gains_from_MLP, ScaledSigmoid, SOSFilter, SVF, SVF_from_MLP
from .utils import to_complex

# pylint: disable=W0718, E1136, E1137


class DiffGFDN(nn.Module):
    """Parent module to do end-to-end optimisation of a Differentiable GFDN"""

    def __init__(
            self,
            sample_rate: int,
            num_groups: int,
            delays: List[int],
            device: torch.device,
            feedback_loop_config: FeedbackLoopConfig,
            use_absorption_filters: bool,
            common_decay_times: List,
            band_centre_hz: Optional[List] = None,
            colorless_fdn_params: Optional[List[ColorlessFDNResults]] = None,
            use_colorless_loss: bool = False):
        """
        Args:
            sample_rate (int): sampling rate of the FDN
            num_groups (int): number of rooms in coupled space
            delays (list): list of delay line lengths (integer)
            device: GPU or CPU for training
            feedback_loop_config (FeedbackLoopConfig): config file for training the feedback loop
            use_absorption_filters (bool): whether to use scalar absorption gains or filters
            common_decay_times (list): list of common decay times (one for each room)
            band_centre_hz (optional, list): frequencies where common decay times are measured
            colorless_fdn_params (ColorlessFDNResults, optional): dataclass containing the optimised
                        input, output gains and feedback matrix from the lossless prototype for each FDN
                        in the GFDN
            use_colorless_loss(bool): whether to use the colorless loss in the DiffGFDN cost function itself,
                                      complementary to colorless_fdn_params, not to be used together

        """
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device

        # input parameters
        self.num_groups = num_groups
        self.delays = delays
        self.num_delay_lines = len(delays)
        self.num_delay_lines_per_group = int(self.num_delay_lines /
                                             self.num_groups)
        self.use_absorption_filters = use_absorption_filters
        self.delays_by_group = [
            torch.tensor(self.delays[i:i + self.num_delay_lines_per_group],
                         device=self.device) for i in
            range(0, self.num_delay_lines, self.num_delay_lines_per_group)
        ]
        self.band_centre_hz = band_centre_hz
        self.common_decay_times = common_decay_times
        self.use_colorless_loss = use_colorless_loss

        self.delays = torch.tensor(delays,
                                   dtype=torch.float32,
                                   device=self.device)
        self.delays = self.delays.to(self.device)
        logger.info(f"The delay line lengths are {self.delays} samples")
        # Register delays as a buffer
        self.register_buffer('delay_buffer', self.delays)

        # initialise input-output gains
        self._init_io_gains(colorless_fdn_params)
        # initialise absorption filters in delay lines
        self._init_absorption(common_decay_times, band_centre_hz)
        # initialise feedback loop
        self._init_feedback(feedback_loop_config, colorless_fdn_params)

        # add a lowpass filter at the end to remove high frequency artifacts
        self.design_lowpass_filter()

    def _init_io_gains(self, colorless_fdn_params: Optional[List] = None):
        """Initialise input/output gains"""
        if colorless_fdn_params is None:
            logger.info('Using learnable gains and feedback matrix')
            self.input_gains = nn.Parameter(
                (2 * torch.randn(self.num_delay_lines, 1) - 1) /
                self.num_delay_lines)

            self.output_gains = nn.Parameter(
                (2 * torch.randn(self.num_delay_lines, 1) - 1) /
                self.num_delay_lines)
        else:
            logger.info(
                "Using gains and feedback matrix from the colorless FDN prototype"
            )
            # these are from the colorless FDN prototype
            # these should be of length Ndel x 1
            self.input_gains = torch.tensor([
                colorless_fdn_params[i].opt_input_gains.tolist()
                for i in range(self.num_groups)
            ],
                                            device=self.device).view(-1, 1)
            self.output_gains = torch.tensor([
                colorless_fdn_params[i].opt_output_gains.tolist()
                for i in range(self.num_groups)
            ],
                                             device=self.device).view(-1, 1)

    def _init_absorption(self,
                         common_decay_times: List,
                         band_centre_hz: Optional[List] = None):
        """Initialise absorption gains/filters in the delay lines"""
        # frequency-dependent absorption filters
        if self.use_absorption_filters:
            # this will be of size (num_groups, num_del_per_group,
            # numerator (filter_order), denominator(filter_order))
            self.gain_per_sample = torch.tensor([
                decay_times_to_gain_filters_geq(
                    band_centre_hz,
                    np.squeeze(common_decay_times)[:, i],
                    self.delays_by_group[i], self.sample_rate).tolist()
                for i in range(self.num_groups)
            ],
                                                device=self.device)
            self.filter_order = self.gain_per_sample.shape[-2]
            try:
                self.gain_per_sample = self.gain_per_sample.view(
                    self.num_delay_lines, self.filter_order, 2)
            except Exception:
                # cascade of SOS filters is detected
                n_filters = self.gain_per_sample.shape[1]
                self.gain_per_sample = self.gain_per_sample.permute(
                    0, 2, 1, 3, 4)
                self.gain_per_sample = self.gain_per_sample.reshape(
                    self.num_delay_lines, n_filters, self.filter_order, 2)
        # broadband absorption gains
        else:
            self.gain_per_sample = torch.flatten(
                torch.tensor([
                    decay_times_to_gain_per_sample(
                        np.squeeze(common_decay_times)[i],
                        self.delays_by_group[i], self.sample_rate).tolist()
                    for i in range(self.num_groups)
                ],
                             device=self.device))
        # logger.info(f"Delay line lengths: {self.delays}")
        # logger.info(f"Gains in delay lines: {self.gain_per_sample}")
        # Register delay filters as a buffer
        self.register_buffer('delay_filters', self.gain_per_sample)

    def _init_feedback(self,
                       feedback_loop_config: FeedbackLoopConfig,
                       colorless_fdn_params: Optional[List] = None):
        """Initialise input-output gain vectors and the feedback matrix"""
        # learnable input and output gains
        if colorless_fdn_params is None:
            self.feedback_loop = FeedbackLoop(
                self.num_groups, self.num_delay_lines_per_group, self.delays,
                self.gain_per_sample, self.use_absorption_filters,
                feedback_loop_config.coupling_matrix_type,
                feedback_loop_config.pu_matrix_order)
        else:
            # convert list of numpy arrays to list of torch tensors
            colorless_feedback_matrix_list = [
                torch.from_numpy(colorless_fdn_params[i].opt_feedback_matrix)
                for i in range(self.num_groups)
            ]
            # convert from list of tensors to num_groups x N x N tensor
            colorless_feedback_matrix = torch.stack(
                colorless_feedback_matrix_list, dim=0)

            self.feedback_loop = FeedbackLoop(
                self.num_groups,
                self.num_delay_lines_per_group,
                self.delays,
                self.gain_per_sample,
                self.use_absorption_filters,
                coupling_matrix_type=feedback_loop_config.coupling_matrix_type,
                colorless_feedback_matrix=colorless_feedback_matrix,
                device=self.device)

    def sub_fdn_output(self, z: torch.Tensor) -> torch.Tensor:
        """Get the magnitude response of each FDN (without the absorption)"""
        num_freq_pts = len(z)
        Hout = torch.zeros((num_freq_pts, self.num_groups),
                           dtype=torch.complex64,
                           device=self.device)
        for k in range(self.num_groups):
            group_idx = torch.arange(k * self.num_delay_lines_per_group,
                                     (k + 1) * self.num_delay_lines_per_group,
                                     dtype=torch.int32)
            C = to_complex(self.output_gains[group_idx].expand(
                self.num_delay_lines_per_group, num_freq_pts))

            B = to_complex(self.input_gains[group_idx].expand(
                self.num_delay_lines_per_group, num_freq_pts))

            A = self.feedback_loop.M[k].unsqueeze(0).repeat(num_freq_pts, 1, 1)
            D = torch.diag_embed(
                torch.unsqueeze(z, dim=-1)**self.delays_by_group[k])
            P = torch.linalg.inv(D - A).to(torch.complex64)
            H = torch.einsum('kn, knm -> km', C.permute(1, 0), P).permute(1, 0)
            Hout[..., k] = torch.einsum('mk, mk -> k', H, B)

        return Hout

    def design_lowpass_filter(
        self,
        filter_order: int = 16,
    ):
        """
        Add a lowpass filter in the end to prevent spurius high frequency artifacts
        Args:
            filter_order (int): IIR filter order
            cutoff_hz (float): cutoff frequency of the lowpass (12k by default)
        """
        cutoff_hz = self.sample_rate / 2 - 1e3
        sos_filter_coeffs = torch.tensor(
            butter(filter_order,
                   cutoff_hz / (self.sample_rate / 2.0),
                   btype='lowpass',
                   output='sos'))
        self.lowpass_biquad = BiquadCascade(
            num_sos=sos_filter_coeffs.shape[0],
            num_coeffs=sos_filter_coeffs[:, :3],
            den_coeffs=sos_filter_coeffs[:, 3:])
        self.lowpass_filter = SOSFilter(sos_filter_coeffs.shape[0],
                                        device=self.device)


class DiffGFDNVarSourceReceiverPos(DiffGFDN):

    def __init__(
            self,
            sample_rate: int,
            num_groups: int,
            delays: List[int],
            device: torch.device,
            feedback_loop_config: FeedbackLoopConfig,
            output_filter_config: OutputFilterConfig,
            input_filter_config: OutputFilterConfig,
            use_absorption_filters: bool,
            common_decay_times: List = None,
            band_centre_hz: Optional[List] = None,
            colorless_fdn_params: Optional[List[ColorlessFDNResults]] = None,
            use_colorless_loss: bool = False):
        """
        Differentiable GFDN module for a grid of source and listener locations, where the input and output filter
        coefficients are learnt with deep learning.
        Args:
            sample_rate (int): sampling rate of the FDN
            num_groups (int): number of rooms in coupled space
            delays (list): list of delay line lengths (integer)
            device: GPU or CPU for training
            feedback_loop_config (FeedbackLoopConfig): config file for training the feedback loop
            output_filter_config (OutputFilterConfig): config file for training the output SVF filters
            input_filter_config (InputFilterConfig): config file for training the input SVF filters
            use_absorption_filters (bool): whether to use scalar absorption gains or filters
            common_decay_times (list): list of common decay times (one for each room)
            band_centre_hz (optional, list): frequencies where common decay times are measured
            colorless_fdn_params (ColorlessFDNResults, optional): dataclass containing the optimised
                        input, output gains and feedback matrix from the lossless prototype
            use_colorless_loss (bool): whether to use the colorless loss in the DiffGFDN cost function itself. 
                                      Complementary to colorless_fdn_params, not to be used together
        """
        super().__init__(sample_rate, num_groups, delays, device,
                         feedback_loop_config, use_absorption_filters,
                         common_decay_times, band_centre_hz,
                         colorless_fdn_params, use_colorless_loss)

        self.use_svf_in_output = output_filter_config.use_svfs
        self.use_svf_in_input = input_filter_config.use_svfs

        if self.use_svf_in_output:
            logger.info("Using filters in output")
            self.output_filters = SVF_from_MLP(
                self.sample_rate,
                self.num_groups,
                self.num_delay_lines_per_group,
                output_filter_config.num_fourier_features,
                output_filter_config.num_hidden_layers,
                output_filter_config.num_neurons_per_layer,
                output_filter_config.encoding_type,
                output_filter_config.compress_pole_factor,
                position_type="output_gains",
                device=self.device)

        else:
            logger.info("Using gains in output")
            self.output_scalars = Gains_from_MLP(
                self.num_groups,
                self.num_delay_lines_per_group,
                output_filter_config.num_fourier_features,
                output_filter_config.num_hidden_layers,
                output_filter_config.num_neurons_per_layer,
                output_filter_config.encoding_type,
                position_type="output_gains",
                device=self.device)

        if self.use_svf_in_input:
            logger.info("Using filters in input")
            self.input_filters = SVF_from_MLP(
                self.sample_rate,
                self.num_groups,
                self.num_delay_lines_per_group,
                input_filter_config.num_fourier_features,
                input_filter_config.num_hidden_layers,
                input_filter_config.num_neurons_per_layer,
                input_filter_config.encoding_type,
                input_filter_config.compress_pole_factor,
                position_type="input_gains",
                device=self.device)

        else:
            logger.info("Using gains in input")
            self.input_scalars = Gains_from_MLP(
                self.num_groups,
                self.num_delay_lines_per_group,
                input_filter_config.num_fourier_features,
                input_filter_config.num_hidden_layers,
                input_filter_config.num_neurons_per_layer,
                input_filter_config.encoding_type,
                position_type="input_gains",
                device=self.device)

    def forward(self, x: Dict) -> torch.tensor:
        """
        Compute H(z) = c(z)^T (D - A(z)Gamma(z))^{-1} b(z) + d(z)
        Args:
            x(dict) : input feature dict
        Returns:
            H (tensor): tensor of size num_src x num_rec x num_freq_points
        """
        z = x['z_values']
        self.batch_size = x['listener_position'].shape[0]

        num_freq_pts = len(z)
        C_init = to_complex(
            self.output_gains.expand(self.batch_size, self.num_delay_lines,
                                     num_freq_pts))
        B_init = to_complex(
            self.input_gains.expand(self.batch_size, self.num_delay_lines,
                                    num_freq_pts))

        # this is of size B x Ndel x num_freq_points
        if self.use_svf_in_output:
            C = self.output_filters(x) * C_init
        else:
            C = to_complex(self.output_scalars(x)) * C_init

        if self.use_svf_in_input:
            B = self.input_filters(x) * B_init
        else:
            B = to_complex(self.input_scalars(x)) * B_init

        # get the output of the feedback loop, this is of size num_freq_points x Ndel x Ndel
        P = self.feedback_loop(z)
        # C.T @ P of size B x Ndel x num_freq_pts
        Htemp = torch.einsum('knb, knm -> kmb', C.permute(-1, 1, 0),
                             P).permute(-1, 1, 0)

        # loop over different sources (input gains)
        direct_filter = x['target_early_response']

        # C.T @ P @ B + d(z)
        H = torch.einsum('bmk, bmk -> bk', Htemp, B) + direct_filter

        # pass through a lowpass filter
        # lowpass_response = self.lowpass_filter(z, self.lowpass_biquad)
        # H_lp = H * lowpass_response

        if self.use_colorless_loss:
            H_sub_fdn = super().sub_fdn_output(z)
            return H, H_sub_fdn
        else:
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
    def get_param_dict_inference(self, data: Dict) -> Dict:
        """Get output of MLP during inference"""
        param_np = {}
        try:
            if self.use_svf_in_output:
                param_out_mlp = self.output_filters.get_param_dict(data)
                param_np['output_svf_params'] = param_out_mlp['svf_params']
                param_np['output_biquad_coeffs'] = param_out_mlp[
                    'biquad_coeffs']

            else:
                param_out_mlp = self.output_scalars.get_param_dict(data)
                param_np['output_scalars'] = param_out_mlp['gains']

            if self.use_svf_in_input:
                param_out_mlp = self.input_filters.get_param_dict(data)
                param_np['input_svf_params'] = param_out_mlp['svf_params']
                param_np['input_biquad_coeffs'] = param_out_mlp[
                    'biquad_coeffs']
            else:
                param_out_mlp = self.input_scalars.get_param_dict(data)
                param_np['input_scalars'] = param_out_mlp['gains']
        except Exception as e:
            logger.warning(e)
        return param_np

    @torch.no_grad()
    def get_param_dict(self) -> Dict:
        """Return the parameters as a dict"""
        param_np = {}
        param_np['delays'] = self.delays.squeeze().cpu().numpy()
        param_np['gains_per_sample'] = self.gain_per_sample.squeeze().cpu(
        ).numpy()
        param_np['input_gains'] = self.input_gains.squeeze().cpu().numpy()
        param_np['output_gains'] = self.output_gains.squeeze().cpu().numpy()

        try:
            if self.feedback_loop.coupling_matrix_type in (
                    CouplingMatrixType.SCALAR, CouplingMatrixType.FILTER):
                param_np[
                    'coupled_feedback_matrix'] = self.feedback_loop.get_coupled_feedback_matrix(
                    ).squeeze().cpu().numpy()
                param_np[
                    'individual_mixing_matrix'] = self.feedback_loop.M.squeeze(
                    ).cpu().numpy()
                if self.feedback_loop.coupling_matrix_type == CouplingMatrixType.SCALAR:
                    param_np[
                        'coupling_matrix'] = self.feedback_loop.nd_unitary(
                            self.feedback_loop.alpha,
                            self.num_groups).squeeze().cpu().numpy()
                else:
                    unitary_matrix = self.feedback_loop.ortho_param(
                        self.feedback_loop.unitary_matrix)
                    unit_vectors = self.feedback_loop.unit_vectors / torch.norm(
                        self.feedback_loop.unit_vectors, dim=0, keepdim=True)
                    param_np[
                        'coupling_matrix'] = self.feedback_loop.fir_paraunitary(
                            unitary_matrix,
                            unit_vectors).squeeze().cpu().numpy()
            else:
                # any unitary matrix without any specific coupling structure
                param_np[
                    'coupled_feedback_matrix'] = self.feedback_loop.coupled_feedback_matrix
        except Exception:
            logger.warning('Parameter not initialised yet in FeedbackLoop!')

        return param_np


class DiffGFDNVarReceiverPos(DiffGFDN):

    def __init__(
            self,
            sample_rate: int,
            num_groups: int,
            delays: List[int],
            device: torch.device,
            feedback_loop_config: FeedbackLoopConfig,
            output_filter_config: OutputFilterConfig,
            use_absorption_filters: bool,
            common_decay_times: List = None,
            band_centre_hz: Optional[List] = None,
            colorless_fdn_params: Optional[List[ColorlessFDNResults]] = None,
            use_colorless_loss: bool = False):
        """
        Differentiable GFDN module for a grid of listener locations, where the output filter
        coefficients are learnt with deep learning.
        Args:
            sample_rate (int): sampling rate of the FDN
            num_groups (int): number of rooms in coupled space
            delays (list): list of delay line lengths (integer)
            device: GPU or CPU for training
            feedback_loop_config (FeedbackLoopConfig): config file for training the feedback loop
            output_filter_config (OutputFilterConfig): config file for training the output SVF filters
            use_absorption_filters (bool): whether to use scalar absorption gains or filters
            common_decay_times (list): list of common decay times (one for each room)
            band_centre_hz (optional, list): frequencies where common decay times are measured
            colorless_fdn_params (ColorlessFDNResults, optional): dataclass containing the optimised
                        input, output gains and feedback matrix from the lossless prototype
            use_colorless_loss (bool): whether to use the colorless loss in the DiffGFDN cost function itself. 
                                      Complementary to colorless_fdn_params, not to be used together
        """
        super().__init__(sample_rate, num_groups, delays, device,
                         feedback_loop_config, use_absorption_filters,
                         common_decay_times, band_centre_hz,
                         colorless_fdn_params, use_colorless_loss)

        self.use_svf_in_output = output_filter_config.use_svfs
        self.input_scalars = torch.ones(self.num_groups, 1)

        if self.use_svf_in_output:
            logger.info("Using filters in output")
            self.output_filters = SVF_from_MLP(
                self.sample_rate,
                self.num_groups,
                self.num_delay_lines_per_group,
                output_filter_config.num_fourier_features,
                output_filter_config.num_hidden_layers,
                output_filter_config.num_neurons_per_layer,
                output_filter_config.encoding_type,
                output_filter_config.compress_pole_factor,
                device=self.device)

        else:
            logger.info("Using gains in output")
            self.output_scalars = Gains_from_MLP(
                self.num_groups,
                self.num_delay_lines_per_group,
                output_filter_config.num_fourier_features,
                output_filter_config.num_hidden_layers,
                output_filter_config.num_neurons_per_layer,
                output_filter_config.encoding_type,
                device=self.device)

    def forward(self, x: Dict) -> torch.tensor:
        """
        Compute H(z) = c(z)^T (D - A(z)Gamma(z))^{-1} b(z) + d(z)
        Args:
            x(dict) : input feature dict
        """
        z = x['z_values']
        self.batch_size = x['listener_position'].shape[0]

        num_freq_pts = len(z)
        C_init = to_complex(
            self.output_gains.expand(self.batch_size, self.num_delay_lines,
                                     num_freq_pts))
        # this is of size B x Ndel x num_freq_points
        if self.use_svf_in_output:
            C = self.output_filters(x) * C_init
        else:
            C = to_complex(self.output_scalars(x)) * C_init

        # this is also of size B x Ndel x num_freq_points
        B = to_complex(
            self.input_gains.expand(self.batch_size, self.num_delay_lines,
                                    num_freq_pts))

        # get the output of the feedback loop, this is of size num_freq_points x Ndel x Ndel
        P = self.feedback_loop(z)
        # C.T @ P of size B x Ndel x num_freq_pts
        Htemp = torch.einsum('knb, knm -> kmb', C.permute(-1, 1, 0),
                             P).permute(-1, 1, 0)
        # C.T @ P @ B + d(z)
        direct_filter = x['target_early_response']
        H = torch.einsum('bmk, bmk -> bk', Htemp, B) + direct_filter

        # pass through a lowpass filter
        # lowpass_response = self.lowpass_filter(z, self.lowpass_biquad)
        # H_lp = H * lowpass_response

        if self.use_colorless_loss:
            H_sub_fdn = super().sub_fdn_output(z)
            return H, H_sub_fdn
        else:
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
    def get_param_dict_inference(self, data: Dict) -> Dict:
        """Get output of MLP during inference"""
        param_np = {}
        try:
            if self.use_svf_in_output:
                param_out_mlp = self.output_filters.get_param_dict(data)
                param_np['output_svf_params'] = param_out_mlp['svf_params']
                param_np['output_biquad_coeffs'] = param_out_mlp[
                    'biquad_coeffs']

            else:
                param_out_mlp = self.output_scalars.get_param_dict(data)
                param_np['output_scalars'] = param_out_mlp['gains']
        except Exception as e:
            logger.warning(e)
        return param_np

    @torch.no_grad()
    def get_param_dict(self) -> Dict:
        """Return the parameters as a dict"""
        param_np = {}
        param_np['delays'] = self.delays.squeeze().cpu().numpy()
        param_np['gains_per_sample'] = self.gain_per_sample.squeeze().cpu(
        ).numpy()
        param_np['input_scalars'] = self.input_scalars.squeeze().cpu().numpy()
        param_np['input_gains'] = self.input_gains.squeeze().cpu().numpy()
        param_np['output_gains'] = self.output_gains.squeeze().cpu().numpy()

        try:
            if self.feedback_loop.coupling_matrix_type in (
                    CouplingMatrixType.SCALAR, CouplingMatrixType.FILTER):
                param_np[
                    'coupled_feedback_matrix'] = self.feedback_loop.get_coupled_feedback_matrix(
                    ).squeeze().cpu().numpy()
                param_np[
                    'individual_mixing_matrix'] = self.feedback_loop.M.squeeze(
                    ).cpu().numpy()
                if self.feedback_loop.coupling_matrix_type == CouplingMatrixType.SCALAR:
                    param_np[
                        'coupling_matrix'] = self.feedback_loop.nd_unitary(
                            self.feedback_loop.alpha,
                            self.num_groups).squeeze().cpu().numpy()
                else:
                    unitary_matrix = self.feedback_loop.ortho_param(
                        self.feedback_loop.unitary_matrix)
                    unit_vectors = self.feedback_loop.unit_vectors / torch.norm(
                        self.feedback_loop.unit_vectors, dim=0, keepdim=True)
                    param_np[
                        'coupling_matrix'] = self.feedback_loop.fir_paraunitary(
                            unitary_matrix,
                            unit_vectors).squeeze().cpu().numpy()
            else:
                # any unitary matrix without any specific coupling structure
                param_np[
                    'coupled_feedback_matrix'] = self.feedback_loop.coupled_feedback_matrix
        except Exception:
            logger.warning('Parameter not initialised yet in FeedbackLoop!')

        return param_np


class DiffGFDNSinglePos(DiffGFDN):
    """Differentiable GFDN module for a single source-listener position"""

    def __init__(
            self,
            sample_rate: int,
            num_groups: int,
            delays: List[int],
            device: torch.device,
            feedback_loop_config: FeedbackLoopConfig,
            output_filter_config: OutputFilterConfig,
            use_absorption_filters: bool,
            common_decay_times: List,
            band_centre_hz: Optional[List] = None,
            colorless_fdn_params: Optional[List[ColorlessFDNResults]] = None,
            use_colorless_loss: bool = False,
            input_filter_config: Optional[OutputFilterConfig] = None):
        """
        Args:
            sample_rate (int): sampling rate of the FDN
            num_groups (int): number of rooms in coupled space
            delays (list): list of delay line lengths (integer)
            device: GPU or CPU for training
            feedback_loop_config (FeedbackLoopConfig): config file for training the feedback loop
            output_filter_config (OutputFilterConfig): config file for training the output SVF filters
            use_absorption_filters (bool): whether to use scalar absorption gains or filters
            common_decay_times (list): list of common decay times (one for each room)
            band_centre_hz (optional, list): frequencies where common decay times are measured
            colorless_fdn_params (ColorlessFDNResults, optional): dataclass containing the optimised
                        input, output gains and feedback matrix from the lossless prototype
            use_colorless_loss(bool): whether to use the colorless loss in the DiffGFDN cost function itself,
                                      complementary to colorless_fdn_params, not to be used together
            input_filter_config (OutputFilterConfig): config file for training the input SVF filters
        """
        super().__init__(sample_rate, num_groups, delays, device,
                         feedback_loop_config, use_absorption_filters,
                         common_decay_times, band_centre_hz,
                         colorless_fdn_params, use_colorless_loss)

        if input_filter_config is not None:
            self.use_svf_in_input = input_filter_config.use_svfs
        else:
            self.use_svf_in_input = False
        self.use_svf_in_output = output_filter_config.use_svfs

        if self.use_svf_in_output or self.use_svf_in_input:
            centre_freq, shelving_crossover = eq_freqs()
            self.svf_cutoff_freqs = torch.pi * torch.cat(
                (torch.tensor([shelving_crossover[0]]), centre_freq,
                 torch.tensor([shelving_crossover[-1]]))).to(
                     self.device) / self.sample_rate
            self.num_biquads = len(self.svf_cutoff_freqs)
            self.compress_pole_factor = output_filter_config.compress_pole_factor

        self._init_source_filters()
        self._init_receiver_filters()

    def _init_source_filters(self):
        """Initialise the source gains/filters to be learnt by the network"""
        # check if the input gains are scalars or filters
        if self.use_svf_in_input:
            # resonance distributed randomly between 0 and 1
            init_params = torch.randn(self.num_groups, self.num_biquads, 2)
            # gains initialised at 0dB
            init_params[..., 1] = torch.zeros(
                (self.num_groups, self.num_biquads))

            # the resonance and the gain of the SVFs are the parameters
            self.input_svf_params = nn.Parameter(init_params)

            self.input_filters = SOSFilter(self.num_biquads,
                                           device=self.device)

            # resonance should be between 0 and 1 for complex conjugate poles
            self.input_scaled_res = ScaledSigmoid(lower_limit=1e-6,
                                                  upper_limit=1.0)
            # SVF gain range in dB
            self.input_scaled_gains = ScaledSigmoid(lower_limit=-6.0,
                                                    upper_limit=6.0)
        else:
            self.input_scalars = nn.Parameter(
                (torch.ones(self.num_groups, 1)) / np.sqrt(self.num_groups))

    def _init_receiver_filters(self):
        """Initialise the listener gains/filters to be learnt by the network"""
        # check if the output gains are filters or scalars
        if self.use_svf_in_output:
            # resonance distributed randomly between 0 and 1
            init_params = torch.randn(self.num_groups, self.num_biquads, 2)
            # gains initialised at 0dB
            init_params[..., 1] = torch.zeros(
                (self.num_groups, self.num_biquads))

            # the resonance and the gain of the SVFs are the parameters
            self.output_svf_params = nn.Parameter(init_params)

            self.output_filters = SOSFilter(self.num_biquads,
                                            device=self.device)

            # resonance should be between 0 and 1 for complex conjugate poles
            self.output_scaled_res = ScaledSigmoid(lower_limit=1e-6,
                                                   upper_limit=1.0)
            # SVF gain range in dB
            self.output_scaled_gains = ScaledSigmoid(lower_limit=-6.0,
                                                     upper_limit=6.0)
        # each group will have a unique scalar gain
        else:
            self.output_scalars = nn.Parameter(
                (torch.ones(self.num_groups, 1)) / np.sqrt(self.num_groups))

    def forward(self, x: Dict) -> torch.tensor:
        """
        Compute H(z) = c(z)^T (D - A(z)Gamma(z))^{-1} b(z) + d(z)
        Args:
            x(dict) : input feature dict
        """
        z = x['z_values']
        num_freq_pts = len(z)
        C_init = to_complex(
            self.output_gains.expand(self.num_delay_lines, num_freq_pts))
        B_init = to_complex(
            self.input_gains.expand(self.num_delay_lines, num_freq_pts))

        if not self.use_svf_in_output:
            output_scalars_expanded = self.output_scalars.expand(
                self.num_groups, num_freq_pts)
            output_scalars_expanded = output_scalars_expanded.repeat_interleave(
                self.num_delay_lines_per_group, dim=0)
            C = to_complex(output_scalars_expanded)
        else:
            # this is of size Ndel x num_freq_points
            C = self.get_filter(z, filt_type='output')

        # of size Ndel x num_freq_points
        C *= C_init

        if not self.use_svf_in_input:
            input_scalars_expanded = self.input_scalars.expand(
                self.num_groups, num_freq_pts)
            # this is also of size Ndel x num_bins
            input_scalars_expanded = input_scalars_expanded.repeat_interleave(
                self.num_delay_lines_per_group, dim=0)
            B = to_complex(input_scalars_expanded)
        else:
            B = self.get_filter(z, filt_type='input')

        # of size Ndel x num_freq_points
        B *= B_init

        # get the output of the feedback loop, this is of size num_freq_points x Ndel x Ndel
        P = self.feedback_loop(z)

        # C.T @ P of size num_freq_pts x Ndel
        Htemp = torch.einsum('kn, knm -> km', C.permute(-1, 0), P)
        # C.T @ P @ B + d(z)
        direct_filter = x['target_early_response']

        H = torch.einsum('ki, ik -> k', Htemp, B)
        H += direct_filter

        # pass through a lowpass filter
        # lowpass_response = self.lowpass_filter(z, self.lowpass_biquad)
        # H_lp = H * lowpass_response

        if self.use_colorless_loss:
            H_sub_fdn = super().sub_fdn_output(z)
            return H, H_sub_fdn
        else:
            return H

    def get_filter(self,
                   z_values: torch.tensor,
                   filt_type: str = 'output') -> torch.tensor:
        """Get the input/output filter response"""
        # always ensure that the filter cutoff frequency and resonance are positive
        # this will be the output tensor
        Hout = torch.zeros((self.num_delay_lines, len(z_values)),
                           dtype=torch.complex64)

        reshape_size = (self.num_groups, self.num_biquads)
        # gradient computation is an issue if we view in place,
        # best to clone this
        if filt_type == 'output':
            svf_params = self.output_svf_params.clone()
        else:
            svf_params = self.input_svf_params.clone()

        flattened_svf_gains = svf_params[..., 1].view(-1)
        flattened_svf_res = svf_params[..., 0].view(-1)

        if filt_type == 'output':
            svf_params[..., 1] = self.output_scaled_gains(
                flattened_svf_gains).view(reshape_size)
            svf_params[..., 0] = self.output_scaled_res(
                flattened_svf_res).view(reshape_size)

        else:
            svf_params[..., 1] = self.input_scaled_gains(
                flattened_svf_gains).view(reshape_size)
            svf_params[..., 0] = self.input_scaled_res(flattened_svf_res).view(
                reshape_size)

        # initialise empty filters
        biquad_cascade = [
            BiquadCascade(self.num_biquads, torch.zeros((self.num_biquads, 3)),
                          torch.zeros((self.num_biquads, 3)))
            for i in range(self.num_groups)
        ]

        # fill the empty filters
        for i in range(self.num_groups):
            svf_params_del_line = svf_params[i, :]
            svf_cascade = [
                SVF(cutoff_frequency=self.svf_cutoff_freqs[k],
                    resonance=svf_params_del_line[k, 0],
                    filter_type=("lowshelf" if k == 0 else
                                 "highshelf" if k == self.num_biquads -
                                 1 else "peaking"),
                    G_db=svf_params_del_line[k, 1],
                    device=self.device) for k in range(self.num_biquads)
            ]

            biquad_cascade[i] = BiquadCascade.from_svf_coeffs(
                svf_cascade, self.compress_pole_factor, device=self.device)

            if filt_type == 'output':
                # all delay lines in a group have the same output filter
                Hout[i * self.num_delay_lines_per_group:(i + 1) *
                     self.num_delay_lines_per_group, :] = torch.tile(
                         self.output_filters(z_values, biquad_cascade[i]),
                         (self.num_delay_lines_per_group, 1))
            else:
                # all delay lines in a group have the same output filter
                Hout[i * self.num_delay_lines_per_group:(i + 1) *
                     self.num_delay_lines_per_group, :] = torch.tile(
                         self.input_filters(z_values, biquad_cascade[i]),
                         (self.num_delay_lines_per_group, 1))

        if filt_type == 'output':
            self.output_biquad_cascade = deepcopy(biquad_cascade)
        else:
            self.input_biquad_cascade = deepcopy(biquad_cascade)

        return Hout

    @torch.no_grad()
    def get_param_dict(self) -> Dict:
        """Return the parameters as a dict"""
        param_np = {}
        param_np['delays'] = self.delays.squeeze().cpu().numpy()
        param_np['gains_per_sample'] = self.gain_per_sample.squeeze().cpu(
        ).numpy()
        param_np['input_gains'] = self.input_gains.squeeze().cpu().numpy()
        param_np['output_gains'] = self.output_gains.squeeze().cpu().numpy()

        try:
            if self.feedback_loop.coupling_matrix_type in (
                    CouplingMatrixType.SCALAR, CouplingMatrixType.FILTER):
                param_np[
                    'coupled_feedback_matrix'] = self.feedback_loop.get_coupled_feedback_matrix(
                    ).squeeze().cpu().numpy()
                param_np[
                    'individual_mixing_matrix'] = self.feedback_loop.M.squeeze(
                    ).cpu().numpy()
                if self.feedback_loop.coupling_matrix_type == CouplingMatrixType.SCALAR:
                    param_np[
                        'coupling_matrix'] = self.feedback_loop.nd_unitary(
                            self.feedback_loop.alpha,
                            self.num_groups).squeeze().cpu().numpy()
                else:
                    unitary_matrix = self.feedback_loop.ortho_param(
                        self.feedback_loop.unitary_matrix)
                    unit_vectors = self.feedback_loop.unit_vectors / torch.norm(
                        self.feedback_loop.unit_vectors, dim=0, keepdim=True)
                    param_np[
                        'coupling_matrix'] = self.feedback_loop.fir_paraunitary(
                            unitary_matrix,
                            unit_vectors).squeeze().cpu().numpy()
            else:
                # any unitary matrix without any specific coupling structure
                param_np[
                    'coupled_feedback_matrix'] = self.feedback_loop.coupled_feedback_matrix
        except Exception:
            logger.warning('Parameter not initialised yet in FeedbackLoop!')

        # get absorption filters
        if self.use_absorption_filters:
            param_np['absorption_filters'] = [
                self.feedback_loop.delay_line_gains[n]
                for n in range(self.num_delay_lines)
            ]
        else:
            param_np['absorption_coeffs'] = self.gain_per_sample

        try:
            if self.use_svf_in_output:
                self.output_svf_params[..., 1] = self.output_scaled_gains(
                    self.output_svf_params[..., 1].view(-1)).view(
                        self.num_groups, self.num_biquads)
                self.output_svf_params[..., 0] = self.output_scaled_res(
                    self.output_svf_params[..., 0].view(-1)).view(
                        self.num_groups, self.num_biquads)

                param_np['output_svf_params'] = self.output_svf_params.squeeze(
                ).cpu().numpy()
                param_np['output_biquad_coeffs'] = [
                    torch.cat((self.output_biquad_cascade[n].num_coeffs,
                               self.output_biquad_cascade[n].den_coeffs),
                              dim=-1).squeeze().cpu().numpy()
                    for n in range(self.num_groups)
                ]
            else:
                param_np['output_scalars'] = self.output_scalars.squeeze().cpu(
                ).numpy()

            if self.use_svf_in_input:
                self.input_svf_params[..., 1] = self.input_scaled_gains(
                    self.input_svf_params[..., 1].view(-1)).view(
                        self.num_groups, self.num_biquads)
                self.input_svf_params[..., 0] = self.input_scaled_res(
                    self.input_svf_params[..., 0].view(-1)).view(
                        self.num_groups, self.num_biquads)

                param_np['input_svf_params'] = self.input_svf_params.squeeze(
                ).cpu().numpy()
                param_np['input_biquad_coeffs'] = [
                    torch.cat((self.input_biquad_cascade[n].num_coeffs,
                               self.input_biquad_cascade[n].den_coeffs),
                              dim=-1).squeeze().cpu().numpy()
                    for n in range(self.num_groups)
                ]
            else:
                param_np['input_scalars'] = self.input_scalars.squeeze().cpu(
                ).numpy()

        except Exception as e:
            logger.warning(e)

        return param_np
