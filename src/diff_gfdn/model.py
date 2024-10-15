from typing import Dict, List, Optional, Tuple

from loguru import logger
from scipy.signal import butter
import torch
from torch import nn

from .config.config import CouplingMatrixType, FeedbackLoopConfig, OutputFilterConfig
from .feedback_loop import FeedbackLoop
from .filters import absorption_to_gain_per_sample, decay_times_to_gain_filters_geq
from .gain_filters import BiquadCascade, ScaledSigmoid, ScaledSoftPlus, SOSFilter, SVF, SVF_from_MLP
from .geq.eq import eq_freqs
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
        room_dims: Optional[List[Tuple]] = None,
        absorption_coeffs: Optional[List] = None,
        common_decay_times: Optional[List] = None,
        band_centre_hz: Optional[List] = None,
    ):
        """
        Args:
            sample_rate (int): sampling rate of the FDN
            num_groups (int): number of rooms in coupled space
            delays (list): list of delay line lengths (integer)
            device: GPU or CPU for training
            feedback_loop_config (FeedbackLoopConfig): config file for training the feedback loop
            use_absorption_filters (bool): whether to use scalar absorption gains or filters
            room_dims (optional, list): dimensions of each room as a tuple
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
        self.delays = delays
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
                decay_times_to_gain_filters_geq(
                    band_centre_hz, common_decay_times[:, i],
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

        else:
            self.gain_per_sample = torch.flatten(
                torch.tensor([
                    absorption_to_gain_per_sample(
                        room_dims[i], absorption_coeffs[i],
                        self.delays_by_group[i], self.sample_rate)[1]
                    for i in range(self.num_groups)
                ],
                             device=self.device))

        # logger.info(f"Gains in delay lines: {self.gain_per_sample}")

        self.delays = torch.tensor(delays,
                                   dtype=torch.float32,
                                   device=self.device)

        self.delays = self.delays.to(self.device)

        self.feedback_loop = FeedbackLoop(
            self.num_groups, self.num_delay_lines_per_group, self.delays,
            self.gain_per_sample, self.use_absorption_filters,
            feedback_loop_config.coupling_matrix_type,
            feedback_loop_config.pu_matrix_order)

        # add a lowpass filter at the end to remove high frequency artifacts
        self.design_lowpass_filter()

    def design_lowpass_filter(self,
                              filter_order: int = 16,
                              cutoff_hz: float = 15e3):
        """
        Add a lowpass filter in the end to prevent spurius high frequency artifacts
        Args:
            filter_order (int): IIR filter order
            cutoff_hz (float): cutoff frequency of the lowpass (12k by default)
        """
        sos_filter_coeffs = torch.tensor(
            butter(filter_order,
                   cutoff_hz / (self.sample_rate / 2.0),
                   btype='lowpass',
                   output='sos'))
        self.lowpass_biquad = BiquadCascade(
            num_sos=sos_filter_coeffs.shape[0],
            num_coeffs=sos_filter_coeffs[:, :3],
            den_coeffs=sos_filter_coeffs[:, 3:])
        self.lowpass_filter = SOSFilter(sos_filter_coeffs.shape[0])


class DiffGFDNVarReceiverPos(DiffGFDN):

    def __init__(self,
                 sample_rate: int,
                 num_groups: int,
                 delays: List[int],
                 device: torch.device,
                 feedback_loop_config: FeedbackLoopConfig,
                 output_filter_config: OutputFilterConfig,
                 use_absorption_filters: bool,
                 room_dims: Optional[List[Tuple]] = None,
                 absorption_coeffs: Optional[List] = None,
                 common_decay_times: Optional[List] = None,
                 band_centre_hz: Optional[List] = None):
        """
        Differentiable GFDN module for a grid of listener locations.
        Args:
            sample_rate (int): sampling rate of the FDN
            num_groups (int): number of rooms in coupled space
            delays (list): list of delay line lengths (integer)
            room_dims (list): dimensions of each room as a tuple
            device: GPU or CPU for training
            feedback_loop_config (FeedbackLoopConfig): config file for training the feedback loop
            output_filter_config (OutputFilterConfig): config file for training the output SVF filters
            room_dims (optional, list): dimensions of each room as a tuple
            use_absorption_filters (bool): whether to use scalar absorption gains or filters
            absorption_coeffs (optional, list): uniform absorption coefficients (one for each room)
            common_decay_times (optional, list): list of common decay times (one for each room)
            band_centre_hz (optional, list): frequencies where common decay times are measured
        """
        super().__init__(sample_rate, num_groups, delays, device,
                         feedback_loop_config, use_absorption_filters,
                         room_dims, absorption_coeffs, common_decay_times,
                         band_centre_hz)

        # unique to the network
        self.input_gains = nn.Parameter(
            torch.randn(self.num_delay_lines, 1) / self.num_delay_lines)

        self.output_filters = SVF_from_MLP(
            self.num_groups,
            self.num_delay_lines_per_group,
            output_filter_config.num_biquads_svf,
            output_filter_config.num_fourier_features,
            output_filter_config.num_hidden_layers,
            output_filter_config.num_neurons_per_layer,
            output_filter_config.encoding_type,
            output_filter_config.compress_pole_factor,
        )

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
        Htemp = torch.einsum('knb, knm -> kmb', C.permute(-1, 1, 0),
                             P).permute(-1, 1, 0)
        # C.T @ P @ B + d(z)
        direct_filter = x['target_early_response']
        H = torch.einsum('bmk, bmk -> bk', Htemp, B) + direct_filter

        # pass through a lowpass filter
        lowpass_response = self.lowpass_filter(z, self.lowpass_biquad)
        H_lp = H * lowpass_response
        return H_lp

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
            param_np[
                'coupled_feedback_matrix'] = self.feedback_loop.get_coupled_feedback_matrix(
                ).squeeze().cpu().numpy()
            if self.feedback_loop.coupling_matrix_type == CouplingMatrixType.SCALAR:
                param_np['coupling_matrix'] = self.feedback_loop.nd_unitary(
                    self.feedback_loop.alpha,
                    self.num_groups).squeeze().cpu().numpy()
            else:
                unitary_matrix = self.feedback_loop.ortho_param(
                    self.feedback_loop.unitary_matrix)
                unit_vectors = self.feedback_loop.unit_vectors / torch.norm(
                    self.feedback_loop.unit_vectors, dim=0, keepdim=True)
                param_np[
                    'coupling_matrix'] = self.feedback_loop.fir_paraunitary(
                        unitary_matrix, unit_vectors).squeeze().cpu().numpy()

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
                for n in range(self.num_groups)
            ] for b in range(self.batch_size)]
        except Exception as e:
            logger.warning(e)

        return param_np


class DiffGFDNSinglePos(DiffGFDN):
    """Differentiable GFDN module for a single source-listener position"""

    def __init__(self,
                 sample_rate: int,
                 num_groups: int,
                 delays: List[int],
                 device: torch.device,
                 feedback_loop_config: FeedbackLoopConfig,
                 output_filter_config: OutputFilterConfig,
                 use_absorption_filters: bool,
                 room_dims: Optional[List[Tuple]] = None,
                 absorption_coeffs: Optional[List] = None,
                 common_decay_times: Optional[List] = None,
                 band_centre_hz: Optional[List] = None):
        """
        Args:
            sample_rate (int): sampling rate of the FDN
            num_groups (int): number of rooms in coupled space
            delays (list): list of delay line lengths (integer)
            device: GPU or CPU for training
            feedback_loop_config (FeedbackLoopConfig): config file for training the feedback loop
            output_filter_config (OutputFilterConfig): config file for training the output SVF filters
            use_absorption_filters (bool): whether to use scalar absorption gains or filters
            room_dims (optional, list): dimensions of each room as a tuple
            absorption_coeffs (optional, list): uniform absorption coefficients (one for each room)
            common_decay_times (optional, list): list of common decay times (one for each room)
            band_centre_hz (optional, list): frequencies where common decay times are measured
        """
        super().__init__(sample_rate, num_groups, delays, device,
                         feedback_loop_config, use_absorption_filters,
                         room_dims, absorption_coeffs, common_decay_times,
                         band_centre_hz)

        # unique to the network - same gains for each group
        self.input_gains = nn.Parameter(
            (2 * torch.randn(self.num_delay_lines, 1) - 1) /
            self.num_delay_lines)
        # each delay line should have a unique scalar gain
        self.output_gains = nn.Parameter(
            torch.randn(self.num_delay_lines, 1) / self.num_delay_lines)

        self.compress_pole_factor = output_filter_config.compress_pole_factor

        # check if the output gains are filters or scalars
        self.use_svf_in_output = output_filter_config.use_svfs
        if self.use_svf_in_output:
            centre_freq, shelving_crossover = eq_freqs()
            self.svf_cutoff_freqs = torch.pi * torch.cat(
                (torch.tensor([shelving_crossover[0]]), centre_freq,
                 torch.tensor([shelving_crossover[-1]]))) / self.sample_rate
            self.num_biquads = len(self.svf_cutoff_freqs)

            # resonance distributed randomly between 0 and 1
            init_params = torch.randn(self.num_groups, self.num_biquads, 2)
            # gains initialised at 0dB
            init_params[..., 1] = torch.zeros(
                (self.num_groups, self.num_biquads))

            # the resonance and the gain of the SVFs are the parameters
            self.output_svf_params = nn.Parameter(init_params)

            self.output_filters = SOSFilter(self.num_biquads)
            # resonance should be between 0 and 1 for complex conjugate poles
            self.scaled_sigmoid = ScaledSigmoid(lower_limit=0.0,
                                                upper_limit=1.0)
            # SVF gain range in dB
            self.soft_plus = ScaledSoftPlus(lower_limit=-3.0, upper_limit=3.0)

    def forward(self, x: Dict) -> torch.tensor:
        """
        Compute H(z) = c(z)^T (D - A(z)Gamma(z))^{-1} b(z) + d(z)
        Args:
            x(dict) : input feature dict
        """
        z = x['z_values']
        num_freq_pts = len(z)
        output_gains = to_complex(
            self.output_gains.expand(self.num_delay_lines, num_freq_pts))

        # this is of size Ndel x num_freq_points
        C = output_gains * self.get_output_filter(
            z) if self.use_svf_in_output else output_gains

        # this is also of size Ndel x 1
        B = to_complex(self.input_gains)

        # get the output of the feedback loop, this is of size num_freq_points x Ndel x Ndel
        P = self.feedback_loop(z)

        # C.T @ P of size Ndel x num_freq_pts
        Htemp = torch.einsum('kn, knm -> km', C.permute(-1, 0), P)
        # C.T @ P @ B + d(z)
        direct_filter = x['target_early_response']

        H = torch.einsum('ik, kj -> ij', Htemp, B).squeeze()
        H += direct_filter

        # pass through a lowpass filter
        lowpass_response = self.lowpass_filter(z, self.lowpass_biquad)
        H_lp = H * lowpass_response
        return H_lp

    def get_output_filter(self, z_values: torch.tensor) -> torch.tensor:
        """Get the output filter response"""
        # always ensure that the filter cutoff frequency and resonance are positive
        # this will be the output tensor
        Hout = torch.zeros((self.num_delay_lines, len(z_values)),
                           dtype=torch.complex64)

        reshape_size = (self.num_groups, self.num_biquads)
        # gradient computation is an issue if we view in place,
        # best to clone this
        output_svf_params = self.output_svf_params.clone()

        flattened_svf_gains = output_svf_params[..., 1].view(-1)
        flattened_svf_res = output_svf_params[..., 0].view(-1)
        output_svf_params[..., 1] = self.soft_plus(flattened_svf_gains).view(
            reshape_size)
        output_svf_params[..., 0] = self.scaled_sigmoid(
            flattened_svf_res).view(reshape_size)

        # initialise empty filters
        self.biquad_cascade = [
            BiquadCascade(self.num_biquads, torch.zeros((self.num_biquads, 3)),
                          torch.zeros((self.num_biquads, 3)))
            for i in range(self.num_groups)
        ]

        # fill the empty filters
        for i in range(self.num_groups):
            svf_params_del_line = output_svf_params[i, :]
            svf_cascade = [
                SVF(cutoff_frequency=self.svf_cutoff_freqs[k],
                    resonance=svf_params_del_line[k, 0],
                    filter_type=("lowshelf" if k == 0 else
                                 "highshelf" if k == self.num_biquads -
                                 1 else "peaking"),
                    G_db=svf_params_del_line[k, 1])
                for k in range(self.num_biquads)
            ]
            self.biquad_cascade[i] = BiquadCascade.from_svf_coeffs(
                svf_cascade, self.compress_pole_factor)

            # all delay lines in a group have the same output filter
            Hout[i * self.num_delay_lines_per_group:(i + 1) *
                 self.num_delay_lines_per_group, :] = torch.tile(
                     self.output_filters(z_values, self.biquad_cascade[i]),
                     (self.num_delay_lines_per_group, 1))

        return Hout

    @torch.no_grad()
    def get_param_dict(self) -> Dict:
        """Return the parameters as a dict"""
        param_np = {}
        param_np['delays'] = self.delays.squeeze().cpu().numpy()
        param_np['gains_per_sample'] = self.gain_per_sample.squeeze().cpu(
        ).numpy()
        param_np['input_gains'] = self.input_gains.squeeze().cpu().numpy()

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

        try:
            self.output_svf_params[..., 1] = self.soft_plus(
                self.output_svf_params[..., 1].view(-1)).view(
                    self.num_groups, self.num_biquads)
            self.output_svf_params[..., 0] = self.scaled_sigmoid(
                self.output_svf_params[..., 0].view(-1)).view(
                    self.num_groups, self.num_biquads)

            param_np['output_svf_params'] = self.output_svf_params.squeeze(
            ).cpu().numpy()
            param_np['output_biquad_coeffs'] = [
                torch.cat((self.biquad_cascade[n].num_coeffs,
                           self.biquad_cascade[n].den_coeffs),
                          dim=-1).squeeze().cpu().numpy()
                for n in range(self.num_groups)
            ]
            param_np['output_gains'] = self.output_gains.squeeze().cpu().numpy(
            )
        except Exception as e:
            logger.warning(e)

        return param_np
