from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torchaudio.functional import filtfilt

from .config.config import FeatureEncodingType
from .dnn import MLP, OneHotEncoding, ScaledSigmoid, SinusoidalEncoding
from .filters.geq import eq_freqs
from .utils import db2lin

# pylint: disable=E0606
# flake8: noqa:E265


#######################################FILTER UTILS#########################################
@dataclass
class SVF:
    """Dataclass representing a state variable filter, which can then be converted to a biquad"""

    cutoff_frequency: float
    resonance: float
    filter_type: str
    m_LP: Optional[float] = 1.0
    m_HP: Optional[float] = 1.0
    m_BP: Optional[float] = 1.0
    G_db: Optional[float] = None
    device: Optional[torch.device] = 'cpu'

    def __post_init__(self):
        """Fix the mixing coefficients based on the type of filter"""
        assert self.resonance >= 0, "Resonance must be positive to ensure stability"

        if self.G_db is None:
            self.G = 1.0
        else:
            self.G = db2lin(self.G_db)
        # assert (self.G <= 1.0), "gain cannot be greater than unity!"

        if self.filter_type == "lowpass":
            m = torch.cat(
                (
                    (torch.ones_like(self.G)).unsqueeze(-1),
                    (torch.zeros_like(self.G)).unsqueeze(-1),
                    torch.zeros_like(self.G).unsqueeze(-1),
                ),
                dim=-1,
            ).to(self.device)
        elif self.filter_type == "highpass":
            m = torch.cat(
                (
                    (torch.zeros_like(self.G)).unsqueeze(-1),
                    (torch.zeros_like(self.G)).unsqueeze(-1),
                    torch.ones_like(self.G).unsqueeze(-1),
                ),
                dim=-1,
            ).to(self.device)
        elif self.filter_type == "bandpass":
            m = torch.cat(
                (
                    (torch.zeros_like(self.G)).unsqueeze(-1),
                    (torch.ones_like(self.G)).unsqueeze(-1),
                    torch.zeros_like(self.G).unsqueeze(-1),
                ),
                dim=-1,
            ).to(self.device)
        elif self.filter_type == "lowshelf":
            m = torch.cat(
                (
                    (self.G * torch.ones_like(self.G)).unsqueeze(-1),
                    (2 * self.resonance * torch.sqrt(self.G)).unsqueeze(-1),
                    (torch.ones_like(self.G)).unsqueeze(-1),
                ),
                dim=-1,
            ).to(self.device)
        elif self.filter_type == "highshelf":
            m = torch.cat(
                (
                    (torch.ones_like(self.G)).unsqueeze(-1),
                    (2 * self.resonance * torch.sqrt(self.G)).unsqueeze(-1),
                    (self.G * torch.ones_like(self.G)).unsqueeze(-1),
                ),
                dim=-1,
            ).to(self.device)
        elif self.filter_type in ("peaking", "notch"):
            m = torch.cat(
                (
                    (torch.ones_like(self.G)).unsqueeze(-1),
                    (2 * self.resonance * self.G).unsqueeze(-1),
                    (torch.ones_like(self.G)).unsqueeze(-1),
                ),
                dim=-1,
            ).to(self.device)
        else:
            print(
                "The filter type not specified or not in the list. Using the given mixing coefficents."
            )
        self.m_LP = m[0]
        self.m_BP = m[1]
        self.m_HP = m[2]


@dataclass
class BiquadCascade:
    """Dataclass representing a biquad cascade"""

    # number of second order sections
    num_sos: int
    # numerator coeffs of size num_sos x 3
    num_coeffs: torch.tensor
    # denominator coeffs of size num_sos x 3
    den_coeffs: torch.tensor

    @staticmethod
    def from_svf_coeffs(svf_coeffs: List[SVF],
                        compress_pole_factor: float = 1.0,
                        device: Optional[torch.device] = 'cpu'):
        """
        Get the biquad cascade from a list of SVF coefficients
        Args:
            svf_coefs (list): list of SVF objects, each representing a filter in the casacade
            compress_pole_factor: number between 0 and 1 that reduces the pole radii of the biquads and
                                prevents time domain aliasing
        """
        num_sos = len(svf_coeffs)
        num_coeffs = torch.zeros((num_sos, 3), device=device)
        den_coeffs = torch.zeros_like(num_coeffs)
        for i in range(num_sos):
            cur_svf = svf_coeffs[i]
            num_coeffs[
                i,
                0] = cur_svf.cutoff_frequency**2 * cur_svf.m_LP + cur_svf.cutoff_frequency * cur_svf.m_BP + cur_svf.m_HP
            num_coeffs[i,
                       1] = (2 * cur_svf.cutoff_frequency**2 * cur_svf.m_LP -
                             2 * cur_svf.m_HP) * compress_pole_factor
            num_coeffs[i, 2] = (cur_svf.cutoff_frequency**2 * cur_svf.m_LP -
                                cur_svf.cutoff_frequency * cur_svf.m_BP +
                                cur_svf.m_HP) * compress_pole_factor**2

            den_coeffs[
                i,
                0] = cur_svf.cutoff_frequency**2 + 2 * cur_svf.resonance * cur_svf.cutoff_frequency + 1
            den_coeffs[i, 1] = (2 * cur_svf.cutoff_frequency**2 -
                                2) * compress_pole_factor
            den_coeffs[i,
                       2] = (cur_svf.cutoff_frequency**2 -
                             2 * cur_svf.resonance * cur_svf.cutoff_frequency +
                             1) * compress_pole_factor**2
        return BiquadCascade(num_sos, num_coeffs, den_coeffs)


class IIRFilter(nn.Module):

    def __init__(self,
                 filt_order: int,
                 num_filters: int,
                 filter_numerator: torch.tensor,
                 filter_denominator: torch.tensor,
                 device: Optional[torch.device] = 'cpu'):
        """
        Filter input with an IIR filter of order filt_order
        Args:
            filt_order (int): order of the IIR fulter
            filter_numerator (torch.tensor): numerator coefficients
            filter_denominator (torch.tensor): denominator coefficients
            device (optional, torch.device): the training device, CPU or GPU
        """
        super().__init__()
        self.filt_order = filt_order
        self.num_filters = num_filters
        self.filter_numerator = filter_numerator
        self.filter_denominator = filter_denominator
        self.device = device

        assert self.filter_numerator.shape == (self.num_filters,
                                               self.filt_order)

    def forward(self, z: torch.tensor):
        """
        Calculate 
        Here, z represents the input frequency sampling points
        """
        H = torch.ones((self.num_filters, len(z)),
                       dtype=torch.complex64,
                       device=self.device)
        Hnum = torch.zeros_like(H)
        Hden = torch.zeros_like(H)

        for k in range(self.filt_order):
            Hnum += torch.einsum('n, k -> nk', self.filter_numerator[:, k],
                                 torch.pow(z, -k))
            Hden += torch.einsum('n, k -> nk', self.filter_denominator[:, k],
                                 torch.pow(z, -k))

        H = torch.div(Hnum, Hden + 1e-9)
        return H


class SOSFilter(nn.Module):

    def __init__(
        self,
        num_biquads: int,
        biquad_cascade: Optional[BiquadCascade] = None,
        device: Optional[torch.device] = 'cpu',
    ):
        """
        Filter input with a cascade of second order sections (either in the time of frequency domain)
        Args:
            num_biquads : number of biquads in the filter

        """
        super().__init__()
        self.device = device
        self.num_biquads = num_biquads
        if biquad_cascade is not None:
            self.biquad_cascade = biquad_cascade

    def forward(self,
                z: torch.Tensor,
                biquad_cascade: Optional[BiquadCascade] = None):
        """
        Calculate prod_i (b0,i + b1,iz^{-1} + b2,i z^{-2}) / (a0,i + a1,i z^{-1} + a2,iz^{-2})
        Here, z represents the input frequency sampling points
        """
        if biquad_cascade is None:
            biquad_cascade = self.biquad_cascade

        H = torch.ones(len(z), dtype=torch.complex64, device=self.device)
        for k in range(self.num_biquads):
            H *= torch.div(
                biquad_cascade.num_coeffs[k, 0] +
                biquad_cascade.num_coeffs[k, 1] * torch.pow(z, -1) +
                biquad_cascade.num_coeffs[k, 2] * torch.pow(z, -2),
                biquad_cascade.den_coeffs[k, 0] +
                biquad_cascade.den_coeffs[k, 1] * torch.pow(z, -1) +
                biquad_cascade.den_coeffs[k, 2] * torch.pow(z, -2))

        return H

    def filter(
        self,
        input_signal: torch.Tensor,
        biquad_cascade: Optional[BiquadCascade] = None,
    ):
        """Filter the input signal in the time domain. This will be useful during inferencing"""
        output = np.zeros_like(input_signal)

        if biquad_cascade is None:
            biquad_cascade = self.biquad_cascade

        # filter in SOS form
        for k in range(self.num_biquads):
            inp = input_signal if k == 0 else output
            output = filtfilt(inp, biquad_cascade.den_coeffs[k, :],
                              biquad_cascade.num_coeffs[k, :])
        return output


class SVF_from_MLP(nn.Module):

    def __init__(self,
                 sample_rate: float,
                 num_groups: int,
                 num_delay_lines_per_group: int,
                 num_fourier_features: int,
                 num_hidden_layers: int,
                 num_neurons: int,
                 encoding_type: FeatureEncodingType,
                 compress_pole_factor: Optional[float] = 1.0,
                 position_type: str = "output_gains",
                 device: Optional[torch.device] = 'cpu'):
        """
        Train the MLP to get SVF coefficients for a biquad cascade
        Args:
            sample_rate (float): sampling rate of the network
            num_groups (int): number of groups in the GFDN
            num_delay_lines_per_group: number of delay lines in each group in the GFDN
            num_fourier_features (int): how much will the spatial locations expand as a feature
            num_hidden_layers (int): Number of hidden layers.
            num_neurons (int): Number of neurons in each hidden layer.
            position_type (str): whether the SVF is driving the input or output gains
            encoding_type (str): whether to use one-hot encoding with the grid geometry information, 
                                 or directly use the sinusoidal encodings of the position 
                                 coordinates of the receiversas inputs to the MLP
            compress_pole_factor (float): number between 0 and 1 that reduces the pole radii of the biquads and
                                    prevents time domain aliasing
        """
        super().__init__()
        self.num_groups = num_groups
        self.num_delay_lines_per_group = num_delay_lines_per_group
        self.num_delay_lines = self.num_groups * self.num_delay_lines_per_group
        self.position_type = position_type
        self.encoding_type = encoding_type
        self.compress_pole_factor = compress_pole_factor
        self.device = device

        centre_freq, shelving_crossover = eq_freqs()
        self.svf_cutoff_freqs = torch.pi * torch.cat(
            (torch.tensor([shelving_crossover[0]]), centre_freq,
             torch.tensor([shelving_crossover[-1]]))).to(
                 self.device) / sample_rate
        self.num_biquads = len(self.svf_cutoff_freqs)

        if self.encoding_type == FeatureEncodingType.SINE:
            # if we were feeding the spatial coordinates directly, then the
            # number of input features would be 3. Since we are encoding them,
            # the number of features is 3 * num_fourier_features * 2
            num_input_features = 3 * num_fourier_features * 2
            self.encoder = SinusoidalEncoding(num_fourier_features)

        elif self.encoding_type == FeatureEncodingType.MESHGRID:
            # in this case, the (x,y) locations of the meshgrid and the
            # corresponding one-hot vector (1s where all the receiver locations are)
            # are inputs to the MLP
            num_input_features = 3
            self.encoder = OneHotEncoding()

        self.mlp = MLP(num_input_features,
                       num_hidden_layers,
                       num_neurons,
                       self.num_groups,
                       self.num_biquads,
                       num_params=2)

        self.sos_filter = SOSFilter(self.num_biquads, device=self.device)
        # constraints on PEQ resonance
        self.scaled_res = ScaledSigmoid(lower_limit=1e-6, upper_limit=1.0)
        # constraints on PEQ gains
        self.scaled_gains = ScaledSigmoid(lower_limit=-6, upper_limit=6)

    def forward(self, x: Dict) -> torch.tensor:
        """
        Run the input features through the MLP, gets the filter coefficients as output of the MLP
        Then returns the frequency response of the cascade of SVF filters
        """
        z_values = x['z_values']
        position = x[
            'listener_position'] if self.position_type == "output_gains" else x[
                'source_position']
        self.batch_size = position.shape[0]

        # this will be the output tensor
        H = torch.zeros((self.batch_size, self.num_delay_lines, len(z_values)),
                        dtype=torch.complex64,
                        device=self.device)

        # encode the position coordinates only
        if self.encoding_type == FeatureEncodingType.SINE:
            encoded_position = self.encoder(position)
        elif self.encoding_type == FeatureEncodingType.MESHGRID:
            encoded_position, _, rec_idx = self.encoder(x['mesh_2D'], position)

        # run the MLP, output of the MLP are the state variable filter coefficients
        self.svf_params = self.mlp(encoded_position)

        # if meshgrid encoding is used, the size of svf_params is (Lx*Ly, N, K, 2).
        # instead, we want the size to be (B, N, K, 2). So, we only take the filters
        # corresponding to the position of the receivers in the meshgrid
        if self.encoding_type == FeatureEncodingType.MESHGRID:
            self.svf_params = self.svf_params[rec_idx, ...]  # pylint: disable=E0601
            assert self.svf_params.shape[0] == self.batch_size

        # always ensure that the filter parameters are constrained
        reshape_size = (self.batch_size, self.num_groups, self.num_biquads)
        self.svf_params[..., 0] = self.scaled_res(
            self.svf_params[..., 0].view(-1)).view(reshape_size)
        self.svf_params[..., 1] = self.scaled_gains(
            self.svf_params[..., 1].view(-1)).view(reshape_size)

        # initialise empty filters
        self.biquad_cascade = [[
            BiquadCascade(self.num_biquads, torch.zeros((self.num_biquads, 3)),
                          torch.zeros((self.num_biquads, 3)))
            for i in range(self.num_groups)
        ] for b in range(self.batch_size)]

        # fill the empty filters
        for b in range(self.batch_size):
            for i in range(self.num_groups):
                svf_params_del_line = self.svf_params[b, i, :]
                svf_cascade = [
                    SVF(cutoff_frequency=self.svf_cutoff_freqs[k],
                        resonance=svf_params_del_line[k, 0],
                        filter_type=("lowshelf" if k == 0 else
                                     "highshelf" if k == self.num_biquads -
                                     1 else "peaking"),
                        G_db=svf_params_del_line[k, 1],
                        device=self.device) for k in range(self.num_biquads)
                ]
                self.biquad_cascade[b][i] = BiquadCascade.from_svf_coeffs(
                    svf_cascade, self.compress_pole_factor, device=self.device)

                # all delay lines in a group have the same output filter
                H[b, i * self.num_delay_lines_per_group:(i + 1) *
                  self.num_delay_lines_per_group, :] = torch.tile(
                      self.sos_filter(z_values, self.biquad_cascade[b][i]),
                      (self.num_delay_lines_per_group, 1))

        return H

    def print(self):
        """Print the value of the parameters"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    def get_parameters(self) -> Tuple:
        """Return the parameters as tuple"""
        svf_params = self.svf_params
        biquad_coeffs = [[
            torch.cat((self.biquad_cascade[b][n].num_coeffs,
                       self.biquad_cascade[b][n].den_coeffs),
                      dim=-1).squeeze().cpu().numpy()
            for n in range(self.num_groups)
        ] for b in range(self.batch_size)]
        return (svf_params, biquad_coeffs)

    @torch.no_grad()
    def get_param_dict(self, x: Dict) -> Dict:
        """Return the parameters as a dict for a new data position - used in inferencing"""
        self.forward(x)
        param_np = {}
        param_np['svf_params'] = self.svf_params.squeeze().cpu().numpy()
        param_np['biquad_coeffs'] = [[
            torch.cat((self.biquad_cascade[b][n].num_coeffs,
                       self.biquad_cascade[b][n].den_coeffs),
                      dim=-1).squeeze().cpu().numpy()
            for n in range(self.num_groups)
        ] for b in range(self.batch_size)]
        return param_np


class Gains_from_MLP(nn.Module):

    def __init__(
        self,
        num_groups: int,
        num_delay_lines_per_group: int,
        num_fourier_features: int,
        num_hidden_layers: int,
        num_neurons: int,
        encoding_type: FeatureEncodingType,
        position_type: str = "output_gains",
        device: Optional[torch.device] = 'cpu',
        gain_limits: Optional[Tuple] = None,
    ):
        """
        Train the MLP to get scalar gains for each delay line
        Args:
            num_groups (int): number of groups in the GFDN
            num_delay_lines_per_group: number of delay lines in each group in the GFDN
            num_fourier_features (int): how much will the spatial locations expand as a feature
            num_hidden_layers (int): Number of hidden layers.
            num_neurons (int): Number of neurons in each hidden layer.
            position_type (str): whether the SVF is driving the input or output gains
            encoding_type (str): whether to use one-hot encoding with the grid geometry information, 
                                 or directly use the sinusoidal encodings of the position 
                                 coordinates of the receiversas inputs to the MLP
            gain_limits (optional, tuple): range of the MLP output in the linear scale, specified as a tuple
        """
        super().__init__()
        self.num_groups = num_groups
        self.num_delay_lines_per_group = num_delay_lines_per_group
        self.position_type = position_type
        self.encoding_type = encoding_type
        self.device = device

        if self.encoding_type == FeatureEncodingType.SINE:
            # if we were feeding the spatial coordinates directly, then the
            # number of input features would be 3. Since we are encoding them,
            # the number of features is 3 * num_fourier_features * 2
            num_input_features = 3 * num_fourier_features * 2
            self.encoder = SinusoidalEncoding(num_fourier_features)

        elif self.encoding_type == FeatureEncodingType.MESHGRID:
            # in this case, the (x,y,z) locations of the meshgrid and the
            # corresponding one-hot vector (1s where all the receiver locations are)
            # are inputs to the MLP
            num_input_features = 3
            self.encoder = OneHotEncoding()

        self.mlp = MLP(num_input_features,
                       num_hidden_layers,
                       num_neurons,
                       self.num_groups,
                       num_biquads_in_cascade=1,
                       num_params=1)

        # constraints on output gains
        gain_limits = (-1.0, 1.0) if gain_limits is None else gain_limits
        self.scaled_sigmoid = ScaledSigmoid(lower_limit=gain_limits[0],
                                            upper_limit=gain_limits[1])

    def forward(self, x: Dict) -> torch.tensor:
        """
        Run the input features through the MLP, gets the filter coefficients as output of the MLP
        Then returns the frequency response of the cascade of SVF filters
        """
        z_values = x['z_values']
        position = x[
            'norm_listener_position'] if self.position_type == "output_gains" else x[
                'source_position']
        self.batch_size = position.shape[0]

        # encode the position coordinates only
        if self.encoding_type == FeatureEncodingType.SINE:
            encoded_position = self.encoder(position)
        elif self.encoding_type == FeatureEncodingType.MESHGRID:
            encoded_position, _, rec_idx = self.encoder(x['mesh_2D'], position)

        # run the MLP, output of the MLP are the state variable filter coefficients
        self.gains = self.mlp(encoded_position)

        # if meshgrid encoding is used, the size of svf_params is (Lx*Ly, Ngroup, K, 2).
        # instead, we want the size to be (B, Ngroup, K, 2). So, we only take the filters
        # corresponding to the position of the receivers in the meshgrid
        if self.encoding_type == FeatureEncodingType.MESHGRID:
            self.gains = self.gains[rec_idx, ...]  # pylint: disable=E0601
            assert self.gains.shape[0] == self.batch_size

        # always ensure that the filter parameters are constrained
        reshape_size = (self.batch_size, self.num_groups)
        self.gains = self.scaled_sigmoid(
            self.gains.view(-1)).view(reshape_size)

        # expand the gains to have shape (B, Ngroup x N_del_per_group, K, 2)
        expanded_gains = self.gains.repeat_interleave(
            self.num_delay_lines_per_group, dim=1)

        # fill the output gains of size B x N x K
        C = expanded_gains.unsqueeze(-1).repeat(1, 1, len(z_values))

        return C

    def print(self):
        """Print the value of the parameters"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    def get_parameters(self) -> Tuple:
        """Return the parameters as tuple"""
        gains = self.gains
        return gains

    @torch.no_grad()
    def get_param_dict(self, x: Dict) -> Dict:
        """Return the parameters as a dict"""
        self.forward(x)
        param_np = {}
        param_np['gains'] = self.gains.squeeze().cpu().numpy()
        return param_np
