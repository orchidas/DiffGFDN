from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

# pylint: disable=E0606


@dataclass
class SVF:
    """Dataclass representing a state variable filter, which can then be converted to a biquad"""

    cutoff_frequency: float
    resonance: float
    m_LP: float
    m_HP: float
    m_BP: float


@dataclass
class BiquadCascade:
    """Dataclass representing a biquad cascade"""

    # number of second order sections
    num_sos: int
    # numerator coeffs of size num_sos x 3
    num_coeffs: torch.Tensor
    # denominator coeffs of size num_sos x 3
    den_coeffs: torch.Tensor

    @staticmethod
    def from_svf_coeffs(svf_coeffs: List[SVF]):
        """Get the biquad cascade from SVF coefficients"""
        num_sos = len(svf_coeffs)
        num_coeffs = torch.zeros((num_sos, 3))
        den_coeffs = torch.zeros_like(num_coeffs)
        for i in range(num_sos):
            cur_svf = svf_coeffs[i]
            num_coeffs[
                i,
                0] = cur_svf.cutoff_frequency**2 * cur_svf.m_LP + cur_svf.cutoff_frequency * cur_svf.m_BP + cur_svf.m_HP
            num_coeffs[
                i,
                1] = 2 * cur_svf.cutoff_frequency**2 * cur_svf.m_LP - 2 * cur_svf.m_HP
            num_coeffs[
                i,
                2] = cur_svf.cutoff_frequency**2 * cur_svf.m_LP - cur_svf.cutoff_frequency * cur_svf.m_BP + cur_svf.m_HP

            den_coeffs[
                i,
                0] = cur_svf.cutoff_frequency**2 + 2 * cur_svf.resonance * cur_svf.cutoff_frequency + 1
            den_coeffs[i, 1] = 2 * cur_svf.cutoff_frequency**2 - 2
            den_coeffs[
                i,
                2] = cur_svf.cutoff_frequency**2 - 2 * cur_svf.resonance * cur_svf.cutoff_frequency + 1

        return BiquadCascade(num_sos, num_coeffs, den_coeffs)


class SoftPlus(nn.Module):

    def forward(self, x: torch.Tensor):
        """Softplus function ensures positive output for SVF resonance"""
        return torch.div(torch.log(1 + torch.exp(x)), np.log(2))


class TanSigmoid(nn.Module):

    def forward(self, x: torch.Tensor):
        """Tan-sigmoid ensures positive output for SVF frequency"""
        sigmoid = torch.div(1, 1 + torch.exp(-x))
        return torch.tan(np.pi * sigmoid * 0.5)


class SOSFilter(nn.Module):

    def __init__(self,
                 num_biquads: int,
                 biquad_cascade: Optional[BiquadCascade] = None):
        """
        Filter input with a cascade of second order sections (either in the time of frequency domain)
        Args:
            num_biquads : number of biquads in the filter
            biquad_cascade (Optional, BiquadCascade): the biquad cascade object
        """
        super().__init__()
        self.num_biquads = num_biquads
        if biquad_cascade is not None:
            self.biquad_cascade = biquad_cascade

    def forward(self, z: torch.Tensor, biquad_cascade: BiquadCascade):
        """
        Calculate prod_i (b0,i + b1,iz^{-1} + b2,i z^{-2}) / (a0,i + a1,i z^{-1} + a2,iz^{-2})
        Here, z represents the input frequency sampling points
        """
        H = torch.ones(len(z), dtype=torch.complex64)
        for k in range(self.num_biquads):
            H *= torch.div(
                biquad_cascade.num_coeffs[k, 0] +
                biquad_cascade.num_coeffs[k, 1] * torch.pow(z, -1) +
                biquad_cascade.num_coeffs[k, 2] * torch.pow(z, -2),
                biquad_cascade.den_coeffs[k, 0] +
                biquad_cascade.den_coeffs[k, 1] * torch.pow(z, -1) +
                biquad_cascade.den_coeffs[k, 2] * torch.pow(z, -2))

        return H

    def filter(self, input_signal: torch.Tensor):
        """Filter the input signal in the time domain. This will be useful during inference"""
        output = np.zeros_like(input_signal)
        for k in range(self.num_biquads):
            output += torch.filtfilt(input_signal,
                                     self.biquad_cascade.den_coeffs[k, :],
                                     self.biquad_cascade.num_coeffs[k, :])
        return output


class SinusoidalEncoding(nn.Module):
    """
    Encode the input features in the Fourier domain before sending them into the MLP
    This increases the input feature dimension
    """

    def __init__(self, num_fourier_features: int):
        """
        Args:
            num_fourier_features: Lower dimensional features are expanded to this size
        """
        super().__init__()
        self.num_fourier_features = num_fourier_features

    def forward(self, pos_coords: torch.tensor):
        """
        Args:
            pos_coords: the position coordinates of the receivers
        """
        # x contains the position coordinates of size num_pos_pts x 3
        num_pos_pts, num_pos_features = pos_coords.shape
        encoded_pos = torch.zeros(
            num_pos_pts, num_pos_features * self.num_fourier_features * 2)

        start_idx = 0
        for k in range(self.num_fourier_features):
            encoded_pos[:, start_idx:start_idx +
                        2 * num_pos_features] = torch.cat(
                            (torch.sin(2**k * np.pi * pos_coords),
                             torch.cos(2**k * np.pi * pos_coords)),
                            dim=-1)
            start_idx += 2 * num_pos_features
        # this is of size num_pos_pts x (3 * num_fourier_features * 2)
        return encoded_pos


class OneHotEncoding(nn.Module):
    """
    Instead of using the spatial coordinates of the receiver positions,
    uses the meshgrid of the entire 3D geometry of the space, and puts a
    1 in the binary encoding tensor (same size as the 3D meshgrid) wherever
    a receiver is present. This makes use of the knowledge of the room's geometry
    """

    def pos_in_meshgrid(
            self, X_flat: torch.tensor, Y_flat: torch.Tensor,
            Z_flat: torch.tensor,
            receiver_pos: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        Return a tensor of the same size as meshgrid, with the position
        containing the receiver denoted as 1. This is one-hot encoding
        Args:
            X_flat, Y_flat, Z_flat : flattened meshgrid points of size (Lx1)
            receiver_pos : list of receiver positions to look for in meshgrid, of size num_receiver_posx3
        Returns:
            Tuple: one hot encoded vector of size Lx1, containing 1s in all 
                   positions where receivers were found in the meshgrid,
                   and the closest corresponding points in the meshgrid
        """
        one_hot_encoding = torch.zeros_like(X_flat)
        num_pos_pts = len(receiver_pos)
        closest_points = torch.zeros_like(receiver_pos)

        for k in range(num_pos_pts):
            # Calculate the distance from the target point to each point in the meshgrid
            distances = torch.sqrt((X_flat[:, 0] - receiver_pos[k, 0])**2 +
                                   (Y_flat[:, 0] - receiver_pos[k, 1])**2 +
                                   (Z_flat[:, 0] - receiver_pos[k, 2])**2)

            # Find the index of the minimum distance
            min_index = torch.argmin(distances)
            one_hot_encoding[min_index, 0] = 1.0

            # find the closest point in the meshgrid
            closest_points[k, 0] = X_flat[min_index]
            closest_points[k, 1] = Y_flat[min_index]
            closest_points[k, 2] = Z_flat[min_index]

        return one_hot_encoding, closest_points

    def forward(self, mesh_3D: torch.tensor, receiver_pos: torch.tensor):
        """
        Args:
            mesh_3D (torch.tensor): Lx * Ly * Lz, 3 3D mesh coordinates
            receiver_pos (torch.tensor): Bx3 positions of the receivers in batch
        """
        # Flatten the meshgrid arrays
        X_flat = mesh_3D[..., 0].view(-1, 1)  # Shape (L_x * L_y * L_z, 1)
        Y_flat = mesh_3D[..., 1].view(-1, 1)  # Shape (L_x * L_y, * L_z, 1)
        Z_flat = mesh_3D[..., 2].view(-1, 1)  # Shape (L_x * L_y * L_z, 1)

        one_hot_vector, closest_points = self.pos_in_meshgrid(
            X_flat, Y_flat, Z_flat, receiver_pos)

        # Shape (L_x * L_y * L_z, 4)
        input_tensor = torch.cat((X_flat, Y_flat, Z_flat, one_hot_vector),
                                 dim=1)
        return input_tensor, closest_points


class MLP(nn.Module):

    def __init__(self, num_pos_features: int, num_hidden_layers: int,
                 num_neurons: int, num_biquads_in_cascade: int,
                 num_delay_lines: int):
        """
        Initialize the MLP.

        Args:
            num_pos_features (int): Number of spatial features
            num_hidden_layers (int): Number of hidden layers.
            num_neurons (int): Number of neurons in each hidden layer.
            num_biquads_in_cascaed (int): Number of biquads in cascade
            num_delay_lines (int): number of delay lines in FDN
        """
        super().__init__()

        # input is only position dependent.
        input_size = num_pos_features
        self.num_biquads = num_biquads_in_cascade
        self.num_delay_lines = num_delay_lines
        # Output layer has (num_del * 5 SVF params * num_biquads) features
        output_size = self.num_delay_lines * 5 * self.num_biquads

        # Create a list of layers for the MLP
        layers = []

        # Input layer -> First hidden layer
        layers.append(nn.Linear(input_size, num_neurons))
        # layer normalisation to ensure that weights and biases are distributed in (0,1)
        layers.append(nn.LayerNorm(num_neurons))
        layers.append(nn.ReLU())  # Activation function

        # Add N hidden layers with L neurons each
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.LayerNorm(num_neurons))
            layers.append(nn.ReLU())  # Activation function

        # Last hidden layer -> Output layer
        layers.append(nn.Linear(num_neurons, output_size))

        # Combine layers into a Sequential model
        self.model = nn.Sequential(*layers)

        # Add an adaptive average pooling layer to pool over the batch dimension
        self.pooling = nn.AdaptiveAvgPool1d(
            1)  # Pool down to 1 value along the batch dimension

    def forward(self, x: torch.tensor):
        """
        Args:
        x (torch.tensor): input feature vector
        """
        x = self.model(x)
        # pool the output layer to ensure that size is 5 x num_delay_lines x num_biquads
        # Reshape x to [batch_size, 5 * N * K, 1] so we can apply 1D pooling
        x = x.unsqueeze(-1)

        # Apply adaptive average pooling to reduce the batch dimension
        x = self.pooling(x.permute(
            1, 2, 0))  # batch along the last axis, shape[5*K*N, 1, B]

        # Remove the last dimension (size 1)
        x = x.squeeze(-1)
        # Average across the batch dimension and reshape to [5, N, K]
        x = x.view(self.num_delay_lines, self.num_biquads, 5)

        return x


class SVF_from_MLP(nn.Module):

    def __init__(self,
                 num_biquads: int,
                 num_delay_lines: int,
                 num_fourier_features: int = 10,
                 num_hidden_layers: int = 3,
                 num_neurons: int = 2**8,
                 position_type: str = "output_gains",
                 encoding_type: str = "direct"):
        """
        Train the MLP to get SVF coefficients for a biquad cascade
        Args:
            num_fourier_features (int): how much will the spatial locations expand as a feature
            num_hidden_layers (int): Number of hidden layers.
            num_neurons (int): Number of neurons in each hidden layer.
            num_biquads (int): Number of biquads in cascade
            num_delay_lines (int): number of delay lines in FDN
            position_type (str): whether the SVF is driving the input or output gains
            encoding_type (str): whether to use one-hot encoding with the grid geometry information, 
                                 or directly use the sinusoidal encodings of the position 
                                 coordinates of the receiversas inputs to the MLP
        """
        super().__init__()
        self.num_biquads = num_biquads
        self.num_delay_lines = num_delay_lines
        self.position_type = position_type
        self.encoding_type = encoding_type

        if encoding_type == "direct":
            # if we were feeding the spatial coordinates directly, then the
            # number of input features would be 3. Since we are encoding them,
            # the number of features is 3 * num_fourier_features * 2
            num_input_features = 3 * num_fourier_features * 2
            self.encoder = SinusoidalEncoding(num_fourier_features)
        elif encoding_type == "one_hot":
            # in this case, the (x,y,z) locations of the meshgrid and the
            # corresponding one-hot vector (1s where all the receiver locations are)
            # are inputs to the MLP
            num_input_features = 4
            self.encoder = OneHotEncoding()

        self.mlp = MLP(num_input_features, num_hidden_layers, num_neurons,
                       self.num_biquads, self.num_delay_lines)

        self.biquad_cascade = [
            BiquadCascade(self.num_biquads, torch.zeros((self.num_biquads, 3)),
                          torch.zeros((self.num_biquads, 3)))
            for k in range(self.num_delay_lines)
        ]

        self.sos_filter = SOSFilter(self.num_biquads)
        self.soft_plus = SoftPlus()
        self.tan_sigmoid = TanSigmoid()

    def forward(self, x: Dict) -> torch.tensor:
        """
        Run the input features through the MLP, gets the filter coefficients as output of the MLP
        Then returns the frequency response of the cascade of SVF filters
        """
        z_values = x['z_values']
        position = x[
            'listener_position'] if self.position_type == "output_gains" else x[
                'source_position']
        mesh_3D = x['mesh_3D']

        # this will be the output tensor
        H = torch.zeros((self.num_delay_lines, len(z_values)),
                        dtype=torch.complex64)

        # encode the position coordinates only
        if self.encoding_type == "direct":
            encoded_position = self.encoder(position)
        elif self.encoding_type == "one_hot":
            encoded_position, _ = self.encoder(mesh_3D, position)

        # run the MLP, output of the MLP are the state variable filter coefficients
        self.svf_params = self.mlp(encoded_position)

        # always ensure that the filter cutoff frequency and resonance are positive
        self.svf_params[..., 0] = self.tan_sigmoid(
            self.svf_params[..., 0].view(-1)).reshape(self.num_delay_lines,
                                                      self.num_biquads)
        self.svf_params[..., 1] = self.soft_plus(
            self.svf_params[..., 1].view(-1)).reshape(self.num_delay_lines,
                                                      self.num_biquads)

        for i in range(self.num_delay_lines):
            svf_params_del_line = self.svf_params[i, ...]
            svf_cascade = [
                SVF(svf_params_del_line[k, 0], svf_params_del_line[k, 1],
                    svf_params_del_line[k, 2], svf_params_del_line[k, 3],
                    svf_params_del_line[k, 4]) for k in range(self.num_biquads)
            ]
            self.biquad_cascade[i] = BiquadCascade.from_svf_coeffs(svf_cascade)
            H[i, :] = self.sos_filter(z_values, self.biquad_cascade[i])

        return H

    def print(self):
        """Print the value of the parameters"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    def get_parameters(self) -> Tuple:
        """Return the parameters as tuple"""
        svf_params = self.svf_params
        biquad_coeffs = torch.cat(
            (self.biquad_cascade.num_coeffs, self.biquad_cascade.den_coeffs),
            dim=-1)
        return (svf_params, biquad_coeffs)

    @torch.no_grad()
    def get_param_dict(self) -> Dict:
        """Return the parameters as a dict"""
        param_np = {}
        param_np['svf_params'] = self.svf_params.squeeze().cpu().numpy()
        param_np['biquad_coeffs'] = torch.cat(
            (self.biquad_cascade.num_coeffs, self.biquad_cascade.den_coeffs),
            dim=-1).squeeze().cpu().numpy()
        return param_np
