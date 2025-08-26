from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import init

# flake8: noqa:E265

###################################CONSTRAINTS###########################################


class Sigmoid(nn.Module):
    """Sigmoid nonlinearity between 0 and 1"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid non-linearity to constrain a function between 0 and 1"""
        return 1.0 / (1 + torch.exp(-x))


class ScaledSigmoid(nn.Module):

    def __init__(self, lower_limit: float, upper_limit: float):
        """
        Args:
            lower_limit (float): lower limit of function value
        """
        super().__init__()
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.sigmoid = Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid non-linearity to constrain a function between lower limit and upper limit"""
        return self.lower_limit + (self.upper_limit -
                                   self.lower_limit) * self.sigmoid(x)


class SoftPlus(nn.Module):

    def forward(self, x: torch.Tensor):
        """Softplus function ensures positive output"""
        return torch.log(1 + torch.exp(x))


class ScaledSoftPlus(nn.Module):

    def __init__(
        self,
        lower_limit: float,
        upper_limit: float,
    ):
        """
        Args:
        upper_limit (float): upper limit of function value
        """
        super().__init__()
        self.soft_plus = SoftPlus()
        self.sigmoid = Sigmoid()
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def forward(self, x: torch.Tensor):
        """Scaled softplus function to get values between lowr_lmit and upper limit"""
        return self.lower_limit + (self.upper_limit -
                                   self.lower_limit) * self.soft_plus(x) / (
                                       1 + self.soft_plus(x))


class TanSigmoid(nn.Module):

    def __init__(self, scale_factor: float = 1.0):
        """
        Args:
            scale_factor (float): scale factor to determine slope of sigmoid transition
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.sigmoid = Sigmoid()

    def forward(self, x: torch.Tensor):
        """Tan-sigmoid ensures positive output for SVF frequency"""
        return torch.tan(np.pi * self.sigmoid(x) * 0.5)


#######################################MLP-RELATED###########################################


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
        f_min, f_max = 1.0, 32.0  # Frequency range
        frequencies = torch.exp(
            torch.linspace(np.log(f_min), np.log(f_max),
                           self.num_fourier_features))

        start_idx = 0
        for k in range(self.num_fourier_features):
            encoded_pos[:, start_idx:start_idx +
                        2 * num_pos_features] = torch.cat(
                            (torch.sin(frequencies[k] * np.pi * pos_coords),
                             torch.cos(frequencies[k] * np.pi * pos_coords)),
                            dim=-1)
            start_idx += 2 * num_pos_features
        # this is of size num_pos_pts x (3 * num_fourier_features * 2)
        return encoded_pos


class OneHotEncoding(nn.Module):
    """
    Instead of using the spatial coordinates of the receiver positions,
    uses the meshgrid of the 2D geometry of the space, and puts a
    1 in the binary encoding tensor (same size as the 2D meshgrid) wherever
    a receiver is present. This makes use of the knowledge of the room's geometry
    """

    def pos_in_meshgrid(
        self, X_flat: torch.tensor, Y_flat: torch.Tensor,
        receiver_pos: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Return a tensor of the same size as meshgrid, with the position
        containing the receiver denoted as 1. This is one-hot encoding
        Args:
            X_flat, Y_flat : flattened meshgrid points of size (Lx1)
            receiver_pos : list of receiver positions to look for in meshgrid, of size num_receiver_posx3
        Returns:
            Tuple: one hot encoded vector of size Lx1, containing 1s in all 
                   positions where receivers were found in the meshgrid,
                   and the closest corresponding points in the meshgrid,
                   and the indices of the receiver positions in the meshgrid
        """
        one_hot_encoding = torch.zeros_like(X_flat)
        num_pos_pts = len(receiver_pos)
        closest_points = torch.zeros_like(receiver_pos[:, :2])
        min_index = torch.zeros(num_pos_pts, dtype=torch.int32)

        for k in range(num_pos_pts):
            # Calculate the distance from the target point to each point in the meshgrid
            distances = torch.sqrt((X_flat[:, 0] - receiver_pos[k, 0])**2 +
                                   (Y_flat[:, 0] - receiver_pos[k, 1])**2)

            # Find the index of the minimum distance
            min_index[k] = torch.argmin(distances)
            one_hot_encoding[min_index, 0] = 1.0

            # find the closest point in the meshgrid
            closest_points[k, 0] = X_flat[min_index[k]]
            closest_points[k, 1] = Y_flat[min_index[k]]

        return one_hot_encoding, closest_points, min_index

    def forward(self, mesh_2D: torch.tensor, receiver_pos: torch.tensor):
        """
        Args:
            mesh_3D (torch.tensor): Lx * Ly, 2 2D mesh coordinates
            receiver_pos (torch.tensor): Bx3 positions of the receivers in batch
        """
        # Flatten the meshgrid arrays
        X_flat = mesh_2D[..., 0].view(-1, 1)  # Shape (L_x * L_y, 1)
        Y_flat = mesh_2D[..., 1].view(-1, 1)  # Shape (L_x * L_y, 1)

        one_hot_vector, closest_points, rec_idx = self.pos_in_meshgrid(
            X_flat, Y_flat, receiver_pos)

        # Shape (L_x * L_y, 3)
        input_tensor = torch.cat((X_flat, Y_flat, one_hot_vector),
                                 dim=1).to(torch.float32)
        return input_tensor, closest_points, rec_idx


class ConvNet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        hidden_channels: int,
        num_layers: int = 3,
        kernel_size: int = 3,
    ):
        """
        Implement a generic CNN (fully connected) for spatial processing.
        Args:
            in_channels (int): Number of input features (channels).
            out_channels (int): Number of output features (channels).
            num_groups (int): Number of output groups per spatial location.
            hidden_channels (int): Number of channels in intermediate layers.
            num_layers (int): Total number of convolution layers.
            kernel_size (Tuple[int, int]): Size of the convolution kernel.
        """
        super().__init__()

        layers = []
        # keeps the same matrix size, HxW
        padding_up = (kernel_size[0] - 1) // 2
        padding_left = (kernel_size[1] - 1) // 2
        # First conv layer
        layers.append(
            nn.Conv2d(in_channels,
                      hidden_channels,
                      kernel_size=kernel_size,
                      padding=(padding_up, padding_left)))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(hidden_channels,
                          hidden_channels,
                          kernel_size=kernel_size,
                          padding=(padding_up, padding_left)))

            layers.append(nn.ReLU())

        # Final conv layer to output `num_groups * out_channels`
        layers.append(
            nn.Conv2d(
                hidden_channels,
                num_groups * out_channels,
                kernel_size=kernel_size,
                padding=(padding_up, padding_left),
            ))

        self.conv_net = nn.Sequential(*layers)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): Shape (in_channels H, W)

        Returns:
            Tensor of shape (H, W, num_groups, out_channels,)
        """
        out = self.conv_net(x)  # (num_groups * out_channels, H, W)
        C, H, W = out.shape
        # shape H, W, C
        out = out.permute(1, -1, 0)
        assert C == self.num_groups * self.out_channels
        return out.view(H, W, self.num_groups, self.out_channels)


class ResidualBlock(nn.Module):
    """ResNet style skip connections"""

    def __init__(self, num_neurons: int):
        super().__init__()
        self.linear = nn.Linear(num_neurons, num_neurons)
        self.norm = nn.LayerNorm(num_neurons)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        residual = x
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        return out + residual


class MLP_SkipConnections(nn.Module):
    """MLP with ResNet style skip connections"""

    def __init__(self, num_pos_features: int, num_hidden_layers: int,
                 num_neurons: int, num_groups: int,
                 num_biquads_in_cascade: int, num_params: int):
        super().__init__()

        input_size = num_pos_features
        self.num_biquads = num_biquads_in_cascade
        self.num_groups = num_groups
        self.num_params = num_params
        output_size = self.num_groups * self.num_params * self.num_biquads

        # Input projection
        self.input_layer = nn.Sequential(nn.Linear(input_size, num_neurons),
                                         nn.LayerNorm(num_neurons), nn.ReLU())

        # Residual hidden layers
        self.hidden_layers = nn.ModuleList(
            [ResidualBlock(num_neurons) for _ in range(num_hidden_layers)])

        # Output projection
        self.output_layer = nn.Linear(num_neurons, output_size)

        self._initialise_weights()

    def _initialise_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = self.input_layer(x)

        for layer in self.hidden_layers:
            x = layer(x)

        x = self.output_layer(x)
        x = x.view(batch_size, self.num_groups, self.num_biquads,
                   self.num_params)
        return x


class MLP(nn.Module):

    def __init__(self, num_pos_features: int, num_hidden_layers: int,
                 num_neurons: int, num_groups: int,
                 num_biquads_in_cascade: int, num_params: int):
        """
        Initialize the MLP.

        Args:
            num_pos_features (int): Number of spatial features
            num_hidden_layers (int): Number of hidden layers.
            num_neurons (int): Number of neurons in each hidden layer.
            num_biquads_in_cascaed (int): Number of biquads in cascade
            num_groups (int): number of groups in GFDN
            num_params (int): number of learnable parameters in each SVF
        """
        super().__init__()

        # input is only position dependent.
        input_size = num_pos_features
        self.num_biquads = num_biquads_in_cascade
        self.num_groups = num_groups
        self.num_params = num_params
        # Output layer has (num_del * num_params * num_biquads) features
        output_size = self.num_groups * self.num_params * self.num_biquads

        # Create a list of layers for the MLP
        layers = []

        # Input layer -> First hidden layer
        layers.append(nn.Linear(input_size, num_neurons))
        # layer normalisation to ensure that weights and biases are distributed in (0,1)
        layers.append(nn.LayerNorm(num_neurons))
        layers.append(nn.ReLU())  # Activation function

        # Add N hidden layers with L neurons each
        for num_layer in range(num_hidden_layers):
            layers.append(nn.Linear(num_neurons, num_neurons))
            # add layer normalisation every 3rd layer
            if num_layer % 3 == 0:
                layers.append(nn.LayerNorm(num_neurons))
            layers.append(nn.ReLU())  # Activation function

        # Last hidden layer -> Output layer
        layers.append(nn.Linear(num_neurons, output_size))

        # Combine layers into a Sequential model
        self.model = nn.Sequential(*layers)

        self._initialise_weights()

    def _initialise_weights(self):
        """Initialize weights and biases of Linear layers."""
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                # Use He initialization for ReLU
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                # Optional: Bias initialization
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)

    def forward(self, x: torch.tensor):
        """
        Args:
        x (torch.tensor): input feature vector
        """
        batch_size = x.shape[0]
        x = self.model(x)
        x = x.view(batch_size, self.num_groups, self.num_biquads,
                   self.num_params)
        return x
