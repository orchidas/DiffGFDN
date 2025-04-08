from typing import Dict, Optional, Tuple

from numpy.typing import NDArray
import spaudiopy as sp
import torch
from torch import nn

from diff_gfdn.gain_filters import FeatureEncodingType, MLP, OneHotEncoding, ScaledSigmoid, SinusoidalEncoding

# pylint: disable=E0606


class Directional_Beamforming_Weights_from_MLP(nn.Module):

    def __init__(
        self,
        num_groups: int,
        ambi_order: int,
        num_fourier_features: int,
        num_hidden_layers: int,
        num_neurons: int,
        encoding_type: FeatureEncodingType,
        device: Optional[torch.device] = 'cpu',
    ):
        """
        Train the MLP to get directional beamformer weights for the amplitudes of each slope, as a function
        of receiver position. These weights will be multiplied with an SH matrix to get direction dependent 
        amplitudes for each slope.
        Args:
                num_groups (int): number of slopes in model
                ambi_order (int): ambisonics order for beamformer design
                num_fourier_features (int): how much will the spatial locations expand as a feature
                num_hidden_layers (int): Number of hidden layers.
                num_neurons (int): Number of neurons in each hidden layer.
                encoding_type (str): whether to use one-hot encoding with the grid geometry information, 
                                     or directly use the sinusoidal encodings of the position 
                                     coordinates of the receiversas inputs to the MLP

        """
        super().__init__()
        self.num_groups = num_groups
        self.encoding_type = encoding_type
        self.device = device
        self.ambi_order = ambi_order
        self.num_out_features = (ambi_order + 1)**2

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
            num_input_features = 4
            self.encoder = OneHotEncoding()

        self.mlp = MLP(num_input_features,
                       num_hidden_layers,
                       num_neurons,
                       self.num_groups,
                       num_biquads_in_cascade=1,
                       num_params=self.num_out_features)

        # constraints on output gains - ensures they are positive, and they sum to one.
        # useful for directional beamforming
        self.scaling = nn.Softmax(dim=-1)

    def forward(self, x: Dict) -> torch.tensor:
        """Run the input features through the MLP. Output is of size batch_size x num_slopes x (N_sp+1)**2"""
        position = x['norm_listener_position']
        self.batch_size = position.shape[0]
        mesh_3D = x['mesh_3D']

        # encode the position coordinates only
        if self.encoding_type == FeatureEncodingType.SINE:
            encoded_position = self.encoder(position)
        elif self.encoding_type == FeatureEncodingType.MESHGRID:
            encoded_position, _, rec_idx = self.encoder(mesh_3D, position)

        # run the MLP
        self.weights = self.mlp(encoded_position)

        # if meshgrid encoding is used, the size of MLP outputs is (Lx*Ly*Lz, Ngroup, (N_sp+1)**2).
        # instead, we want the size to be (B, Ngroup, (N_sp+1)**2). So, we only take the filters
        # corresponding to the position of the receivers in the meshgrid
        if self.encoding_type == FeatureEncodingType.MESHGRID:
            self.gains = self.gains[rec_idx, ...]  # pylint: disable=E0601
            assert self.gains.shape[0] == self.batch_size

        # always ensure that the filter parameters are constrained
        reshape_size = (self.batch_size, self.num_groups,
                        self.num_out_features)
        # ensure weights are between 0-1 and they add to 1
        self.weights = self.scaling(self.weights).reshape(reshape_size)

        return self.weights

    def normalise_beamformer_weights(self):
        """Normalise the beamforming weight matrix for amplitude preservation"""
        return self.weights / (torch.norm(self.weights, dim=-1, keepdim=True) +
                               1e-6)

    def get_directional_amplitudes(
            self, desired_directions: NDArray) -> torch.Tensor:
        """
        Convert beamforming weights into directional amplitudes by multiplying with SH matrix
        Args:
            desired_directions (NDArray): 2 x num_directions matrix of azimuth and polar angles
        Returns:
            torch.Tensor: output matrix of size batch size x num_slopes x num_directions
        """
        # output of size num_directions x (N_sp+1)^2
        sph_matrix = torch.tensor(sp.sph.sh_matrix(self.ambi_order,
                                                   desired_directions[0, :],
                                                   desired_directions[1, :],
                                                   sh_type='real'),
                                  dtype=torch.float32,
                                  device=self.device)
        # we want the output shape to be num_batches, num_slopes, num_directions
        return torch.einsum('bkn, nj -> bjk', self.weights, sph_matrix.T)

    def print(self):
        """Print the value of the parameters"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    def get_parameters(self) -> Tuple:
        """Return the parameters as tuple"""
        weights = self.weights
        return weights

    @torch.no_grad()
    def get_param_dict(self, x: Dict) -> Dict:
        """Return the parameters as a dict"""
        self.forward(x)
        param_np = {}
        param_np['beamformer_weights'] = self.weights.squeeze().cpu().numpy()
        param_np['directional_weights'] = self.get_directional_amplitudes(
            x['sph_directions'])
        return param_np


class Omni_Amplitudes_from_MLP(nn.Module):

    def __init__(
        self,
        num_groups: int,
        num_fourier_features: int,
        num_hidden_layers: int,
        num_neurons: int,
        encoding_type: FeatureEncodingType,
        device: Optional[torch.device] = 'cpu',
        gain_limits: Optional[Tuple] = None,
    ):
        """
        Train the MLP to get omnidirectional amplitudes for each slope
        Args:
            num_groups (int): number of slopes in model
            num_fourier_features (int): how much will the spatial locations expand as a feature
            num_hidden_layers (int): Number of hidden layers.
            num_neurons (int): Number of neurons in each hidden layer.
            encoding_type (str): whether to use one-hot encoding with the grid geometry information, 
                                 or directly use the sinusoidal encodings of the position 
                                 coordinates of the receiversas inputs to the MLP
            gain_limits (optional, tuple): range of the MLP output in the linear scale, specified as a tuple
        """
        super().__init__()
        self.num_groups = num_groups
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
            num_input_features = 4
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
        """Run the input features through the MLP. Output is of size batch size x num_slopes"""
        position = x['norm_listener_position']
        self.batch_size = position.shape[0]
        mesh_3D = x['mesh_3D']

        # encode the position coordinates only
        if self.encoding_type == FeatureEncodingType.SINE:
            encoded_position = self.encoder(position)
        elif self.encoding_type == FeatureEncodingType.MESHGRID:
            encoded_position, _, rec_idx = self.encoder(mesh_3D, position)

        # run the MLP
        self.gains = self.mlp(encoded_position)

        # if meshgrid encoding is used, the size of MLP output is (Lx*Ly*Lz, Ngroup).
        # instead, we want the size to be (B, Ngroup). So, we only take the gains
        # corresponding to the position of the receivers in the meshgrid
        if self.encoding_type == FeatureEncodingType.MESHGRID:
            self.gains = self.gains[rec_idx, ...]  # pylint: disable=E0601
            assert self.gains.shape[0] == self.batch_size

        # always ensure that the filter parameters are constrained
        reshape_size = (self.batch_size, self.num_groups)
        self.gains = self.scaled_sigmoid(
            self.gains.view(-1)).view(reshape_size)

        return self.gains

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
