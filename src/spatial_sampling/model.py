from typing import Dict, Optional, Tuple

from loguru import logger
import numpy as np
from numpy.typing import NDArray
import spaudiopy as sp
import torch
from torch import nn

from diff_gfdn.dnn import ConvNet, MLP, MLP_SkipConnections, ScaledSigmoid, Sigmoid, SinusoidalEncoding

from .config import BeamformerType

# pylint: disable=E0606


class Directional_Beamforming_Weights(nn.Module):
    """Parent class for learning directional beamforming weights with DNN"""

    def __init__(self,
                 num_groups: int,
                 ambi_order: int,
                 num_fourier_features: int,
                 desired_directions: NDArray,
                 device: torch.device,
                 beamformer_type: Optional[BeamformerType] = None):
        """
        Initialise parent class parameters
        Args:
            num_groups (int): number of groups whose parameters need to be learned
            ambi_order (int): order of the SMA recordings
            num_fourier_features (int): number of features used for sinusoidal encoding
            desired_directions (NDArray): 2 x num_directions array with desired azimuth and polar angles
            device (torch.devce) to train on cpu or gpu
            beamformer_type (BeamformerType): type of beamformer used to convert 
                                              from SHD to directional weights
        """
        super().__init__()
        self.num_groups = num_groups
        self.device = device
        self.ambi_order = ambi_order
        self.num_fourier_features = num_fourier_features
        self.num_out_features = (ambi_order + 1)**2
        self.initialise_beamformer(beamformer_type, desired_directions)
        # constraints on directional amplitudes - ensures they are between 0 and 1
        # useful for directional beamforming
        self.scaling = Sigmoid()

    def initialise_beamformer(self, beamformer_type: BeamformerType,
                              desired_directions: NDArray):
        """Initialise the beamformer used to convert from SH to directional amplitudes"""
        if beamformer_type == BeamformerType.MAX_DI:
            self.modal_weights = sp.sph.cardioid_modal_weights(self.ambi_order)
        elif beamformer_type == BeamformerType.MAX_RE:
            self.modal_weights = sp.sph.maxre_modal_weights(self.ambi_order)
        elif beamformer_type == BeamformerType.BUTTER:
            self.modal_weights = sp.sph.butterworth_modal_weights(
                self.ambi_order, k=5, n_c=3)
        else:
            self.modal_weights = np.ones(self.ambi_order + 1)
            logger.warning(
                "Other types of beamformers not available, using unity weights"
            )

        # output of size num_directions x (N_sp+1)^2
        self.analysis_matrix, _ = sp.sph.design_sph_filterbank(
            self.ambi_order,
            desired_directions[0, :],
            desired_directions[1, :],
            self.modal_weights,
            mode='energy',
            sh_type='real')

        self.analysis_matrix = torch.tensor(self.analysis_matrix,
                                            dtype=torch.float32,
                                            device=self.device)

    def normalise_weights(self, weights: torch.Tensor):
        """Normalise the learned weight matrix for energy preservation"""
        return weights / (torch.norm(weights, dim=-1, keepdim=True) + 1e-6)

    def get_directional_amplitudes(self) -> torch.Tensor:
        """
        Convert learned weights into directional amplitudes by multiplying with SH matrix
        Returns:
            torch.Tensor: output matrix of size batch size x num_directions x num_slopes
        """
        # we want the output shape to be num_batches, num_directions, num_slopes
        output = torch.einsum('bkn, nj -> bjk', self.weights,
                              self.analysis_matrix.T)

        # ensure the amplitudes are between 0 and 1
        return self.scaling(output)

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
        ).squeeze().cpu().numpy()
        return param_np


class Directional_Beamforming_Weights_from_MLP(Directional_Beamforming_Weights
                                               ):

    def __init__(
        self,
        num_groups: int,
        ambi_order: int,
        num_fourier_features: int,
        num_hidden_layers: int,
        num_neurons: int,
        desired_directions: NDArray,
        device: Optional[torch.device] = 'cpu',
        beamformer_type: Optional[BeamformerType] = None,
        use_skip_connections: Optional[bool] = False,
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
                use_skip_connections (bool): whether to use ResNet style skip connections

        """
        super().__init__(num_groups, ambi_order, num_fourier_features,
                         desired_directions, device, beamformer_type)
        # if we were feeding the spatial coordinates directly, then the
        # number of input features would be 3. Since we are encoding them,
        # the number of features is 3 * num_fourier_features * 2
        num_input_features = 3 * num_fourier_features * 2
        self.encoder = SinusoidalEncoding(num_fourier_features)

        if use_skip_connections:
            logger.info("Using ResNet style skip connections")
            self.mlp = MLP_SkipConnections(num_input_features,
                                           num_hidden_layers,
                                           num_neurons,
                                           self.num_groups,
                                           num_biquads_in_cascade=1,
                                           num_params=self.num_out_features)
        else:
            self.mlp = MLP(num_input_features,
                           num_hidden_layers,
                           num_neurons,
                           self.num_groups,
                           num_biquads_in_cascade=1,
                           num_params=self.num_out_features)

    def forward(self, x: Dict) -> torch.tensor:
        """Run the input features through the MLP. Output is of size batch_size x num_slopes x (N_sp+1)**2"""
        position = x['norm_listener_position']
        self.batch_size = position.shape[0]

        # encode the position coordinates only
        encoded_position = self.encoder(position)

        # run the MLP
        self.weights = self.mlp(encoded_position)

        reshape_size = (self.batch_size, self.num_groups,
                        self.num_out_features)
        self.weights = self.weights.reshape(reshape_size)

        # normalise weights to have unit energy
        self.weights = super().normalise_weights(self.weights)

        return self.weights


class Directional_Beamforming_Weights_from_CNN(Directional_Beamforming_Weights
                                               ):

    def __init__(
        self,
        num_groups: int,
        ambi_order: int,
        num_fourier_features: int,
        num_hidden_channels: int,
        num_layers: int,
        kernel_size: int,
        desired_directions=NDArray,
        device: Optional[torch.device] = 'cpu',
        beamformer_type: Optional[BeamformerType] = None,
    ):
        """
        Train the CNN to get directional beamformer weights for the amplitudes of each slope, as a function
        of receiver position. These weights will be multiplied with an SH matrix to get direction dependent 
        amplitudes for each slope.
        Args:
            num_groups (int): number of slopes in model
            ambi_order (int): ambisonics order for beamformer design
            num_fourier_features (int): how much will the spatial locations expand as a feature
            num_hidden_channels (int): Number of hidden layers.
            num_layers (int): number of layers in the network
            kernel_size (int): Size of the learnable convolution kernel
        """
        super().__init__(num_groups, ambi_order, num_fourier_features,
                         desired_directions, device, beamformer_type)
        self.num_in_features = 2 * num_fourier_features * 2
        self.num_hidden_channels = num_hidden_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.encoder = SinusoidalEncoding(num_fourier_features)

        self.cnn = ConvNet(self.num_in_features, self.num_out_features,
                           self.num_groups, self.num_hidden_channels,
                           self.num_layers, self.kernel_size)

    def forward(self, x: Dict) -> torch.tensor:
        """Run the input features through the CNN. Output is of size H*W x num_slopes x (N_sp+1)**2"""
        mesh_2D = x['mesh_2D']
        # size is. (H*W, num_in_features)
        H, W, num_coords = mesh_2D.shape
        mesh_2D = mesh_2D.view(H * W, num_coords)
        B = H * W

        encoded_mesh = self.encoder(mesh_2D)
        encoded_mesh = encoded_mesh.view(self.num_in_features, H, W)

        # shape - H, W, num_groups, (N_sp+1)**2
        self.weights = self.cnn(encoded_mesh)

        reshape_size = (B, self.num_groups, self.num_out_features)
        self.weights = self.weights.reshape(reshape_size)

        return self.weights


class Omni_Amplitudes_from_MLP(nn.Module):

    def __init__(
        self,
        num_groups: int,
        num_fourier_features: int,
        num_hidden_layers: int,
        num_neurons: int,
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
        self.device = device

        # if we were feeding the spatial coordinates directly, then the
        # number of input features would be 3. Since we are encoding them,
        # the number of features is 3 * num_fourier_features * 2
        num_input_features = 3 * num_fourier_features * 2
        self.encoder = SinusoidalEncoding(num_fourier_features)

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

        # encode the position coordinates only
        encoded_position = self.encoder(position)

        # run the MLP
        self.gains = self.mlp(encoded_position)

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
