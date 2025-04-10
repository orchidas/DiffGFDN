# pylint: disable=relative-beyond-top-level

from enum import Enum
from typing import Optional

from pydantic import BaseModel, computed_field


class DNNType(Enum):
    """Different types of DNNs for training"""

    CNN = "cnn"
    MLP = "mlp"

    def __repr__(self) -> str:
        return str(self.value)


class CNNConfig(BaseModel):
    # tune hyperparameters of the CNN
    num_hidden_channels: int = 2**6
    num_layers: int = 3
    kernel_size: int = 3


class MLPConfig(BaseModel):
    # tune hyperparameters of the MLP
    num_neurons_per_layer: int = 2**7
    num_hidden_layers: int = 3


class DNNConfig(BaseModel):
    mlp_config: Optional[MLPConfig()] = None
    cnn_config: Optional[CNNConfig()] = None
    num_fourier_features: int = 10


class SpatialSamplingConfig(BaseModel):
    # config file with training parameters
    # where to get the labeled dataset
    room_dataset_path: str = 'resources/Georg_3room_FDTD/srirs.pkl'
    # number of receivers in each training batch
    batch_size: int = 32
    # torch device - cuda or cpu or mps (for apple silicon)
    device: str = 'cpu'
    # seed for random number generation
    seed: int = 241924
    # how many grid spacings to test?
    num_grid_spacing: Optional[int] = None
    # maximum epochs for training
    max_epochs: int = 50
    # learning rate for Adam optimiser
    lr: float = 0.001
    # directory to save results
    train_dir: str = "output/spatial-sampling/"
    # DNN configs
    dnn_config: DNNConfig()

    @computed_field
    @property
    def network_type(self) -> str:
        """Returns the type of the DNN used for training"""
        return DNNType.CNN if self.dnn_config.mlp_config is None else DNNType.MLP
