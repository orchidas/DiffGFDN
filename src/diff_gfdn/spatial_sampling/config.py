# pylint: disable=relative-beyond-top-level

from typing import Tuple

from pydantic import BaseModel

from ..config.config import FeatureEncodingType


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
    # split between traning and validation, specified as a range from min to max
    train_valid_split: Tuple = (0.1, 0.9)
    # how many splits to test?
    num_grid_spacing: int = 10
    # maximum epochs for training
    max_epochs: int = 50
    # learning rate for Adam optimiser
    lr: float = 0.01
    # directory to save results
    train_dir: str = "output/spatial-sampling/"
    # MLP parameters
    num_hidden_layers: int = 3
    num_neurons_per_layer: int = 2**7
    num_fourier_features: int = 10
    encoding_type: FeatureEncodingType = FeatureEncodingType.SINE
