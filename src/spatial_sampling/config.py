# pylint: disable=relative-beyond-top-level

from typing import Optional

from pydantic import BaseModel

from diff_gfdn.config.config import FeatureEncodingType


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
    # MLP parameters
    num_hidden_layers: int = 3
    num_neurons_per_layer: int = 2**7
    num_fourier_features: int = 10
    encoding_type: FeatureEncodingType = FeatureEncodingType.SINE
