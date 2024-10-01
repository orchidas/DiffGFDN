from enum import Enum
from typing import List

import numpy as np
import sympy as sp
from pydantic import BaseModel, computed_field

from ..utils import ms_to_samps


class CouplingMatrixType(Enum):
    """Different types of coupling matrix"""

    SCALAR = "scalar_matrix"
    FILTER = "filter_matrix"

    def __repr__(self) -> str:
        return str(self.value)


class FeatureEncodingType(Enum):
    """Different types of encoding for the input MLP features"""

    SINE = "sinusoidal"
    MESHGRID = "meshgrid"

    def __repr__(self) -> str:
        return str(self.value)


class FeedbackLoopConfig(BaseModel):
    # FIR PU matrix order
    pu_matrix_order: int = 2**5
    # coupling matrix type
    coupling_matrix_type: CouplingMatrixType = CouplingMatrixType.SCALAR


class OutputFilterConfig(BaseModel):
    # config for training the output filters based on listener location
    # number of biquads in each filter
    num_biquads_svf: int = 8
    num_hidden_layers: int = 1
    num_neurons_per_layer: int = 2**7
    num_fourier_features: int = 10
    encoding_type: FeatureEncodingType = FeatureEncodingType.SINE


class TrainerConfig(BaseModel):
    # config file with training parameters
    # number of receivers in each training batch
    batch_size: int = 32
    # torch device - cuda or cpu or mps (for apple silicon)
    device: str = 'cpu'
    # split between traning and validation
    train_valid_split: float = 0.8
    # maximum epochs for training
    max_epochs: int = 5
    # learning rate for Adam optimiser
    lr: float = 0.01
    # directory to save results
    train_dir: str = "../output"
    # where to save the IRs
    ir_dir: str = "../audio"


class DiffGFDNConfig(BaseModel):
    """Config file for training the DiffGFDN"""

    # path to three room dataset
    room_dataset_path: str = '../resources/Georg_3room_FDTD/srirs.pkl'
    # sampling rate of the FDN
    sample_rate: float = 32000.0
    # training config
    trainer_config: TrainerConfig = TrainerConfig()
    # number of delay lines
    num_delay_lines: int = 12
    # delay range in ms - first delay should be after the mixing time
    delay_range_ms: List[float] = [20.0, 50.0]
    # config for the feedback loop
    feedback_loop_config: FeedbackLoopConfig = FeedbackLoopConfig()
    # number of biquads in SVF
    output_filter_config: OutputFilterConfig = OutputFilterConfig()

    @computed_field
    @property
    def delay_length_samps(self) -> List[int]:
        """Co-prime delay line lenghts for a given range"""
        np.random.seed(46434)
        delay_range_samps = ms_to_samps(np.asarray(self.delay_range_ms),
                                        self.sample_rate)
        # generate prime numbers in specified range
        prime_nums = np.array(list(
            sp.primerange(delay_range_samps[0], delay_range_samps[1])),
                              dtype=np.int32)
        rand_primes = prime_nums[np.random.permutation(len(prime_nums))]
        # delay line lengths
        delay_lengths = np.array(np.r_[rand_primes[:self.num_delay_lines - 1],
                                       sp.nextprime(delay_range_samps[1])],
                                 dtype=np.int32).tolist()
        return delay_lengths
