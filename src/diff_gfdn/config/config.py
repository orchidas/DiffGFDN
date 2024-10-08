from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import sympy as sp
from pydantic import BaseModel, Field, computed_field, model_validator

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
    # by how much to constrain the pole radii when calculating output filter coeffs
    compress_pole_factor: float = 1.0


class TrainerConfig(BaseModel):
    # config file with training parameters
    # number of receivers in each training batch
    batch_size: int = 32
    # Number of frequency bins in the magnitude response
    num_freq_bins: Optional[int] = None
    # torch device - cuda or cpu or mps (for apple silicon)
    device: str = 'cpu'
    # split between traning and validation
    train_valid_split: float = 0.8
    # maximum epochs for training
    max_epochs: int = 5
    # learning rate for Adam optimiser
    lr: float = 0.01
    # length of IR filters (needed for calculating reguralisation loss)
    output_filt_ir_len_ms: float = 500
    # whether to use regularisation loss to reduce time domain aliasing
    use_reg_loss: bool = False
    # whether to use perceptual ERB loss
    use_erb_edr_loss: bool = False
    # directory to save results
    train_dir: str = "../output"
    # where to save the IRs
    ir_dir: str = "../audio"
    # by how much to reduce the radius of each pole during frequency sampling
    reduced_pole_radius: Optional[float] = None
    # attenuation in dB that the anti aliasing envelope should be reduced by
    alias_attenuation_db: Optional[int] = None 


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
    # whether to use scalar or frequency-dependent gains in delay lines
    use_absorption_filters: bool = True
    # config for the feedback loop
    feedback_loop_config: FeedbackLoopConfig = FeedbackLoopConfig()
    # number of biquads in SVF
    output_filter_config: OutputFilterConfig = OutputFilterConfig()
    # Validator to ensure the nested TrainerConfig is validated,
    # otherwise new sampling radius won't be set
    @model_validator(mode='before')
    @classmethod
    def validate_trainer_config(cls, values: Dict):
        """Make sure all fields of TrainerConfig are validated"""
        trainer_config = values.get('trainer_config')
        # Explicitly convert to TrainerConfig
        if isinstance(trainer_config, dict):
            values['trainer_config'] = TrainerConfig(**trainer_config)
        return values

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
