# pylint: disable=relative-beyond-top-level

from enum import Enum
from typing import List, Optional, Tuple

from loguru import logger
import numpy as np
from pydantic import BaseModel, computed_field, ConfigDict, Field, field_validator, model_validator
import sympy as sp
import torch

from spatial_sampling.config import BeamformerType

from ..utils import ms_to_samps


class CouplingMatrixType(Enum):
    """Different types of coupling matrix"""

    # scalar, unitary coupling matrix
    SCALAR = "scalar_matrix"
    # paraunitary FIR coupling matrix
    FILTER = "filter_matrix"
    # this gets triggered if we want the coupled feedback matrix to be
    # a random unitary matrix that is optimised, i.e, it has no special
    # structure representing individual mixing matrix and a coupling matrix
    RANDOM = "random_matrix"

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


class MLPTuningConfig(BaseModel):
    # tune hyperparameters of the MLP
    tune_hyperparameters: bool = True
    min_layers: int = 1
    max_layers: int = 20
    min_neurons: int = 2**4
    max_neurons: int = 2**7
    step_size: int = 2**4
    num_trials: int = 50


class SubbandProcessingConfig(BaseModel):
    # config for running DiffGFDNs in parallel, one for each subband
    centre_frequency: float
    frequency_range: Tuple
    num_fraction_octaves: int = 3
    use_amp_preserving_filterbank: bool = True


class OutputFilterConfig(BaseModel):
    # config for training the output filters based on listener location
    # number of biquads in each filter
    # whether to use SVFs or scalar gains
    use_svfs: bool = True
    # by how much to constrain the pole radii when calculating output filter coeffs
    compress_pole_factor: float = 1.0
    # used only if MLP is used for training the SVF filters
    mlp_tuning_config: Optional[MLPTuningConfig] = None
    # or, fix them to these numbers
    num_hidden_layers: int = 3
    num_neurons_per_layer: int = 2**7
    num_fourier_features: int = 10
    # sinusoidal encoding
    encoding_type: FeatureEncodingType = FeatureEncodingType.SINE
    # beamforming type for converting from SHD to directional amplitudes
    beamformer_type: Optional[BeamformerType] = None
    # whether to use skip connections in MLP
    use_skip_connections: bool = False


class DecayFilterConfig(BaseModel):
    # whether to use scalar or frequency-dependent gains in delay lines
    use_absorption_filters: bool = True
    # whether to learn the common decay times or not
    learn_common_decay_times: bool = False
    # whether to initialise decay filters with pre-found values
    initialise_with_opt_values: bool = True


class TestSetConfig(BaseModel):
    seed: int = 4314
    ratio: float = 0.1


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
    # whether to have a separate test set
    hold_out_test_set: Optional[TestSetConfig] = None
    # grid resolution if training on a uniform grid
    grid_resolution_m: Optional[float] = None
    # maximum epochs for training
    max_epochs: int = 5
    # learning rate for Adam optimiser
    lr: float = 0.01
    # learning rate for input output gains
    io_lr: float = 0.01
    # learning rate for coupling angles
    coupling_angle_lr: float = 0.01
    # length of IR filters (needed for calculating reguralisation loss)
    output_filt_ir_len_ms: float = 500
    # whether to use regularisation loss to reduce time domain aliasing
    use_reg_loss: bool = False
    # whether to use perceptual ERB loss
    use_erb_edr_loss: bool = False
    # whether to use colorless loss in the DiffGFDN's loss itself
    use_colorless_loss: bool = False
    use_asym_spectral_loss: bool = False
    # weights for edc and edr loss
    edc_loss_weight: float = 1.0
    edr_loss_weight: float = 1.0
    spectral_loss_weight: float = 1.0
    sparsity_loss_weight: float = 1.0
    # whether to use masking while calculating edc loss
    use_edc_mask: bool = False
    # whether to use frequency-based weighting in loss
    use_frequency_weighting: bool = False
    # whether the GFDN is processing only one subband
    subband_process_config: Optional[SubbandProcessingConfig] = None
    # directory to save results
    train_dir: str = "output/cpu/"
    # where to save the IRs
    ir_dir: str = "audio/cpu/"
    # whether to save the true measured IRs as wave files
    save_true_irs: bool = False
    # attenuation in dB that the anti aliasing envelope should be reduced by
    alias_attenuation_db: Optional[int] = None
    # by how much to reduce the radius of each pole during frequency sampling
    reduced_pole_radius: float = Field(
        default=1.0)  # Default value, to be set dynamically

    # validator for training on GPU
    @field_validator('device', mode='after')
    @classmethod
    def validate_training_device(cls, value):
        """Validate GPU, if it is used for training"""
        if value == 'cuda':
            assert torch.cuda.is_available(
            ), "CUDA is not available for training"
            logger.info(f"Running on GPU: {torch.cuda.get_device_name(0)}")

    # validator for the 'reduced_pole_radius' field
    @model_validator(mode='after')
    @classmethod
    def calculate_reduced_pole_radius(cls, model):
        """Set reduced pole radius based on alias_attenuation_db"""
        alias_attenuation_db = model.alias_attenuation_db
        num_freq_bins = model.num_freq_bins
        if alias_attenuation_db is not None and num_freq_bins is not None:
            model.reduced_pole_radius = 10**(-abs(alias_attenuation_db) /
                                             num_freq_bins / 20)
        return model


class ColorlessFDNConfig(BaseModel):
    """Config file for colorless FDN training"""

    # whether to use colorless FDN to get the input/output gains and feedback matrix
    use_colorless_prototype: bool = False
    # batch size for training
    batch_size: int = 2000
    # maximum epochs
    max_epochs: int = 20
    # training and validation split
    train_valid_split: float = 0.8
    # learning rate for Adam optimiser
    lr: float = 0.01
    # weigth for the sparsity loss
    alpha: float = 1


class DiffGFDNConfig(BaseModel):
    """Config file for training the DiffGFDN"""

    # random seed for reproducibility
    seed: int = 46434
    # path to three room dataset
    room_dataset_path: str = 'resources/Georg_3room_FDTD/srirs.pkl'
    # number of groups in the GFDN
    num_groups: int = 3
    # if a single measurement is being used
    ir_path: Optional[str] = None
    # sampling rate of the FDN
    sample_rate: float = 32000.0
    # training config
    trainer_config: TrainerConfig = TrainerConfig()
    # delay range in ms - first delay should be after the mixing time
    delay_range_ms: List[float] = [20.0, 50.0]
    # ambisonics order for directional FDN
    ambi_order: Optional[int] = None
    # total number of delay lines
    num_delay_lines: Optional[int] = 12

    # config for the feedback loop
    feedback_loop_config: FeedbackLoopConfig = FeedbackLoopConfig()
    # config for decay filters
    decay_filter_config: DecayFilterConfig = DecayFilterConfig()
    # number of biquads in SVF
    output_filter_config: OutputFilterConfig = OutputFilterConfig()
    input_filter_config: Optional[OutputFilterConfig] = OutputFilterConfig()
    # colorless FDN config
    colorless_fdn_config: ColorlessFDNConfig = ColorlessFDNConfig()

    @model_validator(mode="after")
    def set_num_delay_lines(self):
        """Set number of delay lines based on ambisonics order"""
        if self.ambi_order is not None:
            self.num_delay_lines = ((self.ambi_order + 1)**2) * self.num_groups
        return self

    # validator for the 'reduced_pole_radius' field
    @model_validator(mode='after')
    @classmethod
    def set_train_valid_ratio(cls, model):
        """Set training and validation set ratio"""
        if model.trainer_config.grid_resolution_m is not None:
            if model.ambi_order is None:
                raise AttributeError(
                    "Only use grid resolution for directional reverberation training!"
                )
            model.trainer_config.train_valid_split = None
        return model

    @computed_field
    @property
    def delay_length_samps(self) -> List[int]:
        """Co-prime delay line lenghts for a given range"""
        delay_range_samps = ms_to_samps(np.asarray(self.delay_range_ms),
                                        self.sample_rate)
        # generate prime numbers in specified range
        prime_nums = np.array(list(
            sp.primerange(delay_range_samps[0], delay_range_samps[1])),
                              dtype=np.int32)

        np.random.seed(self.seed)
        rand_primes = prime_nums[np.random.permutation(len(prime_nums))]
        # delay line lengths
        delay_lengths = np.array(np.r_[rand_primes[:self.num_delay_lines - 1],
                                       sp.nextprime(delay_range_samps[1])],
                                 dtype=np.int32).tolist()
        return delay_lengths

    # forbid extra fields - adding this to help prevent errors in config file creation
    model_config = ConfigDict(extra="forbid")
