from dataclasses import dataclass
import pickle
from typing import Any, Optional

from numpy.typing import ArrayLike, NDArray


class NAFDatasetUnpickler(pickle.Unpickler):
    """
    I have changed the name from NAFDataset to NAFDatasetTrain
    to remove ambiguity between training and inference datasets
    """

    def __init__(self, file, is_train: bool = True):
        """Initialise data class, set is_train = True 
        if unpickling from NAFDataset to NAFDatasetTrain
        """
        super().__init__(file)
        self.is_train = is_train

    def find_class(self, module, name):
        """Change class for unpickling"""
        if (module.endswith("dataclass")
                or module == "__main__") and name == 'NAFDataset':
            # redirect to the new class
            return NAFDatasetTrain if self.is_train else NAFDatasetInfer
        return super().find_class(module, name)


@dataclass
class NAFDatasetTrain:
    num_train_receivers: int
    num_infer_receivers: int
    train_receiver_pos: NDArray  # of shape num_training_receivers x 3
    infer_receiver_pos: NDArray  # of shape num_infer_receivers x 3
    train_brirs: NDArray  # of shape num_training_receivers x num_orientation x num_time_samples x num_ears
    infer_brirs: NDArray  # of shape num_infer_receivers x num_orientation x num_time_samples  x num_ears
    orientation: ArrayLike  # of length 4


@dataclass
class BarycentricInterpolatedDataset:
    num_infer_receivers: int
    infer_receiver_pos: NDArray
    ref_srir: NDArray  # of shape num_infer_receivers x num_ambi_channels x num_time_samples
    pred_srir: NDArray  # of shape num_infer_receivers x num_ambi_channels x num_time_samples
    mixing_time_samp: ArrayLike  # of length num_infer_receivers
    ref_drir: Optional[
        NDArray] = None  # of shape num_receivers x num_directions x num_time_samples
    pred_drir: Optional[
        NDArray] = None  # of shape num_receivers x num_directions x num_time_samples


@dataclass
class NAFDatasetInfer:
    orientation: Any
    num_infer_receivers: int
    infer_receiver_pos: NDArray
    gt_brirs: NDArray
    infer_brirs: NDArray
