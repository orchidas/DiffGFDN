from dataclasses import dataclass
import os
import pickle
from typing import List

import torch

from ..config.config import DiffGFDNConfig


@dataclass
class ColorlessFDNResults:
    opt_input_gains: torch.tensor
    opt_output_gains: torch.tensor
    opt_feedback_matrix: torch.tensor


def get_colorless_fdn_params(
        config_dict: DiffGFDNConfig) -> List[ColorlessFDNResults]:
    """Return a list of ColorlessFDNResults objects, one for each group in the GFDN"""
    params_opt = []
    colorless_dir = config_dict.trainer_config.train_dir + "colorless-fdn/"
    num_groups = config_dict.num_groups

    for k in range(num_groups):
        filename = f'parameters_opt_group={k + 1}.pkl'
        with open(os.path.join(colorless_dir, filename), "rb") as f:
            cur_params = pickle.load(f)
        params_opt.append(cur_params)
    return params_opt
