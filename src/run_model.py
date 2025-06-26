import argparse
import os
import shutil
import time

import numpy as np
import torch

from diff_gfdn.config.config import DiffGFDNConfig
from diff_gfdn.config.config_loader import dump_config_to_pickle, load_and_validate_config
from diff_gfdn.solver import run_training_single_pos, run_training_var_receiver_pos


def parse_args() -> argparse.Namespace:
    """
    Argument parser

    Returns:
        (Namespace): an argparse namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config_file",
        default=None,
        help="Configuration file (YAML) containing diff GFDN \
        (if none provided the default parameters are loaded).",
    )

    arguments = parser.parse_args()
    return arguments


def main():
    """Read config file and run the training"""
    args = parse_args()
    if args.config_file:
        config_dict = load_and_validate_config(args.config_file,
                                               DiffGFDNConfig)
    else:
        config_dict = DiffGFDNConfig()

    # set random seeds
    np.random.seed(config_dict.seed)
    torch.manual_seed(config_dict.seed)

    # make output directory
    if config_dict.trainer_config.train_dir is not None:

        # remove directory if it already exists, we want it to be overwritten
        if os.path.isdir(config_dict.trainer_config.train_dir):
            shutil.rmtree(config_dict.trainer_config.train_dir)

        # create the output directory
        os.makedirs(config_dict.trainer_config.train_dir)
    else:
        args.train_dir = os.path.join('output', time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(config_dict.trainer_config.train_dir)

    # save arguments
    args_file = os.path.join(config_dict.trainer_config.train_dir,
                             'config_args.pickle')
    dump_config_to_pickle(config_dict, args_file)

    # run the training module either for various source-listener positions
    # or for a single measurement
    if config_dict.ir_path is None:
        run_training_var_receiver_pos(config_dict)
    else:
        run_training_single_pos(config_dict)


if __name__ == '__main__':
    main()
