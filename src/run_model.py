import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Dict

import yaml
from pydantic import BaseModel

from diff_gfdn.config.config import DiffGFDNConfig
from diff_gfdn.solver import run_training

# pylint: disable=W1514


def load_yaml_config(file_path: str):
    """Given a config path, load the config as a YAML file"""
    # Resolve the relative file path
    file_path = Path(file_path).resolve()

    # Read and parse the YAML file
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)

    return config_data


def load_and_validate_config(file_path: str, config_class: BaseModel):
    """
    Given a path, load and validate config
    Args: 
        file_path (str) : Path to config
        config_class (BaseModel): config class (child of BaseModel)
    """
    config_data = load_yaml_config(file_path)
    return config_class(**config_data)


def dump_config_to_pickle(config_data: Dict, output_path: str):
    """
    Dump the contents of the config dictionary into a pickle file
    Args:
        config_data (Dict): dictionary containing config data
        output_path (str): path where to dump the data
    """
    # Resolve the output file path
    output_path = Path(output_path).resolve()

    # Dump the config data to a pickle file
    with open(output_path, 'wb') as pickle_file:
        pickle.dump(config_data, pickle_file)


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
        nargs='*',
        default=None,
        help="Configuration file (YAML) containing upmixer parameters \
        (if none provided the default parameters are loaded).",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':

    args = parse_args()
    if args.config_file:
        config_dict = load_and_validate_config(args.config_path,
                                               DiffGFDNConfig)
    else:
        config_dict = DiffGFDNConfig()

    # make output directory
    if config_dict.trainer_config.train_dir is not None:
        if not os.path.isdir(config_dict.trainer_config.train_dir):
            os.makedirs(config_dict.trainer_config.train_dir)
    else:
        args.train_dir = os.path.join('output', time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(config_dict.trainer_config.train_dir)

    # save arguments
    args_file = os.path.join(config_dict.trainer_config.train_dir,
                             'config_args.pickle')
    dump_config_to_pickle(config_dict, args_file)

    # run the training module
    run_training(config_dict)
