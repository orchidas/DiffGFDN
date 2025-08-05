from pathlib import Path
import pickle
from typing import Dict

from pydantic import BaseModel
import yaml

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
