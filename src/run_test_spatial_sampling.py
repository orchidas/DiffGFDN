import argparse
import os

from loguru import logger
import torch

from run_model import dump_config_to_pickle, load_and_validate_config
from spatial_sampling.config import SpatialSamplingConfig
from spatial_sampling.solver import run_training_spatial_sampling

# pylint: disable=W0621


def main(config_dict: SpatialSamplingConfig, infer_only: bool = False):
    """
    Test how much spatial smpling is needed for the MLP to learn the spatial distribution
    of the common slope amplitudes. The config file contains a range of train-valid split ratios.
    If infer_only is true, only inference is done and plots are made.
    """
    # set random seeds
    torch.manual_seed(config_dict.seed)

    # make output directory
    if config_dict.train_dir is not None:
        # remove directory if it already exists, we want it to be overwritten
        if os.path.isdir(config_dict.train_dir):
            logger.warning("Training directory exists")
            # shutil.rmtree(config_dict.train_dir)
        else:
            # create the output directory
            os.makedirs(config_dict.train_dir)

    # save arguments
    args_file = os.path.join(config_dict.train_dir, 'config_args.pickle')
    dump_config_to_pickle(config_dict, args_file)
    run_training_spatial_sampling(config_dict, infer_only)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training or inference")

    parser.add_argument("-c",
                        "--config_file",
                        type=str,
                        help="Config file for training")

    parser.add_argument(
        "--infer",
        action="store_true",
        help="Set this flag to run inference instead of training")

    args = parser.parse_args()
    if args.config_file:
        config_dict = load_and_validate_config(args.config_file,
                                               SpatialSamplingConfig)
    else:
        config_dict = SpatialSamplingConfig()
    main(config_dict, args.infer)
