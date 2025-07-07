import argparse
import os
from pathlib import Path
import pickle
from typing import List, Optional

from loguru import logger
import torch
from pathlib import Path
from dataclass import NAFDatasetInfer
from run_model import dump_config_to_pickle, load_and_validate_config
from sofa_parser import convert_srir_to_brir, HRIRSOFAReader, save_to_sofa
from spatial_sampling.config import SpatialSamplingConfig
from spatial_sampling.dataloader import parse_room_data
from spatial_sampling.inference import get_ambisonic_rirs
from spatial_sampling.solver import run_training_spatial_sampling

# pylint: disable=W0621
# flake8: noqa=E251


def run_training(config_dict: SpatialSamplingConfig, infer_only: bool):
    """Run training / inference (if inter_only is true) for a single config dict"""
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


def run_inference_on_all_bands(
    output_path: str,
    data_path: str,
    config_path: str,
    infer_dataset_path: str,
    grid_resolution_m: float,
    return_brirs: bool = False,
    hrtf_path: Optional[str] = None,
):
    """
    Once the model is trained for each frequency band, 
    run inference over all bands and save the data.
    Args:
        output_path (str): where to save the output file (pkl for BRIR, sofa for SRIR)
        data_path (str): path where original full band CS dataset pickle file is saved
        config_path (str): path to config files
        infer_dataset_path (str): path to inference dataset containing listener positions. 
                                  Data must be of type NAFDatasetInfer
        grid_resolution_m (float): what is the grid resolution for inferencing?
        return_brirs (bool): whether to return BRIRs or ambisonic RIRs
        hrtf_path (str, optional): path to hrtf, if returning BRIRs
    """
    logger.info("Running inferencing for all octave bands...")
    assert infer_dataset_path is not None, "Must provide path to inference dataset \
            containing infererence positions"

    assert data_path is not None, "Must provide path to full band dataset"

    assert grid_resolution_m is not None, "Must provide grid resolution for inference"

    with open(infer_dataset_path, "rb") as f:
        infer_dataset = pickle.load(f)
    infer_pos_list = infer_dataset.infer_receiver_pos

    # get the original dataset
    room_data = parse_room_data(data_path)
    pred_room_data = get_ambisonic_rirs(infer_pos_list,
                                        room_data,
                                        use_trained_model=True,
                                        config_path=config_path,
                                        grid_resolution_m=grid_resolution_m)
    if return_brirs:
        logger.info("Converting to BRIRs")
        # get the HRTF
        hrtf_reader = HRIRSOFAReader(hrtf_path)

        if hrtf_reader.fs != room_data.sample_rate:
            logger.info(f"Resampling HRTFs to {room_data.sample_rate: .0f} Hz")
            hrtf_reader.resample_hrirs(room_data.sample_rate)

        pred_brirs = convert_srir_to_brir(pred_room_data.rirs, hrtf_reader,
                                          infer_dataset.orientation)

        mlp_brir_dataset = NAFDatasetInfer(
            orientation=infer_dataset.orientation,
            num_infer_receivers=infer_dataset.num_infer_receivers,
            infer_receiver_pos=infer_pos_list,
            gt_brirs=infer_dataset.gt_brirs,
            infer_brirs=pred_brirs)

        logger.info("Saving BRIRs in a pickle file")
        with open(output_path, "wb") as f:
            pickle.dump(mlp_brir_dataset, f)
    else:
        logger.info("Saving SRIRs to SOFA file")
        save_to_sofa(pred_room_data, output_path)


def main(
    config_dict: SpatialSamplingConfig,
    config_path: str,
    freqs_list_train: Optional[List] = None,
    infer_only: bool = False,
):
    """
    Test how much spatial sampling is needed for the MLP to learn the spatial distribution
    of the common slope amplitudes. The config file contains a range of train-valid split ratios.
    Args:
        config_dict (SpatialSamplingConfig): if training a single frequency band, the config dictionary
        config_path (str): path to config files
        freqs_list_train (List): list of frequency bands to train
        infer_only (bool): If true, only inference is done for the specified frequencu bands
                    and plots are made.        
    """
    if freqs_list_train is not None:

        for k in range(len(freqs_list_train)):
            logger.info(f"Running training for {freqs_list_train[k]} Hz band")

            config_dict_path = Path(config_path) / f'treble_data_grid_training_{int(freqs_list_train[k])}Hz_directional_spatial_sampling_test.yml'
            cur_config_dict = load_and_validate_config(str(config_dict_path),
                                                       SpatialSamplingConfig)
            run_training(cur_config_dict, infer_only)
    else:
        run_training(config_dict, infer_only)


#########################################################################

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

    parser.add_argument(
        "--config_path",
        type=str,
        default=Path('data/config/spatial_sampling/').resolve(),
        help="Path to config files")

    parser.add_argument(
        "--freqs",
        nargs="+",  # Accepts multiple values
        type=float,  # Convert to float
        default=None,
        help="List of frequencies for training")

    parser.add_argument(
        "--data_path",
        type=str,
        default=Path('resources/Georg_3room_FDTD/srirs_spatial.pkl').resolve(),
        help="Path to full band dataset (needed for inferencing")

    parser.add_argument("--infer_dataset_path",
                        type=str,
                        default=None,
                        help="Path to Inference Dataset")

    parser.add_argument("--grid_res",
                        type=float,
                        default=None,
                        help="Grid resolution for inference")

    parser.add_argument(
        "--return_brirs",
        action="store_true",
        help="Set this flag to run inference instead of training")

    parser.add_argument(
        "--hrtf_path",
        type=str,
        default=Path(
            'resources/HRTF/48kHz/KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa'
        ).resolve(),
        help="Path to HRTF set")

    parser.add_argument("-o",
                        "--output_path",
                        type=str,
                        default=None,
                        help="Output file path")

    args = parser.parse_args()
    if args.config_file:
        config_dict = load_and_validate_config(args.config_file,
                                               SpatialSamplingConfig)
    else:
        config_dict = SpatialSamplingConfig()

    if args.infer_dataset_path is None:
        main(
            config_dict,
            config_path=args.config_path,
            freqs_list_train=args.freqs,
            infer_only=args.infer,
        )
    else:
        run_inference_on_all_bands(
            output_path=args.output_path,
            data_path=args.data_path,
            config_path=args.config_path,
            infer_dataset_path=args.infer_dataset_path,
            grid_resolution_m=args.grid_res,
            return_brirs=args.return_brirs,
            hrtf_path=args.hrtf_path,
        )
