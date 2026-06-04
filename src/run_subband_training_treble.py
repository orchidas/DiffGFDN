import argparse
import os
from pathlib import Path
import shutil
from typing import List, Optional

from loguru import logger
import numpy as np
import yaml

from diff_gfdn.config.config import ColorlessFDNConfig, DiffGFDNConfig, TestSetConfig
from diff_gfdn.config.config_loader import dump_config_to_pickle, load_and_validate_config
from diff_gfdn.inference import infer_all_octave_bands_omni_gfdn
from diff_gfdn.solver import run_training_var_receiver_pos

# flake8: noqa: E231
# pylint: disable=W0621, W0632, E0402, C0301


def create_config(
    cur_freq_hz: int,
    data_path: str,
    freq_range: List,
    config_path: str,
    train_valid_split: float,
    write_config: bool = True,
    seed_base: int = 23463,
    allow_coupling: bool = False,
    use_same_colorless_params: bool = False,
) -> DiffGFDNConfig:
    """
    Create config file for each subband
    Args:
        cur_freq_hz (int): frequency at which we will evaluate subband GFDN
        data_path (str): path to original dataset
        freq_range (List): maximum and minimum frequencies in octave band processing
        config_path (str): path where config file is to be saved
        train_valid_split (str): how to split dataset into training and validation
        write_config (bool): whether to write the config file to disk
        seed_base (int): base seed for all training
        allow_coupling (bool): whether to allow coupling between groups in each subband GFDN
        use_same_colorless_params (bool): use same m, A, b, c for all subband GFDNs

    """

    # parameters from hypertuning
    if cur_freq_hz == 63:
        num_hidden_layers = 1
        num_neurons_per_layer = 2**3
    elif cur_freq_hz == 125:
        num_hidden_layers = 1
        num_neurons_per_layer = 2**4
    elif cur_freq_hz in (250, 500, 1000):
        num_hidden_layers = 5
        num_neurons_per_layer = 2**4
    else:
        num_hidden_layers = 3
        num_neurons_per_layer = 2**7

    test_set_config = TestSetConfig()
    colorless_config_dict = ColorlessFDNConfig()
    use_colorless_loss = not use_same_colorless_params

    # different options for ablation study
    if allow_coupling:
        str_append = '_colorless_loss_diff_delays_with_coupling'
    elif use_same_colorless_params:
        str_append = '_same_colorless_params'
        colorless_config_dict = {
            'use_colorless_prototype':
            True,
            'batch_size':
            4000,
            'max_epochs':
            15,
            'saved_param_path':
            str(Path('output/fixed_colorless_fdn_params/').resolve())
        }
        colorless_config_dict = ColorlessFDNConfig(**colorless_config_dict)
    else:
        str_append = '_colorless_loss_diff_delays'

    # this ensures that the same delay line lengths are used if we are using the
    # same colorless FDN params, or else different delay line lengths are used for
    # different subband GFDNs
    if use_same_colorless_params:
        seed = seed_base
    else:
        seed = seed_base + cur_freq_hz + np.random.randint(0, 100, 1).item()
    config_dict = {
        'seed': seed,
        'room_dataset_path': data_path,
        'sample_rate': 32000.0,
        'num_delay_lines': 12,
        'decay_filter_config': {
            'use_absorption_filters': False,
            'learn_common_decay_times': False,
            'initialise_with_opt_values': True,
        },
        'trainer_config': {
            'max_epochs': 20,
            'batch_size': 32,
            'device': 'gpu',
            'save_true_irs': True,
            'train_valid_split': train_valid_split,
            'hold_out_test_set': {
                'seed': test_set_config.seed,
                'ratio': 0.1,
            },
            'lr': 1e-3,
            'num_freq_bins': 131072,
            'use_edc_mask': True,
            'use_colorless_loss': use_colorless_loss,
            'edc_loss_weight': 10,
            'sparsity_loss_weight': 2,
            'use_asym_spectral_loss': True,
            'train_dir':
            f'output/train_split_test/{train_valid_split:.1f}/grid_rir_treble_band_centre={cur_freq_hz}Hz{str_append}/',
            'ir_dir':
            f'audio/train_split_test/{train_valid_split:.1f}/grid_rir_treble_band_centre={cur_freq_hz}Hz{str_append}/',
            'subband_process_config': {
                'centre_frequency': cur_freq_hz,
                'num_fraction_octaves': 1,
                'frequency_range': freq_range,
                'use_amp_preserving_filterbank': True,
            },
        },
        'feedback_loop_config': {
            'coupling_matrix_type': 'scalar_matrix',
            'use_zero_coupling': not allow_coupling,
        },
        'output_filter_config': {
            'use_svfs': False,
            'num_hidden_layers': num_hidden_layers,
            'num_neurons_per_layer': num_neurons_per_layer,
            'num_fourier_features': 20,
        },
        'colorless_fdn_config': colorless_config_dict,
    }

    # writing the dictionary to a YAML file
    if write_config:
        logger.info("Writing to config file")
        cur_config_path = f'{config_path}/treble_data_grid_training_{cur_freq_hz}Hz_{str_append}.yml'

        # Convert the Pydantic config to dict
        dict_to_dump = config_dict.copy()

        # this fixes the setting of colorless_fdn_config with a dict
        dict_to_dump["colorless_fdn_config"] = dict_to_dump[
            "colorless_fdn_config"].model_dump()

        with open(cur_config_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(dict_to_dump, file, default_flow_style=False)

    diff_gfdn_config = DiffGFDNConfig(**config_dict)
    return diff_gfdn_config


def training(freqs_list: List, config_dicts: List[DiffGFDNConfig]):
    """Run DiffGFDN training for various subbands"""

    if freqs_list is None:
        logger.info("No frequencies to train on")
        return
    else:
        for k in range(len(freqs_list)):
            logger.info(f'Training GFDN for subband = {freqs_list[k]} Hz')
            # get config file
            config_dict = config_dicts[k]

            # make output directory
            if config_dict.trainer_config.train_dir is not None:
                # remove directory if it already exists, we want it to be overwritten
                if os.path.isdir(config_dict.trainer_config.train_dir):
                    shutil.rmtree(config_dict.trainer_config.train_dir)

                # create the output directory
                os.makedirs(config_dict.trainer_config.train_dir)

                # write arguments to a pickle file
                args_file = os.path.join(config_dict.trainer_config.train_dir,
                                         'config_args.pickle')
                dump_config_to_pickle(config_dict, args_file)

                # run the training
                run_training_var_receiver_pos(config_dict)

        logger.info('Training complete')


def main(freqs_list_train: Optional[List] = None,
         split_ratio: Optional[float] = None,
         allow_coupling: bool = False,
         use_same_colorless_params: bool = False):
    """
    Main function to run the training and inferencing
    Args:
        freqs_list_train (List): list of frequencies to train on
        split_ratio (float): training and validation set split ratios
        allow_coupling (bool): allow coupling in the feedback matrix
        use_same_colorless_params (bool): use same m, A, b, c for all subband GFDNs
    """

    # generate config file
    freqs_list = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    config_path = Path("data/config").resolve()
    data_path = Path('resources/Georg_3room_FDTD').resolve()
    config_dicts = []
    training_complete = freqs_list_train is None

    if not training_complete:
        for k in range(len(freqs_list_train)):
            cur_data_path = f'{data_path}/srirs_band_centre={int(freqs_list_train[k])}Hz.pkl'

            # generate config file
            config_dict = create_config(
                int(freqs_list_train[k]),
                cur_data_path,
                freq_range=[freqs_list[0], freqs_list[-1]],
                config_path=config_path,
                train_valid_split=split_ratio
                if split_ratio is not None else 0.8,
                write_config=not training_complete,
                allow_coupling=allow_coupling,
                use_same_colorless_params=use_same_colorless_params,
            )
            config_dicts.append(config_dict)
        logger.info("Done creating config files")

        # run training
        training(freqs_list_train, config_dicts)

    # inferencing
    if training_complete:
        config_dicts = []
        for k in range(len(freqs_list)):
            # if no explicit split ratio is provided, read the saved config file
            if split_ratio is None:
                cur_config_path = f'{config_path}/treble_data_grid_training_{freqs_list[k]}Hz'\
                + '_colorless_loss.yml'
                cur_config_dict = load_and_validate_config(
                    cur_config_path, DiffGFDNConfig)
            else:
                cur_data_path = f'{data_path}/srirs_band_centre={int(freqs_list[k])}Hz.pkl'

                cur_config_dict = create_config(
                    freqs_list[k],
                    data_path=cur_data_path,
                    freq_range=[freqs_list[0], freqs_list[-1]],
                    config_path=config_path,
                    train_valid_split=split_ratio,
                    write_config=False,
                )
            config_dicts.append(cur_config_dict)

        logger.info("Done reading config files")
        save_filename = Path(
            f'output/treble_data_grid_training_final_rirs_colorless_loss_diff_delays_split={split_ratio:.1f}.pkl'
        ).resolve()
        output_path = Path(
            f"audio/train_split_test/{split_ratio:.1f}/grid_rir_treble_subband_processing_colorless_loss_diff_delays"
        )

        infer_all_octave_bands_omni_gfdn(freqs_list, config_dicts,
                                         save_filename, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process frequency list.")
    parser.add_argument(
        "--freqs",
        nargs="+",  # Accepts multiple values
        type=float,  # Convert to float
        help="List of frequencies for training")
    parser.add_argument("--allow_coupling",
                        action="store_true",
                        help="Allow coupling in feedback matrix")
    parser.add_argument(
        "--use_same_colorless_params",
        action="store_true",
        help="Use same delay line lengths and A, b, c for all subband GFDNs")
    parser.add_argument("--split",
                        type=float,
                        default=0.8,
                        help="Training and validation set split ratio")

    args = parser.parse_args()
    main(args.freqs, args.split, args.allow_coupling,
         args.use_same_colorless_params)
