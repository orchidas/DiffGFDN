import os
from pathlib import Path
import shutil
from typing import List

from loguru import logger
import numpy as np
import pandas as pd
import pyfar as pf
from scipy.signal import fftconvolve
import soundfile as sf
import torch
import yaml

from diff_gfdn.config.config import DiffGFDNConfig
from diff_gfdn.dataloader import load_dataset, ThreeRoomDataset
from diff_gfdn.model import DiffGFDNVarReceiverPos
from diff_gfdn.solver import run_training_var_receiver_pos
from diff_gfdn.utils import get_response
from run_model import dump_config_to_pickle

# flake8: noqa: E231
# pylint: disable=W0621


def sum_arrays(series):
    """sum an array along first dimension"""
    return np.sum(series, axis=0)


def create_config(
    cur_freq_hz: int,
    data_path: str,
    freq_range: List,
    config_path: str,
    write_config: bool = True,
    # seed_base: int = 23463,
) -> DiffGFDNConfig:
    """Create config file for each subband"""
    # seed = seed_base + cur_freq_hz
    config_dict = {
        'room_dataset_path': data_path,
        'sample_rate': 32000.0,
        'num_delay_lines': 12,
        'use_absorption_filters': False,
        # 'seed': seed,
        'trainer_config': {
            'max_epochs': 10,
            'batch_size': 32,
            'save_true_irs': True,
            'train_valid_split': 0.8,
            'num_freq_bins': 131072,
            'use_edc_mask': True,
            'use_colorless_loss': True,
            'train_dir':
            f'output/grid_rir_treble_band_centre={cur_freq_hz}Hz_colorless_loss/',
            'ir_dir':
            f'audio/grid_rir_treble_band_centre={cur_freq_hz}Hz_colorless_loss/',
            'subband_process_config': {
                'centre_frequency': cur_freq_hz,
                'num_fraction_octaves': 1,
                'frequency_range': freq_range,
            },
        },
        'feedback_loop_config': {
            'coupling_matrix_type': 'scalar_matrix',
        },
        'output_filter_config': {
            'use_svfs': False,
            'num_hidden_layers': 3,
            'num_neurons_per_layer': 128,
            'num_fourier_features': 20,
        },
    }

    # writing the dictionary to a YAML file
    if write_config:
        cur_config_path = f'{config_path}/treble_data_grid_training_{cur_freq_hz}Hz_colorless_loss.yml'
        with open(cur_config_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(config_dict, file, default_flow_style=False)

    diff_gfdn_config = DiffGFDNConfig(**config_dict)
    return diff_gfdn_config


def training(freqs_list: int, data_path: str, training_complete: bool = False):
    """Run DiffGFDN training for various subbands"""
    subband_config_dicts = []
    config_path = Path("data/config").resolve()

    for k in range(len(freqs_list)):
        logger.info(f'Training GFDN for subband = {freqs_list[k]} Hz')
        cur_data_path = f'{data_path}/srirs_band_centre={freqs_list[k]}Hz.pkl'

        # generate config file
        config_dict = create_config(freqs_list[k],
                                    cur_data_path,
                                    freq_range=[freqs_list[0], freqs_list[-1]],
                                    config_path=config_path,
                                    write_config=training_complete)
        subband_config_dicts.append(config_dict)

        if not training_complete:
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
    return subband_config_dicts


def inferencing(freqs_list: List, config_dicts: List[DiffGFDNConfig],
                save_filename: str):
    """Run inferencing and save the full band RIRs for each position"""
    if not os.path.exists(save_filename):
        synth_subband_rirs = pd.DataFrame(
            columns=['frequency', 'position', 'time_samples'])

        # prepare the reconstructing filterbank
        subband_filters, _ = pf.dsp.filter.reconstructing_fractional_octave_bands(
            None,
            num_fractions=config_dicts[0].trainer_config.
            subband_process_config.num_fraction_octaves,
            frequency_range=(freqs_list[0], freqs_list[-1]),
            sampling_rate=config_dicts[0].sample_rate,
        )

        # loop through all subband frequencies
        for k in range(len(freqs_list)):
            logger.info(
                f'Running inferencing for subband = {freqs_list[k]} Hz')

            config_dict = config_dicts[k]
            room_data = ThreeRoomDataset(
                Path(config_dict.room_dataset_path).resolve(), config_dict)

            config_dict = config_dict.model_copy(
                update={"num_groups": room_data.num_rooms})
            trainer_config = config_dict.trainer_config

            # force the trainer config device to be CPU
            if trainer_config.device != 'cpu':
                trainer_config = trainer_config.model_copy(
                    update={"device": 'cpu'})

            # prepare the training and validation data for DiffGFDN
            train_dataset, _ = load_dataset(
                room_data,
                trainer_config.device,
                train_valid_split_ratio=1.0,
                batch_size=trainer_config.batch_size,
                shuffle=False)

            # initialise the model
            model = DiffGFDNVarReceiverPos(
                config_dict.sample_rate,
                config_dict.num_groups,
                config_dict.delay_length_samps,
                trainer_config.device,
                config_dict.feedback_loop_config,
                config_dict.output_filter_config,
                use_absorption_filters=False,
                common_decay_times=room_data.common_decay_times,
                use_colorless_loss=trainer_config.use_colorless_loss)

            checkpoint_dir = Path(trainer_config.train_dir +
                                  'checkpoints/').resolve()

            # load the trained weights for the particular epoch
            checkpoint = torch.load(
                f'{checkpoint_dir}/model_e{trainer_config.max_epochs-1}.pt',
                weights_only=True,
                map_location=torch.device('cpu'))
            # Load the trained model state
            model.load_state_dict(checkpoint)
            # in eval mode, no gradients are calculated
            model.eval()

            # loop through all positions
            for data in train_dataset:
                position = data['listener_position'].detach().cpu().numpy()
                _, _, h = get_response(data, model)

                # loop over all positions for a particular frequency band and add it to a dataframe
                for num_pos in range(position.shape[0]):
                    cur_rir = h[num_pos, :].detach().cpu().numpy()
                    cur_rir_filtered = fftconvolve(
                        cur_rir, subband_filters.coefficients[k, :])
                    # position should be saved as tuple because numpy array is unhashable
                    new_row = pd.DataFrame({
                        'frequency': [freqs_list[k]],
                        'position':
                        [(position[num_pos, 0], position[num_pos,
                                                         1], position[num_pos,
                                                                      2])],
                        'time_samples': [cur_rir_filtered]
                    })
                    synth_subband_rirs = pd.concat(
                        [synth_subband_rirs, new_row], ignore_index=True)

        synth_subband_rirs.to_pickle(save_filename)
    else:
        logger.info('Reading saved pickle file')
        synth_subband_rirs = pd.read_pickle(save_filename)

    # Save the synthesised RIRs
    logger.info('Saving synthesised RIRs')

    # Group by 'position' and sum the 'time_samples' over each frequency band
    synth_rirs = synth_subband_rirs.groupby('position')['time_samples'].apply(
        sum_arrays)

    # Convert to DataFrame if needed
    synth_rirs_df = synth_rirs.reset_index()
    synth_rirs_df.columns = ['position', 'time_samples']
    output_path = Path("audio/grid_rir_treble_subband_processing")

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # Save each row's 'time_samples' as a WAV file
    for _, row in synth_rirs_df.iterrows():
        position = row['position']
        values = row['time_samples']

        filename = f'{output_path.resolve()}/ir_({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}).wav'
        sf.write(filename, values, int(config_dicts[0].sample_rate))

    logger.info("Done...")


if __name__ == '__main__':
    freqs_list = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    data_path = Path('resources/Georg_3room_FDTD').resolve()
    config_dicts = training(freqs_list, data_path, training_complete=True)
    save_filename = Path(
        'output/treble_data_grid_training_final_rirs.pkl').resolve()
    inferencing(freqs_list, config_dicts, save_filename)
