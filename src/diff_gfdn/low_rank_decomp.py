import os
from pathlib import Path
from typing import Tuple

from loguru import logger
import numpy as np
from numpy.typing import NDArray
import soundfile as sf
import torch
from torch import nn
from tqdm import tqdm

from .config.config import DiffGFDNConfig
from .dataloader import load_dataset, RIRData, RoomDataset
from .model import DiffGFDNSinglePos
from .solver import data_parser_var_receiver_pos, run_training_colorless_fdn
from .utils import get_response, ms_to_samps

# pylint: disable=W0632, E1136


def get_source_receiver_gains(
        room_data: RoomDataset) -> Tuple[NDArray, NDArray]:
    """
    Do a rank-1 decomposition of the matrix of common slope amplitudes to get the source and
    receiver gains (one for each centre frequency)
    Args:
        room_data (RoomDataset): room dataset object
    Returns:
        Tuple[NDArray, NDArray]: the input and output gains of size 
                                 num_src x num_slopes, num_rec x num_slopes
    """
    # do a rank-1 decomposition of the matrix
    g_in = np.zeros((room_data.num_src, room_data.num_rooms))
    g_out = np.zeros((room_data.num_rec, room_data.num_rooms))
    A_matrix = room_data.amplitudes
    A_recons = np.zeros_like(A_matrix)

    for k in range(room_data.num_rooms):
        cur_amp_matrix = A_matrix[..., k]
        [U, S, Vh] = np.linalg.svd(cur_amp_matrix)
        max_svd_idx = np.argmax(np.abs(S), axis=0)
        g_in[:, k] = np.sqrt(S[max_svd_idx]) * U[max_svd_idx, :]
        g_out[:, k] = np.sqrt(S[max_svd_idx]) * Vh[:, max_svd_idx]
        cur_recons_matrix = g_in[..., k] @ g_out[..., k].T
        A_recons[..., k] = cur_recons_matrix
        logger.info(
            f'Variance explained by first principal component: {S[max_svd_idx] / np.sum(S)}'
        )
        logger.info(
            f'Reconstruction error: {np.linalg.norm(cur_amp_matrix - cur_recons_matrix)}'
        )

    return g_in, g_out


def run_low_rank_decomp(config_dict: DiffGFDNConfig) -> NDArray:
    """
    Determine source and receiver gains by doing a rank-1 decomposition 
    of the matrix of common slope amplitudes
    Returns:
        NDArray: num_src x num_rec x ir_length RIRs at all positions
    """
    room_data = data_parser_var_receiver_pos(
        config_dict.room_dataset_path,
        num_freq_bins=config_dict.trainer_config.num_freq_bins,
    )

    ir_dir = Path(config_dict.trainer_config.ir_dir).resolve()
    if not os.path.exists(ir_dir):
        os.makedirs(ir_dir)
    config_dict = config_dict.copy(update={"num_groups": room_data.num_rooms})

    trainer_config = config_dict.trainer_config
    # prepare the training and validation data for DiffGFDN
    if trainer_config.batch_size != room_data.num_freq_bins:
        trainer_config = trainer_config.copy(
            update={"batch_size": room_data.num_freq_bins})

    # get the source and receiver gains
    g_in, g_out = get_source_receiver_gains(room_data)

    # get the colorless FDN params
    if config_dict.colorless_fdn_config.use_colorless_prototype:
        colorless_fdn_params = run_training_colorless_fdn(
            config_dict, num_freq_bins=trainer_config.num_freq_bins)
    else:
        colorless_fdn_params = None

    # initialise the model
    model = DiffGFDNSinglePos(
        config_dict.sample_rate,
        config_dict.num_groups,
        config_dict.delay_length_samps,
        trainer_config.device,
        config_dict.feedback_loop_config,
        config_dict.output_filter_config,
        use_absorption_filters=False,
        common_decay_times=room_data.common_decay_times,
        colorless_fdn_params=colorless_fdn_params,
    )

    # loop over all source and receiver positions
    all_rir_recons = np.zeros(
        (room_data.num_src, room_data.num_rec, room_data.rir_length),
        dtype=np.float32)

    mixing_time_samp = ms_to_samps(room_data.mixing_time_ms,
                                   config_dict.sample_rate)

    for src_idx in tqdm(range(room_data.num_src)):
        src_pos_to_investigate = np.squeeze(
            np.round(room_data.source_position[src_idx, :], 2))
        logger.info(f'Running GFDN for source pos: {src_pos_to_investigate}')
        for rec_idx in range(room_data.num_rec):
            rec_pos_to_investigate = np.squeeze(
                np.round(room_data.receiver_position[rec_idx, :], 2))
            true_ir = np.squeeze(room_data.rirs[src_idx, rec_idx, :])
            amplitudes = np.squeeze(room_data.amplitudes[src_idx, rec_idx, :])
            filename = f'ir_src={src_pos_to_investigate}_rec={rec_pos_to_investigate}.wav'
            save_ir_path = Path(trainer_config.ir_dir + 'true_' +
                                filename).resolve()
            if not os.path.exists(save_ir_path):
                sf.write(save_ir_path, true_ir, int(config_dict.sample_rate))

            # create RIRDataset
            rir_data = RIRData(
                rir=true_ir,
                sample_rate=config_dict.sample_rate,
                common_decay_times=room_data.common_decay_times,
                band_centre_hz=room_data.band_centre_hz,
                amplitudes=amplitudes,
                nfft=config_dict.trainer_config.num_freq_bins,
            )

            # prepare the training and validation data for DiffGFDN
            train_dataset = load_dataset(rir_data,
                                         trainer_config.device,
                                         train_valid_split_ratio=1.0,
                                         batch_size=trainer_config.batch_size,
                                         shuffle=False)

            model.eval()
            cur_gin = g_in[src_idx, :]
            cur_gout = g_out[rec_idx, :]
            model.input_scalars = nn.Parameter(
                torch.tensor(cur_gin[:, np.newaxis], dtype=torch.float32))
            model.output_scalars = nn.Parameter(
                torch.tensor(cur_gout[:, np.newaxis], dtype=torch.float32))

            with torch.no_grad():
                for data in train_dataset:
                    _, approx_ir = get_response(data, model)

                    # the input and output scalars should be scaled to match the energy of the desired RIR
                    H_late = torch.fft.rfft(approx_ir[mixing_time_samp:])
                    energyH = torch.mean(torch.pow(torch.abs(H_late), 2))
                    energyH_target = torch.mean(
                        torch.pow(torch.abs(data['target_late_response']), 2))
                    energy_diff = torch.div(energyH, energyH_target)
                    for name, prm in model.named_parameters():
                        if name in ('input_scalars', 'output_scalars'):
                            prm.data.copy_(
                                torch.div(prm.data,
                                          torch.pow(energy_diff, 1 / 4)))
                    _, approx_ir = get_response(data, model)
                    sf.write(
                        Path(trainer_config.ir_dir + 'approx_' +
                             filename).resolve(), approx_ir,
                        int(config_dict.sample_rate))
                    all_rir_recons[
                        src_idx,
                        rec_idx, :] = approx_ir[:room_data.rir_length].detach(
                        ).numpy()

    return all_rir_recons
