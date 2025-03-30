import os
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger
import numpy as np
from numpy.typing import ArrayLike, NDArray
import soundfile as sf
import torch
from torch import nn
from tqdm import tqdm

from .config.config import DiffGFDNConfig
from .dataloader import load_dataset, RIRData, RoomDataset
from .filters.geq import design_geq
from .gain_filters import BiquadCascade, SOSFilter
from .model import DiffGFDNSinglePos
from .solver import data_parser_var_receiver_pos, run_training_colorless_fdn
from .utils import db, get_response, ms_to_samps

# pylint: disable=W0632, E1136


def fit_filters_to_gains(target_gains: ArrayLike,
                         band_centre_hz: ArrayLike,
                         fs: float,
                         device: Optional[str] = 'cpu') -> SOSFilter:
    """Fit SOSFilters to gains in subbands"""
    shelving_crossover_hz = [
        band_centre_hz[0] / pow(2, 1 / 2), band_centre_hz[-1] * pow(2, 1 / 2)
    ]
    b, a = design_geq(db(target_gains),
                      center_freq=torch.tensor(band_centre_hz),
                      shelving_crossover=torch.tensor(shelving_crossover_hz),
                      fs=fs)
    filter_order = b.shape[0]
    biquads = BiquadCascade(filter_order, b, a)
    return SOSFilter(filter_order, biquads, device=device)


def get_source_receiver_filters(
        room_data: RoomDataset
) -> Tuple[NDArray[np.object_], NDArray[np.object_]]:
    """
    Do a rank-1 decomposition of the matrix of common slope amplitudes to get the source and
    receiver filters
    Args:
        room_data (RoomDataset): room dataset object
    Returns:
        Tuple[SOSFilter, SOSFilter]: the input and output filters of size 
                                 num_src x num_slopes, num_rec x num_slopes
    """
    # this is of size num_src_pos x num_rec_pos x num_slopes x nbands
    A_matrix = room_data.amplitudes
    A_recons = np.zeros_like(A_matrix)
    band_centre_hz = room_data.band_centre_hz
    num_subbands = len(room_data.band_centre_hz)
    assert A_matrix.shape[
        -1] == num_subbands, "Centre frequencies must be in octave bands"

    g_in = np.empty((room_data.num_src, room_data.num_rooms), dtype=SOSFilter)
    g_out = np.empty((room_data.num_rec, room_data.num_rooms), dtype=SOSFilter)

    g_in = np.zeros((room_data.num_src, room_data.num_rooms, num_subbands))
    g_out = np.zeros((room_data.num_rec, room_data.num_rooms, num_subbands))

    for k in range(room_data.num_rooms):
        cur_gin = np.zeros((room_data.num_src, num_subbands))
        cur_gout = np.zeros((room_data.num_rec, num_subbands))

        for b in range(num_subbands):
            cur_amp_matrix = A_matrix[..., k, b]
            [U, S, Vh] = np.linalg.svd(cur_amp_matrix)
            max_svd_idx = np.argmax(np.abs(S), axis=0)
            cur_gin[:, b] = np.sqrt(S[max_svd_idx]) * U[max_svd_idx, :]
            cur_gout[:, b] = np.sqrt(S[max_svd_idx]) * Vh[:, max_svd_idx]
            cur_recons_matrix = cur_gin[:, b] @ cur_gout[:, b].T
            A_recons[..., k, b] = cur_recons_matrix
            logger.info(
                f'Variance explained by first principal component: {S[max_svd_idx] / np.sum(S)}'
            )
            logger.info(
                f'Reconstruction error: {np.linalg.norm(cur_amp_matrix - cur_recons_matrix)}'
            )
        # fit filters to get the desired gains in octave bands
        for src_idx in range(room_data.num_src):
            g_in[src_idx, k] = fit_filters_to_gains(cur_gin[src_idx, :],
                                                    band_centre_hz,
                                                    fs=room_data.sample_rate)
        for rec_idx in range(room_data.num_rec):
            g_out[rec_idx, k] = fit_filters_to_gains(cur_gout[rec_idx, :],
                                                     band_centre_hz,
                                                     fs=room_data.sample_rate)

    return g_in, g_out


def get_source_receiver_gains(
        room_data: RoomDataset) -> Tuple[NDArray, NDArray]:
    """
    Do a rank-1 decomposition of the matrix of common slope amplitudes to get the source and
    receiver gains
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

    # are the RIRs in frequency bands?
    is_in_subbands = room_data.band_centre_hz is not None

    # get the source and receiver gains
    if is_in_subbands:
        g_in, g_out = get_source_receiver_filters(room_data)
    else:
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
        use_absorption_filters=is_in_subbands,
        common_decay_times=room_data.common_decay_times,
        colorless_fdn_params=colorless_fdn_params,
        input_filter_config=config_dict.output_filter_config,
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

            if is_in_subbands:
                model.input_filters = cur_gout
                model.output_filters = cur_gin
            else:
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
