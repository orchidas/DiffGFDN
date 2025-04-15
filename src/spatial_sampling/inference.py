from copy import deepcopy
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple

from loguru import logger
import numpy as np
from numpy.typing import NDArray
from slope2noise.generate import shaped_wgn
import spaudiopy as sp
import torch
from torch.utils.data import DataLoader

from diff_gfdn.utils import db2lin, ms_to_samps
from src.run_model import load_and_validate_config

from .config import BeamformerType, DNNType, SpatialSamplingConfig
from .dataloader import custom_collate_spatial_sampling, get_dataloader, SpatialRoomDataset, SpatialSamplingDataset
from .model import (
    Directional_Beamforming_Weights_from_CNN,
    Directional_Beamforming_Weights_from_MLP,
    Omni_Amplitudes_from_MLP,
)

# flake8: noqa:E231, E722, F841, W0612
# pylint: disable=W0702, E0606


def get_ambisonic_rirs(rec_pos_list: NDArray, output_pkl_path: str,
                       full_band_room_data: SpatialRoomDataset,
                       config_path: str) -> SpatialRoomDataset:
    """
    Get ambisonic / omni RIRs predicted by the neural net using the
    common slopes model
    Args:
        rec_pos_list (NDArray): positions at which to get RIR
        output_pkl_path (str): path where output pkl file is to be saved
        full_band_room_data (SpatialRoomDataset): the OG dataset
        config_path (str): path to te config files
    """

    freq_bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    config_paths = [
        Path(
            f'{config_path}/treble_data_grid_training_{freq:.0f}Hz_directional_spatial_sampling_test.yml'
        ).resolve() for freq in freq_bands
    ]
    config_dicts = [
        load_and_validate_config(config_path, SpatialSamplingConfig)
        for config_path in config_paths
    ]

    # copy the full room dataset
    cs_room_data = deepcopy(full_band_room_data)
    # update the positions in the dataset
    cs_room_data.update_receiver_pos(rec_pos_list)

    # get the spatial room impulse responses
    est_srirs, _ = get_soundfield_from_trained_model(
        config_dicts,
        cs_room_data,
        rec_pos_list,
    )

    # update the RIRs
    cs_room_data.update_rirs(est_srirs)

    # Save to a file
    logger.info("Saving to pkl file")
    with open(output_pkl_path, "wb") as f:
        pickle.dump(cs_room_data, f)

    return cs_room_data


def convert_directional_rirs_to_ambisonics(ambi_order: int,
                                           desired_directions: NDArray,
                                           beamformer_type: BeamformerType,
                                           directional_rirs: NDArray):
    """
    Convert directional RIRs into the ambisonics domain by passing through a synthesis Spherical filterbank
    Args:
        ambi_order (int): order of the ambisonics signal
        desired_directions (NDArray): 2 x num_directions array of directions on a uniform grid
        beamformer_type (BeamformerType): type of beamforming done - Butterworth, max-DI, or max-RE
        directional_rirs (NDArray): directional RIRs of shape num_directions x num_pos x num_time_samples
    Returns:
        NDArray: ambisonics RIRs of shape num_ambi_channels x num_pos x num_time_samples
    """
    # get the modal beamformer weights
    if beamformer_type == BeamformerType.MAX_DI:
        modal_weights = sp.sph.cardioid_modal_weights(ambi_order)
    elif beamformer_type == BeamformerType.MAX_RE:
        modal_weights = sp.sph.max_re_modal_weights(ambi_order)
    elif beamformer_type == BeamformerType.BUTTER:
        modal_weights = sp.sph.butterworth_modal_weights(ambi_order,
                                                         k=5,
                                                         n_c=3)
    else:
        raise NameError("Other types of beamformers not available")

    # size is num_ambi_channels x num_directions
    _, synthesis_matrix = sp.sph.design_sph_filterbank(
        ambi_order,
        desired_directions[0, :],
        desired_directions[1, :],
        modal_weights,
        mode='energy',
        sh_type='real')

    ambi_rirs = np.einsum('nj, jbt -> nbt', synthesis_matrix, directional_rirs)
    return ambi_rirs


def get_soundfield_from_trained_model(
    config_dicts: List[SpatialSamplingConfig],
    full_band_room_data: SpatialRoomDataset,
    rec_pos_list: NDArray,
) -> Tuple[NDArray, NDArray]:
    """
    For each frequency band, read the optimised model weights and generate
    the spherical harmonic weighting function for each position and group. Then generate
    the RIRs in the SH domain using the common slopes model.
    Args:
        config_dicts (List): list of config files, one for each frequency band
        full_band_room_data (SpatialRoomDataset): dataset containing parameters for all frequency bands
        rec_pos_list (List): list of receiver positions for which we want the SH weights
    Returns:
        NDArray, NDArray: the omni / ambisonics RIRs of shape num_channels x num_pos x ir_len and 
                         the learned amplitudes of shape num_pos x num_directions x  num_groups x num_bands
    """
    sample_rate = full_band_room_data.sample_rate
    freq_bands = full_band_room_data.band_centre_hz
    ir_len_samps = min(full_band_room_data.rir_length,
                       ms_to_samps(2000, sample_rate))
    num_slopes = full_band_room_data.num_rooms
    num_directions = full_band_room_data.num_directions
    ambi_order = full_band_room_data.ambi_order
    desired_directions = full_band_room_data.sph_directions

    num_pos = rec_pos_list.shape[0]
    num_bands = len(freq_bands)

    assert num_bands == len(
        config_dicts
    ) == 8, "number of config files should be same as the number of frequencies"

    # prepare the dataset
    dataset = SpatialSamplingDataset(config_dicts[0].device,
                                     full_band_room_data)

    learned_amplitudes = np.zeros(
        (num_pos, num_slopes, num_bands)) if ambi_order is None else np.zeros(
            (num_pos, num_directions, num_slopes, num_bands))
    decay_times = np.squeeze(full_band_room_data.common_decay_times)
    t_vals_expanded = np.repeat(np.array(decay_times.T)[np.newaxis, ...],
                                num_pos,
                                axis=0)

    dict_keys = [tuple(np.round(pos, 3)) for pos in rec_pos_list]

    for b_idx, config_dict in enumerate(config_dicts):
        logger.info(
            f"Getting DNN output for frequency = {freq_bands[b_idx]:.0f} Hz")

        config_dict = config_dict.model_copy(update={
            'use_directional_rirs':
            full_band_room_data.sph_directions is not None
        })

        dataloader = get_dataloader(
            dataset,
            batch_size=config_dict.batch_size,
            shuffle=False,
            device=config_dict.device,
            drop_last=False,
            custom_collate_fn=lambda batch: custom_collate_spatial_sampling(
                batch, config_dict.network_type, dataset),
        )

        # get the output of the DNN
        # dictionary with rec_pos_list as keys
        learned_amplitudes_cur_band = get_output_from_trained_model(
            config_dict, desired_directions, rec_pos_list,
            full_band_room_data.grid_spacing_m, dataloader, num_slopes,
            ambi_order)

        # list of (num_groups, num_directions)
        amp_values = [learned_amplitudes_cur_band[key] for key in dict_keys]

        # shape: (num_pos, num_groups, num_directions)
        learned_amplitudes[..., b_idx] = np.stack(
            [v.detach().numpy() for v in amp_values], axis=0).T

    if ambi_order is not None:
        directional_rirs = np.zeros((num_directions, num_pos, ir_len_samps))
        for n in range(num_directions):
            logger.info(f"Getting shaped noise output for direction {n}")
            _, directional_rirs[n, ...] = shaped_wgn(
                t_vals_expanded,
                learned_amplitudes[:, n, ...],
                sample_rate,
                ir_len_samps,
                freq_bands,
            )
        # convert to ambisonic RIRs
        logger.info("Converting directional RIRs into the SH domain")
        ambi_rirs = convert_directional_rirs_to_ambisonics(
            ambi_order, desired_directions,
            config_dicts[0].dnn_config.beamformer_type, directional_rirs)

        return ambi_rirs, learned_amplitudes
    else:
        _, omni_rirs = shaped_wgn(
            t_vals_expanded,
            learned_amplitudes,
            sample_rate,
            ir_len_samps,
            freq_bands,
        )
        return omni_rirs, learned_amplitudes


def get_output_from_trained_model(config_dict: SpatialSamplingConfig,
                                  rec_pos_list: List,
                                  desired_directions: NDArray,
                                  grid_resolution_m: float,
                                  dataloader: DataLoader,
                                  num_rooms: int,
                                  ambi_order: Optional[int] = None) -> Dict:
    """
    Get the learned beamforming weights for a particular frequency band
    Args:
        config_dict (SpatialSamplingConfig): config parameters
        desired_directions (NDArray):2 x num_directions containing az and polar angles 
                                     for directions in which we want to estimate the amplitudes.
        rec_pos_list (List): list of receiver positions for which to get the amplitudes
        grid_resolution_m (float): resolution of the grid on which training was done
        dataloader (Dataloader): the dataset for which we want the amplitudes
        num_rooms (int): number of rooms in the dataset
        ambi_order (int): ambisonics order
    Returns:
        Dict: dictionary of amplitudes, where the keys are receiver positions, and each
              key contains num_direction x num_rooms amplitudes
    """
    # are we learning OMNI amplitudes or directional amplitudes?
    if config_dict.use_directional_rirs:

        if config_dict.network_type == DNNType.MLP:
            logger.info("Using MLP")
            model = Directional_Beamforming_Weights_from_MLP(
                num_rooms,
                ambi_order,
                config_dict.dnn_config.num_fourier_features,
                config_dict.dnn_config.mlp_config.num_hidden_layers,
                config_dict.dnn_config.mlp_config.num_neurons_per_layer,
                desired_directions=desired_directions,
                beamformer_type=config_dict.dnn_config.beamformer_type,
                device=config_dict.device,
            )
        elif config_dict.network_type == DNNType.CNN:
            logger.info("Using CNN")
            model = Directional_Beamforming_Weights_from_CNN(
                num_rooms,
                ambi_order,
                config_dict.dnn_config.num_fourier_features,
                config_dict.dnn_config.cnn_config.num_hidden_channels,
                config_dict.dnn_config.cnn_config.num_layers,
                config_dict.dnn_config.cnn_config.kernel_size,
                desired_directions=desired_directions,
                beamformer_type=config_dict.dnn_config.beamformer_type,
                device=config_dict.device,
            )

    else:
        model = Omni_Amplitudes_from_MLP(
            num_rooms,
            config_dict.dnn_config.num_fourier_features,
            config_dict.dnn_config.mlp_config.num_hidden_layers,
            config_dict.dnn_config.mlp_config.num_neurons_per_layer,
            device=config_dict.device,
            gain_limits=(db2lin(-100), db2lin(0)),
        )

    # load the trained weights for the particular epoch
    max_epochs = config_dict.max_epochs
    checkpoint_found = False
    while not checkpoint_found:
        try:
            checkpoint_dir = Path(
                config_dict.train_dir +
                f'checkpoints/grid_resolution={grid_resolution_m:.1f}'
            ).resolve()
            checkpoint = torch.load(
                f'{checkpoint_dir}/model_e{max_epochs - 1}.pt',
                weights_only=True,
                map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint)

            checkpoint_found = True
            logger.debug(f'Checkpoint found for epoch = {max_epochs}')
            break
        except:
            max_epochs -= 1

    # Load the trained model state
    # in eval mode, no gradients are calculated
    model.eval()
    all_amplitudes = {tuple(np.round(pos, 3)): None for pos in rec_pos_list}

    for data in dataloader:
        position = data['listener_position'].detach().numpy()
        # this is in the SH domain
        dnn_output = model(data)
        # directional amplitudes - size is num_pos x num_directions x num_slopes
        beamformer_output = model.get_directional_amplitudes()
        for num_pos in range(position.shape[0]):
            # collate all RIRs at all positions
            all_amplitudes[tuple(np.round(position[num_pos],
                                          3))] = beamformer_output[num_pos,
                                                                   ...]

    return all_amplitudes
