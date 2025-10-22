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
from tqdm import tqdm

from diff_gfdn.config.config_loader import load_and_validate_config
from diff_gfdn.utils import db2lin, ms_to_samps

from .config import BeamformerType, DNNType, SpatialSamplingConfig
from .dataloader import custom_collate_spatial_sampling, get_dataloader, SpatialRoomDataset, SpatialSamplingDataset
from .model import (
    Directional_Beamforming_Weights_from_CNN,
    Directional_Beamforming_Weights_from_MLP,
    Omni_Amplitudes_from_MLP,
)

# flake8: noqa:E231, E722, F841, W0612
# pylint: disable=W0702, E0606, W0640, W0612. W0707


def get_ambisonic_rirs(
    rec_pos_list: NDArray,
    full_band_room_data: SpatialRoomDataset,
    use_trained_model: bool = True,
    config_path: Optional[str] = None,
    grid_resolution_m: Optional[float] = None,
    output_pkl_path: Optional[str] = None,
    apply_spatial_bandlimiting: bool = False,
) -> SpatialRoomDataset:
    """
    Get ambisonic / omni RIRs predicted by the neural net using the
    common slopes model
    Args:
        rec_pos_list (NDArray): positions at which to get RIR
        output_pkl_path (str): path where output pkl file is to be saved
        full_band_room_data (SpatialRoomDataset): the OG dataset
        config_path (str, optional): path to te config files
        use_trained_model (bool): whether to use trained model, or amplitudes from the dataset
        grid_resolution_m (float, optional): what grid resolution did we use to train the MLP?
        apply_spatial_bandlimiting (bool): if true, spatial bandlimiting is applied to the 
                                           synthesised SRIRs
    """

    freq_bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]

    if use_trained_model:
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

    ir_len_samps = min(full_band_room_data.rir_length,
                       ms_to_samps(2000, cs_room_data.sample_rate))

    # get the spatial room impulse responses
    if use_trained_model:
        logger.info("Using trained model")
        # update the positions in the dataset
        cs_room_data.update_receiver_pos(rec_pos_list)
        est_srirs, _ = get_soundfield_from_trained_model(
            config_dicts,
            cs_room_data,
            rec_pos_list,
            ir_len_samps,
            grid_resolution_m,
            apply_spatial_bandlimiting=apply_spatial_bandlimiting)
    else:
        logger.info("Using common slope amplitudes from dataset")
        # find the closest positions from rec_pos_list in the dataset,
        # and find the RIRs for those positions only
        distances = np.linalg.norm(cs_room_data.receiver_position[:, None, :] -
                                   rec_pos_list,
                                   axis=2)
        indices = np.argmin(distances, axis=0)
        # make sure the shape is num_pos, num_directions, num_slopes, num_bands
        cs_amps = cs_room_data.amplitudes[indices, ...].transpose(0, 2, 1, -1)
        est_srirs = get_rirs_from_common_slopes_model(
            cs_room_data.sample_rate,
            rec_pos_list,
            freq_bands,
            ir_len_samps,
            cs_amps,
            np.squeeze(cs_room_data.common_decay_times),
            cs_room_data.ambi_order,
            cs_room_data.sph_directions,
            beamformer_type=BeamformerType.MAX_DI,
            apply_spatial_bandlimiting=apply_spatial_bandlimiting,
        )
        cs_room_data.update_receiver_pos(rec_pos_list)

    # update the RIRs
    cs_room_data.update_rirs(est_srirs)

    # Save to a file
    if output_pkl_path is not None:
        logger.info("Saving to pkl file")
        with open(output_pkl_path, "wb") as f:
            pickle.dump(cs_room_data, f)

    return cs_room_data


def spatial_bandlimiting(ambi_order: int, des_dir: NDArray, drirs: NDArray,
                         modal_weights: NDArray):
    """Ensure spatial band limitation of directional RIRs - see Holdt et al"""
    sh_matrix = sp.sph.sh_matrix(ambi_order, des_dir[0, :],
                                 np.pi / 2 - des_dir[1, :])
    # size is num_directions x num_directions
    spatial_cov_matrix = sh_matrix @ np.diag(
        sp.sph.repeat_per_order(modal_weights)) @ sh_matrix.T

    # sum over any one dimension since the matrix is symmetric
    norm_factor = spatial_cov_matrix / np.sum(
        spatial_cov_matrix, axis=1, keepdims=True)
    bandlimited_drirs = np.einsum('jk, kbt -> jbt', norm_factor, drirs)
    return bandlimited_drirs


def convert_directional_rirs_to_ambisonics(
        ambi_order: int,
        desired_directions: NDArray,
        beamformer_type: BeamformerType,
        directional_rirs: NDArray,
        apply_spatial_bandlimiting: bool = False):
    """
    Convert directional RIRs into the ambisonics domain by passing through a synthesis Spherical filterbank
    Args:
        ambi_order (int): order of the ambisonics signal
        desired_directions (NDArray): 2 x num_directions array of directions on a uniform grid
        beamformer_type (BeamformerType): type of beamforming done - Butterworth, max-DI, or max-RE
        directional_rirs (NDArray): directional RIRs of shape num_directions x num_pos x num_time_samples
    Returns:
        NDArray: ambisonics RIRs of shape num_pos x num_ambi_channels x num_time_samples 
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

    if apply_spatial_bandlimiting:
        bandlimited_directional_rirs = spatial_bandlimiting(
            ambi_order, desired_directions, directional_rirs, modal_weights)
    else:
        bandlimited_directional_rirs = directional_rirs.copy()

    # size is num_ambi_channels x num_directions
    _, synthesis_matrix = sp.sph.design_sph_filterbank(
        ambi_order,
        desired_directions[0, :],
        np.pi / 2 - desired_directions[1, :],
        modal_weights,
        mode='energy',
        sh_type='real')

    ambi_rirs = np.einsum('nj, jbt -> nbt', synthesis_matrix,
                          bandlimited_directional_rirs)
    return ambi_rirs.transpose(1, 0, -1)


def get_rirs_from_common_slopes_model(
    sample_rate: float,
    rec_pos_list: NDArray,
    freq_bands: List,
    ir_len_samps: int,
    amplitudes: NDArray,
    common_decay_times: List,
    ambi_order: Optional[int] = None,
    des_directions: Optional[NDArray] = None,
    beamformer_type: Optional[BeamformerType] = None,
    batch_size: int = 100,
    apply_spatial_bandlimiting: bool = False,
) -> NDArray:
    """
    Use shaped Gaussian noise to return directional / omni RIRs using the common slopes model
    Args:
        amplitudes (NDArray): common slope amps of size num_pos, num_slopes, num_bands / 
                             num_pos, num_directions, num_slopes, num_bands
    Returns:
        NDArray: directional / omni RIRs of shape num_directions x num_pos x ir_len_sampes/
                 num_pos x ir_len_samps
    """
    num_pos = rec_pos_list.shape[0]
    num_directions = des_directions.shape[-1]
    run_in_batches = num_pos > batch_size
    num_batches = int(np.ceil(num_pos / batch_size))

    t_vals_expanded = np.repeat(np.array(common_decay_times.T)[np.newaxis,
                                                               ...],
                                num_pos,
                                axis=0)

    logger.info(f"Running in batches? {run_in_batches}")

    if ambi_order is not None:
        directional_rirs = np.zeros((num_directions, num_pos, ir_len_samps))
        for n in range(num_directions):
            logger.info(f"Getting shaped noise output for direction {n}")
            if run_in_batches:
                for batch_idx in tqdm(range(num_batches)):
                    cur_idx = slice(batch_idx * batch_size,
                                    min((batch_idx + 1) * batch_size, num_pos))
                    _, directional_rirs[n, cur_idx, :] = shaped_wgn(
                        t_vals_expanded[cur_idx, ...],
                        amplitudes[cur_idx, n, ...],
                        sample_rate,
                        ir_len_samps,
                        f_bands=freq_bands,
                    )
            else:
                _, directional_rirs[n, ...] = shaped_wgn(
                    t_vals_expanded,
                    amplitudes[:, n, ...],
                    sample_rate,
                    ir_len_samps,
                    f_bands=freq_bands,
                )
        # convert to ambisonic RIRs
        logger.info("Converting directional RIRs into the SH domain")
        ambi_rirs = convert_directional_rirs_to_ambisonics(
            ambi_order,
            des_directions,
            beamformer_type,
            directional_rirs,
            apply_spatial_bandlimiting=apply_spatial_bandlimiting)

        return ambi_rirs
    else:
        _, omni_rirs = shaped_wgn(
            t_vals_expanded,
            amplitudes,
            sample_rate,
            ir_len_samps,
            f_bands=freq_bands,
        )
        return omni_rirs


def get_soundfield_from_trained_model(
        config_dicts: List[SpatialSamplingConfig],
        full_band_room_data: SpatialRoomDataset,
        rec_pos_list: NDArray,
        ir_len_samps: int,
        grid_resolution_m: float,
        apply_spatial_bandlimiting: bool = False) -> Tuple[NDArray, NDArray]:
    """
    For each frequency band, read the optimised model weights and generate
    the spherical harmonic weighting function for each position and group. Then generate
    the directional RIRs using the common slopes model. Convert the directional RIRs to 
    ambisonics RIRs after doing spatial bandlimiting and beamforming,
    Args:
        config_dicts (List): list of config files, one for each frequency band
        full_band_room_data (SpatialRoomDataset): dataset containing parameters for all frequency bands
        rec_pos_list (List): list of receiver positions for which we want the SH weights
        ir_len_samps (int): length of the desired RIRs in samples
        grid_resolution_m (float): for what grid resolution do we want to do the load the models for
                                    inferencing?
    Returns:
        NDArray, NDArray: the omni / ambisonics RIRs of shape num_pos x num_ambi_channels x ir_len and 
                         the learned amplitudes of shape num_pos x num_directions x  num_groups x num_bands
    """
    sample_rate = full_band_room_data.sample_rate
    freq_bands = full_band_room_data.band_centre_hz

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
            config_dict, rec_pos_list, desired_directions, grid_resolution_m,
            dataloader, num_slopes, ambi_order)

        # list of (num_groups, num_directions)
        amp_values = [learned_amplitudes_cur_band[key] for key in dict_keys]

        # shape: (num_pos, num_groups, num_directions)
        learned_amplitudes[..., b_idx] = np.stack(
            [v.detach().numpy() for v in amp_values], axis=0)

    rirs = get_rirs_from_common_slopes_model(
        sample_rate,
        rec_pos_list,
        freq_bands,
        ir_len_samps,
        learned_amplitudes,
        decay_times,
        ambi_order,
        desired_directions,
        config_dicts[0].dnn_config.beamformer_type,
        apply_spatial_bandlimiting=apply_spatial_bandlimiting)

    return rirs, learned_amplitudes


def get_output_from_trained_model(config_dict: SpatialSamplingConfig,
                                  rec_pos_list: List,
                                  desired_directions: NDArray,
                                  grid_resolution_m: float,
                                  dataloader: DataLoader,
                                  num_rooms: int,
                                  ambi_order: Optional[int] = None) -> Dict:
    """
    Get the learned directional amplitudes for a particular frequency band
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
            if max_epochs < 0:
                raise FileNotFoundError('Trained model does not exist!')

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
