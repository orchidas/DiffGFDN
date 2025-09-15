from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
import pyfar as pf
from scipy.signal import fftconvolve
from slope2noise.rooms import RoomGeometry
from slope2noise.utils import decay_kernel, schroeder_backward_int
import torch
from torch import nn
from tqdm import tqdm

from spatial_sampling.config import DNNType
from spatial_sampling.dataloader import load_dataset as load_spatial_dataset
from spatial_sampling.dataloader import parse_three_room_data, SpatialRoomDataset
from spatial_sampling.inference import convert_directional_rirs_to_ambisonics

from .colorless_fdn.utils import get_colorless_fdn_params
from .config.config import DiffGFDNConfig, SubbandProcessingConfig
from .dataloader import load_dataset, RoomDataset
from .model import DiffDirectionalFDNVarReceiverPos
from .plot import order_position_matrices, plot_edr
from .utils import db, db2lin, get_response, ms_to_samps

# flake8: noqa:E231
# pylint: disable=W0632, E0606


@dataclass
class DiffGFDNParams:
    output_gains: List
    input_gains: List
    input_scalars: List
    coupled_feedback_matrix: List
    coupling_matrix: List
    output_scalars: Optional[List] = None
    output_biquad_coeffs: Optional[List] = None
    output_sh_gains: Optional[List] = None


class InferDiffGFDN:
    """Class for infering DiffGFDN parameters"""

    def __init__(self,
                 room_data: RoomDataset,
                 config_dict: DiffGFDNConfig,
                 model: nn.Module,
                 use_direct_cs_params: bool = False):
        """
        Initialise inference class
        Args:
            room_data (RoomDataset): room dataset for inference
            config_dict (DiffGFDNConfig): config parameters
            model (DiffGFDNVarReceiverPos) : Differentiable GFDN model (trained)
            use_direct_cs_params (bool): whether to use CS amplitudes directly as output scalars,
                                         instead of learning them with an MLP
        """
        self.room_data = room_data
        self.config_dict = config_dict
        self.trainer_config = config_dict.trainer_config
        self.model = model
        self.checkpoint_dir = Path(self.trainer_config.train_dir +
                                   'checkpoints/').resolve()
        self.max_epochs = self.trainer_config.max_epochs

        output_gains = []
        input_gains = []
        input_scalars = []
        output_scalars = []
        output_biquad_coeffs = []
        coupled_feedback_matrix = []
        coupling_matrix = []

        self.all_learned_params = DiffGFDNParams(
            output_gains, input_gains, input_scalars, coupled_feedback_matrix,
            coupling_matrix, output_scalars, output_biquad_coeffs)

        self.all_pos = [
            np.empty((self.room_data.num_rec, 3))
            for i in range(-1, self.max_epochs)
        ]
        self.all_rirs = [
            np.empty((self.room_data.num_rec, self.room_data.num_freq_bins))
            for i in range(-1, self.max_epochs)
        ]
        self.all_output_scalars = [
            np.empty((self.room_data.num_rec, self.room_data.num_rooms))
            for i in range(-1, self.max_epochs)
        ]
        self.h_approx_list = []
        self.use_direct_cs_params = use_direct_cs_params

        # prepare the dataset
        self.train_dataset, _ = load_dataset(
            self.room_data,
            self.trainer_config.device,
            train_valid_split_ratio=1.0,
            batch_size=self.trainer_config.batch_size,
            shuffle=False)

        # get normalising factor to compensate for subband filtering
        if self.trainer_config.subband_process_config is not None:
            self.subband_filter_norm_factor = self.get_norm_factor(
                self.config_dict.trainer_config.subband_process_config,
                self.room_data.sample_rate)

    @staticmethod
    def find_listener_pos_in_room_data(list_pos: NDArray,
                                       room_data: RoomDataset) -> ArrayLike:
        """Return the indices of list_pos found in room_data.receiver_position"""
        index = np.full(len(list_pos), -1,
                        dtype=np.int32)  # Default to -1 for non-matches

        for i, pos in enumerate(list_pos):
            match = np.where(
                (room_data.receiver_position == pos).all(axis=-1))[0]
            if match.size > 0:
                index[i] = match[0]  # Take the first match if multiple exist
        return index

    @staticmethod
    def get_norm_factor(subband_process_config: SubbandProcessingConfig,
                        fs: float):
        """Normalise the RIR by the filter's energy"""
        subband_filters, subband_freqs = pf.dsp.filter.reconstructing_fractional_octave_bands(
            None,
            num_fractions=subband_process_config.num_fraction_octaves,
            frequency_range=subband_process_config.frequency_range,
            sampling_rate=fs,
        )
        subband_filter_idx = np.argmin(
            np.abs(subband_freqs - subband_process_config.centre_frequency))
        norm_factor = np.sqrt(
            np.sum(
                np.power(subband_filters.coefficients[subband_filter_idx, :],
                         2)))
        return norm_factor

    def get_model_output(self,
                         epoch_list: List[int],
                         desired_filename: str,
                         plot: bool = False,
                         h_true: Optional[NDArray] = None,
                         config_name: Optional[str] = None,
                         fig_path: Optional[str] = None) -> Tuple:
        """
        Get model output for multiple epochs in epoch_list
        Args:
            epoch_list (List): epoch numbers to iterate over
            desired_filename (str): if investigating a single RIR, the name of the RIR
            plot (bool): whether to plot the EDR
            h_true (NDArray): the ground truth RIR
            config_name (str): name of the config file (needed for saving plots)
            fig_path (str): where to save the plots
        Returns:
            Tuple: the single RIR under investigation over multiple epochs, 
                   all positions and all RIRs over all epochs, learned DiffGFDN 
                   parameters over all epochs.
        """
        for epoch in epoch_list:
            # load the trained weights for the particular epoch
            checkpoint = torch.load(f'{self.checkpoint_dir}/model_e{epoch}.pt',
                                    weights_only=True,
                                    map_location=torch.device('cpu'))
            # Load the trained model state
            self.model.load_state_dict(checkpoint, strict=False)
            # in eval mode, no gradients are calculated
            self.model.eval()
            npos = 0

            with torch.no_grad():
                param_dict = self.model.get_param_dict()
                self.all_learned_params.input_gains.append(
                    param_dict['input_gains'])
                self.all_learned_params.input_scalars.append(
                    param_dict['input_scalars'])
                self.all_learned_params.output_gains.append(
                    param_dict['output_gains'])
                self.all_learned_params.coupled_feedback_matrix.append(
                    param_dict['coupled_feedback_matrix'])
                self.all_learned_params.coupling_matrix.append(
                    param_dict['coupling_matrix'])

                for data in self.train_dataset:
                    position = data['listener_position']

                    if self.trainer_config.subband_process_config is not None and self.use_direct_cs_params:
                        # try using CS amps as receiver gains
                        pos_idxs = self.find_listener_pos_in_room_data(
                            position, self.room_data)
                        cs_output_scalars = np.sqrt(
                            self.room_data.amplitudes[pos_idxs, :])

                        if self.trainer_config.use_colorless_loss:
                            _, _, h = get_response(
                                data, self.model,
                                torch.tensor(cs_output_scalars))
                        else:
                            _, h = get_response(
                                data, self.model,
                                torch.tensor(cs_output_scalars))
                    else:
                        if self.trainer_config.use_colorless_loss:
                            _, _, h = get_response(data, self.model)
                        else:
                            _, h = get_response(data, self.model)

                    # this needs to be added to compensate for subband filter energy
                    if self.trainer_config.subband_process_config is not None:
                        h *= self.subband_filter_norm_factor

                    # get parameter dictionary used in inferencing
                    inf_param_dict = self.model.get_param_dict_inference(data)

                    for num_pos in range(position.shape[0]):
                        self.all_pos[epoch + 1][npos, :] = position[num_pos]
                        self.all_rirs[epoch + 1][npos, :] = h[num_pos, :]
                        if 'output_scalars' in inf_param_dict.keys():
                            self.all_output_scalars[epoch + 1][
                                npos, :] = deepcopy(
                                    inf_param_dict['output_scalars'][num_pos])
                        npos += 1
                        filename = f'ir_({position[num_pos,0]:.2f}, {position[num_pos, 1]:.2f},' \
                        + f' {position[num_pos, 2]:.2f}).wav'

                        if filename == desired_filename:

                            # get the ir at this position
                            self.h_approx_list.append(h[num_pos, :])

                            # get the gains for this position
                            if 'output_scalars' in inf_param_dict.keys():
                                self.all_learned_params.output_scalars.append(
                                    deepcopy(inf_param_dict['output_scalars']
                                             [num_pos]))
                            elif 'output_biquad_coeffs' in inf_param_dict.keys(
                            ):
                                self.all_learned_params.output_biquad_coeffs.append(
                                    deepcopy(
                                        inf_param_dict['output_biquad_coeffs']
                                        [num_pos]))

                            if plot:
                                # plot the EDRs of the true and estimated RIRs
                                plot_edr(
                                    torch.tensor(h_true),
                                    self.model.sample_rate,
                                    title=f'True RIR EDR, epoch={epoch}',
                                    save_path=
                                    f'{fig_path}/true_edr_{filename}_{config_name}_epoch={epoch}.png'
                                )

                                plot_edr(
                                    h[num_pos, :],
                                    self.model.sample_rate,
                                    title=f'Estimated RIR EDR, epoch={epoch}',
                                    save_path=
                                    f'{fig_path}/approx_edr_{filename}_{config_name}_epoch={epoch}.png'
                                )

        return (self.h_approx_list, self.all_pos, self.all_rirs,
                self.all_output_scalars, self.all_learned_params)


############################################################################


class InferDiffDirectionalFDN:
    """Class for making plots for the Differentiable Directional FDN"""

    def __init__(
        self,
        room_data: SpatialRoomDataset,
        config_dict: DiffGFDNConfig,
        model: nn.Module,
        apply_filter_norm: bool = False,
        edc_len_ms: Optional[float] = None,
    ):
        """
        Initialise parameters for the class
        Args:
            room_data (SpatialRoomDataset): object of SpatialRoomDataset dataclass
            config_dict (DiffGFDNConfig): config file, read as dictionary
            model (DiffDirectionalFDNVarReceiverPos): the NN model to be tested
            apply_filter_norm (bool): whether to apply the normalising factor for subband filtering
            edc_len_ms (float): length of the RIR to be generated in ms
        """
        self.room_data = deepcopy(room_data)
        self.model = deepcopy(model)
        self.config_dict = deepcopy(config_dict)
        self.room = RoomGeometry(room_data.sample_rate,
                                 room_data.num_rooms,
                                 np.array(room_data.room_dims),
                                 np.array(room_data.room_start_coord),
                                 aperture_coords=room_data.aperture_coords)
        self.num_ambi_channels = (room_data.ambi_order + 1)**2
        self.apply_filter_norm = apply_filter_norm

        # prepare the training and validation data
        self.train_dataset, self.valid_dataset, _ = load_spatial_dataset(
            room_data,
            config_dict.trainer_config.device,
            network_type=DNNType.MLP,
            batch_size=config_dict.trainer_config.batch_size,
            grid_resolution_m=config_dict.trainer_config.grid_resolution_m,
            shuffle=False,
        )

        # get the reference output
        self.src_pos = np.array(self.room_data.source_position).squeeze()
        self.true_points = torch.tensor(self.room_data.receiver_position,
                                        dtype=torch.float32)
        self.true_amps = torch.tensor(self.room_data.amplitudes,
                                      dtype=torch.float32)
        self.mixing_time_samps = ms_to_samps(self.room_data.mixing_time_ms,
                                             self.room_data.sample_rate)

        output_gains = []
        input_gains = []
        input_scalars = []
        output_sh_gains = []
        coupled_feedback_matrix = []
        coupling_matrix = []

        self.all_learned_params = DiffGFDNParams(
            output_gains,
            input_gains,
            input_scalars,
            coupled_feedback_matrix,
            coupling_matrix,
            output_sh_gains=output_sh_gains)

        self.all_output_sh_gains = [
            np.empty((self.room_data.num_rec, self.room_data.num_rooms,
                      (self.room_data.ambi_order + 1)**2))
            for i in range(-1, self.config_dict.trainer_config.max_epochs)
        ]

        # get normalising factor to compensate for subband filtering
        if self.apply_filter_norm:
            logger.info("Applying filter gain normalisation")
            if self.config_dict.trainer_config.subband_process_config is not None:
                self.subband_filter_norm_factor = InferDiffGFDN.get_norm_factor(
                    self.config_dict.trainer_config.subband_process_config,
                    self.room_data.sample_rate)
        self._init_decay_kernel(edc_len_ms)

    def _init_decay_kernel(self, edc_len_ms: Optional[float] = None):
        """Initialise the decay kernels for calculating EDC errors"""
        num_slopes = self.room_data.num_rooms
        if edc_len_ms is None:
            edc_len_ms = self.model.common_decay_times.max() * 1e3
        self.edc_len_samps = ms_to_samps(edc_len_ms,
                                         self.room_data.sample_rate)
        self.envelopes = np.zeros((num_slopes, self.edc_len_samps))
        time_axis = np.linspace(0, (self.edc_len_samps - 1) /
                                self.room_data.sample_rate, self.edc_len_samps)

        for k in range(num_slopes):
            self.envelopes[k, :] = decay_kernel(np.expand_dims(
                self.room_data.common_decay_times[:, k], axis=-1),
                                                time_axis,
                                                self.room_data.sample_rate,
                                                normalize_envelope=True,
                                                add_noise=False).squeeze()

        self.envelopes = torch.tensor(self.envelopes, dtype=torch.float32)

    def convert_ambi_rir_to_directional_rir(self, h_sh: torch.Tensor):
        """
        Convert SH domain RIRs to directional RIRs
        Args:
            H_sh : impulse response DiffDFDN in SH domain of shape num_pos, num_ambi_channels, num_freq_pts
        Returns:
            torch.Tensor: directional impulse response of DiffDFDN of shape 
                          num_pos, num_directions, num_freq_pts
        """
        # get SH conversion matrix
        sh_matrix = self.model.sh_output_scalars.analysis_matrix
        # convert to directional response
        h_dir = torch.einsum('blk, lj -> bjk', h_sh, sh_matrix.T)
        return h_dir

    def get_model_output(
            self,
            num_epochs: int,
            return_directional_rirs: bool = True) -> Tuple[NDArray, NDArray]:
        """
        Get the estimated common slope amplitudes.
        Returns the positions and the directional RIRs at those positions
        """
        # load the trained weights for the particular epoch
        checkpoint_found = False
        while not checkpoint_found:
            try:
                checkpoint = torch.load(Path(
                    f'{self.config_dict.trainer_config.train_dir}/checkpoints/'
                    + f'model_e{num_epochs - 1}.pt').resolve(),
                                        weights_only=True,
                                        map_location=torch.device('cpu'))
                # Load the trained model state
                self.model.load_state_dict(checkpoint, strict=False)
                checkpoint_found = True
                logger.debug(f'Checkpoint found for epoch = {num_epochs}')
                break
            except FileNotFoundError as exc:
                num_epochs -= 1
                if num_epochs < 0:
                    raise FileNotFoundError(
                        'Trained model does not exist!') from exc

        # run the model in eval mode
        self.model.eval()

        est_pos = torch.empty((0, 3))
        est_ambi_rirs = torch.empty(
            (0, self.num_ambi_channels, self.room_data.num_freq_bins))
        with torch.no_grad():
            param_dict = self.model.get_param_dict()
            self.all_learned_params.input_gains.append(
                param_dict['input_gains'])
            self.all_learned_params.input_scalars.append(
                param_dict['input_scalars'])
            self.all_learned_params.output_gains.append(
                param_dict['output_gains'])
            self.all_learned_params.coupled_feedback_matrix.append(
                param_dict['coupled_feedback_matrix'])
            self.all_learned_params.coupling_matrix.append(
                param_dict['coupling_matrix'])
            npos = 0

            for data in tqdm(self.valid_dataset):
                position = data['listener_position']

                # get parameter dictionary used in inferencing
                inf_param_dict = self.model.get_param_dict_inference(data)
                for num_pos in range(position.shape[0]):
                    if 'output_scalars' in inf_param_dict.keys():
                        self.all_output_sh_gains[num_epochs][
                            npos, :] = deepcopy(
                                inf_param_dict['output_scalars'][num_pos])
                    npos += 1

                # get RIRs
                if self.config_dict.trainer_config.use_colorless_loss:
                    _, _, cur_ambi_rir = get_response(data, self.model)
                else:
                    _, cur_ambi_rir = get_response(data, self.model)
                est_pos = torch.vstack((est_pos, position))
                est_ambi_rirs = torch.vstack((est_ambi_rirs, cur_ambi_rir))

            if self.apply_filter_norm:
                # needed for subband filtering
                est_ambi_rirs *= self.subband_filter_norm_factor

            # convert from ambisonics to directional RIRs
            if return_directional_rirs:
                est_dir_rirs = self.convert_ambi_rir_to_directional_rir(
                    est_ambi_rirs)
                return est_pos, est_dir_rirs[..., :self.edc_len_samps]
            else:
                return est_pos, est_ambi_rirs[..., :self.edc_len_samps]

    def plot_edc_error_in_space(self,
                                est_dir_rirs: NDArray,
                                est_points: NDArray,
                                epoch_num: int,
                                idx_in_valid_set: Optional[ArrayLike] = None):
        """
        Plot the error between the CS EDC and MLP EDC in space
        Args:
            grid_resolution_m (float): resolution of the uniform grid used for training
            est_dir_rirs (NDArray): estimated directional RIRs  
                                NN (num_pos, num_directions, num_time_samples)
            est_points (NDArray): receiver positions at which the amplitudes were estimatied
            epoch_num (int): epoch number
            idx_in_valid_set (ArrayLike, optional): the indices of the receiver positions in the valid set,
                                                    if None, all receiver positions are plotted
        """
        logger.info("Making EDC error plots")

        # order the position indices in the estimated data according to the
        # reference dataset
        if idx_in_valid_set is None:
            idx_in_valid_set = np.arange(0,
                                         self.room_data.num_rec,
                                         dtype=np.int32)
            extend = ''
        else:
            extend = '_valid_set'

        # returns idx in est_points that are closest to true_points[idx_in_valid_set]
        ordered_pos_idx = order_position_matrices(
            self.true_points[idx_in_valid_set], est_points)

        est_edc = db(schroeder_backward_int(
            est_dir_rirs[ordered_pos_idx, :, :self.envelopes.shape[-1]]),
                     is_squared=True)
        original_edc = db(torch.einsum('bjk, kt -> bjt',
                                       self.true_amps[idx_in_valid_set],
                                       self.envelopes),
                          is_squared=True)

        error_db = torch.mean(torch.abs(original_edc - est_edc), dim=-1)

        logger.info(f'Mean EDC error in dB is {error_db.mean():.3f} dB')
        to_append = f'grid_resolution={self.config_dict.trainer_config.grid_resolution_m}m' \
                    if self.config_dict.trainer_config.grid_resolution_m is not None else \
                    f'split_ratio={np.round(self.config_dict.trainer_config.train_valid_split, 1)}'

        for j in range(self.room_data.num_directions):
            save_dir = Path(
                f'{self.config_dict.trainer_config.train_dir}/direction={j+1}'
            ).resolve()
            save_dir.mkdir(parents=True, exist_ok=True)

            self.room.plot_edc_error_at_receiver_points(
                self.true_points[idx_in_valid_set],
                self.src_pos,
                db2lin(error_db[:, j]),
                scatter_plot=True,
                cur_freq_hz=None,
                save_path=f'{save_dir}/edc_error_in_space_' + to_append +
                extend + f'_epoch={epoch_num}.png',
                # title=
                # f'az = {np.degrees(self.room_data.sph_directions[0, j]):.2f} deg,'
                # +
                # f' pol = {np.degrees(self.room_data.sph_directions[1, j]):.2f} deg'
            )

        return original_edc.detach().cpu().numpy(), est_edc


#######################################################################################


def sum_arrays(group):
    "Sum rows sharing the same position coordinates"
    all_rirs = group["filtered_time_samples"].to_numpy()
    out = np.zeros_like(all_rirs[0])
    for rir in all_rirs:
        out += rir
    return out


def infer_all_octave_bands_directional_fdn(
    freqs_list: List,
    config_dicts: List[DiffGFDNConfig],
    save_dir: str,
    fullband_room_data: SpatialRoomDataset,
    rec_pos_list: NDArray,
) -> SpatialRoomDataset:
    """
    Run inference on all trained DiffDirectionalFDNs operating in all octave bands and save it in a dataframe
    Args:
        freqs_list (List): list of all frequencies
        config_dicts (List): list of all config files
        save_dir (str): path where file is to be saved
        fullband_room_dataset_path (SpatialRoomDataset): dataset of ground truth fullband RIRs
        rec_pos_list (NDArray): receiver positions over which to carry out inference
    Returns:
        SpatialRoomDataset: room data with synthesised RIRs 
    """

    # prepare the reconstructing filterbank
    subband_filters, _ = pf.dsp.filter.reconstructing_fractional_octave_bands(
        None,
        num_fractions=config_dicts[0].trainer_config.subband_process_config.
        num_fraction_octaves,
        frequency_range=(freqs_list[0], freqs_list[-1]),
        sampling_rate=config_dicts[0].sample_rate,
    )

    if freqs_list != [63, 125, 250, 500, 1000, 2000, 4000, 8000
                      ] or not os.path.exists(save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for k in range(len(freqs_list)):
            band_filename = f"{save_dir}/synth_band_{freqs_list[k]}Hz.pkl"
            if os.path.exists(band_filename):
                logger.info(f"Skipping {freqs_list[k]} Hz (already computed)")
                continue
            logger.info(
                f'Running inferencing for subband = {freqs_list[k]} Hz')

            # loop through all subband frequencies
            df_band = pd.DataFrame(columns=[
                'frequency', 'position', 'time_samples',
                'filtered_time_samples'
            ])

            config_dict = config_dicts[k]
            trainer_config = config_dict.trainer_config

            if "3room_FDTD" in config_dict.room_dataset_path:
                room_data = parse_three_room_data(
                    Path(config_dict.room_dataset_path).resolve())
            else:
                logger.error("Other room data not supported currently")

            # update the receiver positions in room dataset so that
            # the dataloader reads the updated positions in rec_pos_list
            # for inference
            room_data.update_receiver_pos(rec_pos_list)

            config_dict = config_dict.model_copy(
                update={"num_groups": room_data.num_rooms})
            assert config_dict.num_delay_lines % config_dict.num_groups == 0, "Delay lines must be \
                divisible by number of groups in network"

            # update ambisonics order
            config_dict = config_dict.model_copy(
                update={"ambi_order": room_data.ambi_order})

            if config_dict.sample_rate != room_data.sample_rate:
                logger.warning(
                    "Config sample rate does not match data, altering it")
                config_dict.sample_rate = room_data.sample_rate

            # get the training config
            trainer_config = config_dict.trainer_config
            # update num_freq_bins in pydantic class
            trainer_config = trainer_config.model_copy(
                update={"num_freq_bins": room_data.num_freq_bins})

            # are we using a colorless FDN to get the feedback matrix?
            if config_dict.colorless_fdn_config.use_colorless_prototype:
                colorless_fdn_params = get_colorless_fdn_params(config_dict)
            else:
                colorless_fdn_params = None

            # initialise the model
            model = DiffDirectionalFDNVarReceiverPos(
                room_data.sample_rate,
                room_data.num_rooms,
                config_dict.delay_length_samps,
                trainer_config.device,
                config_dict.feedback_loop_config,
                config_dict.output_filter_config,
                ambi_order=config_dict.ambi_order,
                desired_directions=room_data.sph_directions,
                common_decay_times=room_data.common_decay_times
                if config_dict.decay_filter_config.initialise_with_opt_values
                else None,
                band_centre_hz=room_data.band_centre_hz,
                colorless_fdn_params=colorless_fdn_params,
                use_colorless_loss=trainer_config.use_colorless_loss,
            )

            # create the inference object
            cur_infer_fdn = InferDiffDirectionalFDN(
                room_data,
                config_dict,
                model,
                apply_filter_norm=True,
                edc_len_ms=2000,
            )

            # get the ambisonics RIRs for the current frequency band
            position, est_dir_rirs = cur_infer_fdn.get_model_output(
                trainer_config.max_epochs, return_directional_rirs=True)

            # loop over all positions for a particular frequency band and add it to a dataframe
            for num_pos in range(position.shape[0]):
                cur_rir = est_dir_rirs[num_pos, ...].detach().cpu().numpy()

                # filter the current SRIR
                cur_rir_filtered = fftconvolve(
                    cur_rir,
                    subband_filters.coefficients[k, :][None, :],
                    mode='same')

                # position should be saved as tuple because numpy array is unhashable
                new_row = pd.DataFrame({
                    'frequency': [freqs_list[k]],
                    'position': [(position[num_pos,
                                           0], position[num_pos,
                                                        1], position[num_pos,
                                                                     2])],
                    'filtered_time_samples': [cur_rir_filtered],
                    'time_samples': [cur_rir],
                })
                df_band = pd.concat([df_band, new_row], ignore_index=True)

            df_band.to_pickle(band_filename)
            logger.info(f"Saved RIRs for band {freqs_list[k]} Hz")
            del model
            del room_data
            del cur_infer_fdn
        return

    else:
        # inference for all bands is complete, read the band wise dataframes
        # Dictionary: pos_key -> summed filtered_time_samples
        pos_to_rir = defaultdict(lambda: 0)
        pos_to_pos = {}

        for k, freq in enumerate(freqs_list):
            band_filename = f"{save_dir}/synth_band_{freq}Hz.pkl"
            df_band = pd.read_pickle(band_filename)

            # round positions and create pos_key
            df_band["pos_key"] = df_band["position"].apply(lambda pos: tuple(
                np.round(np.asarray(pos, dtype=np.float64), 3)))

            # accumulate into dictionary instead of building giant DataFrame
            for _, row in df_band.iterrows():
                pos_key = row["pos_key"]
                rir = row["filtered_time_samples"].astype(np.float32)

                if isinstance(pos_to_rir[pos_key], int):  # first time
                    pos_to_rir[pos_key] = rir
                    pos_to_pos[pos_key] = np.array(row["position"],
                                                   dtype=np.float64)
                else:
                    # add in-place to avoid new allocations
                    pos_to_rir[pos_key] += rir

        # Now stack results into arrays
        synth_rirs = list(pos_to_rir.values())
        est_dir_rirs = np.stack(
            synth_rirs, axis=0)  # (num_positions, num_channels, num_samples)

        # convert to ambisonics
        est_srirs = convert_directional_rirs_to_ambisonics(
            fullband_room_data.ambi_order, fullband_room_data.sph_directions,
            config_dicts[0].output_filter_config.beamformer_type,
            est_dir_rirs.transpose(1, 0, -1))

        # get receiver positions
        new_rec_pos_list = np.vstack(list(pos_to_pos.values()))

        # update dataset
        dfdn_room_data = deepcopy(fullband_room_data)
        dfdn_room_data.update_receiver_pos(new_rec_pos_list)
        dfdn_room_data.update_rirs(est_srirs)
        return dfdn_room_data
