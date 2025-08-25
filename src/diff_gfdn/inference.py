from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pyfar as pf
from slope2noise.rooms import RoomGeometry
from slope2noise.utils import decay_kernel, schroeder_backward_int
import torch
from torch import nn
from tqdm import tqdm

from spatial_sampling.config import DNNType
from spatial_sampling.dataloader import SpatialRoomDataset
from spatial_sampling.dataloader import load_dataset as load_spatial_dataset

from .config.config import DiffGFDNConfig, SubbandProcessingConfig
from .dataloader import load_dataset, RoomDataset
from .plot import order_position_matrices, plot_edr
from .utils import db, db2lin, get_response, ms_to_samps

# flake8: noqa:E231
# pylint: disable=W0632


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
                    deepcopy(param_dict['input_gains']))
                self.all_learned_params.input_scalars.append(
                    deepcopy(param_dict['input_scalars']))
                self.all_learned_params.output_gains.append(
                    deepcopy(param_dict['output_gains']))
                self.all_learned_params.coupled_feedback_matrix.append(
                    deepcopy(param_dict['coupled_feedback_matrix']))
                self.all_learned_params.coupling_matrix.append(
                    deepcopy(param_dict['coupling_matrix']))

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

    def __init__(self, room_data: SpatialRoomDataset,
                 config_dict: DiffGFDNConfig, model: nn.Module):
        """
        Initialise parameters for the class
        Args:
            room_data (SpatialRoomDataset): object of SpatialRoomDataset dataclass
            config_dict (DiffGFDNConfig): config file, read as dictionary
            model (DiffDirectionalFDNVarReceiverPos): the NN model to be tested
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

        # prepare the training and validation data
        self.train_dataset, _, _ = load_spatial_dataset(
            room_data,
            config_dict.trainer_config.device,
            network_type=DNNType.MLP,
            batch_size=config_dict.trainer_config.batch_size,
            train_valid_split_ratio=1.0,
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
        if self.config_dict.trainer_config.subband_process_config is not None:
            self.subband_filter_norm_factor = InferDiffGFDN.get_norm_factor(
                self.config_dict.trainer_config.subband_process_config,
                self.room_data.sample_rate)
        self._init_decay_kernel()

    def _init_decay_kernel(self):
        """Initialise the decay kernels for calculating EDC errors"""
        num_slopes = self.room_data.num_rooms
        edc_len_samps = ms_to_samps(self.model.common_decay_times.max() * 1e3,
                                    self.room_data.sample_rate)
        self.envelopes = np.zeros((num_slopes, edc_len_samps))
        time_axis = np.linspace(0, (edc_len_samps - 1) /
                                self.room_data.sample_rate, edc_len_samps)

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

    def get_model_output(self, num_epochs: int) -> Tuple[NDArray, NDArray]:
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

            for data in tqdm(self.train_dataset):
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
                # compensate for subband filtering
                est_ambi_rirs_comp = est_ambi_rirs * self.subband_filter_norm_factor

            # convert from ambisonics to directional RIRs
            est_dir_rirs = self.convert_ambi_rir_to_directional_rir(
                est_ambi_rirs_comp)

        return est_pos, est_dir_rirs

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
                save_path=f'{save_dir}/edc_error_in_space_' +
                f'split_ratio={np.round(self.config_dict.trainer_config.train_valid_split, 1)}'
                + extend + f'_epoch={epoch_num}.png',
                title=
                f'az = {np.degrees(self.room_data.sph_directions[0, j]):.2f} deg,'
                +
                f' pol = {np.degrees(self.room_data.sph_directions[1, j]):.2f} deg'
            )

        return original_edc, est_edc
