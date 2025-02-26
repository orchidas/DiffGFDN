import os
from pathlib import Path
import time
from typing import Dict, Optional, Tuple

from loguru import logger
import numpy as np
import pyfar as pf
import torch
from torch.utils.data import DataLoader
import torchaudio
from tqdm import trange

from .colorless_fdn.losses import amse_loss, mse_loss, sparsity_loss
from .config.config import TrainerConfig
from .losses import edc_loss, edr_loss, reg_loss
from .model import DiffGFDN, DiffGFDNSinglePos, DiffGFDNVarReceiverPos
from .utils import get_response, get_str_results, ms_to_samps

# flake8: noqa: E231
# pylint: disable=W0632, E0606


class Trainer:
    """Parent class for training DiffGFDN for a grid of source-listener positions and for one static position"""

    def __init__(self, net: DiffGFDN, trainer_config: TrainerConfig):
        """Class to train the DiffGFDN"""
        self.net = net
        self.device = trainer_config.device
        self.max_epochs = trainer_config.max_epochs
        self.patience = 5
        self.early_stop = 0
        self.train_dir = Path(trainer_config.train_dir).resolve()
        self.ir_dir = Path(trainer_config.ir_dir).resolve()
        self.use_reg_loss = trainer_config.use_reg_loss
        self.use_colorless_loss = trainer_config.use_colorless_loss
        self.reduced_pole_radius = trainer_config.reduced_pole_radius
        self.subband_process_config = trainer_config.subband_process_config

        if self.subband_process_config is not None:
            subband_filters, subband_freqs = pf.dsp.filter.reconstructing_fractional_octave_bands(
                None,
                num_fractions=self.subband_process_config.num_fraction_octaves,
                frequency_range=self.subband_process_config.frequency_range,
                sampling_rate=self.net.sample_rate,
            )
            subband_filter_idx = np.argmin(
                np.abs(subband_freqs -
                       self.subband_process_config.centre_frequency))
            subband_filter = torch.tensor(
                subband_filters.coefficients[subband_filter_idx])
            self.subband_filter_freq_resp = torch.fft.rfft(
                subband_filter, n=trainer_config.num_freq_bins)

            print(subband_filter_idx)

        self.init_scheduler(trainer_config)

        if not os.path.exists(self.ir_dir):
            os.makedirs(self.ir_dir)

        if trainer_config.use_edc_mask:
            logger.info("Using masked EDC loss")

        self.criterion = [
            edr_loss(
                self.net.sample_rate,
                reduced_pole_radius=self.reduced_pole_radius,
                use_erb_grouping=trainer_config.use_erb_edr_loss,
                use_weight_fn=trainer_config.use_frequency_weighting,
            ),
            edc_loss(self.net.common_decay_times.max() * 1e3,
                     self.net.sample_rate,
                     use_mask=trainer_config.use_edc_mask),
        ]
        self.loss_weights = torch.tensor(
            [trainer_config.edr_loss_weight, trainer_config.edc_loss_weight])

        if trainer_config.use_reg_loss:
            logger.info(
                'Using regularisation loss to reduce time domain aliasing in output filters'
            )
            self.criterion.append(
                reg_loss(
                    ms_to_samps(trainer_config.output_filt_ir_len_ms,
                                self.net.sample_rate), self.net.num_groups,
                    self.net.output_filters.num_biquads))
            self.loss_weights = torch.cat(self.loss_weights,
                                          torch.tensor([1.0]))

        if self.use_colorless_loss:
            logger.info('Using colorless FDN loss for each sub-FDN')
            if trainer_config.use_asym_spectral_loss:
                self.colorless_criterion = [amse_loss(), sparsity_loss()]
            else:
                self.colorless_criterion = [mse_loss(), sparsity_loss()]
            self.colorless_loss_weights = torch.tensor([
                trainer_config.spectral_loss_weight,
                trainer_config.sparsity_loss_weight
            ])

    def init_scheduler(self, trainer_config: TrainerConfig):
        """
        Initialise scheduler, set faster learning rate for input-output gains
        specified learning rate for all other params
        """
        param_groups = [
            {
                'params': [
                    param for name, param in self.net.named_parameters()
                    if 'feedback_loop.alpha' in name
                ],
                'lr':
                trainer_config.coupling_angle_lr,
            },
            {
                'params': [
                    param for name, param in self.net.named_parameters()
                    if 'output_gains' in name
                ],
                'lr':
                trainer_config.io_lr,
            },
            {
                'params': [
                    param for name, param in self.net.named_parameters()
                    if 'input_gains' in name
                ],
                'lr':
                trainer_config.io_lr
            },
            {
                'params': [
                    param for name, param in self.net.named_parameters()
                    if 'output_svf_params' in name
                ],
                'lr':
                trainer_config.io_lr
            },
            {
                'params': [
                    param for name, param in self.net.named_parameters()
                    if 'input_scalars' in name
                ],
                'lr':
                trainer_config.io_lr
            },
            {
                'params': [
                    param for name, param in self.net.named_parameters()
                    if 'output_scalars' in name
                ],
                'lr':
                trainer_config.io_lr
            },
            # Add more groups as needed
        ]

        # Gather parameters that are not in the specified groups
        other_params = [
            param for name, param in self.net.named_parameters()
            if not ('feedback_loop.alpha' in name or 'input_gains' in name
                    or 'output_gains' in name or 'output_svf_params' in name
                    or 'output_scalars' in name or 'input_scalars' in name)
        ]

        # Add the other parameters with a learning rate of 0.01
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': trainer_config.lr
            })

        self.optimizer = torch.optim.Adam(param_groups)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=10,
                                                         gamma=0.1)

    def to_device(self):
        """Return the device to train on - CPU or GPU"""
        for i in range(len(self.criterion)):
            self.criterion[i] = self.criterion[i].to(self.device)

        if self.colorless_criterion:
            for i in range(len(self.colorless_criterion)):
                self.colorless_criterion[0] = self.colorless_criterion[i].to(
                    self.device)

    def print_results(self, e: int, e_time):
        """Print results of training"""
        print(
            get_str_results(epoch=e,
                            train_loss=self.train_loss,
                            time=e_time,
                            individual_losses=self.individual_train_loss if
                            hasattr(self, 'individual_train_loss') else None))

        # for debugging
        # for name, param in self.net.named_parameters():
        #     if name in ('input_scalars',
        #                 'output_scalars') and param.requires_grad:
        #         print(f"Parameter {name}: {param.data}")
        #         print(f"Parameter {name} gradient: {param.grad.norm()}")

    def save_model(self, e: int):
        """Save the model at epoch number e"""
        dir_path = os.path.join(self.train_dir, 'checkpoints')
        # create checkpoint folder
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # save model
        torch.save(self.net.state_dict(),
                   os.path.join(dir_path, 'model_e' + str(e) + '.pt'))

    def calculate_losses(self,
                         data: Dict,
                         H: torch.tensor,
                         H_sub_fdn: Optional[Tuple] = None) -> Dict:
        """
        Avoid repetition of code by by using single function to calculate losses
        Args:
            data (Dict): data dictionary
            H (torch.tensor): transfer function of DiffGFDN
            H_sub_fdn (Tuple): transfer function of each FDN
        Returns:
            Dict: dictionary of all losses
        """

        edr_loss_val = self.loss_weights[0] * self.criterion[0](
            data['target_rir_response'], H)
        edc_loss_val = self.loss_weights[1] * self.criterion[1](
            data['target_rir_response'], H)
        all_losses = {
            'edc_loss': edc_loss_val,
            'edr_loss': edr_loss_val,
        }

        if self.use_reg_loss:
            reg_loss_val = self.loss_weights[2] * self.criterion[2](
                self.net.biquad_cascade)
            all_losses.update({'reg_loss': reg_loss_val})

        if self.use_colorless_loss:
            spectral_loss_val = 0.0
            sparsity_loss_val = 0.0
            for k in range(self.net.num_groups):
                # mean over all freq bins
                spectral_loss_val += self.colorless_loss_weights[
                    0] * self.colorless_criterion[0](H_sub_fdn[0][..., k],
                                                     torch.ones_like(
                                                         H_sub_fdn[0][..., k]))

                sparsity_loss_val = self.colorless_loss_weights[
                    1] * self.colorless_criterion[1](
                        self.net.feedback_loop.ortho_param(
                            self.net.feedback_loop.M[k]))
            colorless_losses = {
                'spectral_loss': spectral_loss_val,
                'sparsity_loss': sparsity_loss_val
            }
            all_losses.update(colorless_losses)
        return all_losses


class VarReceiverPosTrainer(Trainer):
    """Class for training DiffGFDN for a grid of receiver positions"""

    def __init__(self, net: DiffGFDNVarReceiverPos,
                 trainer_config: TrainerConfig):
        super().__init__(net, trainer_config)

    def train(self, train_dataset: DataLoader):
        """Train the network"""
        self.train_loss = []
        self.individual_train_loss = []

        st = time.time()  # start time
        # save initial parameters
        super().save_model(-1)

        for epoch in trange(self.max_epochs, desc='Training'):
            logger.info(f'Epoch #{epoch}')
            st_epoch = time.time()

            # normalise b, c at each epoch to ensure the sub-FDNs have
            # unit energy
            data = next(iter(train_dataset))
            self.normalize(data)

            # training
            epoch_loss = 0.0
            all_loss = {}
            for data in train_dataset:
                cur_loss, cur_all_loss = self.train_step(data)
                epoch_loss += cur_loss

                if not all_loss:
                    all_loss = {key: 0.0 for key in cur_all_loss}

                for key, value in cur_all_loss.items():
                    all_loss[key] += value

            self.scheduler.step()
            self.train_loss.append(epoch_loss / len(train_dataset))
            for key, value in all_loss.items():
                all_loss[key] /= len(train_dataset)
            self.individual_train_loss.append(all_loss)
            et_epoch = time.time()
            super().save_model(epoch)
            super().print_results(epoch, et_epoch - st_epoch)

            # early stopping
            if epoch >= 1:
                if abs(self.train_loss[-2] - self.train_loss[-1]) <= 0.0001:
                    self.early_stop += 1
                else:
                    self.early_stop = 0
            if self.early_stop == self.patience:
                break

        et = time.time()  # end time
        logger.info('Training time: {:.3f}s'.format(et - st))

        # save the trained IRs
        logger.info("Saving the trained IRs...")
        for data in train_dataset:
            src_position = data['source_position']
            rec_position = data['listener_position']
            self.save_ir(
                data,
                directory=self.ir_dir,
                src_pos=src_position,
                rec_pos=rec_position,
            )

    def train_step(self, data):
        """Single step of training"""
        self.optimizer.zero_grad()
        if self.use_colorless_loss:
            H, H_sub_fdn = self.net(data)
            if self.subband_process_config is not None:
                # filter H in subbands before calculating loss
                H_subband = H * self.subband_filter_freq_resp
            all_losses = super().calculate_losses(data, H_subband, H_sub_fdn)
        else:
            H = self.net(data)
            if self.subband_process_config is not None:
                # filter H in subbands before calculating loss
                H_subband = H * self.subband_filter_freq_resp
            all_losses = super().calculate_losses(data, H_subband)

        loss = sum(all_losses.values())
        loss.backward()
        self.optimizer.step()
        return loss.item(), all_losses

    @torch.no_grad()
    def validate(self, valid_dataset: DataLoader):
        """Validate the training with unseen data and save the resulting IRs"""
        total_loss = 0
        self.valid_loss = []
        self.individual_valid_loss = []

        for data in valid_dataset:
            rec_position = data['listener_position']
            src_position = data['source_position']
            logger.info("Running the network for new batch of positiions")
            cur_all_losses = {}
            if self.use_colorless_loss:
                H, H_sub_fdn = self.save_ir(data,
                                            directory=self.ir_dir,
                                            src_pos=src_position,
                                            rec_pos=rec_position,
                                            filename_prefix="valid_ir")
                if self.subband_process_config is not None:
                    # filter H in subbands before calculating loss
                    H_subband = H * self.subband_filter_freq_resp
                cur_all_losses = super().calculate_losses(
                    data, H_subband, H_sub_fdn)

            else:
                H = self.save_ir(data,
                                 directory=self.ir_dir,
                                 src_pos=src_position,
                                 rec_pos=rec_position,
                                 filename_prefix="valid_ir")
                if self.subband_process_config is not None:
                    # filter H in subbands before calculating loss
                    H_subband = H * self.subband_filter_freq_resp
                cur_all_losses = super().calculate_losses(data, H_subband)

            cur_loss = sum(cur_all_losses.values())
            total_loss += cur_loss

            self.valid_loss.append(cur_loss)
            self.individual_valid_loss.append(cur_all_losses)
            logger.info(
                f"The validation loss for the current position is {cur_loss:.4f}"
            )

        net_valid_loss = total_loss / len(valid_dataset)
        logger.info(f"The net validation loss is {net_valid_loss:.4f}")

    @torch.no_grad()
    def normalize(self, data: Dict):
        # average energy normalization - this normalises the energy
        # of each of the sub-FDNs to be unity

        if self.use_colorless_loss:
            _, H_sub_fdn, _ = get_response(data, self.net)
            energyH_sub = torch.mean(torch.pow(torch.abs(H_sub_fdn[0]), 2),
                                     dim=0)
            for name, prm in self.net.named_parameters():
                if name in ('input_gains', 'output_gains'):
                    for k in range(self.net.num_groups):
                        ind_slice = torch.arange(
                            k * self.net.num_delay_lines_per_group,
                            (k + 1) * self.net.num_delay_lines_per_group,
                            dtype=torch.int32)
                        prm.data[ind_slice].copy_(
                            torch.div(prm.data[ind_slice],
                                      torch.pow(energyH_sub[k], 1 / 4)))

    @torch.no_grad()
    def save_ir(
        self,
        input_features: Dict,
        directory: str,
        src_pos: torch.tensor,
        rec_pos: torch.tensor,
        filename_prefix: str = "ir",
        norm: bool = True,
    ) -> torch.tensor:
        """
        Save the impulse response generated from the model
        Args:
            input_features (Dict): dictionary of input features
            directory (str): where to save the audio
            pos_list (torch.tensor): B x 3 position coordinates
            norm (bool): whether to normalise the RIR
        Returns:
            torch.tensor - the frequency response at the given input features
        """
        if self.use_colorless_loss:
            H, H_sub_fdn, h = get_response(input_features, self.net)
        else:
            H, h = get_response(input_features, self.net)

        # undo sampling outside the unit circle by multiplying IR with an exponentiated envelope

        if self.reduced_pole_radius is not None:
            h *= torch.pow(1.0 / self.reduced_pole_radius,
                           torch.arange(0, h.shape[-1]))

        if norm:
            h = torch.div(h, torch.max(torch.abs(h)))

        num_src = 1 if src_pos.ndim == 1 or torch.all(
            src_pos == src_pos[0]) else src_pos.shape[0]

        for src_idx in range(num_src):
            for num_pos in range(rec_pos.shape[0]):

                if num_src == 1:
                    filename = (
                        f'{filename_prefix}_({rec_pos[num_pos,0]:.2f}, '
                        f'{rec_pos[num_pos, 1]:.2f}, {rec_pos[num_pos, 2]:.2f}).wav'
                    )
                else:
                    filename = (
                        f'{filename_prefix}_src_pos=({src_pos[src_idx,0]:.2f}, '
                        f'{src_pos[src_idx, 1]:.2f}, {src_pos[src_idx, 2]:.2f})'
                        f'_rec_pos=({rec_pos[num_pos,0]:.2f}, '
                        f'{rec_pos[num_pos, 1]:.2f}, {rec_pos[num_pos, 2]:.2f}).wav'
                    )

                filepath = os.path.join(directory, filename)
                # for some reason torch audio expects a 2D tensor
                torchaudio.save(filepath,
                                torch.stack((h[num_pos, :], h[num_pos, :]),
                                            dim=1).cpu(),
                                int(self.net.sample_rate),
                                bits_per_sample=32,
                                channels_first=False)
        return (H, H_sub_fdn) if self.use_colorless_loss else H


class SinglePosTrainer(Trainer):
    """Trainer class for training DiffGFDN on a single measured RIR"""

    def __init__(self, net: DiffGFDNSinglePos, trainer_config: TrainerConfig,
                 filename: str):
        super().__init__(net, trainer_config)
        self.filename = filename

    def train(self, train_dataset: DataLoader):
        # Training cannot be done batch-wise in this instance, similarly we cannot have
        # a train-valid split. This is because we need the ENTIRE sampled unit circle
        # response to calculate the loss function

        # normalize input and output gains based on energy of FDN's IR
        data = next(iter(train_dataset))
        self.normalize(data)

        self.train_loss = []
        self.individual_train_loss = []
        st = time.time()  # start time
        for epoch in trange(self.max_epochs, desc='Training'):
            st_epoch = time.time()

            # training - looping over a single batch
            for data in train_dataset:
                epoch_loss, individual_losses = self.train_step(data)

            self.scheduler.step()
            self.train_loss.append(epoch_loss)
            self.individual_train_loss.append(individual_losses)
            et_epoch = time.time()

            super().print_results(epoch, et_epoch - st_epoch)
            super().save_model(epoch)

            # early stopping
            if epoch >= 1:
                if abs(self.train_loss[-2] - self.train_loss[-1]) <= 1e-4:
                    self.early_stop += 1
                else:
                    self.early_stop = 0
            if self.early_stop == self.patience:
                break

        et = time.time()  # end time
        print('Training time: {:.3f}s'.format(et - st))

        # save the trained IRs
        logger.info("Saving the trained IR...")
        for data in train_dataset:
            self.save_ir(
                data,
                directory=self.ir_dir,
                filename_prefix='approx_' + self.filename,
            )

    def train_step(self, data: Dict) -> Tuple[float, Dict]:
        """Single step for training"""
        self.optimizer.zero_grad()
        if self.use_colorless_loss:
            H, H_sub_fdn = self.net(data)
            if self.subband_process_config is not None:
                # filter H in subbands before calculating loss
                H_subband = H * self.subband_filter_freq_resp
            all_losses = super().calculate_losses(data, H_subband, H_sub_fdn)
        else:
            H = self.net(data)
            if self.subband_process_config is not None:
                # filter H in subbands before calculating loss
                H_subband = H * self.subband_filter_freq_resp
            all_losses = super().calculate_losses(data, H_subband)

        loss = sum(all_losses.values())
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item(), all_losses

    @torch.no_grad()
    def normalize(self, data: Dict):
        # average energy normalization - this normalises the energy
        # of the initial FDN to the target impulse response

        if self.use_colorless_loss:
            H, H_sub_fdn, _ = get_response(data, self.net)
            energyH_sub = torch.mean(torch.pow(torch.abs(H_sub_fdn[0]), 2),
                                     dim=0)
            for name, prm in self.net.named_parameters():
                if name in ('input_gains', 'output_gains'):
                    for k in range(self.net.num_groups):
                        ind_slice = torch.arange(
                            k * self.net.num_delay_lines_per_group,
                            (k + 1) * self.net.num_delay_lines_per_group,
                            dtype=torch.int32)
                        prm.data[ind_slice].copy_(
                            torch.div(prm.data[ind_slice],
                                      torch.pow(energyH_sub[k], 1 / 4)))
        else:
            H, _ = get_response(data, self.net)
        energyH = torch.mean(torch.pow(torch.abs(H), 2))
        energyH_target = torch.mean(
            torch.pow(torch.abs(data['target_rir_response']), 2))
        energy_diff = torch.div(energyH, energyH_target)
        # apply energy normalization on input and output gains only
        for name, prm in self.net.named_parameters():
            if name in ('input_scalars', 'output_scalars'):
                prm.data.copy_(
                    torch.div(prm.data, torch.pow(energy_diff, 1 / 4)))

    @torch.no_grad()
    def save_ir(self,
                data: Dict,
                directory: str,
                filename_prefix: str = 'ir',
                norm=False):
        if self.use_colorless_loss:
            _, _, h = get_response(data, self.net)
        else:
            _, h = get_response(data, self.net)
        # undo sampling outside the unit circle by multiplying IR with an exponentiated envelope
        if self.reduced_pole_radius is not None:
            h *= torch.pow(1.0 / self.reduced_pole_radius,
                           torch.arange(0, h.shape[-1]))
        if norm:
            h = torch.div(h, torch.max(torch.abs(h)))
        filepath = os.path.join(directory, filename_prefix + '.wav')
        torchaudio.save(filepath,
                        torch.stack((h, h), dim=1).cpu(),
                        int(self.net.sample_rate),
                        bits_per_sample=32,
                        channels_first=False)
