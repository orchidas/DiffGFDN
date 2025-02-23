import os
import pickle
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
from scipy.io import savemat
import torch

from .colorless_fdn.model import ColorlessFDN
from .colorless_fdn.utils import ColorlessFDNResults
from .model import DiffGFDN


def save_diff_gfdn_parameters(net: DiffGFDN, dir_path: str, filename: str):
    """
    Save parameters of DiffGFDN() net to .mat file 
    Args    net (nn.Module): trained FDN() network
            dir_path (string): path to output firectory
            filename (string): name of the file 
    Output  param (dictionary of tensors): FDN() net parameters
            param_np (dictionary of numpy arrays): DiffGFDN() net parameters
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    param = fdn2dir(net)
    param_np = {}
    for name, value in param.items():
        try:
            param_np[name] = value.squeeze().cpu().numpy()
        except AttributeError:
            param_np[name] = value
    # save parameters in numpy format
    savemat(os.path.join(dir_path, filename), param_np)

    return param, param_np


def save_colorless_fdn_parameters(net: ColorlessFDN, dir_path: str,
                                  filename: str) -> ColorlessFDNResults:
    """
    Save parameters of ColorlessFDN() net to .pkl file 
    Args    net (nn.Module): trained FDN() network
            dir_path (string): path to output firectory
            filename (string): name of the file 
    Output  ColorlessFDNResults: dataclass containing the results of optimisation
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    param = fdn2dir(net)
    param_np = {}
    for name, value in param.items():
        try:
            param_np[name] = value.squeeze().cpu().numpy() if isinstance(
                value, torch.Tensor) else value
        except AttributeError:
            param_np[name] = value

    colorless_fdn_params = ColorlessFDNResults(
        opt_input_gains=param_np['input_gains'],
        opt_output_gains=param_np['output_gains'],
        opt_feedback_matrix=param_np['feedback_matrix'])
    # save parameters in numpy format
    with open(os.path.join(dir_path, filename), "wb") as f:
        pickle.dump(colorless_fdn_params, f)

    return colorless_fdn_params


def fdn2dir(net: Union[DiffGFDN, ColorlessFDN]):
    """
    Save learnable parameters to a dictionary  
    Args    net (nn.Module): trained FDN() network
    Output  d (dictionary of tensors): FDN() net parameters 
    """
    d = {}  # empty dictionary
    # from parameter dictionary of model
    d = net.get_param_dict()

    # MLP learned weights and biases
    for name, param in net.named_parameters():
        if param.requires_grad and name not in d.keys():
            d[name] = param.data

    return d


def save_loss(train_loss: List,
              output_dir: str,
              save_plot=True,
              filename: str = '',
              xaxis_label: Optional[str] = "epoch #",
              individual_losses: Optional[List[Dict]] = None):
    """
    Save training and validation loss values in .mat format
    Args    train_loss (list): training loss values at each epoch
            output_dir (string): path to output directory
            save_plot (bool): if True saves the plot of the losses in .pdf format
            filename (string): additional string to add before .pdf and .mat
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_epochs = len(train_loss)
    losses = {}
    losses['train'] = train_loss
    individual_losses_mat = {}

    if save_plot:
        plt.figure()
        plt.plot(range(1, n_epochs + 1), train_loss)
        plt.xlabel(xaxis_label)
        plt.ylabel('loss')
        plt.savefig(os.path.join(output_dir, filename + '.pdf'))

        if individual_losses is not None:
            keys = individual_losses[0].keys()
            plt.figure()

            for key in keys:
                loss_values = [
                    d[key] for d in individual_losses
                ]
                individual_losses_mat[key] = loss_values
                plt.semilogy(range(1, n_epochs + 1), loss_values, label=key)

            plt.xlabel(xaxis_label)
            plt.ylabel('loss (log)')
            plt.legend()
            plt.savefig(
                os.path.join(output_dir, filename + '_individual_loss.pdf'))
            plt.close()

    savemat(os.path.join(output_dir, 'losses_' + filename + '.mat'), losses)

    if individual_losses is not None:
        savemat(
            os.path.join(output_dir, 'individual_losses_' + filename + '.mat'),
            individual_losses_mat)
