import numpy as np
import torch
from torch import nn


# wrapper for the sparsity loss
class sparsity_loss(nn.Module):
    """Sparsity loss of feedback matrix"""

    def forward(self, A: torch.tensor):
        """
        Args:
            A (torch.tensor): 2D feedback matrix
        """
        N = A.shape[-1]
        return -(torch.sum(torch.abs(A)) -
                 (N * np.sqrt(N))) / (N * (np.sqrt(N) - 1))


class mse_loss(nn.Module):
    """Means squared error between abs(x1) and x2"""

    def forward(self, y_pred: torch.tensor, y_true: torch.tensor):
        """
        Args:
            y_pred (torch.tensor): output of the model
            y_true (torch.tensor): expected output
        """
        # loss on system's output
        y_pred_sum = torch.sum(y_pred, dim=-1)
        loss = torch.mean(
            torch.pow(
                torch.abs(y_pred_sum) - torch.abs(y_true),
                2 * torch.ones(y_pred.size(0))))

        return loss


class amse_loss(nn.Module):
    """Asymmetric Means squared error between abs(x1) and x2"""

    def forward(self, y_pred: torch.tensor, y_true: torch.tensor):
        """
        Args:
            y_pred (torch.tensor): output of the model
            y_true (torch.tensor): expected output
        """
        # loss on system's output
        loss = self.p_loss(y_pred, y_true)
        return loss

    def p_loss(self, y_pred: torch.tensor, y_true: torch.tensor):
        """Higher loss if the magnitude exceeds the desired magnitude"""
        gT = 2 * torch.ones(len(y_pred))
        gT = gT + 2 * torch.gt(
            (torch.abs(y_pred) - torch.abs(y_true)), 1).type(torch.uint8)
        loss = torch.mean(
            torch.pow((torch.abs(y_pred) - torch.abs(y_true)), gT))

        return loss
