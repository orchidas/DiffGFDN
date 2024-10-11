import torch
from torch import nn
from torch.optim import LBFGS
import torch 
import soundfile as sf
from itertools import product

def get_device():
    r""" Output 'cuda' if gpu is available, 'cpu' otherwise """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_complex(x):
    r"""
    Converts a real tensor to a complex tensor.

        **Args**:
            x (torch.Tensor): The input tensor.

        **Returns**:
            torch.Tensor: The complex tensor with the same shape as the input tensor.
    """
    return torch.complex(x, torch.zeros_like(x))

def save_audio(filepath, x, fs=48000):
    sf.write(filepath, x.cpu().numpy(), fs)

class RegularGridInterpolator:
    """
    Interpolates values on a regular grid.

        **Args**:
            - points (tuple or list): The grid points.
            - values (torch.Tensor): The corresponding values.

        **Attribute**s:
            - points (tuple or list): The grid points.
            - values (torch.Tensor): The corresponding values.
            - ms (list): The shape of the values tensor.
            - n (int): The number of grid points.

        **Methods**:
            - __call__(points_to_interp): Interpolates the values at the given points.

        **Returns**:
            - torch.Tensor: The interpolated values.
    """

    def __init__(self, points, values):
        self.points = points
        self.values = values

        assert isinstance(self.points, tuple) or isinstance(self.points, list)
        assert isinstance(self.values, torch.Tensor)

        self.ms = list(self.values.shape)
        self.n = len(self.points)

        assert len(self.ms) == self.n

        for i, p in enumerate(self.points):
            assert isinstance(p, torch.Tensor)
            assert p.shape[0] == self.values.shape[i]

    def __call__(self, points_to_interp):
        assert self.points is not None
        assert self.values is not None

        assert len(points_to_interp) == len(self.points)
        K = points_to_interp[0].shape[0]
        for x in points_to_interp:
            assert x.shape[0] == K

        idxs = []
        dists = []
        overalls = []
        for p, x in zip(self.points, points_to_interp):
            idx_right = torch.bucketize(x, p)
            idx_right[idx_right >= p.shape[0]] = p.shape[0] - 1
            idx_left = (idx_right - 1).clamp(0, p.shape[0] - 1)
            dist_left = x - p[idx_left]
            dist_right = p[idx_right] - x
            dist_left[dist_left < 0] = 0.
            dist_right[dist_right < 0] = 0.
            both_zero = (dist_left == 0) & (dist_right == 0)
            dist_left[both_zero] = dist_right[both_zero] = 1.

            idxs.append((idx_left, idx_right))
            dists.append((dist_left, dist_right))
            overalls.append(dist_left + dist_right)

        numerator = 0.
        for indexer in product([0, 1], repeat=self.n):
            as_s = [idx[onoff] for onoff, idx in zip(indexer, idxs)]
            bs_s = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]
            numerator += self.values[as_s] * \
                torch.prod(torch.stack(bs_s), dim=0)
        denominator = torch.prod(torch.stack(overalls), dim=0)
        return numerator / denominator
    
class MLS(nn.Module):
    def __init__(self, G, target_interp):
        super().__init__()
        self.G = G
        self.target_interp = target_interp

    def forward(self, x):
        return torch.mean(torch.pow(torch.matmul(self.G, x) - self.target_interp, 2))


def minimize_LBFGS(G, target_interp, lower_bound, upper_bound, num_freq, max_iter=100):

    initial_guess = nn.Parameter(torch.ones(num_freq + 1))
    assert len(lower_bound) == len(upper_bound) == len(initial_guess), 'The number of bounds must be equal to the number of gains.'
    
    # Create an instance of LBFGS optimizer
    optimizer = LBFGS([initial_guess])
    criterion = MLS(G, target_interp)
    # Define a closure for the LBFGS optimizer
    def closure():
        optimizer.zero_grad()
        loss = criterion(initial_guess)
        loss.backward()
        initial_guess.data.clamp_(lower_bound, upper_bound) 
        return loss
    
    # Perform optimization
    for i in range(max_iter):
        optimizer.step(closure)

    # Get the optimized result
    return initial_guess
