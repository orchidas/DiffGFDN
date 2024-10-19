from typing import Tuple

import numpy as np
from scipy.signal import chirp
import torch

from .functional import bandpass_filter, sosfreqz


def signal_gallery(
    batch_size: int,
    n_samples: int,
    n: int,
    signal_type: str = "impulse",
    fs: int = 48000,
    rate: float = 1.0,
    reference=None,
    device=None,
):
    r"""
    Generate a tensor containing a signal based on the specified signal type.

    Supported signal types are:
    - impulse: A single impulse at the first sample, followed by :attr:`n_samples-1` zeros.
    - sine: A sine wave of frequency :attr:`rate` Hz, if given. Otherwise, a sine wave of frequency 1 Hz.
    - sweep: A linear sweep from 20 Hz to 20 kHz.
    - wgn: White Gaussian noise.
    - exp: An exponential decay signal.
    - reference: A reference signal.

        **Args**:
            - batch_size (int): The number of signal batches to generate.
            - n_samples (int): The number of samples in each signal.
            - n (int): The number of channels in each signal.
            - signal_type (str, optional): The type of signal to generate. Defaults to 'impulse'.
            - fs (int, optional): The sampling frequency of the signals. Defaults to 48000.
            - reference (torch.Tensor, optional): A reference signal to use. Defaults to None.
            - device (torch.device, optional): The device of constructed tensors. Defaults to None.

        **Returns**:
            - torch.Tensor: A tensor of shape (batch_size, n_samples, n) containing the generated signals.
    """
    signal_types = {
        "impulse",
        "sine",
        "sweep",
        "wgn",
        "exp",
        "reference",
    }

    if signal_type not in signal_types:
        raise ValueError(f"Matrix type {signal_type} not recognized.")
    if signal_type == "impulse":
        x = torch.zeros(batch_size, n_samples, n)
        x[:, 0, :] = 1
        return x.to(device)
    elif signal_type == "sine":
        if rate is not None:
            return torch.sin(2 * np.pi * rate / fs * torch.linspace(
                0, n_samples / fs, n_samples)).unsqueeze(-1).expand(
                    batch_size, n_samples, n).to(device)
        else:
            return torch.sin(
                torch.linspace(0, 2 * np.pi, n_samples).unsqueeze(-1).expand(
                    batch_size, n_samples, n)).to(device)
    elif signal_type == "sweep":
        t = torch.linspace(0, n_samples / fs - 1 / fs, n_samples)
        x = torch.tensor(chirp(t, f0=20, f1=20000, t1=t[-1], method="linear"),
                         device=device).unsqueeze(-1)
        return x.expand(batch_size, n_samples, n)
    elif signal_type == "wgn":
        return torch.randn((batch_size, n_samples, n), device=device)
    elif signal_type == "exp":
        return torch.exp(-rate * torch.arange(n_samples) /
                         fs).unsqueeze(-1).expand(batch_size, n_samples,
                                                  n).to(device)
    elif signal_type == "reference":
        if isinstance(reference, torch.Tensor):
            return reference.expand(batch_size, n_samples, n).to(device)
        else:
            return torch.tensor(reference,
                                device=device).expand(batch_size, n_samples, n)


def WGN_reverb(matrix_size: Tuple = (1, 1),
               t60: float = 1.0,
               samplerate: int = 48000,
               device=None) -> torch.Tensor:
    r"""
    Generate White-Gaussian-Noise-reverb impulse responses.

        **Args**:
            - matrix_size (tuple, optional): (output_channels, input_channels). Defaults to (1,1).
            - t60 (float, optional): Reverberation time. Defaults to 1.0.
            - samplerate (int, optional): Sampling frequency. Defaults to 48000.
            - nfft (int, optional): Number of frequency bins. Defaults to 2**11.

        **Returns**:
            torch.Tensor: Matrix of WGN-reverb impulse responses.
    """
    # Number of samples
    n_samples = int(1.5 * t60 * samplerate)
    # White Guassian Noise
    noise = torch.randn(n_samples, *matrix_size, device=device)
    # Decay
    dr = t60 / torch.log(torch.tensor(1000, dtype=torch.float32,
                                      device=device))
    decay = torch.exp(-1 / dr * torch.linspace(0, t60, n_samples))
    decay = decay.view(-1,
                       *(1, ) * (len(matrix_size))).expand(-1, *matrix_size)
    # Decaying WGN
    IRs = torch.mul(noise, decay)
    # Go to frequency domain
    TFs = torch.fft.rfft(input=IRs, n=n_samples, dim=0)

    # Generate bandpass filter
    fc_left = torch.tensor([20], dtype=torch.float32, device=device)
    fc_right = torch.tensor([20000], dtype=torch.float32, device=device)
    g = torch.tensor([1], dtype=torch.float32, device=device)
    b, a = bandpass_filter(fc1=fc_left,
                           fc2=fc_right,
                           gain=g,
                           fs=samplerate,
                           device=device)
    sos = torch.cat((b.reshape(1, 3), a.reshape(1, 3)), dim=1)
    bp_H = sosfreqz(sos=sos, nfft=n_samples).squeeze()
    bp_H = bp_H.view(*bp_H.shape,
                     *(1, ) * (len(TFs.shape) - 1)).expand(*TFs.shape)

    # Apply bandpass filter
    TFs = torch.mul(TFs, bp_H)

    # Return to time domain
    IRs = torch.fft.irfft(input=TFs, n=n_samples, dim=0)

    # Normalize
    vec_norms = torch.linalg.vector_norm(IRs, ord=2, dim=0)
    return IRs / vec_norms
