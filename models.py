import os
import typing as tp
import warnings

import torch
from functools import partial
import julius
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Union
from torch import nn


class MelSpectrum:
    def __init__(self, sample_rate: int, n_mels=40, n_fft=512, hop_length=None,
                 normalized=True, use_log_scale=True, log_scale_eps=1e-5):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.normalized = normalized
        self.use_log_scale = use_log_scale
        self.log_scale_eps = log_scale_eps
        self.trans = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels,
            n_fft=n_fft, hop_length=self.hop_length, normalized=normalized
        )
    def _compute(self, wav: torch.Tensor) -> torch.Tensor:
        # Assume wav is a 1D tensor of audio samples
        if self.use_log_scale:
            melspec = torch.log10(self.trans(wav) + self.log_scale_eps)
        else:
            melspec = self.trans(wav)

        return melspec

    def get_feature(self, wav: torch.Tensor) -> torch.Tensor:
        return self._compute(wav)

    """ 
   def _compute(self, filepath: Union[str, Path], duration: float) -> torch.Tensor:
        wav, sr = torchaudio.load(filepath)
        if sr != self.sample_rate:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(wav)

        wav = torch.mean(wav, dim=0)  # Convert stereo to mono
        melspec = self.trans(wav)  # Compute Mel spectrogram

        if self.use_log_scale:
            melspec = torch.log10(melspec + self.log_scale_eps)

        # Calculate target length based on duration and interpolate
        target_length = int(round(duration * self.sample_rate))
        melspec = torch.nn.functional.interpolate(melspec.unsqueeze(0), size=(target_length,), mode='linear',
                                                  align_corners=False).squeeze(0)

        return melspec

    def get_feature(self, filepath: Union[str, Path], duration: float) -> torch.Tensor:
        return self._compute(filepath, duration)
        """


class PhaseSpectrum:
    def __init__(self, sample_rate: int, n_fft: int = 512, hop_length: int = None):
        """
        Initializes the PhaseSpectrum class with necessary parameters.

        Parameters:
        sample_rate (int): The sample rate of the audio data.
        n_fft (int): The number of points in the FFT, affects frequency resolution.
        hop_length (int): The number of samples between successive frames. Default is n_fft // 4.
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4

    def _compute(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Computes the phase spectrum of an audio tensor.

        Parameters:
        wav (torch.Tensor): A tensor of audio samples.

        Returns:
        torch.Tensor: The phase spectrum of the audio tensor.
        """
        if wav.dim() > 1:
            wav = torch.mean(wav, dim=0)  # Convert stereo to mono if necessary
        window = torch.hann_window(self.n_fft, periodic=True)
        # Compute the Short-Time Fourier Transform (STFT)
        stft = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True, window=window)
        # Extract the phase
        phase = torch.angle(stft)

        return phase

    def get_feature(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Public method to get the phase spectrum feature of an audio tensor.

        Parameters:
        wav (torch.Tensor): A tensor of audio samples.

        Returns:
        torch.Tensor: The phase spectrum of the audio tensor.
        """
        return self._compute(wav)


class SpeechEnvelope:
    def __init__(self, sample_rate: int):
        """
        Initializes the SpeechEnvelope class with the necessary parameters.

        Parameters:
        sample_rate (int): The sample rate of the audio data.
        """
        self.sample_rate = sample_rate

    def _compute(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Computes the speech envelope of an audio tensor.

        Parameters:
        wav (torch.Tensor): A tensor of audio samples.

        Returns:
        torch.Tensor: The envelope of the audio tensor.
        """
        if wav.dim() > 1:
            wav = torch.mean(wav, dim=0)  # Convert stereo to mono if necessary

        # Apply the STFT to obtain the frequency-time representation of the signal
        stft_result = torch.stft(wav, n_fft=len(wav), hop_length=len(wav) // 2, window=torch.hann_window(len(wav)),
                                 return_complex=True)

        # Compute the envelope as the magnitude of the complex numbers
        envelope = torch.abs(stft_result)

        # Collapse the frequency dimension by taking the maximum, which gives us the envelope over time
        envelope = torch.max(envelope, dim=-2)[0]  # Assuming frequency dimension is the last but one

        return envelope

    def get_feature(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Public method to get the speech envelope feature of an audio tensor.

        Parameters:
        wav (torch.Tensor): A tensor of audio samples.

        Returns:
        torch.Tensor: The envelope of the audio tensor.
        """
        return self._compute(wav)


class MFCCSpectrum:
    def __init__(self, sample_rate: int, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = None, n_mels: int = 40):
        """
        Initializes the MFCCSpectrum class with necessary parameters.

        Parameters:
        sample_rate (int): The sample rate of the audio files.
        n_mfcc (int): The number of MFCCs to return.
        n_fft (int): The number of points in the FFT, affects frequency resolution.
        hop_length (int): The number of samples between successive frames.
        n_mels (int): The number of Mel filterbanks.
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 2
        self.n_mels = n_mels

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'n_mels': self.n_mels,
                'hop_length': self.hop_length
            }
        )

    def _compute(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Computes the MFCCs of an audio tensor.

        Parameters:
        wav (torch.Tensor): A tensor of audio samples.

        Returns:
        torch.Tensor: The MFCCs of the audio tensor.
        """
        if wav.dim() > 1:
            wav = torch.mean(wav, dim=0)  # Convert stereo to mono if necessary

        mfccs = self.mfcc_transform(wav)  # Compute MFCC

        return mfccs

    def get_feature(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Public method to get the MFCC features of an audio tensor.

        Parameters:
        wav (torch.Tensor): A tensor of audio samples.

        Returns:
        torch.Tensor: The MFCCs of the audio tensor.
        """
        return self._compute(wav)


class SimpleConv(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, depth=4):
        super().__init__()
        # Create a sequence of channels from input to hidden depths
        channels = [in_channels] + [hidden_channels] * (depth - 1) + [out_channels]
        self.network = ConvSequence(channels, kernel=5, stride=1, dilation_growth=2, activation=nn.ReLU())

    def forward(self, x):
        return self.network(x)

class ConvSequence(nn.Module):

    def __init__(self, channels: tp.Sequence[int], kernel: int = 4, dilation_growth: int = 1,
                 dilation_period: tp.Optional[int] = None, stride: int = 2,
                 dropout: float = 0.0, leakiness: float = 0.0, groups: int = 1,
                 decode: bool = False, batch_norm: bool = False, dropout_input: float = 0,
                 skip: bool = False, scale: tp.Optional[float] = None, rewrite: bool = False,
                 activation_on_last: bool = True, post_skip: bool = False, glu: int = 0,
                 glu_context: int = 0, glu_glu: bool = True, activation: tp.Any = None) -> None:
        super().__init__()
        dilation = 1
        channels = tuple(channels)
        self.skip = skip
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()
        if activation is None:
            activation = partial(nn.LeakyReLU, leakiness)
        Conv = nn.Conv1d if not decode else nn.ConvTranspose1d
        # build layers
        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers: tp.List[nn.Module] = []
            is_last = k == len(channels) - 2

            # Set dropout for the input of the conv sequence if defined
            if k == 0 and dropout_input:
                assert 0 < dropout_input < 1
                layers.append(nn.Dropout(dropout_input))

            # conv layer
            if dilation_growth > 1:
                assert kernel % 2 != 0, "Supports only odd kernel with dilation for now"
            if dilation_period and (k % dilation_period) == 0:
                dilation = 1
            pad = kernel // 2 * dilation
            layers.append(Conv(chin, chout, kernel, stride, pad,
                               dilation=dilation, groups=groups if k > 0 else 1))
            dilation *= dilation_growth
            # non-linearity
            if activation_on_last or not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(num_features=chout))
                layers.append(activation())
                if dropout:
                    layers.append(nn.Dropout(dropout))
                if rewrite:
                    layers += [nn.Conv1d(chout, chout, 1), nn.LeakyReLU(leakiness)]
                    # layers += [nn.Conv1d(chout, 2 * chout, 1), nn.GLU(dim=1)]
            if chin == chout and skip:
                if scale is not None:
                    layers.append(LayerScale(chout, scale))
                if post_skip:
                    layers.append(Conv(chout, chout, 1, groups=chout, bias=False))

            self.sequence.append(nn.Sequential(*layers))
            if glu and (k + 1) % glu == 0:
                ch = 2 * chout if glu_glu else chout
                act = nn.GLU(dim=1) if glu_glu else activation()
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(chout, ch, 1 + 2 * glu_context, padding=glu_context), act))
            else:
                self.glus.append(None)

    def forward(self, x: tp.Any) -> tp.Any:
        for module_idx, module in enumerate(self.sequence):
            old_x = x
            x = module(x)
            if self.skip and x.shape == old_x.shape:
                x = x + old_x
            glu = self.glus[module_idx]
            if glu is not None:
                x = glu(x)
        return x

class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """
    def __init__(self, channels: int, init: float = 0.1, boost: float = 5.):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init / boost
        self.boost = boost

    def forward(self, x):
        return (self.boost * self.scale[:, None]) * x