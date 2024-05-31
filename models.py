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
from scipy.signal import hilbert

class MelSpectrum:
    def __init__(self, sample_rate: int, n_mels=20, n_fft=16, hop_length=None,
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
    def __init__(self, sample_rate: int, n_fft: int = 16, hop_length: int = None):
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
    def __init__(self, sample_rate: int, n_mfcc: int = 4, n_fft: int = 16, hop_length: int = None, n_mels: int = 16):
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
        self.window_length_sec = 0.03  # 10 ms window length
        self.n_fft = n_fft  # Calculate n_fft based on window length and sample rate
        self.hop_length = hop_length if hop_length is not None else max(1, self.n_fft // 2)  # Ensure at least 1
        self.n_mels = n_mels

        # Initialize the MFCC transform with the calculated parameters
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'n_mels': self.n_mels,
                'hop_length': self.hop_length,
                'win_length': self.n_fft,  # Optionally set win_length to n_fft
                'window_fn': torch.hamming_window
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

class DeltaDeltaMFCC:
    def __init__(self, sample_rate: int, n_mfcc: int = 4, n_fft: int = 16, hop_length: int = None, n_mels: int = 16):
        """
        Initializes the DeltaDeltaMFCC class with necessary parameters.

        Parameters:
        sample_rate (int): The sample rate of the audio files.
        n_mfcc (int): The number of MFCCs to return.
        n_fft (int): The number of points in the FFT, affects frequency resolution.
        hop_length (int): The number of samples between successive frames.
        n_mels (int): The number of Mel filterbanks.
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.window_length_sec = 0.03  # 30 ms window length
        self.n_fft = n_fft  # Calculate n_fft based on window length and sample rate
        self.hop_length = hop_length if hop_length is not None else max(1, self.n_fft // 2)  # Ensure at least 1
        self.n_mels = n_mels

        # Initialize the MFCC transform with the calculated parameters
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'n_mels': self.n_mels,
                'hop_length': self.hop_length,
                'win_length': self.n_fft,  # Optionally set win_length to n_fft
                'window_fn': torch.hamming_window
            }
        )

    def _compute(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Computes the Delta-Delta MFCCs of an audio tensor.

        Parameters:
        wav (torch.Tensor): A tensor of audio samples.

        Returns:
        torch.Tensor: The Delta-Delta MFCCs of the audio tensor.
        """
        if wav.dim() > 1:
            wav = torch.mean(wav, dim=0)  # Convert stereo to mono if necessary
        mfccs = self.mfcc_transform(wav)  # Compute MFCC
        deltas = torchaudio.functional.compute_deltas(mfccs)
        delta_deltas = torchaudio.functional.compute_deltas(deltas)
        return delta_deltas

    def get_feature(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Public method to get the Delta-Delta MFCC features of an audio tensor.

        Parameters:
        wav (torch.Tensor): A tensor of audio samples.

        Returns:
        torch.Tensor: The Delta-Delta MFCCs of the audio tensor.
        """
        return self._compute(wav)

class PhaseOfEnvelope:
    def __init__(self, sample_rate: int):
        """
        Initializes the PhaseOfEnvelope class with the necessary parameter.

        Parameters:
        sample_rate (int): The sample rate of the audio data.
        """
        self.sample_rate = sample_rate

    def _compute(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Computes the phase of the envelope of an audio waveform using the Hilbert transform.

        Parameters:
        wav (torch.Tensor): A 1D tensor of audio samples.

        Returns:
        torch.Tensor: The phase of the envelope.
        """
        wav_np = wav.numpy()  # Convert to NumPy array for Hilbert transform
        analytic_signal = hilbert(wav_np)  # Compute the analytic signal
        envelope_phase = np.angle(analytic_signal)  # Extract the phase
        return torch.from_numpy(envelope_phase)

    def get_feature(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Public method to get the phase of the envelope feature of an audio tensor.

        Parameters:
        wav (torch.Tensor): A tensor of audio samples.

        Returns:
        torch.Tensor: The phase of the envelope.
        """
        return self._compute(wav)
