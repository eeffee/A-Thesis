import mne
from dataPrep import Brennan2019Recording
import torch
from torch import nn, optim
import numpy as np
from models import MelSpectrum, PhaseSpectrum, MFCCSpectrum, SpeechEnvelope, PhaseOfEnvelope,DeltaDeltaMFCC
import os
import random
from sklearn.decomposition import PCA
from scipy.signal import resample, hilbert
from scipy.io import wavfile

torch.autograd.set_detect_anomaly(True)

configurations = {
    'eeg_conditions': ['raw', 'phase'],
    'frequency_bands': ['theta', 'gamma', 'delta', 'beta', 'speech'],
    'audio_features': {
        'PhaseOfEnvelope': PhaseOfEnvelope,
        'Mel': MelSpectrum,
        'Phase': PhaseSpectrum,
        'MFCC': MFCCSpectrum,
        'DeltaDeltaMFCC': DeltaDeltaMFCC,
        'Envelope': SpeechEnvelope
    },
    'output_paths': {
        'model': '/home/oztufan/resultsDCCA/models/',
        'training': '/home/oztufan/resultsDCCA/trainresults/',
        'evaluation': '/home/oztufan/resultsDCCA/evalresults/',
        'eeg' : '/home/oztufan/resultsDCCA/eeg/',
        'audio' : '/home/oztufan/resultsDCCA/audio/'
    },
    'audio_path': '/home/oztufan/bg257f92t/audio/AllStory.wav',
    'data_path': 'data/'
}

# Define constants and parameters
SAMPLE_RATE_AUDIO = 44100  # Audio sample rate
SAMPLE_RATE_EEG = 500  # EEG sample rate
SEGMENT_LENGTH_SEC = 30  # Length of each audio and EEG segment in seconds

def bandpass_filter(data, sfreq, l_freq, h_freq):
    # Create a filter for the band range [l_freq, h_freq]
    filter_design = mne.filter.create_filter(data, sfreq, l_freq, h_freq, method='iir')
    filtered_data = mne.filter.filter_data(data, sfreq, l_freq, h_freq, method='iir', iir_params=filter_design)
    return filtered_data

def clamp_data(tensor, min_value=-1e6, max_value=1e6, name=""):
    clamped_tensor = torch.clamp(tensor, min=min_value, max=max_value)
    check_data(clamped_tensor, name)
    return clamped_tensor

def extract_phase(data):
    analytic_signal = hilbert(data)
    phase_data = np.angle(analytic_signal)
    return phase_data

def compute_correlation_matrix(u, v):
    u_mean = torch.mean(u, dim=0)
    v_mean = torch.mean(v, dim=0)
    u_centered = u - u_mean  # Out-of-place operation
    v_centered = v - v_mean  # Out-of-place operation

    # Compute covariance matrices
    N = u.size(0)
    sigma_uu = torch.matmul(u_centered.T, u_centered) / (N - 1)
    sigma_vv = torch.matmul(v_centered.T, v_centered) / (N - 1)
    sigma_uv = torch.matmul(u_centered.T, v_centered) / (N - 1)


    # Compute the matrix product
    inv_sigma_uu = torch.linalg.inv(sigma_uu)
    inv_sigma_vv = torch.linalg.inv(sigma_vv)
    correlation_matrix = torch.matmul(inv_sigma_uu, torch.matmul(sigma_uv, torch.matmul(inv_sigma_vv, sigma_uv.T)))

    return correlation_matrix

def apply_pca_to_eeg_data(eeg_data, n_components=17):
    # Assuming original eeg_data shape is (channels, samples)
    pca = PCA(n_components=n_components)
    reshaped_data = eeg_data.T  # Shape becomes (samples, channels)
    transformed_data = pca.fit_transform(reshaped_data)
    # Transformed data should now be (samples, components)
    #print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    # Check for NaNs after PCA transformation
    if np.isnan(transformed_data).any():
        raise ValueError("PCA transformed data contains NaN values")
    return transformed_data

class EEGNet(nn.Module):
    def __init__(self, input_channels=17, transform_type=None): # After PCA, 17 channels obtained.
        super(EEGNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=3, padding=1),  # Convolution over time for each channel
            nn.BatchNorm1d(128),  # Batch normalization
            nn.LogSigmoid(),
            nn.MaxPool1d(2),  # Reduce dimensionality
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LogSigmoid(),
            nn.MaxPool1d(2),  # Further reduction by factor of 2
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LogSigmoid(),
            nn.MaxPool1d(2)
        )
        self._initialize_weights()  # Initialize weights


        # Calculate size after convolutions
        # After each MaxPool1d, the length is halved
        # Initial length: 15000
        # After first pool: 7500
        # After second pool: 3750
        # Calculate output size based on transform_type
        if transform_type == 'raw':
            output_size = 64 * 937  # The final sequence length after pooling is 937
        elif transform_type == 'phase':
            output_size = 64 * 937 # Adjust this based on the actual length post-transformations for phase
        self.fc_layers = nn.Sequential(
            nn.Linear(output_size, 128),
            nn.LogSigmoid(),
            nn.Linear(128, 32)
        )
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  # Xavier initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        x = torch.clamp(x, min=-1e6, max=1e6)  # Clamp input values
        if torch.isnan(x).any():
            raise ValueError("Input to EEGNet contains NaN values")

        x = self.conv_layers[0](x)
        if torch.isnan(x).any():
            raise ValueError("NaN values found after first conv layer in EEGNet")
        x = self.conv_layers[1](x)
        if torch.isnan(x).any():
            raise ValueError("NaN values found after first batch norm in EEGNet")
        x = self.conv_layers[2](x)
        if torch.isnan(x).any():
            raise ValueError("NaN values found after first activation in EEGNet")
        x = self.conv_layers[3](x)
        if torch.isnan(x).any():
            raise ValueError("NaN values found after first pooling in EEGNet")

        x = self.conv_layers[4](x)
        if torch.isnan(x).any():
            raise ValueError("NaN values found after second conv layer in EEGNet")
        x = self.conv_layers[5](x)
        if torch.isnan(x).any():
            raise ValueError("NaN values found after second batch norm in EEGNet")
        x = self.conv_layers[6](x)
        if torch.isnan(x).any():
            raise ValueError("NaN values found after second activation in EEGNet")
        x = self.conv_layers[7](x)
        if torch.isnan(x).any():
            raise ValueError("NaN values found after second pooling in EEGNet")

        x = self.conv_layers[8](x)
        if torch.isnan(x).any():
            raise ValueError("NaN values found after third conv layer in EEGNet")
        x = self.conv_layers[9](x)
        if torch.isnan(x).any():
            raise ValueError("NaN values found after third batch norm in EEGNet")
        x = self.conv_layers[10](x)
        if torch.isnan(x).any():
            raise ValueError("NaN values found after third activation in EEGNet")
        x = self.conv_layers[11](x)
        if torch.isnan(x).any():
            raise ValueError("NaN values found after third pooling in EEGNet")

        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        if torch.isnan(x).any():
            raise ValueError("NaN values found after fc_layers in EEGNet")

        return x


class EEGNet2D(nn.Module):
    def __init__(self, input_channels=1, input_height=17, input_width=7500, transform_type=None):
        super(EEGNet2D, self).__init__()
        self.transform_type = transform_type

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.LogSigmoid(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # Ensure stride <= kernel_size
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.LogSigmoid(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # Ensure stride <= kernel_size
            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.LogSigmoid(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))  # Ensure stride <= kernel_size
        )
        self._initialize_weights()

        height = input_height  # No change in height dimension for kernel_size=(1, 3)
        width = input_width // 8  # Three max pooling layers with pool size (1, 2)
        output_size = 128 * height * width
        self.fc_layers = nn.Sequential(
            nn.Linear(output_size, 128),
            nn.LogSigmoid(),
            nn.Linear(128, 32)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if torch.isnan(x).any():
            raise ValueError("Input to EEGNet2D contains NaN values")

        # Continue with the rest of the layers...
        x = self.conv_layers(x)
        if torch.isnan(x).any():
            raise ValueError(f"NaN values found after conv layer in EEGNet2D")

        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        if torch.isnan(x).any():
            raise ValueError("NaN values found after fc_layers in EEGNet2D")

        return x


def load_and_segment_eeg(subject_id, segment_length_sec, sample_rate_eeg, batch_size=256, frequency_band=None,
                         transform_type=None):
    # Initialize and load data using some EEG data loader
    recording = Brennan2019Recording(subject_uid=subject_id, recording_uid=None)
    eeg_raw = recording._load_raw()  # Load EEG data
    eeg_data = eeg_raw.get_data()
    eeg_data = clamp_data(torch.tensor(eeg_data), name="Raw EEG Data")

    sample_rate_eeg = 250
    if eeg_raw.info['sfreq'] != sample_rate_eeg:
        num_samples = eeg_data.shape[1]
        new_num_samples = int(num_samples * sample_rate_eeg / eeg_raw.info['sfreq'])
        eeg_data = resample(eeg_data, new_num_samples, axis=1)

    # Frequency bands
    freq_bands = {
        'gamma': (30, 100),
        'beta': (13, 30),
        'speech': (1, 10),
        'delta': (1, 4),
        'theta': (4, 8)
    }

    print(
        f"EEG data before PCA: mean={np.mean(eeg_data)}, std={np.std(eeg_data)}, min={np.min(eeg_data)}, max={np.max(eeg_data)}")

    # Filter EEG data if a frequency band is specified
    if frequency_band in freq_bands:
        l_freq, h_freq = freq_bands[frequency_band]
        eeg_data = bandpass_filter(eeg_data, eeg_raw.info['sfreq'], l_freq, h_freq)

    # Extract phase if requested
    if transform_type == 'phase':
        eeg_data = extract_phase(eeg_data)
        # Check for NaN and infinite values after extracting phase
    check_for_nan_inf(eeg_data, "Phase Extracted EEG Data")
    print(f"Phase Extracted EEG data: mean={np.mean(eeg_data)}, std={np.std(eeg_data)}, min={np.min(eeg_data)}, max={np.max(eeg_data)}")

    # Apply PCA
    eeg_data = apply_pca_to_eeg_data(eeg_data)

    eeg_data = clamp_data(torch.tensor(eeg_data), name="EEG Data Last")

    # Process and batch data
    def process_data(data, segment_length_sec, sample_rate_eeg, batch_size):
        data = data.T
        num_components, total_samples = data.shape
        samples_per_segment = int(segment_length_sec * sample_rate_eeg)
        segments = []

        for start in range(0, total_samples - samples_per_segment + 1, samples_per_segment):
            segment = data[:, start:start + samples_per_segment]
            segments.append(segment[np.newaxis, :, :])

        if not segments:
            return []

        segments_tensor = torch.tensor(np.concatenate(segments, axis=0), dtype=torch.float32)
        std = segments_tensor.std(dim=2, keepdim=True)
        std[std < 1e-6] = 1e-6  # Prevent division by zero or very small std
        segments_tensor = (segments_tensor - segments_tensor.mean(dim=2, keepdim=True)) / std
        check_data(segments_tensor, "Processed EEG Data")
        segments_tensor = segments_tensor.view(-1, 1, num_components, samples_per_segment)

        total_batches = (len(segments) + batch_size - 1) // batch_size
        batched_segments = [segments_tensor[i * batch_size:min((i + 1) * batch_size, len(segments_tensor))] for i in
                            range(total_batches)]

        return batched_segments

    segments = process_data(eeg_data, segment_length_sec, sample_rate_eeg, batch_size)

    return {
        'segments': segments
    }


class AudioFeatureNet(nn.Module):
    def __init__(self, feature_type):
        super(AudioFeatureNet, self).__init__()
        self.feature_type = feature_type

        # Define convolutional layer sizes based on feature type
        if feature_type == 'Mel':
            input_channels = 40  # Example, adjust as necessary
        elif feature_type == 'Phase':
            input_channels = 257
        elif feature_type in ['MFCC', 'DeltaDeltaMFCC']:
            input_channels = 13
        elif feature_type == 'Envelope':
            input_channels = 1
        elif feature_type == 'PhaseOfEnvelope':
            input_channels = 1

            # Configure convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.Sigmoid(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.MaxPool1d(2)
        )
        self._initialize_weights()  # Initialize weights

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  # Xavier initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if torch.isnan(x).any():
            raise ValueError("Input to AudioFeatureNet contains NaN values")
        x = self.conv_layers(x)
        if torch.isnan(x).any():
            raise ValueError("NaN values found after conv_layers in AudioFeatureNet")

        num_features = x.size(1) * x.size(2)
        fc_layers = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.Sigmoid(),
            nn.Linear(128, 32))
        x = x.view(x.size(0), -1)
        x = fc_layers(x)
        if torch.isnan(x).any():
            raise ValueError("NaN values found after fc_layers in AudioFeatureNet")
        return x


class AudioFeatureNet2D(nn.Module):
    def __init__(self, feature_type):
        super(AudioFeatureNet2D, self).__init__()
        self.feature_type = feature_type

        # Define convolutional layer sizes based on feature type
        if feature_type == 'Mel':
            input_channels = 1
            input_height, input_width = 40, 59
        elif feature_type == 'Phase':
            input_channels = 1
            input_height, input_width = 257, 59
        elif feature_type in ['MFCC', 'DeltaDeltaMFCC']:
            input_channels = 1
            input_height, input_width = 13, 8
        elif feature_type == 'Envelope':
            input_channels = 1
            input_height, input_width = 1, 3
        elif feature_type == 'PhaseOfEnvelope':
            input_channels = 1
            input_height, input_width = 1, 7500
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        # Separate handling for the Envelope feature type
        if feature_type == 'Envelope' or feature_type == 'PhaseOfEnvelope':
            self.conv_layers = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=(1, 1)),  # Use kernel size (1, 1) to avoid dimension issues
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=(1, 1)),  # Use kernel size (1, 1)
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=(1, 1)),  # Use kernel size (1, 1)
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
            height = input_height  # No change in height dimension for kernel_size=(1, 1)
            width = input_width
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d((2, 2))
            )
            # Calculate the size after convolutions and pooling
            height = input_height // 8  # Three max pooling layers with pool size (2, 2)
            width = input_width // 8

        output_size = 128 * height * width

        self.fc_layers = nn.Sequential(
            nn.Linear(output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if torch.isnan(x).any():
            raise ValueError("Input to AudioFeatureNet2D contains NaN values")

        x = self.conv_layers(x)

        if torch.isnan(x).any():
            raise ValueError("NaN values found after conv_layers in AudioFeatureNet2D")

        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        if torch.isnan(x).any():
            raise ValueError("NaN values found after fc_layers in AudioFeatureNet2D")

        return x




def segment_and_extract_features(audio_path, segment_length_sec, sample_rate_audio, feature_extractor, batch_size=256):
    # Load audio data
    sr, audio = wavfile.read(audio_path)
    sample_rate_audio = 250  # Set your desired sample rate
    if sr != sample_rate_audio:
        audio = resample(audio, int(len(audio) * sample_rate_audio / sr))

    samples_per_segment = int(segment_length_sec * sample_rate_audio)
    segments = [audio[i:i + samples_per_segment] for i in range(0, len(audio), samples_per_segment)]
    features = [feature_extractor.get_feature(torch.tensor(segment, dtype=torch.float32)) for segment in segments if len(segment) == samples_per_segment]

    if not features:
        return []  # Handle case where no features are extracted

    features_tensor = torch.stack(features)
    if len(features_tensor.shape) == 2:
        features_tensor = features_tensor.unsqueeze(1)  # Add a channel dimension

    # Adjust for 2D Convolutions
    features_tensor = features_tensor.view(-1, 1, features_tensor.shape[1], features_tensor.shape[2])
    features_tensor = (features_tensor - features_tensor.mean(dim=(2, 3), keepdim=True)) / features_tensor.std(dim=(2, 3), keepdim=True)

    batched_features = [features_tensor[i * batch_size:min((i + 1) * batch_size, len(features_tensor))] for i in range((len(features_tensor) + batch_size - 1) // batch_size)]

    return batched_features


class DCCALoss(nn.Module):
    def __init__(self):
        super(DCCALoss, self).__init__()

    def forward(self, u, v):
        # Mean centering
        u_mean = torch.mean(u, dim=0)
        v_mean = torch.mean(v, dim=0)
        u_centered = u - u_mean  # Out-of-place operation
        v_centered = v - v_mean  # Out-of-place operation

        # Compute covariance matrices
        N = u.size(0)
        sigma_uu = torch.matmul(u_centered.T, u_centered) / (N - 1)
        sigma_vv = torch.matmul(v_centered.T, v_centered) / (N - 1)
        sigma_uv = torch.matmul(u_centered.T, v_centered) / (N - 1)

        # Regularize covariances by adding small identity matrices
        d_u = sigma_uu.size(0)
        d_v = sigma_vv.size(0)
        identity_u = torch.eye(d_u, dtype=sigma_uu.dtype, device=sigma_uu.device)
        identity_v = torch.eye(d_v, dtype=sigma_vv.dtype, device=sigma_vv.device)
        sigma_uu = sigma_uu + 1e-4 * identity_u  # Out-of-place operation
        sigma_vv = sigma_vv + 1e-4 * identity_v  # Out-of-place operation

        # Compute the matrix product
        inv_sigma_uu = torch.linalg.inv(sigma_uu)
        inv_sigma_vv = torch.linalg.inv(sigma_vv)
        T = torch.matmul(inv_sigma_uu, torch.matmul(sigma_uv, torch.matmul(inv_sigma_vv, sigma_uv.T)))

        # Loss is the negative sum of diagonal elements of T
        loss = -torch.trace(T)
        return loss



def run_phase(subjects, audio_path, feature_extractor, eeg_net, audio_net, optimizer, loss_fn, phase, batch_size,
              frequency_band, eeg_condition):
    total_loss = 0.0
    num_batches = 0

    for subject_id in subjects:
        print(subject_id)
        eeg_batches = load_and_segment_eeg(subject_id, SEGMENT_LENGTH_SEC, SAMPLE_RATE_EEG, batch_size, frequency_band,
                                           eeg_condition)['segments']
        audio_batches = segment_and_extract_features(audio_path, SEGMENT_LENGTH_SEC, SAMPLE_RATE_AUDIO,
                                                     feature_extractor, batch_size)

        for eeg_batch, audio_batch in zip(eeg_batches, audio_batches):
            # Example usage before model forward pass
            check_data(eeg_batch, "EEG Batch")
            eeg_batch = (eeg_batch - eeg_batch.mean(dim=0, keepdim=True)) / eeg_batch.std(dim=0, keepdim=True)
            check_data(audio_batch, "Audio Batch")

            if phase == 'train' and optimizer:
                optimizer.zero_grad()

            eeg_features = eeg_net(eeg_batch)
            check_data(eeg_features, "EEG Features After Forward Pass")
            audio_features = audio_net(audio_batch)
            check_data(audio_features, "Audio Features After Forward Pass")
            loss = loss_fn(eeg_features, audio_features)
            if torch.isnan(loss):
                raise ValueError("Loss contains NaN values")
            if phase == 'train' and optimizer:
                loss.backward()
                #nn.utils.clip_grad_norm_(eeg_net.parameters(), max_norm=1.0)
                #nn.utils.clip_grad_norm_(audio_net.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    average_loss = total_loss / num_batches
    return average_loss


def check_data(tensor, name=""):
    if torch.isnan(tensor).any():
        raise ValueError(f"Data contains NaN values: {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Data contains Inf values: {name}")

def check_for_nan_inf(data, name=""):
    """ Check for NaN and infinite values in the data. """
    if np.isnan(data).any():
        raise ValueError(f"Data contains NaN values: {name}")
    if np.isinf(data).any():
        raise ValueError(f"Data contains infinite values: {name}")

def check_grads(eeg_net, audio_net):
    for name, param in eeg_net.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN gradients in EEGNet parameter: {name}")
            raise ValueError(f"NaN gradients in EEGNet parameter: {name}")
        if param.grad is not None:
            print(f"Gradient for {name}: mean={param.grad.mean().item()}, std={param.grad.std().item()}, min={param.grad.min().item()}, max={param.grad.max().item()}")

    for name, param in audio_net.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN gradients in AudioNet parameter: {name}")
            raise ValueError(f"NaN gradients in AudioNet parameter: {name}")
        if param.grad is not None:
            print(f"Gradient for {name}: mean={param.grad.mean().item()}, std={param.grad.std().item()}, min={param.grad.min().item()}, max={param.grad.max().item()}")


def train_model(eeg_net, audio_net, loss_fn, optimizer, train_subjects, valid_subjects, test_subjects, feature_extractor, condition, band, feature_name, audio_path, batch_size, num_epochs=13):

    print(f"Training for {condition} in {band} band using {feature_name} features.")
    training_results_path = os.path.join(configurations['output_paths']['training'],
                                         f"{condition}_{band}_{feature_name}_training_results.txt")
    os.makedirs(os.path.dirname(training_results_path), exist_ok=True)
    # Prepare to track losses for reporting
    train_losses = []
    validation_losses = []

    with open(training_results_path, 'w') as file:
        # Run training for a fixed number of epochs
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Training phase
            train_loss = run_phase(train_subjects, audio_path, feature_extractor, eeg_net, audio_net, optimizer,
                                   loss_fn, 'train', batch_size,band,condition)
            train_losses.append(train_loss)

            # Validation phase
            valid_loss = run_phase(valid_subjects, audio_path, feature_extractor, eeg_net, audio_net, None,
                                   loss_fn, 'validate', batch_size, band, condition)
            file.write(f'Epoch {epoch + 1}: Training Loss = {train_loss}, Validation Loss = {valid_loss}\n')
            validation_losses.append(valid_loss)

            # Optionally run a testing phase after the last training epoch
            if test_subjects:
                test_loss = run_phase(test_subjects, audio_path, feature_extractor, eeg_net, audio_net, None,
                                      loss_fn, 'test', batch_size, band, condition)
                file.write(f'Final Test Loss = {test_loss}\n')

            # Log average losses for this configuration
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_valid_loss = sum(validation_losses) / len(validation_losses)
            print(f"Average Training Loss for {condition} {band} {feature_name}: {avg_train_loss}")
            print(f"Average Validation Loss for {condition} {band} {feature_name}: {avg_valid_loss}")


def evaluate_model(configurations, condition, band, feature_name, subjects):
    output_base_path = configurations['output_paths']['evaluation']
    os.makedirs(output_base_path, exist_ok=True)
    eeg_model_path = os.path.join(configurations['output_paths']['eeg'],
                                  f"{condition}_{band}_{feature_name}_eeg_net.pth")
    audio_model_path = os.path.join(configurations['output_paths']['audio'],
                                    f"{condition}_{band}_{feature_name}_audio_net.pth")

    # Load the models
    eeg_net = EEGNet2D(input_channels=1, input_height=17, input_width=7500, transform_type=condition)
    audio_net = AudioFeatureNet2D(feature_type=feature_name)  # Ensure this aligns with your updated class definition
    eeg_net.load_state_dict(torch.load(eeg_model_path))
    audio_net.load_state_dict(torch.load(audio_model_path))
    eeg_net.eval()
    audio_net.eval()

    audio_path = configurations['audio_path']
    feature_extractor = configurations['audio_features'][feature_name](SAMPLE_RATE_AUDIO)

    for subject_id in subjects:
        # Load and process data for each subject
        eeg_batches = load_and_segment_eeg(subject_id, SEGMENT_LENGTH_SEC, SAMPLE_RATE_EEG, 256, band, condition)[
            'segments']
        audio_batches = segment_and_extract_features(audio_path, SEGMENT_LENGTH_SEC, SAMPLE_RATE_AUDIO,
                                                     feature_extractor, 256)

        # Accumulate and average features over all batches
        all_eeg_features, all_audio_features = [], []
        for eeg_batch, audio_batch in zip(eeg_batches, audio_batches):
            eeg_features = eeg_net(eeg_batch)
            check_tensor(eeg_features)
            audio_features = audio_net(audio_batch)
            check_tensor(audio_features)
            all_eeg_features.append(eeg_features)
            all_audio_features.append(audio_features)

        mean_eeg_features = torch.cat(all_eeg_features)
        mean_audio_features = torch.cat(all_audio_features)

        # Compute correlation matrix
        correlations = compute_correlation_matrix(mean_eeg_features, mean_audio_features)
        average_correlation = torch.mean(correlations).item()
        avg_diagonal = torch.mean(torch.diag(correlations)).item()

        # Save correlation results for each subject
        detailed_output_path = os.path.join(output_base_path,
                                            f"{condition}_{band}_{feature_name}_{subject_id}_correlation.npy")
        summary_output_path = os.path.join(output_base_path,
                                           f"{condition}_{band}_{feature_name}_{subject_id}_average_correlation.txt")
        np.save(detailed_output_path, correlations.detach().numpy())
        with open(summary_output_path, 'w') as f:
            f.write(f"Average Correlation: {average_correlation}\nDiagonal Average: {avg_diagonal}\n")


        print(f"Correlation output saved successfully for {condition} {band} {feature_name} with subject {subject_id}.")
        print(f"Average correlation ({average_correlation}) saved to {summary_output_path}.")


def save_model_parameters(eeg_net, audio_net, condition, band, feature_name, configurations):
    eeg_base_path = configurations['output_paths']['eeg']
    audio_base_path = configurations['output_paths']['audio']
    os.makedirs(eeg_base_path, exist_ok=True)
    os.makedirs(audio_base_path, exist_ok=True)
    eeg_model_path = os.path.join(eeg_base_path, f"{condition}_{band}_{feature_name}_eeg_net.pth")
    audio_model_path = os.path.join(audio_base_path, f"{condition}_{band}_{feature_name}_audio_net.pth")
    torch.save(eeg_net.state_dict(), eeg_model_path)
    torch.save(audio_net.state_dict(), audio_model_path)
    print(f"EEG model parameters saved to {eeg_model_path}")
    print(f"Audio model parameters saved to {audio_model_path}")

def ensure_directories_exist(configurations):
    for path in configurations['output_paths'].values():
        os.makedirs(path, exist_ok=True)

def check_tensor(tensor):
    if torch.isnan(tensor).any():
        raise ValueError("Tensor contains NaN values")

def run_full_analysis(configurations):
    # Set random seed for reproducibility, manage subject lists, etc.
    random.seed(42)
    torch.manual_seed(42)

    # Call this function at the start of your `run_full_analysis`
    ensure_directories_exist(configurations)

    subjects = ['S01', 'S03', 'S04', 'S05', 'S06', 'S08', 'S10', 'S11', 'S12',
                'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22',
                'S25', 'S26', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40', 'S41', 'S42']
    bads = ['S02','S07', 'S09', 'S23', 'S24', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33',
            'S43', 'S46', 'S47', 'S49', 'S44', 'S45', 'S48']  # 'S44', 'S45', 'S48'  not bad use for evaluation
    subjects = [s for s in subjects if s not in bads]
    random.shuffle(subjects)
    train_subjects = subjects[:23]
    valid_subjects = subjects[23:28]
    test_subjects = subjects[28:]
    subjects_to_evaluate = ['S44', 'S45', 'S48']

    # Loop over conditions, bands, and feature types
    for condition in configurations['eeg_conditions']:
        for band in configurations['frequency_bands']:
            for feature_name, feature_class in configurations['audio_features'].items():
                # Initialize networks and optimizer
                eeg_net = EEGNet2D(input_channels=1, input_height=17, input_width=7500, transform_type=condition)
                audio_net = AudioFeatureNet2D(feature_type=feature_name)
                loss_fn = DCCALoss()
                optimizer = optim.Adam(list(eeg_net.parameters()) + list(audio_net.parameters()), lr=0.05)

                feature_extractor = feature_class(SAMPLE_RATE_AUDIO)
                audio_path = configurations['audio_path']

                # Train model
                train_model(eeg_net, audio_net, loss_fn, optimizer, train_subjects, valid_subjects, test_subjects, feature_extractor, condition, band, feature_name, audio_path, 256, 1)

                # Save models
                save_model_parameters(eeg_net,audio_net,condition,band,feature_name,configurations)

                # Evaluate on specific subjects
                evaluate_model(configurations, condition, band, feature_name, subjects_to_evaluate)

def main():
    print(ensure_directories_exist(configurations))
    run_full_analysis(configurations)


if __name__ == "__main__":
    main()


