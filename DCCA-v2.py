import mne
from Codes.dataPrep import Brennan2019Recording
import librosa
import torch
from torch import nn, optim
import numpy as np
from Codes.models import MelSpectrum, PhaseSpectrum, MFCCSpectrum, SpeechEnvelope, PhaseOfEnvelope,DeltaDeltaMFCC
import os
import random
from sklearn.decomposition import PCA
from scipy.signal import resample, hilbert
import torchaudio
from torchaudio.transforms import Resample
from scipy.io import wavfile

configurations = {
    'eeg_conditions': ['phase', 'raw'],
    'frequency_bands': ['theta', 'gamma', 'delta', 'beta', 'speech'],
    'audio_features': {
        'Mel': MelSpectrum,
        'Phase': PhaseSpectrum,
        'MFCC': MFCCSpectrum,
        'DeltaDeltaMFCC': DeltaDeltaMFCC,
        'Envelope': SpeechEnvelope,
        'PhaseOfEnvelope': PhaseOfEnvelope,
    },
    'output_paths': {
        'model': '/Users/efeoztufan/Desktop/A-Thesis/test/models/',
        'training': '/Users/efeoztufan/Desktop/A-Thesis/test/trainresults/',
        'evaluation': '/Users/efeoztufan/Desktop/A-Thesis/test/evalresults/'
    },
    'audio_path': '/Users/efeoztufan/Desktop/A-Thesis/Datasets/bg257f92t/audio/AllStory.wav',
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

def extract_phase(data):
    analytic_signal = hilbert(data)
    phase_data = np.angle(analytic_signal)
    return phase_data

def compute_correlation_matrix(H1, H2, reg_lambda=1e-4):
    # Subtract mean
    H1 -= H1.mean(0, keepdim=True)
    H2 -= H2.mean(0, keepdim=True)

    # Clipping to prevent numerical instability
    H1 = torch.clamp(H1, min=-1e5, max=1e5)
    H2 = torch.clamp(H2, min=-1e5, max=1e5)

    # Compute covariance matrices
    cov_H1 = torch.matmul(H1.T, H1) / (H1.size(0) - 1)
    cov_H2 = torch.matmul(H2.T, H2) / (H2.size(0) - 1)
    cov_H1H2 = torch.matmul(H1.T, H2) / (H1.size(0) - 1)

    # Regularization
    cov_H1 += reg_lambda * torch.eye(cov_H1.size(0), device=cov_H1.device)
    cov_H2 += reg_lambda * torch.eye(cov_H2.size(0), device=cov_H2.device)

    # Check for numerical stability
    #if not torch.isfinite(cov_H1).all() or not torch.isfinite(cov_H2).all():
        #raise RuntimeError("Covariance matrices contain infs or NaNs after regularization")

    # Compute inverse covariance matrices
    inv_cov_H1 = torch.linalg.inv(cov_H1)
    inv_cov_H2 = torch.linalg.inv(cov_H2)

    # Compute matrix T
    T = torch.matmul(torch.matmul(torch.matmul(inv_cov_H1, cov_H1H2), inv_cov_H2), cov_H1H2.T)

    # Check for numerical stability
    #if not torch.isfinite(T).all():
        #raise RuntimeError("Matrix T contains infs or NaNs before eigenvalue decomposition")

    # Eigenvalue decomposition
    eigenvalues, _ = torch.linalg.eigh(T, UPLO='L')

    # Ensure eigenvalues are real and non-negative
    real_eigenvalues = torch.clamp(eigenvalues, min=0)

    # Sort eigenvalues in descending order and compute correlations
    correlations = torch.sqrt(real_eigenvalues).sort(descending=True)[0]

    # Check for numerical stability
    #if not torch.isfinite(correlations).all():
        #raise RuntimeError("Eigenvalues contain infs or NaNs")

    return correlations

def apply_pca_to_eeg_data(eeg_data, n_components=17):
    # Assuming original eeg_data shape is (channels, samples)
    pca = PCA(n_components=n_components)
    reshaped_data = eeg_data.T  # Shape becomes (samples, channels)
    transformed_data = pca.fit_transform(reshaped_data)
    # Transformed data should now be (samples, components)
    return transformed_data

class EEGNet(nn.Module):
    def __init__(self, input_channels=17, transform_type=None): # After PCA, 17 channels obtained.
        super(EEGNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=3, padding=1),  # Convolution over time for each channel
            nn.BatchNorm1d(128),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool1d(2),  # Reduce dimensionality
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # Further reduction by factor of 4
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # Calculate size after convolutions
        # After each MaxPool1d, the length is halved
        # Initial length: 15000
        # After first pool: 7500
        # After second pool: 3750
        # Calculate output size based on transform_type
        print(transform_type)
        if transform_type == 'raw':
            output_size = 64 * 937  # The final sequence length after pooling is 937
        elif transform_type == 'phase':
            output_size = 120000  # Adjust this based on the actual length post-transformations for phase
        self.fc_layers = nn.Sequential(
            nn.Linear(output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )


    def forward(self, x):
        # Ensure the tensor is properly shaped
        if x.dim() == 4 and x.shape[1] == 1:  # Assuming unnecessary singleton dimension present
            x = x.squeeze(1)  # Remove the singleton dimension
        # Now unpack the dimensions
        batch_size, num_channels, L = x.shape
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc_layers(x)
        #x = x.view(batch_size, -1).mean(dim=1)  # Average the outputs across batches if needed
        return x


#transform_type = can be raw or phase, frequency bands are theta, gamma, delta, speech
def load_and_segment_eeg(subject_id, segment_length_sec, sample_rate_eeg, batch_size=256, frequency_band=None, transform_type=None):
    # Initialize and load data using some EEG data loader
    recording = Brennan2019Recording(subject_uid=subject_id, recording_uid=None)
    eeg_raw = recording._load_raw()  # Load EEG data
    eeg_data = eeg_raw.get_data()

    # Frequency bands
    freq_bands = {
        'gamma': (30, 100),
        'beta': (13, 30),
        'speech': (1, 10),
        'delta': (1, 4),
        'theta': (4, 8)
    }

    # Filter EEG data if a frequency band is specified
    if frequency_band in freq_bands:
        l_freq, h_freq = freq_bands[frequency_band]
        eeg_data = bandpass_filter(eeg_data, eeg_raw.info['sfreq'], l_freq, h_freq)

    # Extract phase if requested
    if transform_type == 'phase':
        eeg_data = extract_phase(eeg_data)

    # Apply PCA
    eeg_data = apply_pca_to_eeg_data(eeg_data)

    sample_rate_eeg = 250
    # Resample post-PCA if needed
    if eeg_raw.info['sfreq'] != sample_rate_eeg:
        num_samples = eeg_data.shape[0]
        new_num_samples = int(num_samples * sample_rate_eeg / eeg_raw.info['sfreq'])
        eeg_data = resample(eeg_data, new_num_samples, axis=0)

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
        segments_tensor = (segments_tensor - segments_tensor.mean(dim=2, keepdim=True)) / segments_tensor.std(dim=2, keepdim=True)
        total_batches = (len(segments) + batch_size - 1) // batch_size
        batched_segments = [segments_tensor[i * batch_size:min((i + 1) * batch_size, len(segments_tensor))] for i in range(total_batches)]

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
            #post_conv_size = 64 * 7  # Adjust based on the output size after conv layers
            post_conv_size = 64 * 1292
        elif feature_type == 'Phase':
            input_channels = 257
            #post_conv_size = 64 * 7
            post_conv_size = 64 * 1292
        elif feature_type in ['MFCC', 'DeltaDeltaMFCC']:
            input_channels = 13
            #post_conv_size = 64
            post_conv_size= 64 * 161
        elif feature_type == 'Envelope':
            input_channels = 1
            #post_conv_size = 64
            post_conv_size = 128
        elif feature_type == 'PhaseOfEnvelope':
            input_channels = 1
            #post_conv_size = 60032  # Example, adjust based on actual size
            post_conv_size = 165375 * 64


            # Configure convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Configure fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(post_conv_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

    def forward(self, x):
        print("Initial shape:", x.shape)
        x = self.conv_layers[0:3](x)  # Up to first pooling
        if x.shape[-1] > 2:
            x = self.conv_layers[3:6](x)  # Next conv and pooling
        if x.shape[-1] > 2:
            x = self.conv_layers[6:](x)  # Final layers
        print("Post-conv shape:", x.shape)
        x = x.view(x.size(0), -1)
        print("Flattened shape:", x.shape)
        x = self.fc_layers(x)
        return x

def segment_and_extract_features(audio_path, segment_length_sec, sample_rate_audio, feature_extractor, batch_size=256):

    # Load audio data
    sr, audio = wavfile.read(audio_path)
    # Check if the sampling frequency is not equal to 250 Hz
    sample_rate_audio = 250
    # Resample post-PCA if needed
    if sr != 250:
        num_samples = audio.shape[0]
        new_num_samples = int(num_samples * sample_rate_audio / sr)
        audio = resample(audio, new_num_samples, axis=0)
    samples_per_segment = int(segment_length_sec * sr)

    segments = [audio[i:i + samples_per_segment] for i in range(0, len(audio), samples_per_segment)]
    features = [feature_extractor.get_feature(torch.tensor(segment, dtype=torch.float32)) for segment in segments if
                len(segment) == samples_per_segment]

    if not features:
        return []  # Handle case where no features are extracted

    features_tensor = torch.stack(features)
    if len(features_tensor.shape) == 2:  # If there are only two dimensions
        features_tensor = features_tensor.unsqueeze(1)  # Add a channel dimension

    # Ensure the tensor shape is compatible with what your model expects
    features_tensor = features_tensor.view(-1, features_tensor.shape[1], features_tensor.shape[2])

    # Normalize the features
    features_tensor = (features_tensor - features_tensor.mean(dim=2, keepdim=True)) / features_tensor.std(dim=2,
                                                                                                          keepdim=True)

    # Allow for the last batch to be smaller than batch_size
    batched_features = [features_tensor[i * batch_size:min((i + 1) * batch_size, len(features_tensor))]
                        for i in range((len(features_tensor) + batch_size - 1) // batch_size)]

    return batched_features

class DCCALoss(torch.nn.Module):
    def __init__(self, use_all_singular_values=False, outdim_size=1):
        super(DCCALoss, self).__init__()
        self.device = 'gpu'
        self.use_all_singular_values = use_all_singular_values
        self.outdim_size = outdim_size

    def forward(self, H1, H2):
        r1 = 1e-4
        r2 = 1e-4
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()
        o1 = o2 = H1.size(0)
        m = H1.size(1)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar, H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.t()) + r2 * torch.eye(o2, device=self.device)

        # Using torch.linalg.eigh instead of torch.symeig
        D1, V1 = torch.linalg.eigh(SigmaHat11, UPLO='L')
        D2, V2 = torch.linalg.eigh(SigmaHat22, UPLO='L')

        posInd1 = torch.gt(D1, eps).nonzero(as_tuple=True)[0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]

        posInd2 = torch.gt(D2, eps).nonzero(as_tuple=True)[0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = torch.matmul(torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

        if self.use_all_singular_values:
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
        else:
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, torch.eye(trace_TT.shape[0], device=self.device) * r1)
            U, V = torch.linalg.eigh(trace_TT, UPLO='L')
            U = torch.where(U > eps, U, torch.ones(U.shape, device=self.device) * eps)
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))

        return -corr

def run_phase(subjects, audio_path, feature_extractor, eeg_net, audio_net, optimizer, loss_fn, phase, batch_size,frequency_band,eeg_condition):
    total_loss = 0.0
    num_batches = 0

    for subject_id in subjects:
        eeg_batches = load_and_segment_eeg(subject_id, SEGMENT_LENGTH_SEC, SAMPLE_RATE_EEG, batch_size, frequency_band, eeg_condition)['segments']
        audio_batches = segment_and_extract_features(audio_path, SEGMENT_LENGTH_SEC, SAMPLE_RATE_AUDIO, feature_extractor, batch_size)

        for eeg_batch, audio_batch in zip(eeg_batches, audio_batches):
            if phase == 'train' and optimizer:
                optimizer.zero_grad()

            eeg_features = eeg_net(eeg_batch)
            audio_features = audio_net(audio_batch)
            loss = loss_fn(eeg_features, audio_features)

            if phase == 'train' and optimizer:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    average_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return average_loss

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

            # Validation phase
            valid_loss = run_phase(valid_subjects, audio_path, feature_extractor, eeg_net, audio_net, None,
                                   loss_fn, 'validate', batch_size, band, condition)
            file.write(f'Epoch {epoch + 1}: Training Loss = {train_loss}, Validation Loss = {valid_loss}\n')


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
    # Load the models
    eeg_net = EEGNet(transform_type=condition)
    audio_net = AudioFeatureNet(feature_type=feature_name)  # ensure this aligns with your updated class definition
    eeg_net.load_state_dict(torch.load(configurations['model_paths']['eeg']))
    audio_net.load_state_dict(torch.load(configurations['model_paths']['audio']))
    eeg_net.eval()
    audio_net.eval()

    audio_path = configurations['audio_path']
    feature_extractor = configurations['audio_features'][feature_name](SAMPLE_RATE_AUDIO)

    for subject_id in subjects:
        # Load and process data for each subject
        eeg_batches = load_and_segment_eeg(subject_id, SEGMENT_LENGTH_SEC, SAMPLE_RATE_EEG, 256, band, condition)['segments']
        audio_batches = segment_and_extract_features(audio_path, SEGMENT_LENGTH_SEC, SAMPLE_RATE_AUDIO, feature_extractor, 256)

        # Accumulate and average features over all batches
        all_eeg_features, all_audio_features = [], []
        for eeg_batch, audio_batch in zip(eeg_batches, audio_batches):
            all_eeg_features.append(eeg_net(eeg_batch))
            all_audio_features.append(audio_net(audio_batch))
        mean_eeg_features = torch.cat(all_eeg_features).mean(dim=0)
        mean_audio_features = torch.cat(all_audio_features).mean(dim=0)

        # Compute correlation matrix
        correlations = compute_correlation_matrix(mean_eeg_features, mean_audio_features)
        average_correlation = torch.mean(correlations).item()

        # Save correlation results for each subject
        detailed_output_path = os.path.join(output_base_path,
                                            f"{condition}_{band}_{feature_name}_{subject_id}_correlation.npy")
        summary_output_path = os.path.join(output_base_path,
                                           f"{condition}_{band}_{feature_name}_{subject_id}_average_correlation.txt")
        np.save(detailed_output_path, correlations.numpy())
        with open(summary_output_path, 'w') as f:
            f.write(f"Average Correlation: {average_correlation}\n")

        print(f"Correlation output saved successfully for {condition} {band} {feature_name} with subject {subject_id}.")
        print(f"Average correlation ({average_correlation}) saved to {summary_output_path}.")

def save_model_parameters(eeg_net, audio_net, condition, band, feature_name, configurations):
    model_base_path = configurations['output_paths']['model']
    os.makedirs(model_base_path, exist_ok=True)
    eeg_model_path = os.path.join(model_base_path, f"{condition}_{band}_{feature_name}_eeg_net.pth")
    audio_model_path = os.path.join(model_base_path, f"{condition}_{band}_{feature_name}_audio_net.pth")
    torch.save(eeg_net.state_dict(), eeg_model_path)
    torch.save(audio_net.state_dict(), audio_model_path)
    print(f"EEG model parameters saved to {eeg_model_path}")
    print(f"Audio model parameters saved to {audio_model_path}")

def ensure_directories_exist(configurations):
    for path in configurations['output_paths'].values():
        os.makedirs(path, exist_ok=True)


def run_full_analysis(configurations):
    # Set random seed for reproducibility, manage subject lists, etc.
    random.seed(42)
    torch.manual_seed(42)

    # Call this function at the start of your `run_full_analysis`
    ensure_directories_exist(configurations)

    subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12',
                'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24',
                'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36',
                'S37', 'S38', 'S39', 'S40', 'S41', 'S42']
    bads = ["S24", "S26", "S27", "S30", "S32", "S34", "S35", "S36", "S02", "S25", "S07"]  # S25,S07 not bad use later
    subjects = [s for s in subjects if s not in bads]
    random.shuffle(subjects)
    train_subjects = subjects[:23]
    valid_subjects = subjects[23:28]
    test_subjects = subjects[28:]
    subjects_to_evaluate = ['S25', 'S07']

    # Loop over conditions, bands, and feature types
    for condition in configurations['eeg_conditions']:
        for band in configurations['frequency_bands']:
            for feature_name, feature_class in configurations['audio_features'].items():
                # Initialize networks and optimizer
                eeg_net = EEGNet(input_channels=17, transform_type=condition)
                audio_net = AudioFeatureNet(feature_type=feature_name)  # Adjust this to correctly initialize
                loss_fn = DCCALoss(use_all_singular_values=True, outdim_size=1)
                optimizer = optim.Adam(list(eeg_net.parameters()) + list(audio_net.parameters()), lr=0.001)

                feature_extractor = feature_class(SAMPLE_RATE_AUDIO)
                audio_path = configurations['audio_path']

                # Train model
                train_model(eeg_net, audio_net, loss_fn, optimizer, train_subjects, valid_subjects, test_subjects, feature_extractor, condition, band, feature_name, audio_path, 256, 13)

                # Save models
                torch.save(eeg_net.state_dict(), configurations['model_paths']['eeg'])
                torch.save(audio_net.state_dict(), configurations['model_paths']['audio'])
                print("Models saved successfully for", condition, band, feature_name)

                # Evaluate on specific subjects
                evaluate_model(configurations, condition, band, feature_name, subjects_to_evaluate)

def main():
    print(ensure_directories_exist(configurations))
    run_full_analysis(configurations)


if __name__ == "__main__":
    main()


