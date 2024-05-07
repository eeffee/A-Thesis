import mne

from Codes.dataPrep import Brennan2019Recording
import librosa
import torch
from torch import nn, optim
import numpy as np
from Codes.models import MelSpectrum, PhaseSpectrum, MFCCSpectrum, SpeechEnvelope
import os
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import resample



def segment_audio(audio_path, segment_length=30, sr=44100):
    audio, sr = librosa.load(audio_path, sr=sr)
    for start in range(0, len(audio), segment_length * sr):
        yield audio[start:start + segment_length * sr]


class EEGNet(nn.Module):
    def __init__(self, input_channels=17):
        super(EEGNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=3, padding=1),  # Convolution over time for each channel
            nn.BatchNorm1d(128),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool1d(2),  # Reduce dimensionality
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool1d(2),  # Further reduction by factor of 4
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # Calculate size after convolutions
        # After each MaxPool1d, the length is halved
        # Initial length: 15000
        # After first pool: 7500
        # After second pool: 3750
        output_size = 1875
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * output_size, 128),  # Correctly calculate the flattened size
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
        x = x.view(batch_size, -1).mean(dim=1)  # Average the outputs across batches if needed
        return x


def apply_pca_to_eeg_data(eeg_data, n_components=17):
    # Assuming original eeg_data shape is (channels, samples)
    pca = PCA(n_components=n_components)
    reshaped_data = eeg_data.T  # Shape becomes (samples, channels)
    transformed_data = pca.fit_transform(reshaped_data)
    # Transformed data should now be (samples, components)
    return transformed_data




class AudioFeatureNet(nn.Module):
    def __init__(self):
        super(AudioFeatureNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(40, 64, kernel_size=3, stride=1, padding=1),  # for Mel Spectrogram feature
            #nn.Conv1d(257, 64, kernel_size=3, stride=1, padding=1),  # for Phase feature
            #nn.Conv1d(13, 64, kernel_size=3, stride=1, padding=1),  # for MFCC feature
            #nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=4), # for Speech envelope
            nn.BatchNorm1d(64),  # Batch normalization
            nn.ReLU(),
            #nn.MaxPool1d(4),  # Reduces size by factor of 4
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool1d(2),  # Further reduction by factor of 4
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool1d(2)  # Final reduction
        )
        # Assume initial length = 10336 (make sure this is your actual initial length)
        final_length = 10336 // 2 // 2 // 2  # Adjust based on the actual size after pooling
        self.fc_layers = nn.Sequential(
            #nn.Linear(128 * final_length, 128),  # Ensure 256 * final_length is correct
            nn.Linear(64 * 14, 128), #for phase, for Mel
            #nn.Linear(2560, 128),  # for MFCC
            #nn.Linear(128, 128), #for speech envelope
            #nn.Linear(64, 128), #for speech envelope
            nn.ReLU(),
            nn.Linear(128, 32)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten for FC
        x = self.fc_layers(x)
        return x


"""class DCCALoss(nn.Module):
    def forward(self, H1, H2):
        if H1.size(0) != H2.size(0):
            raise ValueError(f"Batch sizes do not match: {H1.size(0)} != {H2.size(0)}")



        H1 -= H1.mean(0, keepdim=True)
        H2 -= H2.mean(0, keepdim=True)
        H1 = H1 / H1.norm(dim=1, keepdim=True)  # Normalize each sample in batch
        H2 = H2 / H2.norm(dim=1, keepdim=True)  # Normalize each sample in batch
        H1 = torch.clamp(H1, min=-1e5, max=1e5)
        H2 = torch.clamp(H2, min=-1e5, max=1e5)

        cov_H1 = H1.T @ H1 / (H1.size(0) - 1)
        cov_H2 = H2.T @ H2 / (H2.size(0) - 1)
        cov_H1H2 = H1.T @ H2 / (H1.size(0) - 1)

        reg_lambda = 1e-3
        cov_H1 += reg_lambda * torch.eye(cov_H1.size(0), device=cov_H1.device)
        cov_H2 += reg_lambda * torch.eye(cov_H2.size(0), device=cov_H2.device)

        if not torch.isfinite(cov_H1).all() or not torch.isfinite(cov_H2).all():
            raise RuntimeError("Covariance matrices contain infs or NaNs after regularization")

        inv_cov_H1 = torch.linalg.inv(cov_H1)
        inv_cov_H2 = torch.linalg.inv(cov_H2)
        T = inv_cov_H1 @ cov_H1H2 @ inv_cov_H2 @ cov_H1H2.T

        if not torch.isfinite(T).all():
            raise RuntimeError("Matrix T contains infs or NaNs before eigenvalue decomposition")

        eigenvalues = torch.linalg.eigvals(T).real

        if not torch.isfinite(eigenvalues).all():
            raise RuntimeError("Eigenvalues contain infs or NaNs")

        # Safeguard against negative eigenvalues
        eigenvalues = torch.clamp(eigenvalues, min=0)

        corr = torch.sqrt(eigenvalues).sort(descending=True)[0]
        return -corr.sum()"""


class DCCALoss(torch.nn.Module):
    def __init__(self, use_all_singular_values=False, outdim_size=1):
        super(DCCALoss, self).__init__()
        self.device = 'cpu'
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


def segment_and_extract_features(audio_path, segment_length_sec, sample_rate_audio, feature_extractor, batch_size=256):
    audio, sr = librosa.load(audio_path, sr=sample_rate_audio)
    if sr != 125:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=125)
        sr = 125

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



def bandpass_filter(data, sfreq, l_freq, h_freq):
    # Create a filter for the band range [l_freq, h_freq]
    filter_design = mne.filter.create_filter(data, sfreq, l_freq, h_freq, method='iir')
    filtered_data = mne.filter.filter_data(data, sfreq, l_freq, h_freq, method='iir', iir_params=filter_design)
    return filtered_data


def load_and_segment_eeg(subject_id, segment_length_sec, sample_rate_eeg, batch_size=256):
    # Instantiate and load data using Brennan2019Recording
    recording = Brennan2019Recording(subject_uid=subject_id, recording_uid=None)
    eeg_raw = recording._load_raw()  # Load EEG data

    # Extract data as a NumPy array from the MNE Raw object
    eeg_data = eeg_raw.get_data()

    # Apply PCA
    eeg_data = apply_pca_to_eeg_data(eeg_data)
    print(f"After PCA, data shape: {eeg_data.shape}")

    # Correct resampling post-PCA
    if eeg_raw.info['sfreq'] != 125:
        num_samples = eeg_data.shape[0]  # Total number of samples available
        new_num_samples = int(num_samples * 125 / eeg_raw.info['sfreq'])  # Calculate new number of samples
        eeg_data = resample(eeg_data, new_num_samples, axis=0)  # Resample along the samples dimension
        sample_rate_eeg = 250
        print(f"After resampling, data shape: {eeg_data.shape}")  # Should show (new_num_samples, 17)

    def process_data(data, segment_length_sec, sample_rate_eeg, batch_size):
        # Data should be shaped as (num_components, total_samples)
        data = data.T
        num_components, total_samples = data.shape

        samples_per_segment = int(segment_length_sec * sample_rate_eeg)
        print(f"Total samples available: {total_samples}")
        print(f"Samples required per segment: {samples_per_segment}")

        if samples_per_segment > total_samples:
            print("Error: Not enough samples to form a segment.")
            return []

        segments = []
        for start in range(0, total_samples - samples_per_segment + 1, samples_per_segment):
            # Take a slice for each segment and ensure components are treated as channels
            segment = data[:, start:start + samples_per_segment]
            segments.append(segment[np.newaxis, :, :])  # Adds a batch dimension

        if not segments:
            print("No segments were created. Returning empty list.")
            return []

        # Concatenate segments and convert to tensor
        segments_tensor = torch.tensor(np.concatenate(segments, axis=0), dtype=torch.float32)
        segments_tensor = (segments_tensor - segments_tensor.mean(dim=2, keepdim=True)) / segments_tensor.std(dim=2,
                                                                                                              keepdim=True)

        # Ensure each batch has the correct size, handling the last batch separately if it's smaller
        total_batches = (len(segments) + batch_size - 1) // batch_size
        batched_segments = [segments_tensor[i * batch_size:min((i + 1) * batch_size, len(segments_tensor))]
                            for i in range(total_batches)]

        return batched_segments

    # Filter the data for different bands and process each
    #gamma_data = bandpass_filter(eeg_data, sample_rate_eeg, 30, 100)  # Gamma: 30-100 Hz
    #beta_data = bandpass_filter(eeg_data, sample_rate_eeg, 13, 30)  # Beta: 13-30 Hz
    #alpha_data = bandpass_filter(eeg_data, sample_rate_eeg, 8, 12)  # Alpha: 8-12 Hz
    #delta_data = bandpass_filter(eeg_data, sample_rate_eeg, 1, 4)  # Delta: 1-4 Hz
    theta_data = bandpass_filter(eeg_data, sample_rate_eeg, 4, 8)  # Theta: 4-8 Hz

    # Process and batch each frequency band's data
    #raw_segments = process_data(eeg_data, segment_length_sec, sample_rate_eeg, batch_size)
    #gamma_segments = process_data(gamma_data, segment_length_sec, sample_rate_eeg, batch_size)
    #beta_segments = process_data(beta_data, segment_length_sec, sample_rate_eeg, batch_size)
    #alpha_segments = process_data(alpha_data, segment_length_sec, sample_rate_eeg, batch_size)
    #delta_segments = process_data(delta_data, segment_length_sec, sample_rate_eeg, batch_size)
    theta_segments = process_data(theta_data, segment_length_sec, sample_rate_eeg, batch_size)



    return {
        "raw": None,
        "gamma":None,
        "beta": None,
        "alpha": None,
        "delta": None,
        "theta": theta_segments,
    }



def load_model(model_path, model_class, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


def check_tensor(x, name="Tensor"):
    if torch.isnan(x).any() or torch.isinf(x).any():
        raise ValueError(f"{name} contains NaNs or Infs.")


def run_phase(subjects, audio_path, feature_extractor, eeg_net, audio_net, optimizer, loss_fn, phase='train', batch_size=256):
    total_loss = 0
    num_batches = 0

    for subject_id in subjects:
        eeg_batches = load_and_segment_eeg(subject_id, SEGMENT_LENGTH_SEC, SAMPLE_RATE_EEG, batch_size)
        eeg_batches = eeg_batches['theta']
        audio_batches = segment_and_extract_features(audio_path, SEGMENT_LENGTH_SEC, SAMPLE_RATE_AUDIO,
                                                     feature_extractor, batch_size)

        for eeg_batch, audio_batch in zip(eeg_batches, audio_batches):
            if phase == 'train' and optimizer:
                optimizer.zero_grad()

            # Normalizing EEG and Audio features
            print(f"Shape of eeg_batch: {eeg_batch.shape}")
            eeg_features = eeg_net(eeg_batch)
            eeg_features = (eeg_features - eeg_features.mean(dim=1, keepdim=True)) / eeg_features.std(dim=1,
                                                                                                      keepdim=True)

            audio_features = audio_net(audio_batch)
            audio_features = (audio_features - audio_features.mean(dim=1, keepdim=True)) / audio_features.std(
                dim=1, keepdim=True)

            loss = loss_fn(eeg_features, audio_features)
            if phase == 'train' and optimizer:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    average_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return average_loss


# Define constants and parameters
SAMPLE_RATE_AUDIO = 44100  # Audio sample rate
SAMPLE_RATE_EEG = 500  # EEG sample rate
SEGMENT_LENGTH_SEC = 30  # Length of each audio and EEG segment in seconds


def main():
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    # List of subjects, excluding the bad ones
    subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12',
                'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24',
                'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36',
                'S37', 'S38', 'S39', 'S40', 'S41', 'S42']
    bads = ["S24", "S26", "S27", "S30", "S32", "S34", "S35", "S36", "S02", "S25"]  # S25 not bad use later
    subjects = [s for s in subjects if s not in bads]
    random.shuffle(subjects)
    train_subjects = subjects[:23]
    valid_subjects = subjects[23:28]
    test_subjects = subjects[28:]
    train_losses = []
    validation_losses = []


    audio_path = '/Users/efeoztufan/Desktop/A-Thesis/Datasets/bg257f92t/audio/AllStory.wav'  # Update with actual path

    # Initialize networks and loss function
    eeg_net = EEGNet()
    audio_net = AudioFeatureNet()
    loss_fn = DCCALoss()
    optimizer = optim.Adam(list(eeg_net.parameters()) + list(audio_net.parameters()), lr=0.001)
    # Load MelSpectrum feature extractor
    audio_feature_ext = MelSpectrum(SAMPLE_RATE_AUDIO)
    #audio_feature_ext = PhaseSpectrum(SAMPLE_RATE_AUDIO)
    #audio_feature_ext = MFCCSpectrum(SAMPLE_RATE_AUDIO)
    #audio_feature_ext = SpeechEnvelope(SAMPLE_RATE_AUDIO)

    # Adding a test phase
    for epoch in range(13):
        print(f"Epoch {epoch + 1}/{13}")
        train_loss = run_phase(train_subjects, audio_path, audio_feature_ext, eeg_net, audio_net, optimizer, loss_fn, phase='train')
        train_losses.append(train_loss)  # Extend the main train loss list with this epoch's losses

        valid_loss = run_phase(valid_subjects, audio_path, audio_feature_ext,
                               eeg_net, audio_net, None, loss_fn, phase='validate')
        validation_losses.append(valid_loss)  # Extend the main validation loss list
        print(f'Epoch {epoch + 1}: Training Loss = {train_loss}, Validation Loss = {valid_loss}')

        if epoch == 12:  # Run the test phase after the last training epoch
            test_loss = run_phase(test_subjects, audio_path, audio_feature_ext,
                                  eeg_net, audio_net, None, loss_fn, phase='test')
            print(f'Final Test Loss = {test_loss}')

    print('train losses:', train_losses)
    print('validation losses', validation_losses)

    # Save the model parameters for both networks
    torch.save(eeg_net.state_dict(), 'Analysis/DCCA-ModelsParameters/eeg_net_mel_theta.pth')
    torch.save(audio_net.state_dict(), 'Analysis/DCCA-ModelsParameters/audio_net_mel_theta.pth')
    print("Models saved successfully.")


if __name__ == "__main__":
    main()