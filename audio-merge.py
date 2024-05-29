import scipy.io
import numpy as np
import pandas as pd
import h5py
from pydub import AudioSegment
import os
from scipy.signal import butter, lfilter
from scipy.io import wavfile
from scipy.signal import resample

# Set the path to your directory containing the WAV files
directory_path = '/Users/efeoztufan/Desktop/A-Thesis/Datasets/Ali/Audio'

# Create an empty AudioSegment instance for accumulating the audio
combined_audio = AudioSegment.silent(duration=0)

# Loop through all WAV files in the directory
for filename in sorted(os.listdir(directory_path)):
    if filename.endswith('.wav'):
        # Load the WAV file
        audio = AudioSegment.from_wav(os.path.join(directory_path, filename))

        # Repeat the audio three times
        repeated_audio = audio * 3

        # Append the repeated audio to the combined audio
        combined_audio += repeated_audio

# Ensure the combined audio is at 44.1 kHz
combined_audio = combined_audio.set_frame_rate(44100)

# Convert to raw data for filtering
raw_data = combined_audio.get_array_of_samples()

sample_rate_audio = 44100
# Downsample to 250 Hz
#downsampled_audio = filtered_audio.set_frame_rate(250)

goal_sample_rate = 250 # Set your desired sample rate
if goal_sample_rate != sample_rate_audio:
    audio_data = resample(raw_data, int(len(raw_data) * goal_sample_rate / sample_rate_audio))
    sample_rate_audio = goal_sample_rate

# Create a new AudioSegment with the filtered data
resampled_audio = AudioSegment(
    data=bytes(audio_data),
    sample_width=combined_audio.sample_width,
    frame_rate=sample_rate_audio,
    channels=combined_audio.channels
)

# Export the combined and downsampled audio to a new file
resampled_audio.export("/Users/efeoztufan/Desktop/A-Thesis/Datasets/Ali/Audio/AllStories-250Hz-v2.wav", format="wav")

print("Audio files have been successfully merged, filtered, and downsampled to 250 Hz, then saved.")