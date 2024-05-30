from __future__ import annotations
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import typing as tp
import os

import torch
#import xlsxwriter
from pandas._typing import Frequency
from scipy.io import loadmat
import torchaudio
from typing import Union
from Codes import dataevents
from Codes.dataevents import extract_sequence_info, split_wav_as_block
from Codes.models import MelSpectrum, PhaseSpectrum, MFCCSpectrum

SFREQ = 500.0


def get_paths():
    # Directly define the paths as a dictionary
    return {
        'download': '/Users/efeoztufan/Desktop/A-Thesis/Datasets/bg257f92t',
        'audio': '/Users/efeoztufan/Desktop/A-Thesis/Datasets/bg257f92t/audio',
        'proc': '/Users/efeoztufan/Desktop/A-Thesis/Datasets/bg257f92t/proc',
        'audio_chunks': '/Users/efeoztufan/Desktop/A-Thesis/Datasets/bg257f92t/audio_chunks',
        'mel_spectrograms': '/Users/efeoztufan/Desktop/A-Thesis/Analysis/mel_spectrograms',
        'phase_spectrograms': '/Users/efeoztufan/Desktop/A-Thesis/Analysis/phase_spectrograms',
        'mfcc': '/Users/efeoztufan/Desktop/A-Thesis/Analysis/mfcc'
    }


def _read_meta(fname):
    proc = loadmat(
        fname,
        squeeze_me=True,
        chars_as_strings=True,
        struct_as_record=True,
        simplify_cells=True,
    )["proc"]

    # ref = proc["implicitref"]
    # ref_channels = proc["refchannels"]

    # subject_id = proc["subject"]
    meta = proc["trl"]

    # TODO artefacts, ica, rejected components etc
    assert len(meta) == proc["tot_trials"]
    assert proc["tot_chans"] == 61
    bads = list(proc["impedence"]["bads"])
    bads += list(proc["rejections"]["badchans"])
    # proc['varnames'], ['segment', 'tmin', 'Order']

    # summary = proc["rejections"]["final"]["artfctdef"]["summary"]
    # bad_segments = summary["artifact"]

    #     meta = pd.DataFrame(meta[:, 0].astype(int), columns=['start'])

    #     meta['start_offset'] = meta[:, 1].astype(int) # wave?
    #     meta['wav_file'] = meta[:, 3].astype(int)
    #     meta['start_sec'] = meta[:, 4]
    #     meta['mat_index'] = meta[:, 5].astype(int)
    columns = list(proc["varnames"])
    if len(columns) != meta.shape[1]:
        columns = ["start_sample", "stop_sample", "offset"] + columns
        assert len(columns) == meta.shape[1]
    meta = pd.DataFrame(meta, columns=["_" + i for i in columns])
    assert len(meta) == 2129  # FIXME retrieve subjects who have less trials?

    # Add Brennan's annotations
    paths = get_paths()
    source_path = paths['download']
    story = pd.read_csv(source_path + "/" + "AliceChapterOne-EEG.csv")
    events = meta.join(story)

    events["kind"] = "word"
    events["condition"] = "sentence"
    events["duration"] = events.offset - events.onset
    columns = dict(Word="word", Position="word_id", Sentence="sequence_id")
    events = events.rename(columns=columns)
    events["start"] = events["_start_sample"] / SFREQ

    # add audio events
    audio_path = paths['audio']
    wav_file = (
            audio_path + "/DownTheRabbitHoleFinal_SoundFile%i.wav"
    )
    sounds = []
    for segment, d in events.groupby("Segment"):
        # Some wav files start BEFORE the onset of eeg recording...
        start = d.iloc[0].start - d.iloc[0].onset
        sound = dict(
            kind="sound", start=start, filepath=str(wav_file) % segment
        )
        sounds.append(sound)
    events = pd.concat([events, pd.DataFrame(sounds)], ignore_index=True)
    events = events.sort_values("start").reset_index()

    # clean up
    keep = [
        "start",
        "duration",
        "kind",
        "word",
        "word_id",
        "sequence_id",
        "condition",
        "filepath",
    ]
    events = events[keep]
    events[['language', 'modality']] = 'english', 'audio'
    events = extract_sequence_info(events)
    events = events.event.create_blocks(groupby='sentence')
    events = events.event.validate()

    return events



def apply( recording: Brennan2019Recording,
        blocks: tp.Optional[tp.List[tp.Tuple[float, float]]] = None
    ) -> tp.Optional["Dataset"]:

        """Apply the epochs extraction procedure to the raw file and create a SegmentDataset.

        Parameters
        ----------
        recording:
            Recording on which to apply the epochs definition.
        blocks:
            List of tuples representing available ranges for building the dataset.
        """
        if blocks is not None and not blocks:
            raise ValueError("No blocks provided.")
        data = recording._load_raw()
        #sample_rate = data.info["sfreq"]
        sample_rate = Frequency(data.info["sfreq"])

        assert int(sample_rate.sample_rate) == int(sample_rate.sample_rate)

        # hack to discriminate between a condition and a query
        query = "kind=='word'"
        # Load the events DataFrame for the subject
        events_temp= recording._load_events()
        meta = events_temp.query(query)
        times = meta.start.values

        events = events_temp.copy()
        events = events.sort_values('start')

        events = split_wav_as_block(events, blocks)
        tmin= -0.5
        tmax= 2.5
        delta = 0.5 / sample_rate.sample_rate
        mask = np.logical_and(times + tmin >= 0,
                              times + tmax < data.times[-1] + delta)

        # We only keep extracts that are fully contained in at least one of the given blocks.
        in_any_split = False
        for start, stop in blocks:
            in_split = times + tmin >= start
            margin = tmax - delta
            in_split &= times + margin < stop
             # Keep around for debugging
            # print("block", start, stop)
            # print("need", start - self._opts["tmin"], stop + delta - self._opts["tmax"])
            # print("ok", sum(in_split))
            in_any_split |= in_split
        mask &= in_any_split

        # assert mask.any(), "empty dataset"
        # TODO understand why samples/times some are not unique nor ordered

        #samples = to_ind(sample_rate, times[mask])
        samples = sample_rate.to_ind(times[mask])
        unique_samples = np.unique(samples)
        if len(unique_samples) != len(samples):
            print(f"Found {len(samples) - len(unique_samples)} duplicates out of "
                           f"{len(samples)} events")
        if len(np.where(np.diff(times[mask]) < 0)[0]) > 0:
            print(f"Times are not sorted in meg events data at indices "
                           f"{np.where(np.diff(times[mask]) < 0)[0]}. "
                           f"SubjectID={recording.subject_uid}")


        meta = meta.iloc[np.where(mask)].reset_index()
        mne_events = np.concatenate([samples[:, None], np.ones(
            (len(samples), 2), dtype=np.int64)], 1)  # why long?
        # create
        baseline = (None, 0)
        epochs = mne.Epochs(data, events=mne_events,
                            preload=False, baseline=baseline,
                            metadata=meta,  tmin=-0.5, tmax=2.5, decim=1, event_repeated='drop')
        epochs._bad_dropped = True  # Hack: avoid checking

        dset = Dataset(
            recording, epochs, events=events,
            features=None, features_params=None,
            event_mask=None, meg_dimension=None)
        dset.blocks = blocks  # type: ignore
        return dset

def _read_eeg(fname):
    fname = Path(fname)
    mat = loadmat(
        fname,
        squeeze_me=True,
        chars_as_strings=True,
        struct_as_record=True,
        simplify_cells=True,
    )
    mat = mat["raw"]

    # sampling frequency
    sfreq = mat["hdr"]["Fs"]
    assert sfreq == 500.0
    assert mat["fsample"] == sfreq

    # channels
    n_chans = mat["hdr"]["nChans"]
    n_samples = mat["hdr"]["nSamples"]
    ch_names = list(mat["hdr"]["label"])
    assert len(ch_names) == n_chans

    # vertical EOG
    assert ch_names[60] == "VEOG"

    # audio channel
    add_audio_chan = False
    if len(ch_names) == 61:
        ch_names += ["AUD"]
        add_audio_chan = True
    assert ch_names[61] in ("AUD", "Aux5")

    # check name
    for i, ch in enumerate(ch_names[:-2]):
        assert ch == str(i + 1 + (i >= 28))

    # channel type
    assert set(mat["hdr"]["chantype"]) == set(["eeg"])
    ch_types = ["eeg"] * 60 + ["eog", "misc"]
    assert set(mat["hdr"]["chanunit"]) == set(["uV"])

    # create MNE info
    info = mne.create_info(ch_names, sfreq, ch_types, verbose=False)
    subject_id = fname.name.split(".mat")[0]
    info["subject_info"] = dict(his_id=subject_id, id=int(subject_id[1:]))

    # time
    diff = np.diff(mat["time"]) - 1 / sfreq
    tol = 1e-5
    assert np.all(diff < tol)
    assert np.all(diff > -tol)
    start, stop = mat["sampleinfo"]
    assert start == 1
    assert stop == n_samples
    assert mat["hdr"]["nSamplesPre"] == 0
    assert mat["hdr"]["nTrials"] == 1

    # data
    data = mat["trial"]
    assert data.shape[0] == n_chans
    assert data.shape[1] == n_samples
    if add_audio_chan:
        data = np.vstack((data, np.zeros_like(data[0])))

    # create mne objects
    info = mne.create_info(ch_names, sfreq, ch_types, verbose=False)
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)
    montage = mne.channels.make_standard_montage("easycap-M10")
    raw.set_montage(montage)

    assert raw.info["sfreq"] == SFREQ
    assert len(raw.ch_names) == 62



    return raw


class Brennan2019Recording():
    data_url = "https://deepblue.lib.umich.edu/data/concern/data_sets/"
    data_url += "bg257f92t"
    paper_url = "https://journals.plos.org/plosone/"
    paper_url += "article?id=10.1371/journal.pone.0207741"
    doi = "https://doi.org/10.1371/journal.pone.0207741"
    licence = "CC BY 4.0"
    modality = "audio"
    language = "english"
    device = "eeg"
    description = """EEG of Alice in WonderLand, By Brennan and Hale 2019.
        The eeg data was bandpassed between 0.1 and 200. Hz"""

    def __init__(self, *, subject_uid: str, recording_uid: str) -> None:

        self.subject_uid = subject_uid
        self.recording_uid = recording_uid
        self._subject_index: tp.Optional[int] = None  # specified during training
        self._recording_index: tp.Optional[int] = None  # specified during training
        self._mne_info: tp.Optional[mne.Info] = None
        # cache system
        self._arrays: tp.Dict[tp.Tuple[int, float], mne.io.RawArray] = {}
        self._events: tp.Optional[pd.DataFrame] = None

    def _load_raw(self) -> mne.io.RawArray:
        paths = get_paths()
        source = paths['download']
        file = source + "/" + f"{self.subject_uid}.mat"
        raw = _read_eeg(file)
        return raw

    def _load_events(self) -> pd.DataFrame:
        paths = get_paths()
        source = paths['proc']
        file = source + "/" + f"{self.subject_uid}.mat"
        events = _read_meta(file)
        return events

class Frequency:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def __int__(self):
        """Allows the object to be converted to an integer (sample rate)."""
        return self.sample_rate

    def to_ind(self, seconds):
        """Converts a time in seconds (or multiple times in an array)
           to a sample index.
        """
        if isinstance(seconds, np.ndarray):
            return np.round(seconds * self.sample_rate).astype(int)
        return int(round(seconds * self.sample_rate))

class Dataset:
    def __init__(self, recording: Brennan2019Recording, epochs: mne.Epochs,
                 features: tp.Sequence[str], events: pd.DataFrame,
                 features_params: tp.Optional[dict] = None, event_mask: bool = False,
                 meg_dimension: tp.Optional[int] = None) -> None:
        self.recording = recording
        self.epochs = epochs
        self.events = events
        self.sample_rate = None
        self.features_params = None # type: ignore
        self.features = None
        self.meg_dimension = meg_dimension

'''def _extract_wav_part(filepath: Union[Path, str], onset: float, offset: float) -> tp.Tuple[torch.Tensor, int]:
    """Extract a chunk of a wave file based on onset and offset in seconds."""
    info = torchaudio.info(str(filepath))
    sample_rate = info.sample_rate
    frame_start = int(sample_rate * onset)
    frame_end = int(sample_rate * offset) - frame_start
    waveform, _ = torchaudio.load(filepath, frame_offset=frame_start, num_frames=frame_end)
    return waveform, sample_rate'''

def _extract_wav_part(
    filepath: Union[Path, str], onset: float, offset: float
) -> tp.Tuple[torch.Tensor, Frequency]:
    """Extract a chunk of a wave file based on onset and offset in seconds
    """
    info = torchaudio.info(str(filepath))
    sr = Frequency(info.sample_rate)
    wav = torchaudio.load(
        filepath, frame_offset=sr.to_ind(onset), num_frames=sr.to_ind(offset - onset))[0]
    expected_duration = offset - onset
    actual_duration = wav.shape[-1] / sr.sample_rate
    delta = abs(actual_duration - expected_duration)
    assert delta <= 0.1, (delta, filepath, onset, offset, onset - offset)
    return wav, sr


def save_audio_word(filepath, onset, offset, save_dir, index):
    waveform, sample_rate = _extract_wav_part(filepath, onset, offset)
    save_path = os.path.join(save_dir, f"audio_word_{index}.wav")
    torchaudio.save(save_path, waveform, sample_rate)
    return save_path
def save_audio_sentence(filepath, onset, offset, save_dir, index):
    waveform, sample_rate = _extract_wav_part(filepath, onset, offset)
    save_path = os.path.join(save_dir, f"audio_sentence_{index}.wav")
    torchaudio.save(save_path, waveform, sample_rate)
    return save_path

def main():
    # Specify the subject IDs you want to process
    subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12'
        , 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24'
        , 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37'
        , 'S38', 'S39', 'S40', 'S41', 'S42']

    bads = ["S24", "S26", "S27", "S30", "S32", "S34", "S35", "S36"]
    bads += ["S02.mat"]  # bad proc.trl?
    subjects = [s for s in subjects if s not in bads]
    all_recordings = []
    dsets = []
    for subject_id in subjects:
        # Create an instance of Brennan2019Recording for each subject
        recording = Brennan2019Recording(subject_uid=subject_id, recording_uid=None)

        events = recording._load_events()
        blocks = events[events.kind == 'block']
        blocks = dataevents.merge_blocks(blocks, min_block_duration_s=int(6))
        #blocks = dataevents.assign_blocks( blocks, [0.2, 0.1], remove_ratio=0, seed=12,min_n_blocks_per_split=1)
        #for j in range(3):
            #split_blocks = blocks[blocks.split == j]
            #if not split_blocks.empty:
                #start_stops = [(b.start, b.start + b.duration) for b in split_blocks.itertuples()]
                #dset = apply(recording, blocks=start_stops)
                #dsets.append(dset)
        # Generate start and stop tuples for all blocks
        start_stops = [(b.start, b.start + b.duration) for b in blocks.itertuples()]

        # Apply the function to the entire dataset
        dset = apply(recording, blocks=start_stops)
            # prepare events

        event_kinds = {'sound', 'word'}

        # copy otherwise this is a view and we can't assign _stop
        dset.events = dset.events.loc[[c in event_kinds for c in dset.events.kind], :].copy()
        #dset.events.loc[:, "_stop"] = dset.events.start + dset.events.duration  # TODO move
        dset.events['stop'] = dset.events['start'] + dset.events['duration']

        # First, ensure the DataFrame is sorted by the 'start' time, if it's not already.
        dset.events.sort_values(by='start', inplace=True)

        # Then use forward fill to propagate the file paths down the 'filepath' column
        # but only for the rows where 'kind' is 'word'.
        mask = dset.events['kind'] == 'word'
        dset.events.loc[mask, 'filepath'] = dset.events['filepath'].ffill()

        #writer = pd.ExcelWriter('/Users/efeoztufan/Desktop/A-Thesis/Analysis/datasetevents.xlsx', engine='xlsxwriter')
        # Convert the DataFrame to an XlsxWriter Excel object.
        #dset.events.to_excel(writer, sheet_name='Sheet1', index=False)
        # Close the Pandas Excel writer and output the Excel file.
        #writer.close()

        # Initialize and adjust times
        dset.events['adjusted_start'] = dset.events['start']
        dset.events['adjusted_stop'] = dset.events['start'] + dset.events['duration']

        last_filepath = None
        reference_start_time = 0

        for index, row in dset.events.iterrows():
            if row['kind'] == 'sound' and row['filepath'] != last_filepath:
                reference_start_time = row['start']
                last_filepath = row['filepath']
            adjusted_start = row['start'] - reference_start_time
            adjusted_stop = adjusted_start + row['duration']
            dset.events.at[index, 'adjusted_start'] = adjusted_start
            dset.events.at[index, 'adjusted_stop'] = adjusted_stop

        # Ensure the 'adjusted_stop' time never goes negative due to the correction
        dset.events['adjusted_stop'] = dset.events['adjusted_stop'].clip(lower=0)
        # Now you can apply _extract_wav_part to each row and store the result in a new column.
        # Here we initialize the column with None to set the correct datatype
        dset.events['audio_data_path'] = None

        paths = get_paths()
        audio_save_directory = paths['audio_chunks']

        # Instantiate features
        mel_spectrum_extractor = MelSpectrum(sample_rate=44100)  # Example sample rate
        phase_spectrum_extractor = PhaseSpectrum(sample_rate=44100)
        mfcc_extractor = MFCCSpectrum(sample_rate=1600, n_mfcc=13)

        # Path where you want to save Mel spectrogram files
        mel_spectrogram_dir = paths['mel_spectrograms']
        phase_spectrogram_dir = paths['phase_spectrograms']
        mfcc_dir = paths['mfcc']
        os.makedirs(mel_spectrogram_dir, exist_ok=True)
        os.makedirs(phase_spectrogram_dir, exist_ok=True)
        os.makedirs(mfcc_dir, exist_ok=True)

        for index, row in dset.events.iloc[:250].iterrows(): # for index, row in dset.events.iterrows():
            if row['kind'] == 'word' and pd.notna(row['filepath']):
                duration = row['duration']
                audio_word_path = save_audio_word(
                    row['filepath'],
                    row['adjusted_start'],
                    row['adjusted_stop'],
                    audio_save_directory,
                    index
                )
                dset.events.at[index, 'audio_data_path'] = audio_word_path


                # Compute Phase spectrum
              #  mel_spectrogram = mel_spectrum_extractor.get_feature(audio_word_path,duration)
                # Save the Mel spectrogram tensor to a file
               # mel_spectrogram_file = os.path.join(mel_spectrogram_dir, f'mel_spectrogram_{index}.pt')
                #torch.save(mel_spectrogram, mel_spectrogram_file)
                # Store the file path in the dataframe
                #dset.events.at[index, 'mel_spectrogram_path'] = mel_spectrogram_file

                # Compute Phase spectrum
                phase_spectrum = phase_spectrum_extractor.get_feature(audio_word_path, duration)
                phase_spectrogram_file = os.path.join(phase_spectrogram_dir, f'phase_spectrogram_{index}.pt')
                torch.save(phase_spectrum, phase_spectrogram_file)
                dset.events.at[index, 'phase_spectrogram_path'] = phase_spectrogram_file

                # Compute  MFCC
                mfccs = mfcc_extractor.get_feature(audio_word_path, duration)
                mfccs_file = os.path.join(mfcc_dir, f'mfccs_{index}.pt')
                torch.save(mfccs, mfccs_file)
                dset.events.at[index, 'mfccs_path'] = mfccs_file


            elif row['kind'] == 'sound' and pd.notna(row['filepath']):
                duration = row['duration']
                audio_sentence_path = save_audio_sentence(
                    row['filepath'],
                    row['adjusted_start'],
                    row['adjusted_stop'],
                    audio_save_directory,
                    index
                )
                dset.events.at[index, 'audio_data_path'] = audio_sentence_path


                #mel_spectrogram = mel_spectrum_extractor.get_feature(audio_sentence_path, duration)
                # Save the Mel spectrogram tensor to a file
                #mel_spectrogram_file = os.path.join(mel_spectrogram_dir, f'mel_spectrogram_{index}.pt')
                #torch.save(mel_spectrogram, mel_spectrogram_file)
                # Store the file path in the dataframe
                #dset.events.at[index, 'mel_spectrogram_path'] = mel_spectrogram_file

                # Compute Phase spectrum
                phase_spectrum = phase_spectrum_extractor.get_feature(audio_sentence_path, duration)
                phase_spectrogram_file = os.path.join(phase_spectrogram_dir, f'phase_spectrogram_{index}.pt')
                torch.save(phase_spectrum, phase_spectrogram_file)
                dset.events.at[index, 'phase_spectrogram_path'] = phase_spectrogram_file

                # Compute  MFCC
                mfccs = mfcc_extractor.get_feature(audio_sentence_path, duration)
                mfccs_file = os.path.join(mfcc_dir, f'mfccs_{index}.pt')
                torch.save(mfccs, mfccs_file)
                dset.events.at[index, 'mfccs_path'] = mfccs_file
        break



if __name__ == "__main__":
    main()
