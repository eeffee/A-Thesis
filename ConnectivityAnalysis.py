import mne
import mne_connectivity 

import matplotlib.pyplot as plt
from Codes.dataPrep import Brennan2019Recording


def load_data(subject_uid):
    """Load EEG data using the Brennan2019Recording class."""
    recording = Brennan2019Recording(subject_uid=subject_uid, recording_uid=None)
    raw = recording._load_raw()  # Load the EEG data
    return raw


def perform_connectivity_analysis(raw):
    """Perform connectivity analysis on the raw data."""
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)

    # Filtering to isolate the frequency band of interest
    raw.filter(1, 30, fir_design='firwin')

    # Compute spectral connectivity
    con, freqs, times, _, _ = spectral_connectivity(
        [raw], method='pli', indices=None, mode='multitaper', sfreq=raw.info['sfreq'],
        fmin=1, fmax=10, faverage=True, tmin=None, tmax=None, mt_adaptive=True, n_jobs=1,picks=picks)

    con_matrix = con[:, :, 0]  # assuming interest in the first frequency band
    return con_matrix


def plot_connectivity_matrix(con_matrix):
    """Plot the connectivity matrix."""
    # Building a graph from the connectivity matrix
    import networkx as nx
    G = nx.from_numpy_array(con_matrix)

    # Drawing the graph
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color='blue', alpha=0.6)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title('EEG Connectivity Network')
    plt.show()


def main():
    # Example subject UID; replace with actual subject ID
    subject_uid = 'S01'
    raw = load_data(subject_uid)
    con_matrix = perform_connectivity_analysis(raw)
    plot_connectivity_matrix(con_matrix)


if __name__ == "__main__":
    main()
