import pickle
import numpy as np
import torch

class WESADProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        # Labels we care about: 1 (Baseline), 2 (Stress), 3 (Amusement)
        self.valid_labels = [1, 2, 3]

    def load_subject(self, subject_id):
        file_path = f"{self.data_path}/{subject_id}/{subject_id}.pkl"
        with open(file_path, 'rb') as f:
            # WESAD uses latin1 encoding for its pickles
            data = pickle.load(f, encoding='latin1')
        
        signals = data['signal']['chest']
        labels = data['label']

        # Filter for only Baseline, Stress, and Amusement
        indices = np.isin(labels, self.valid_labels)
        
        # Extract Modalities
        # WESAD Chest Data indices: ECG (heart), EDA (skin), EMG (muscle)
        processed_data = {
            'ecg': signals['ECG'][indices],
            'eda': signals['EDA'][indices],
            'emg': signals['EMG'][indices],
            'label': labels[indices] - 1 # Shift labels to 0, 1, 2 for CrossEntropy
        }
        
        return processed_data

    def create_windows(self, data, window_size=3500, step_size=350):

        windows = {k: [] for k in ['ecg', 'eda', 'emg', 'label']}
        num_samples = len(data['label'])

        for i in range(0, num_samples - window_size + 1, step_size):

            for key in ['ecg', 'eda', 'emg']:
                window_slice = data[key][i:i+window_size].flatten()
                windows[key].append(window_slice)

            label_slice = data['label'][i:i+window_size]
            windows['label'].append(np.bincount(label_slice).argmax())

        return {k: np.array(v) for k, v in windows.items()}

# Usage Example
# processor = WESADProcessor('path/to/WESAD')
# subject_data = processor.load_subject('S2')
# windowed_data = processor.create_windows(subject_data)