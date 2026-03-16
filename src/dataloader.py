import torch
from torch.utils.data import Dataset, DataLoader
from src.preprocess import WESADProcessor # Assuming you saved the previous script
import numpy as np
class FedBioDataset(Dataset):
    def __init__(self, subject_id, data_path="./data/WESAD"):
        self.subject_id = subject_id # Need this for the print statement
        self.processor = WESADProcessor(data_path)
        
        # Load and window the data
        raw_data = self.processor.load_subject(subject_id)
        self.windowed = self.processor.create_windows(raw_data)
        
        self.length = len(self.windowed['label'])
        
        # Call normalization
        self.normalize_data()
    def normalize_data(self):
        sensors = ['ecg', 'eda', 'emg']

        for s in sensors:
            data = self.windowed[s]

            mean = data.mean()
            std = data.std()

            self.windowed[s] = (data - mean) / (std + 1e-8)

        print(f"✅ Data for {self.subject_id} normalized per-sensor (global).")
    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        ecg_raw = self.windowed['ecg'][idx]
        eda_raw = self.windowed['eda'][idx]
        emg_raw = self.windowed['emg'][idx]

        def get_moments(sig):
            return np.array([
                np.mean(sig),
                np.std(sig),
                np.min(sig),
                np.max(sig)
            ])

        context_feats = np.concatenate([
            get_moments(ecg_raw),
            get_moments(eda_raw),
            get_moments(emg_raw)
        ])

        context_feats = (context_feats - context_feats.mean()) / (context_feats.std() + 1e-8)

        return {
            'ecg': torch.tensor(ecg_raw).float(),
            'eda': torch.tensor(eda_raw).float(),
            'emg': torch.tensor(emg_raw).float(),
            'context': torch.tensor(context_feats).float(),
            'label': torch.tensor(self.windowed['label'][idx]).long()
        }