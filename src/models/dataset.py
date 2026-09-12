import torch
from torch.utils.data import Dataset
import numpy as np

import functools

@functools.lru_cache(maxsize=10000)
def load_numpy_file(file_path):
    return np.load(file_path)

class MIMIIDataset(Dataset):
    def __init__(self, df, max_time_frames=313):
        self.df = df
        self.max_time_frames = max_time_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['file_path']
        label = row['label']
        
        # Load precomputed numpy array with RAM caching
        mel_spec = load_numpy_file(file_path)
        
        # Pad or truncate the time axis to a fixed length (max_time_frames)
        # Shape is usually (n_mels, time_frames)
        time_frames = mel_spec.shape[1]
        
        if time_frames < self.max_time_frames:
            # Pad
            pad_width = self.max_time_frames - time_frames
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        elif time_frames > self.max_time_frames:
            # Truncate
            mel_spec = mel_spec[:, :self.max_time_frames]
            
        # Add channel dimension for CNN: (1, n_mels, time_frames)
        mel_spec = np.expand_dims(mel_spec, axis=0)
        
        x = torch.tensor(mel_spec, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        
        if 'logit_0' in row.index and 'logit_1' in row.index:
            logits = torch.tensor([row['logit_0'], row['logit_1']], dtype=torch.float32)
            return x, y, logits
        
        return x, y, torch.tensor([float('nan'), float('nan')], dtype=torch.float32)
