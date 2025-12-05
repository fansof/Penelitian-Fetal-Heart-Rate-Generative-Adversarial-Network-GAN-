import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class myDataset(Dataset):
    """
    Load preprocessed FHR data with flexible train/eval splits.
    Preprocessed files already have:
    - Zero gaps fixed
    - Spikes smoothed
    - Out-of-range values replaced
    - Normalized to [-1, 1] using tanh-range
    
    This loader is FAST because all heavy preprocessing is already done!
    """
    
    def __init__(
        self,
        data_folder,           # Path to PREPROCESSED data folder
        ph_file,               # Path to pH labels CSV
        sequence_length=1000,  # Target sequence length (1000 = 15 min at 4 Hz)
        max_normal=447,
        max_pathological=105,
        train_normal=70,
        train_pathological=70,
        eval_normal=30,
        eval_pathological=30,
        mode='train',          # 'train' or 'eval'
        random_seed=81,
        strict_length=True,    # If True: drop records < sequence_length
        crop_strategy='last'  # 'from_first_valid' | 'center' | 'last'
    ):
        self.data_folder = data_folder
        self.sequence_length = sequence_length
        self.mode = mode
        self.strict_length = strict_length
        self.crop_strategy = crop_strategy
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Load pH labels and build label dict
        ph_df = pd.read_csv(ph_file)
        ph_df["Label"] = (ph_df["pH"] >= 7.15).astype(int)
        self.ph_dict = dict(zip(ph_df["Record"].astype(str), ph_df["Label"]))
        
        # Gather all CSV files (excluding ph_labels.csv itself)
        all_files = sorted([
            f for f in os.listdir(data_folder)
            if f.endswith('.csv') and f != os.path.basename(ph_file)
        ])
        
        # Split files by class
        normal_files, pathological_files = [], []
        for f in all_files:
            rid = f.replace('.csv', '')
            lab = self.ph_dict.get(rid, None)
            if lab == 1:
                normal_files.append(f)
            elif lab == 0:
                pathological_files.append(f)
        
        # Validate we have enough files
        total_normal_needed = train_normal + eval_normal
        total_pathological_needed = train_pathological + eval_pathological
        
        if len(normal_files) < total_normal_needed:
            raise ValueError(f"Not enough normal files: need {total_normal_needed}, have {len(normal_files)}")
        if len(pathological_files) < total_pathological_needed:
            raise ValueError(f"Not enough pathological files: need {total_pathological_needed}, have {len(pathological_files)}")
        
        # Sample files
        selected_normal = random.sample(normal_files, min(max_normal, len(normal_files)))
        selected_path = random.sample(pathological_files, min(max_pathological, len(pathological_files)))
        
        # Split into train/eval
        self.train_normal_files = selected_normal[:train_normal]
        self.eval_normal_files = selected_normal[train_normal:train_normal + eval_normal]
        self.train_pathological_files = selected_path[:train_pathological]
        self.eval_pathological_files = selected_path[train_pathological:train_pathological + eval_pathological]
        
        # Set file list based on mode
        if mode == 'train':
            self.file_list = self.train_normal_files + self.train_pathological_files
        else:
            self.file_list = self.eval_normal_files + self.eval_pathological_files
        
        # Store all train/eval files for helper methods
        self.all_train_files = self.train_normal_files + self.train_pathological_files
        self.all_eval_files = self.eval_normal_files + self.eval_pathological_files
        
        # Print summary
        print(f"Dataset ready → {mode}")
        print(f"- Train: {len(self.all_train_files)} files ({len(self.train_normal_files)}N + {len(self.train_pathological_files)}P)")
        print(f"- Eval: {len(self.all_eval_files)} files ({len(self.eval_normal_files)}N + {len(self.eval_pathological_files)}P)")
        print(f"- Total: {len(self.all_train_files) + len(self.all_eval_files)} files")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        rid = file_name.replace('.csv', '')
        fpath = os.path.join(self.data_folder, file_name)
        
        # Get label
        label = self.ph_dict.get(rid, 0)
        
        try:
            # Load PREPROCESSED FHR signal (skip 2 header rows, column 1 = FHR)
            df = pd.read_csv(fpath, skiprows=2, header=None, usecols=[1], names=["FHR"])
            preprocessed = df['FHR'].values.astype(np.float32)
            
            if len(preprocessed) == 0:
                data = torch.zeros((1, self.sequence_length), dtype=torch.float32)
                return data, torch.tensor(label, dtype=torch.long)
            
            # Extract desired sequence length
            finalized = self._finalize_length(preprocessed)
            
            if finalized is None:
                data = torch.zeros((1, self.sequence_length), dtype=torch.float32)
                return data, torch.tensor(label, dtype=torch.long)
            
            # Convert to tensor [1, sequence_length] for Conv1d
            return (
                torch.tensor(finalized, dtype=torch.float32).unsqueeze(0),
                torch.tensor(label, dtype=torch.long)
            )
        
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            data = torch.zeros((1, self.sequence_length), dtype=torch.float32)
            return data, torch.tensor(label, dtype=torch.long)
    
    # ---------------------------- Length Handling ---------------------------- #
    
    def _finalize_length(self, x):
        """Extract desired sequence_length from preprocessed signal."""
        n = len(x)
        L = self.sequence_length
        
        # If signal too short
        if n < L:
            if self.strict_length:
                return None
            # Pad with last value
            pad_val = x[-1] if n > 0 else 0.0
            return np.pad(x, (0, L - n), mode='constant', constant_values=pad_val)
        
        # If exact length
        if n == L:
            return x
        
        # If signal too long, crop based on strategy
        if self.crop_strategy == 'from_first_valid':
            return x[:L]
        elif self.crop_strategy == 'center':
            start = max(0, (n - L) // 2)
            return x[start:start + L]
        else:  # 'last'
            return x[-L:]
    
    # ---------------------------- Helper Methods ---------------------------- #
    
    def get_eval_files_info(self):
        """Return information about eval files."""
        return {
            'eval_normal_files': self.eval_normal_files,
            'eval_pathological_files': self.eval_pathological_files,
            'all_eval_files': self.all_eval_files,
            'data_folder': self.data_folder,
            'sequence_length': self.sequence_length,
            'eval_labels': [self.ph_dict.get(f.replace('.csv', ''), 0) for f in self.all_eval_files],
            'total_eval_normal': len(self.eval_normal_files),
            'total_eval_pathological': len(self.eval_pathological_files),
            'total_eval_files': len(self.all_eval_files)
        }
    
    def get_train_files_info(self):
        """Return information about train files."""
        return {
            'train_normal_files': self.train_normal_files,
            'train_pathological_files': self.train_pathological_files,
            'all_train_files': self.all_train_files,
            'data_folder': self.data_folder,
            'sequence_length': self.sequence_length,
            'train_labels': [self.ph_dict.get(f.replace('.csv', ''), 0) for f in self.all_train_files],
            'total_train_normal': len(self.train_normal_files),
            'total_train_pathological': len(self.train_pathological_files),
            'total_train_files': len(self.all_train_files)
        }