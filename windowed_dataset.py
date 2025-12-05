import torch
from torch.utils.data import Dataset

class WindowedFromLongDataset(Dataset):

    def __init__(self, base_dataset: Dataset, win_len: int = 1000):
        self.base = base_dataset
        self.win_len = int(win_len)
        # inspect first item to infer L_long
        x0, _ = self.base[0]
        self.L_long = x0.shape[-1]
        assert self.L_long % self.win_len == 0
        self.num_win = self.L_long // self.win_len

    def __len__(self):
        return len(self.base) * self.num_win

    def _map_index(self, idx):
        base_idx = idx // self.num_win
        w = idx % self.num_win
        start = w * self.win_len
        end = start + self.win_len
        return base_idx, start, end

    def __getitem__(self, idx):
        base_idx, s, e = self._map_index(idx)
        x, y = self.base[base_idx]
        # x: (1, L_long)
        xw = x[..., s:e]  # (1, win_len)
        return xw, y
