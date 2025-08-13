from data_hamiltonian import DEFAULT_DURATION, DEFAULT_TIME_STEPS
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from dataset_hamiltonian import compute_v_and_a

class HamiltonFullDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        window_len: int = 10,
        horizon: int = 1,
        stride: int = 1,
        duration: float = 10.0,
        time_steps: int = 100,
        return_meta: bool = False,
        random_windows: bool = False
    ):
        self.df = pd.read_csv(csv_path)
        self.window_len = window_len
        self.horizon = horizon
        self.stride = stride
        self.dt = duration / (time_steps - 1)
        self.time_steps = time_steps
        self.return_meta = return_meta
        self.random_windows = random_windows

        # Meta
        self.meta_x0 = self.df["x0"].values.astype("float32")
        self.meta_v0 = self.df["v0"].values.astype("float32")
        self.meta_k  = self.df["k"].values.astype("float32")

        # Full trajectories (N, T)
        x_cols = [f"x_{i}" for i in range(time_steps)]
        self.X = self.df[x_cols].values.astype("float32")

        # Precompute indices if not using random windows
        self.index = []
        if not self.random_windows:
            max_start = time_steps - window_len - (horizon - 1)
            for tid in range(len(self.X)):
                for s in range(0, max_start + 1, stride):
                    self.index.append((tid, s))

    def __len__(self):
        if self.random_windows:
            max_start = self.time_steps - self.window_len - (self.horizon - 1)
            return max(len(self.X) * max(1, max_start), 1)
        return len(self.index)

    def __getitem__(self, idx):
        if self.random_windows:
            tid = np.random.randint(0, len(self.X))
            max_start = self.time_steps - self.window_len - (self.horizon - 1)
            s = np.random.randint(0, max_start + 1)
        else:
            tid, s = self.index[idx]

        # Slice window and target
        x_seq = self.X[tid, s : s + self.window_len]  # (W,)
        target = self.X[tid, s + self.window_len + self.horizon - 1]  # scalar

        # ---- CHANGED: reuse your helper for derivatives ----
        v, a = compute_v_and_a(x_seq, self.dt)  # returns numpy arrays

        # Pack tensors
        inp = torch.tensor(np.stack([x_seq, v.astype(np.float32), a.astype(np.float32)]),
                           dtype=torch.float32)  # (3, W)
        tgt = torch.tensor(target, dtype=torch.float32)

        if self.return_meta:
            meta = {
                "x0": torch.tensor(self.meta_x0[tid]),
                "v0": torch.tensor(self.meta_v0[tid]),
                "k":  torch.tensor(self.meta_k[tid]),
                "start_idx": torch.tensor(s, dtype=torch.long),
            }
            return inp, tgt, meta

        return inp, tgt
