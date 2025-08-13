import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ------------------------------
# Finite-difference helpers
# ------------------------------
def compute_v_and_a(x: np.ndarray, dt: float):
    """
    NumPy helper for per-sample derivatives.
    x: (W,) array of positions
    returns:
        v: (W,) velocity
        a: (W,) acceleration
    """
    x = x.astype(np.float32, copy=False)
    W = x.shape[0]
    v = np.zeros_like(x, dtype=np.float32)
    a = np.zeros_like(x, dtype=np.float32)

    if W >= 3:
        v[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
        v[0]    = (x[1] - x[0]) / dt
        v[-1]   = (x[-1] - x[-2]) / dt

        a[1:-1] = (x[2:] - 2 * x[1:-1] + x[:-2]) / (dt ** 2)
        a[0]    = (x[2] - 2 * x[1] + x[0]) / (dt ** 2)
        a[-1]   = (x[-1] - 2 * x[-2] + x[-3]) / (dt ** 2)
    elif W == 2:
        v[:] = (x[1] - x[0]) / dt
        a[:] = 0.0
    else:
        v[:] = 0.0
        a[:] = 0.0

    return v, a


def compute_v_and_a_torch(x_batch: torch.Tensor, dt: float):
    """
    Torch helper for batched derivatives (for rollout training).
    x_batch: (B, W) positions
    returns:
        v: (B, W)
        a: (B, W)
    """
    B, W = x_batch.shape
    v = torch.zeros_like(x_batch)
    a = torch.zeros_like(x_batch)

    if W >= 3:
        v[:, 1:-1] = (x_batch[:, 2:] - x_batch[:, :-2]) / (2 * dt)
        v[:, 0]     = (x_batch[:, 1] - x_batch[:, 0]) / dt
        v[:, -1]    = (x_batch[:, -1] - x_batch[:, -2]) / dt

        a[:, 1:-1] = (x_batch[:, 2:] - 2 * x_batch[:, 1:-1] + x_batch[:, :-2]) / (dt ** 2)
        a[:, 0]     = (x_batch[:, 2] - 2 * x_batch[:, 1] + x_batch[:, 0]) / (dt ** 2)
        a[:, -1]    = (x_batch[:, -1] - 2 * x_batch[:, -2] + x_batch[:, -3]) / (dt ** 2)
    elif W == 2:
        v[:] = (x_batch[:, 1] - x_batch[:, 0]) / dt
        a[:] = 0.0
    else:
        v[:] = 0.0
        a[:] = 0.0

    return v, a


# ---------------------------------------------------------
# HamiltonFullDataset — slice windows from full trajectories
# ---------------------------------------------------------
class HamiltonFullDataset(Dataset):
    """
    Expects a CSV with columns:
        x0, v0, k, x_0, x_1, ..., x_{time_steps-1}

    Yields (per item):
        input  : (3, window_len) tensor  [rows = x, v, a]
        target : scalar (next x at start+window_len+horizon-1)
        meta   : dict with {x0,v0,k,start_idx} if return_meta=True
    """
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

        self.window_len = int(window_len)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.dt = float(duration) / float(time_steps - 1)
        self.time_steps = int(time_steps)
        self.return_meta = bool(return_meta)
        self.random_windows = bool(random_windows)

        # meta
        self.meta_x0 = self.df["x0"].values.astype("float32")
        self.meta_v0 = self.df["v0"].values.astype("float32")
        self.meta_k  = self.df["k"].values.astype("float32")

        # trajectories matrix (N, T)
        x_cols = [f"x_{i}" for i in range(self.time_steps)]
        self.X = self.df[x_cols].values.astype("float32")

        # precompute indices if using deterministic windows
        self.index = []
        if not self.random_windows:
            max_start = self.time_steps - self.window_len - (self.horizon - 1)
            max_start = max(0, max_start)
            for tid in range(len(self.X)):
                for s in range(0, max_start + 1, self.stride):
                    self.index.append((tid, s))

    def __len__(self):
        if self.random_windows:
            max_start = self.time_steps - self.window_len - (self.horizon - 1)
            max_start = max(1, max_start)
            # effective epoch length: trajectories × valid starts
            return max(len(self.X) * max_start, 1)
        return len(self.index)

    def __getitem__(self, idx):
        if self.random_windows:
            tid = np.random.randint(0, len(self.X))
            max_start = self.time_steps - self.window_len - (self.horizon - 1)
            s = np.random.randint(0, max_start + 1)
        else:
            tid, s = self.index[idx]

        # slice window and target
        x_seq = self.X[tid, s : s + self.window_len]  # (W,)
        tgt_idx = s + self.window_len + self.horizon - 1
        target = self.X[tid, tgt_idx].astype(np.float32)

        # derivatives via shared helper
        v, a = compute_v_and_a(x_seq, self.dt)  # numpy outputs

        inp = torch.tensor(
            np.stack([x_seq, v.astype(np.float32), a.astype(np.float32)]),
            dtype=torch.float32
        )  # (3, W)
        tgt = torch.tensor(target, dtype=torch.float32)

        if self.return_meta:
            meta = {
                "x0": torch.tensor(self.meta_x0[tid], dtype=torch.float32),
                "v0": torch.tensor(self.meta_v0[tid], dtype=torch.float32),
                "k":  torch.tensor(self.meta_k[tid],  dtype=torch.float32),
                "start_idx": torch.tensor(s, dtype=torch.long),
            }
            return inp, tgt, meta

        return inp, tgt
