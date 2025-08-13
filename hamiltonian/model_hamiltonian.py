import torch
import torch.nn as nn

class HamiltonPredictorCNN(nn.Module):
    def __init__(self, window_len=15):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # probe size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, window_len)
            conv_out = self.convs(dummy)      # (1, C, L)
            flat_dim = conv_out.numel()       # C*L

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h = self.convs(x)
        out = self.mlp(h)
        return out.squeeze(-1)
