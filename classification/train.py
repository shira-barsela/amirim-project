import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
import numpy as np
import os

# ============================================
# Configuration flags
# ============================================
MODEL_PATH = "trajectory_classifier.pth"
TRAIN_FROM_SCRATCH = False
CONTINUE_TRAINING = True
EPOCHS = 100
BATCH_SIZE = 32
NOISE_STD = 0.02
X0_RANGE = (-1.2, 1.2)
V0_RANGE = (-1.2, 1.2)
K_RANGE = (0.5, 3.0)


# ============================================
# 1. Generate synthetic labeled dataset
# ============================================

def generate_trajectory(x0, v0, k, t, potential_type):
    def rhs(t, state):
        x, v = state[0], state[1]
        dxdt = v
        if potential_type == "harmonic":
            dvdt = -k * x
        elif potential_type == "quartic":
            dvdt = -k * x ** 3
        else:
            raise ValueError("Unknown potential type")
        return torch.tensor([dxdt, dvdt])

    state0 = torch.tensor([x0, v0])
    with torch.no_grad():
        trajectory = odeint(rhs, state0, t)
    return trajectory[:, 0]  # return only x(t)


class TrajectoryDataset(Dataset):
    def __init__(self, num_samples=500, time_steps=100, duration=10.0, noise_std=0.0):
        self.t = torch.linspace(0, duration, time_steps)
        self.samples = []
        self.labels = []
        self.noise_std = noise_std

        def safe_generate(x0, v0, k, potential_type):
            try:
                x_traj = generate_trajectory(x0, v0, k, self.t, potential_type)
                if torch.isnan(x_traj).any() or torch.abs(x_traj).max() > 1e3:
                    return None
                return x_traj
            except Exception:
                return None

        while len(self.labels) < num_samples // 2:
            x0, v0, k = np.random.uniform(X0_RANGE[0], X0_RANGE[1]), \
                np.random.uniform(V0_RANGE[0], V0_RANGE[1]), np.random.uniform(K_RANGE[0], K_RANGE[1])
            x_traj = safe_generate(x0, v0, k, "harmonic")
            if x_traj is not None:
                if self.noise_std > 0:
                    x_traj = x_traj + torch.randn_like(x_traj) * self.noise_std
                self.samples.append(x_traj)
                self.labels.append(0)

            x0, v0, k = np.random.uniform(X0_RANGE[0], X0_RANGE[1]), \
                np.random.uniform(V0_RANGE[0], V0_RANGE[1]), np.random.uniform(K_RANGE[0], K_RANGE[1])
            x_traj = safe_generate(x0, v0, k, "quartic")
            if x_traj is not None:
                if self.noise_std > 0:
                    x_traj = x_traj + torch.randn_like(x_traj) * self.noise_std
                self.samples.append(x_traj)
                self.labels.append(1)

        self.samples = torch.stack(self.samples)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        unique, counts = torch.unique(self.labels, return_counts=True)
        print(f"📊 Dataset class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


# ============================================
# 2. Define classifier network
# ============================================

class PotentialClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


# ============================================
# 3. Training logic
# ============================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PotentialClassifier().to(device)

    if os.path.exists(MODEL_PATH) and not TRAIN_FROM_SCRATCH:
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"✅ Loaded model from {MODEL_PATH}")

    if TRAIN_FROM_SCRATCH or CONTINUE_TRAINING:
        print("🛠 Training classifier...")
        dataset = TrajectoryDataset(num_samples=1000, noise_std=NOISE_STD)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            total_loss = 0
            correct = 0
            total = 0

            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                loss = loss_fn(output, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_x.size(0)
                preds = output.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

            acc = correct / total
            print(f"Epoch {epoch:03d}: Loss = {total_loss / total:.4f}, Accuracy = {acc:.4f}")

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"💾 Saved model to {MODEL_PATH}")

