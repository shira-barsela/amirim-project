# test_hamiltonian.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

from model_hamiltonian import HamiltonPredictorCNN
from data_hamiltonian import generate_dataset, DEFAULT_DURATION, DEFAULT_TIME_STEPS
from dataset_hamiltonian import HamiltonDataset

# ========= CONFIG =========
NUM_TRAJECTORIES = 50
MODEL_PATH = "models/hamiltonian_cnn.pth"
BATCH_SIZE = 64
NUM_PLOTS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========= Custom Dataset Using In-Memory Array =========
class InMemoryHamiltonDataset(Dataset):
    def __init__(self, data_array, duration=DEFAULT_DURATION, time_steps=DEFAULT_TIME_STEPS):
        self.raw_data = data_array
        self.inputs = data_array[:, 4:-1]  # x_0 to x_9
        self.targets = data_array[:, -1]   # target x_10
        self.dt = duration / (time_steps - 1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx].astype(np.float32)

        # Compute velocity
        v = np.zeros_like(x)
        v[1:-1] = (x[2:] - x[:-2]) / (2 * self.dt)
        v[0] = (x[1] - x[0]) / self.dt
        v[-1] = (x[-1] - x[-2]) / self.dt

        # Compute acceleration
        a = np.zeros_like(x)
        a[1:-1] = (x[2:] - 2 * x[1:-1] + x[:-2]) / (self.dt ** 2)
        a[0] = (x[2] - 2 * x[1] + x[0]) / (self.dt ** 2)
        a[-1] = (x[-1] - 2 * x[-2] + x[-3]) / (self.dt ** 2)

        input_tensor = torch.tensor(np.stack([x, v, a]), dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return input_tensor, target


# ========= Generate Fresh Test Data =========
print("🔄 Generating fresh test dataset...")
raw_data = generate_dataset(NUM_TRAJECTORIES)
test_dataset = InMemoryHamiltonDataset(raw_data)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ========= Load Model =========
model = HamiltonPredictorCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Loaded model from {MODEL_PATH}")

# ========= Evaluate =========
all_targets = []
all_preds = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        all_targets.append(y.cpu().numpy())
        all_preds.append(pred.cpu().numpy())

all_targets = np.concatenate(all_targets)
all_preds = np.concatenate(all_preds)
mse = np.mean((all_targets - all_preds) ** 2)
print(f"📉 Test MSE on fresh data: {mse:.6f}")

# ========= Plot Predictions =========
# Randomly select which samples to plot
random_indices = np.random.choice(len(test_dataset), size=NUM_PLOTS, replace=False)

# Time axis for 11 points (0 to 10 steps of dt)
dt = DEFAULT_DURATION / (DEFAULT_TIME_STEPS - 1)
t_values = np.linspace(0, dt * 10, 11)

for plot_idx, i in enumerate(random_indices):
    x, y_true = test_dataset[i]
    x_vals = x[0].numpy()
    input_tensor = x.unsqueeze(0).to(DEVICE)
    y_pred = model(input_tensor).item()

    plt.figure(figsize=(8, 4))
    plt.plot(t_values[:10], x_vals, marker='o', label="Input x(t)")
    plt.plot(t_values[10], y_true.item(), 'go', label="True x₁₀")
    plt.plot(t_values[10], y_pred, 'rx', label="Predicted x₁₀")
    plt.title(f"Sample #{i+1}")
    plt.xlabel("Time Step")
    plt.ylabel("Position x(t)")
    plt.grid(True)
    plt.legend()
    plt.show()
