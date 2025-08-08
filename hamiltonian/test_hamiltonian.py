# test_hamiltonian.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from model_hamiltonian import HamiltonPredictorCNN
from data_hamiltonian import generate_dataset, harmonic_trajectory, DEFAULT_DURATION, DEFAULT_TIME_STEPS
from dataset_hamiltonian import compute_v_and_a
from data_hamiltonian import X0_RANGE, V0_RANGE, K_RANGE
import random


# ========= CONFIG =========
MODEL_PATH = "models/hamiltonian_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = DEFAULT_DURATION / (DEFAULT_TIME_STEPS - 1)  # default time step


# ========= Batch Evaluation =========
def run_batch_evaluation(model, num_trajectories=50, num_plots=5, batch_size=64):
    class InMemoryHamiltonDataset(Dataset):
        def __init__(self, data_array, dt):
            self.inputs = data_array[:, 4:-1]
            self.targets = data_array[:, -1]
            self.dt = dt

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            x = self.inputs[idx].astype(np.float32)
            v, a = compute_v_and_a(x, self.dt)
            input_tensor = torch.tensor(np.stack([x, v, a]), dtype=torch.float32)
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return input_tensor, target

    print("🔄 Generating fresh test data...")
    raw_data = generate_dataset(num_trajectories)
    dataset = InMemoryHamiltonDataset(raw_data, dt=DT)
    loader = DataLoader(dataset, batch_size=batch_size)

    print("📊 Evaluating...")
    model.eval()
    all_targets, all_preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            all_targets.append(y.cpu().numpy())
            all_preds.append(pred.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    mse = np.mean((all_targets - all_preds) ** 2)
    print(f"📉 Test MSE on fresh data: {mse:.6f}")

    print("📈 Plotting random samples...")
    indices = np.random.choice(len(dataset), size=num_plots, replace=False)
    t_values = np.linspace(0, DT * 10, 11)

    for i, idx in enumerate(indices):
        x, y_true = dataset[idx]
        x_vals = x[0].numpy()
        input_tensor = x.unsqueeze(0).to(DEVICE)
        y_pred = model(input_tensor).item()

        plt.figure(figsize=(8, 4))
        plt.plot(t_values[:10], x_vals, marker='o', label="Input x(t)")
        plt.plot(t_values[10], y_true.item(), 'go', label="True x₁₀")
        plt.plot(t_values[10], y_pred, 'rx', label="Predicted x₁₀")
        plt.title(f"Sample #{i+1} (Index {idx})")
        plt.xlabel("Time (s)")
        plt.ylabel("Position x(t)")
        plt.grid(True)
        plt.legend()
        plt.show()


# ========= Multi-Step Rollout =========
def rollout_from_initial_condition(model, x0, v0, k, steps=DEFAULT_TIME_STEPS):
    t = np.linspace(0, DT * (steps - 1), steps)
    print("duration: ", DT * (steps - 1), ", steps: ", steps)
    true_x = harmonic_trajectory(x0, v0, k, t)

    current_window = true_x[:10].astype(np.float32).copy()
    predicted = list(current_window)

    model.eval()
    for i in range(10, steps):
        v, a = compute_v_and_a(current_window, DT)
        input_tensor = torch.tensor(np.stack([current_window, v, a]), dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            next_x = model(input_tensor).item()

        predicted.append(next_x)
        current_window = np.roll(current_window, -1)
        current_window[-1] = next_x

    print("t\ttrue_x\tpredicted")
    for ti, tx, px in zip(t, true_x, predicted):
        print(f"{ti:.4f}\t{tx:.6f}\t{px:.6f}")

    end_idx = int(len(t) * 0.6)  # 20% of the total samples
    plt.figure(figsize=(10, 4))
    plt.plot(t[:end_idx], true_x[:end_idx], label="Analytic Trajectory", linewidth=2)
    plt.plot(t[:end_idx], predicted[:end_idx], '--', label="Predicted (Multi-step)", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("x(t)")
    plt.title(f"Rollout from x₀={x0}, v₀={v0}, k={k}")
    plt.grid(True)
    plt.legend()
    plt.show()


# ========= MAIN DRIVER =========
if __name__ == "__main__":
    MODE = "rollout"  # "batch" / "rollout"

    model = HamiltonPredictorCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"✅ Loaded model from {MODEL_PATH}")

    if MODE == "batch":
        run_batch_evaluation(model)
    elif MODE == "rollout":
        # Try manual or random values
        x0, v0, k = 2.0, 0.3, 1.2
        # x0 = round(random.uniform(*X0_RANGE), 2)
        # v0 = round(random.uniform(*V0_RANGE), 2)
        # k = round(random.uniform(*K_RANGE), 2)

        rollout_from_initial_condition(model, x0, v0, k)
