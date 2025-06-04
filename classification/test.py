import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from model import TrajectoryCNN
from data_generation import generate_dataset, compute_velocity_acceleration

MODEL_PATH = "trained_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OMEGA_RANGE = (5.0, 10.0)


def simulate_cos_potential(x0, v0, k, omega, t, power):
    dt = t[1] - t[0]
    x, v = [x0], [v0]
    for i in range(1, len(t)):
        xi, vi = x[-1], v[-1]
        ti = t[i - 1]

        # a1 at start
        a1 = -k * 0.5 * power * np.cos(omega * ti) * xi ** (power - 1)
        vi_half = vi + 0.5 * a1 * dt
        xi_half = xi + 0.5 * vi * dt

        # a2 at midpoint
        a2 = -k * 0.5 * power * np.cos(omega * (ti + 0.5 * dt)) * xi_half ** (power - 1)
        vi_new = vi + a2 * dt
        xi_new = xi + vi_half * dt

        x.append(xi_new)
        v.append(vi_new)
    return np.array(x)

def evaluate_harmonic_quartic_testset(csv_path=None, num_samples=200, time_steps=200, batch_size=32):
    if csv_path:
        dataset = TensorDataset(*TrajectoryCNN.load_from_csv(csv_path))
    else:
        t = np.linspace(0, 10, time_steps)
        df = generate_dataset(num_samples=num_samples, t=t)
        all_x, all_y, all_k = [], [], []
        for i in range(len(df)):
            x = np.array(df.iloc[i]["x"])
            v = np.array(df.iloc[i]["v"])
            a = np.array(df.iloc[i]["a"])
            traj = np.stack([x, v, a], axis=0)
            all_x.append(traj)
            all_y.append(int(df.iloc[i]["label"]))
            all_k.append(float(df.iloc[i]["k"]))
        x_tensor = torch.tensor(np.array(all_x), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(all_y), dtype=torch.long)
        k_tensor = torch.tensor(np.array(all_k), dtype=torch.float32)
        dataset = TensorDataset(x_tensor, y_tensor, k_tensor)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = TrajectoryCNN(time_steps=time_steps)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    TP = FP = TN = FN = 0
    predictions = []
    with torch.no_grad():
        for batch_x, batch_y, batch_k in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            logits, _ = model(batch_x)
            preds = logits.argmax(dim=1)
            for i in range(len(batch_y)):
                y_true = batch_y[i].item()
                y_pred = preds[i].item()
                if y_true == 0 and y_pred == 0: TP += 1
                elif y_true == 0 and y_pred == 1: FN += 1
                elif y_true == 1 and y_pred == 1: TN += 1
                elif y_true == 1 and y_pred == 0: FP += 1
                predictions.append((y_true, y_pred, batch_x[i].cpu().numpy(), batch_k[i].item()))

    print("Harmonic vs Quartic Test Results:")
    print(f"TP: {TP} | FN: {FN} | TN: {TN} | FP: {FP}")
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(f"Accuracy: {accuracy:.2%}")

    import random
    label_map = {0: "harmonic", 1: "quartic"}
    for y_true, y_pred, traj, k in random.sample(predictions, min(5, len(predictions))):
        x = traj[0]  # channel 0 is x(t)
        t = np.linspace(0, 10, len(x))
        plt.figure(figsize=(8, 3))
        plt.plot(t, x, label="x(t)")
        plt.title(f"True: {label_map[y_true]} | Pred: {label_map[y_pred]} (k={k:.2f})")
        plt.xlabel("Time")
        plt.ylabel("x(t)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



def evaluate_effective_cos_testset(num_samples=200, time_steps=200, duration=10.0, batch_size=32):
    t = np.linspace(0, duration, time_steps)
    all_x, all_y, all_k, all_info = [], [], [], []

    while len(all_x) < num_samples:
        x0 = np.random.uniform(-2.0, 2.0)
        v0 = np.random.uniform(-2.0, 2.0)
        k = np.random.uniform(0.5, 3.0)
        omega = np.random.uniform(*OMEGA_RANGE)
        label = np.random.randint(0, 2)
        power = 2 if label == 0 else 4

        try:
            x = simulate_cos_potential(x0, v0, k, omega, t, power)
        except:
            continue
        if np.isnan(x).any() or np.abs(x).max() > 1e2:
            continue

        v, a = compute_velocity_acceleration(x, t)
        traj = np.stack([x, v, a], axis=0)
        all_x.append(traj)
        all_y.append(label)
        all_k.append(k)
        all_info.append((label, k, omega, x0, v0, x))

    x_tensor = torch.tensor(np.array(all_x), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(all_y), dtype=torch.long)
    k_tensor = torch.tensor(np.array(all_k), dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor, k_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = TrajectoryCNN(time_steps=time_steps)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    TP = FP = TN = FN = 0
    with torch.no_grad():
        for batch_x, batch_y, _ in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            logits, _ = model(batch_x)
            preds = logits.argmax(dim=1)
            for y_true, y_pred in zip(batch_y.cpu(), preds.cpu()):
                if y_true == 0 and y_pred == 0: TP += 1
                elif y_true == 0 and y_pred == 1: FN += 1
                elif y_true == 1 and y_pred == 1: TN += 1
                elif y_true == 1 and y_pred == 0: FP += 1

    print("\n🧪 Cosine Potential Test Results:")
    print(f"TP: {TP} | FN: {FN} | TN: {TN} | FP: {FP}")
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(f"Accuracy: {accuracy:.2%}\n")

    import random
    for i in random.sample(range(len(all_info)), min(5, len(all_info))):
        label, k, omega, x0, v0, x = all_info[i]
        x_tensor = torch.tensor(np.stack([x, *compute_velocity_acceleration(x, t)], axis=0), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _ = model(x_tensor)
            pred = torch.argmax(logits, dim=1).item()
        label_map = {0: "harmonic", 1: "quartic"}
        plt.figure(figsize=(8, 3))
        plt.plot(t, x, label="x(t)")
        plt.title(f"True: {label_map[label]} | Pred: {label_map[pred]}\n(k={k:.2f}, omega={omega:.2f}, x0={x0:.2f}, v0={v0:.2f})")
        plt.xlabel("Time")
        plt.ylabel("x(t)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def predict_single_trajectory(x: np.ndarray, v: np.ndarray, a: np.ndarray, time_steps: int = 200):
    assert x.shape == v.shape == a.shape == (time_steps,), "Each input must be of shape (time_steps,)"
    traj = np.stack([x, v, a], axis=0)  # shape: (3, time_steps)
    input_tensor = torch.tensor(traj, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # shape: (1, 3, time_steps)

    model = TrajectoryCNN(time_steps=time_steps)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        logits, k_pred = model(input_tensor)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1)[0, predicted_class].item()
        k_value = k_pred.item()

    label_map = {0: "harmonic", 1: "quartic"}
    print(f"🔍 Prediction: {label_map[predicted_class]} (confidence: {confidence:.2%}, predicted k: {k_value:.4f})")
    return predicted_class, confidence, k_value
