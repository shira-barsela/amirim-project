import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import TrajectoryCNN
from data_generation import generate_dataset, compute_velocity_acceleration

MODEL_PATH = "trained_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_test_tensor_dataset(num_samples=200, time_steps=200, duration=10.0):
    t = np.linspace(0, duration, time_steps)
    df = generate_dataset(num_samples=num_samples, t=t)

    all_x = []
    all_y = []
    all_k = []

    for i in range(len(df)):
        x = np.array(df.iloc[i]["x"])
        v = np.array(df.iloc[i]["v"])
        a = np.array(df.iloc[i]["a"])
        traj = np.stack([x, v, a], axis=0)  # (3, time_steps)
        label = int(df.iloc[i]["label"])
        k = float(df.iloc[i]["k"])
        all_x.append(traj)
        all_y.append(label)
        all_k.append(k)

    x_tensor = torch.tensor(np.array(all_x), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(all_y), dtype=torch.long)
    k_tensor = torch.tensor(np.array(all_k), dtype=torch.float32)
    return TensorDataset(x_tensor, y_tensor, k_tensor)


def evaluate_dataset(num_samples=200, time_steps=200, batch_size=32):
    dataset = generate_test_tensor_dataset(num_samples=num_samples, time_steps=time_steps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = TrajectoryCNN(time_steps=time_steps)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    correct, total = 0, 0
    k_errors = []
    with torch.no_grad():
        for batch_x, batch_y, batch_k in dataloader:
            batch_x, batch_y, batch_k = batch_x.to(DEVICE), batch_y.to(DEVICE), batch_k.to(DEVICE)
            logits, k_pred = model(batch_x)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            k_errors.extend((k_pred.squeeze() - batch_k).abs().cpu().tolist())

    accuracy = correct / total
    mae_k = sum(k_errors) / len(k_errors)
    print(f"🧪 Test Accuracy: {accuracy:.2%} | MAE(k): {mae_k:.4f}")


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
