import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import TrajectoryDataset
from model import TrajectoryCNN
import numpy as np

MODEL_PATH = "trained_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_dataset(csv_path: str, batch_size: int = 32, time_steps: int = 200):
    dataset = TrajectoryDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = TrajectoryCNN(time_steps=time_steps)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total
    print(f"🧪 Test Accuracy: {accuracy:.2%}")


def predict_single_trajectory(x: np.ndarray, v: np.ndarray, a: np.ndarray, time_steps: int = 200):
    assert x.shape == v.shape == a.shape == (time_steps,), "Each input must be of shape (time_steps,)"
    traj = np.stack([x, v, a], axis=0)  # shape: (3, time_steps)
    input_tensor = torch.tensor(traj, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # shape: (1, 3, time_steps)

    model = TrajectoryCNN(time_steps=time_steps)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, predicted_class].item()

    label_map = {0: "harmonic", 1: "quartic"}
    print(f"🔍 Prediction: {label_map[predicted_class]} (confidence: {confidence:.2%})")
    return predicted_class, confidence
