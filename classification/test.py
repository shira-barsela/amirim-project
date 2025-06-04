import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from train import PotentialClassifier  # Adjust import path as needed
from train import X0_RANGE
from train import V0_RANGE
from train import K_RANGE

# ===============================
# Config
# ===============================
MODEL_PATH = "trajectory_classifier.pth"
TEST_TYPE = "quad_quartic"  # Options: "quad", "quartic", "quad_quartic", "cos", "all"
OMEGA_RANGE = (1.0, 5.0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t = torch.linspace(0, 10, 100)  # 10 seconds, 100 steps


# ===============================
# Trajectory Simulators
# ===============================
def simulate_trajectory_quadratic(x0, v0, k, t):
    def rhs(_, state):
        x, v = state[0], state[1]
        return torch.tensor([v, -k * x])

    state = torch.tensor([x0, v0])
    traj = [state]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        state = state + dt * rhs(t[i - 1], state)
        traj.append(state)
    return torch.stack(traj)


def simulate_trajectory_quartic(x0, v0, k, t):
    def rhs(_, state):
        x, v = state[0], state[1]
        return torch.tensor([v, -k * x ** 3])

    state = torch.tensor([x0, v0])
    traj = [state]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        state = state + dt * rhs(t[i - 1], state)
        traj.append(state)
    return torch.stack(traj)


def simulate_trajectory_cos_potential(x0, v0, k, omega, t):
    def rhs(ti, state):
        x, v = state[0], state[1]
        return torch.tensor([v, -k * torch.cos(omega * ti) * x])

    state = torch.tensor([x0, v0])
    traj = [state]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        state = state + dt * rhs(t[i - 1], state)
        traj.append(state)
    return torch.stack(traj)


# ===============================
# Dataset Generator
# ===============================
def generate_test_set(num_samples, t, test_type):
    data = []
    labels = []

    while len(labels) < num_samples:
        x0 = torch.empty(1).uniform_(X0_RANGE[0], X0_RANGE[1]).item()
        v0 = torch.empty(1).uniform_(V0_RANGE[0], V0_RANGE[1]).item()
        k = torch.empty(1).uniform_(K_RANGE[0], K_RANGE[1]).item()

        if test_type == "quad":
            traj = simulate_trajectory_quadratic(x0, v0, k, t)
            label = 0
        elif test_type == "quartic":
            traj = simulate_trajectory_quartic(x0, v0, k, t)
            label = 1
        elif test_type == "cos":
            omega = torch.empty(1).uniform_(OMEGA_RANGE[0], OMEGA_RANGE[1]).item()
            traj = simulate_trajectory_cos_potential(x0, v0, k, omega, t)
            label = 2
        elif test_type == "all":
            choice = torch.randint(0, 3, (1,)).item()
            if choice == 0:
                traj = simulate_trajectory_quadratic(x0, v0, k, t)
            elif choice == 1:
                traj = simulate_trajectory_quartic(x0, v0, k, t)
            else:
                omega = torch.empty(1).uniform_(OMEGA_RANGE[0], OMEGA_RANGE[1]).item()
                traj = simulate_trajectory_cos_potential(x0, v0, k, omega, t)
            label = choice
        elif test_type == "quad_quartic":
            choice = torch.randint(0, 2, (1,)).item()
            if choice == 0:
                traj = simulate_trajectory_quadratic(x0, v0, k, t)
            elif choice == 1:
                traj = simulate_trajectory_quartic(x0, v0, k, t)
            else:
                raise ValueError("Invalid random choice int")
            label = choice
        else:
            raise ValueError("Invalid test type")

        if not (torch.isnan(traj[:, 0]).any() or torch.abs(traj[:, 0]).max() > 1e3):
            data.append(traj[:, 0])
            labels.append(label)

    return torch.stack(data), torch.tensor(labels)


# ===============================
# Load Model
# ===============================
model = PotentialClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ===============================
# Generate Test Set
# ===============================
x_test, y_test = generate_test_set(15, t, test_type=TEST_TYPE)
x_test = x_test.to(device)
y_test = y_test.to(device)

# ===============================
# Predict
# ===============================
with torch.no_grad():
    logits = model(x_test)
    predictions = torch.argmax(logits, dim=1)
    print(f"Test type: {TEST_TYPE}")
    if TEST_TYPE == "quad" or TEST_TYPE == "quartic" or TEST_TYPE == "quad_quartic":
        correct = (predictions == y_test).sum().item()
        total = y_test.size(0)
        accuracy = correct / total
        print(f"🎯 Test Accuracy: {accuracy:.2%}")

# ===============================
# Plot Some Results
# ===============================
for i in range(5):
    plt.figure(figsize=(8, 4))
    plt.plot(t.cpu(), x_test[i].cpu(), label="x(t)")
    plt.title(f"True: {y_test[i].item()}, Predicted: {predictions[i].item()}")
    plt.xlabel("Time")
    plt.ylabel("x(t)")
    plt.grid(True)
    plt.legend()
    plt.show()
