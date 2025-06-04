import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================
# Configuration flags
# ============================================
MODEL_PATH = "neural_ode_fit_model.pth"
TRAIN_FROM_SCRATCH = False
CONTINUE_TRAINING = True

# ============================================
# 1. Generate synthetic dataset for training
# ============================================

def generate_trajectory(x0, v0, k_over_m, omega, t):
    """
    Generate x(t) using the true differential equation:
        d^2x/dt^2 + (k/m) * cos(wt) * x = 0

    Returns only x(t) values.
    """
    def true_rhs(t, state):
        x, v = state[0], state[1]
        dxdt = v
        dvdt = -k_over_m * torch.cos(omega * t) * x
        return torch.tensor([dxdt, dvdt])

    state0 = torch.tensor([x0, v0])
    with torch.no_grad():
        trajectory = odeint(true_rhs, state0, t)
    return trajectory  # shape: [len(t), 2] (x and v)

# Create dataset
num_samples = 200
num_timesteps = 100
t = torch.linspace(0, 10, num_timesteps)
x_data = []

for _ in range(num_samples):
    x0 = np.random.uniform(-2, 2)
    v0 = np.random.uniform(-2, 2)
    k_over_m = np.random.uniform(0.5, 3.0)
    omega = np.random.uniform(1.0, 5.0)
    traj = generate_trajectory(x0, v0, k_over_m, omega, t)
    x_data.append(traj)

x_data = torch.stack(x_data)  # shape: [num_samples, len(t), 2]

# ============================================
# 2. Define neural network for f(t, [x, v])
# ============================================

class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, t, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)  # shape [1, 2]

        t_expanded = t.expand(state.shape[0], 1)  # shape [batch_size, 1]
        input = torch.cat([t_expanded, state], dim=1)  # shape [batch_size, 3]
        return self.net(input)


# ============================================
# 3. Training loop
# ============================================

dynamics = ODEFunc()

if os.path.exists(MODEL_PATH) and not TRAIN_FROM_SCRATCH:
    dynamics.load_state_dict(torch.load(MODEL_PATH))
    print(f"✅ Loaded model from {MODEL_PATH}")

if TRAIN_FROM_SCRATCH or CONTINUE_TRAINING:
    print("🛠 Training Neural ODE model...")
    optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    epochs = 300
    batch_size = 32

    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            idx = perm[i:i + batch_size]
            batch = x_data[idx]  # shape: [batch_size, T, 2]
            x0_batch = batch[:, 0, :]            # shape: [batch_size, 2]
            target_traj = batch                  # shape: [batch_size, T, 2]

            pred_traj = odeint(dynamics, x0_batch, t)  # shape: [T, batch_size, 2]
            pred_traj = pred_traj.permute(1, 0, 2)     # shape: [batch_size, T, 2]

            loss = loss_fn(pred_traj, target_traj)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    torch.save(dynamics.state_dict(), MODEL_PATH)
    print(f"💾 Saved model to {MODEL_PATH}")

# ============================================
# 4. Test on new trajectory
# ============================================

# True parameters for test
x0_test, v0_test = 1.0, 0.0
k_over_m_test, omega_test = 2.0, 2.5

true_traj = generate_trajectory(x0_test, v0_test, k_over_m_test, omega_test, t)
true_traj_x = true_traj[:, 0]

with torch.no_grad():
    pred_traj = odeint(dynamics, true_traj[0], t)
    pred_traj_x = pred_traj[:, 0]

# ============================================
# 5. Plot result
# ============================================

plt.figure(figsize=(10, 6))
plt.plot(t, true_traj_x, label="True x(t)", linestyle="--")
plt.plot(t, pred_traj_x, label="Learned x(t)")
plt.title("Neural ODE Learned Dynamics vs True Trajectory")
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.legend()
plt.grid(True)
plt.show()
