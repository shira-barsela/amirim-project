import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

import constants as cons
import functions as func

f_func = func.zero_f
g_chosen_func = func.harmonic_function

x_array = np.zeros(cons.TIME_STEPS)
f_array = np.zeros(cons.TIME_STEPS)

def fill_f_array(f_array, f: Callable[[float], float]) -> None:
    for i in range(cons.TIME_STEPS):
        t = func.int_to_time(i)
        f_array[i] = f(t)

def calc_xi(g_func, i: int) -> None:
    fi = f_array[i]
    if i == 0:
        x_array[0] = cons.X0
        return

    sum = 0
    sum += 0.5 * g_func(i, 0) * x_array[0]

    for j in range(1, i):
        sum += g_func(i, j) * x_array[j]

    sum *= cons.DELTA_T

    denom = 1.0 - 0.5 * cons.DELTA_T * g_func(i, i)
    x_array[i] = (fi + sum) / denom

def plot_x_array():
    t = np.arange(cons.TIME_STEPS) * cons.DELTA_T
    plt.figure(figsize=(8, 4))
    plt.plot(t, x_array, label="x(t)")
    plt.xlabel("time")
    plt.ylabel("x")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def _step_contribution_arrays(i: int, g_func, x_array: np.ndarray):
    """
    Build arrays for a single step i:
    - js: indices 0..i
    - ts: times t_j
    - Gij: kernel values G(t_i - t_j) for j=0..i
    - w: trapezoid weights (0.5 at j=0 and j=i, 1 otherwise)
    - contrib: Δt * w_j * Gij * x_j   (note: the j=i term is moved to denom in your solver)
    """
    js = np.arange(i + 1)
    ts = js * cons.DELTA_T
    Gij = np.array([g_func(i, j) for j in js], dtype=float)

    # trapezoidal weights
    w = np.ones_like(js, dtype=float)
    w[0] = 0.5
    w[-1] = 0.5

    # contributions used on RHS are only j=0..i-1 (you move j=i to the denominator)
    contrib = cons.DELTA_T * w * Gij * x_array[js]
    contrib_for_rhs = contrib.copy()
    contrib_for_rhs[-1] = 0.0  # exclude j=i from RHS (your denominator handles it)

    return js, ts, Gij, w, contrib_for_rhs, Gij[-1]  # also return Gii

def visualize_kernel_and_x(i: int, g_func, x_array: np.ndarray, title_prefix: str = ""):
    """
    Three panels:
      1) x(t) over full time with markers on the first few points and a line at t_i
      2) kernel row G(t_i - t_j) for j=0..i (stem plot)
      3) per-index contributions Δt * w_j * Gij * x_j (bars), showing which j drives the update

    Notes:
      - The j=i term is *not* included in the contributions bar (it sits in the denominator).
      - Use this to tune EPS in constants.py so more than just the self-term contributes.
    """
    if i < 0 or i >= len(x_array):
        raise ValueError("i out of range")

    js, ts, Gij, w, contrib_rhs, Gii = _step_contribution_arrays(i, g_func, x_array)

    t_all = np.arange(len(x_array)) * cons.DELTA_T
    ti = i * cons.DELTA_T

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (1) x(t)
    ax = axes[0]
    ax.plot(t_all, x_array, label="x(t)")
    m = min(10, len(x_array))
    ax.scatter(t_all[:m], x_array[:m], s=16)  # mark early points
    ax.axvline(ti, ls='--', alpha=0.5)
    ax.set_title(f"{title_prefix} x(t)  (i={i}, t_i={ti:.3g}s)")
    ax.set_xlabel("time")
    ax.set_ylabel("x")
    ax.grid(True)
    ax.legend()

    # (2) kernel row
    ax = axes[1]
    markerline, stemlines, baseline = ax.stem(ts, Gij)  # no use_line_collection (deprecated)
    ax.axvline(ti, ls='--', alpha=0.5)
    ax.set_title("Kernel row  G(t_i - t_j)")
    ax.set_xlabel("t_j")
    ax.set_ylabel("G_ij")
    ax.grid(True)

    # (3) contributions
    ax = axes[2]
    ax.bar(ts, contrib_rhs, width=0.8 * cons.DELTA_T)
    ax.set_title(r"Per-index RHS contribution  $\Delta t\,w_j\,G_{ij}\,x_j$ (j=0..i-1)")
    ax.set_xlabel("t_j")
    ax.set_ylabel("contribution")
    ax.grid(True)

    # helpful text box: denominator term and EPS
    denom = 1.0 - 0.5 * cons.DELTA_T * Gii
    txt = (f"Gii = {Gii:.3e}\n"
           f"0.5·Δt·Gii = {0.5*cons.DELTA_T*Gii:.3e}\n"
           f"denom = {denom:.3e}\n"
           f"EPS = {getattr(cons, 'EPS', 'N/A')}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.7"))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # define G_function_by_indexes with the current G function
    def g_by_idx(t1: int, t2: int) -> float:
        return func.g_function_by_ind(g_chosen_func, t1, t2)

    # fill f array
    fill_f_array(f_array, f_func)

    # calc x array
    for i in range(cons.TIME_STEPS):
        calc_xi(g_by_idx, i)

    # plot x array
    # print(x_array)
    # plot_x_array()

    # visualize at a few steps
    visualize_kernel_and_x(i=5, g_func=g_by_idx, x_array=x_array, title_prefix="EPS-tuning: ")
    visualize_kernel_and_x(i=10, g_func=g_by_idx, x_array=x_array, title_prefix="EPS-tuning: ")
