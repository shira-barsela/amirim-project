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
    fi = f_array[i] # sum = f[i]
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
    print(x_array)
    plot_x_array()
