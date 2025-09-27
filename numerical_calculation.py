import numpy as np
from typing import Callable
import constants as cons
import functions as func

f_func = func.zero_f
G_func = func.harmonic_function

x_array = np.zeros(cons.TIME_STEPS)
f_array = np.zeros(cons.TIME_STEPS)

def fill_f_array(f_array, f: Callable[[float], float]) -> None:
    for i in range(cons.TIME_STEPS):
        t = func.int_to_time(i)
        f_array[i] = f(t)

def calc_xi(i: int) -> None:
    fi = f_array[i] # sum = f[i]
    if i == 0:
        x_array[0] = cons.X0
    sum = 0
    sum += func.G_function_by_indexes(i, 0) * x_array[0] / 2

    for j in range(1, i+1):
        sum + func.G_function_by_indexes(i, j) * x_array[i]
    sum *= cons.DELTA_T
    sum += fi

    denominator = 1 - (cons.DELTA_T * func.G_function_by_indexes(i, i) / 2)

    x_array[i] = sum / denominator


if __name__ == "__main__":
    # fill f array
    fill_f_array(f_array, f_func)
    # calc x array
    for i in range(cons.TIME_STEPS):
        calc_xi(i)
