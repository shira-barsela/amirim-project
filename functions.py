import typing
from typing import Callable
import numpy as np
import constants as cons
from constants import EPS


# G functions
def delta_approx(x: float) -> float:
    """Approximation of δ(x) with Gaussian of width eps."""
    return np.exp(-(x/EPS)**2) / (np.sqrt(np.pi) * EPS)

def delta_prime(x: float) -> float:
    """Approximation of δ'(x)"""
    return -2 * x / (EPS**2) * delta_approx(x, EPS)

def delta_double_prime(x: float) -> float:
    """Approximation of δ''(x)"""
    return (4 * x**2 / EPS**4 - 2 / EPS**2) * delta_approx(x, EPS)

def harmonic_function(x: float) -> float:
    """
    U = 0.5kx^2 --> U` = kx & -U` = mx``
    x`` = ∫ x(t) δ``(t) dt
    -->  -U`(x) = ∫ x(t') m δ``(t'-t) dt'
    -->  x(t) = U`/k = - ∫ x(t') m δ``(t'-t) dt' / k
                     = ∫ x(t') (-m/k) δ``(t'-t) dt'
    """
    return (-cons.M/cons.K) * delta_double_prime(x)

# f functions
def zero_f(t: float) -> float:
    return 0

# other
def int_to_time(i: int) -> float:
    return i * cons.DELTA_T

def G_function_by_indexes(g: Callable[[float], float], t1: int, t2: int) -> float:
    return g(int_to_time(t1) - int_to_time(t2))





