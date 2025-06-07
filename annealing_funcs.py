import numpy as np
from lib.funcs import armijo, wolfe
from lib.funcs import BiFunc
from annealing import Vector

def commivoyager_F(xi: Vector) -> Vector:
    i = np.random.randint(0, len(xi))
    j = np.random.randint(0, len(xi))
    i, j = min(i, j), max(i, j)
    return np.concatenate((xi[:i], xi[i:j+1][::-1], xi[j+1:]))

def constraint_annealing_F(x: np.ndarray, func: BiFunc, grad: np.ndarray) -> np.ndarray:
    return x - 0.1*grad

def armijo_annealing_F(x: np.ndarray, func: BiFunc, grad: np.ndarray) -> np.ndarray:
    return x - armijo(x, func, grad)*grad

def wolfe_annealing_F(x: np.ndarray, func: BiFunc, grad: np.ndarray) -> np.ndarray:
    return x - wolfe(x, func, grad)[0]*grad

