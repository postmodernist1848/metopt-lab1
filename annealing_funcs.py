import numpy as np
from lib.funcs import armijo, wolfe
from lib.funcs import BiFunc

def commivoyager(x: np.ndarray) -> float:
    n = len(x)
    res = 0
    for i in range(n):
        res += np.linalg.norm(x[(i+1) % n] - x[i])
    return res

def commivoyager_F(xi: np.ndarray) -> np.ndarray:
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

