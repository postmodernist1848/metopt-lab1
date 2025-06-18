import numpy as np
from lab4.annealing import Vector

def commivoyager(x: Vector) -> float:
    n = len(x)
    res = 0
    for i in range(n):
        res += np.linalg.norm(x[(i+1) % n] - x[i])
    return res
