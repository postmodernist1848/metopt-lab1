import numpy as np
from lab4.annealing import Vector

def commivoyager_F(xi: Vector) -> Vector:
    i = np.random.randint(0, len(xi))
    j = np.random.randint(0, len(xi))
    i, j = min(i, j), max(i, j)
    return np.concatenate((xi[:i], xi[i:j+1][::-1], xi[j+1:]))

def random_F(x: Vector) -> Vector:
    return x + np.random.uniform(-0.3, 0.3, x.shape)
