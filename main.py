import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Protocol
import numpy as np
from typing import Callable, Protocol
import random

type LearningRateFunc = Callable[[int], float]
type StopCondition = Callable[[np.ndarray, np.ndarray], bool]

MAX_ITERATION_LIMIT = 10000

type Func = Callable[[float], float]


class BiFunc(Protocol):
    def __call__(self, x: np.ndarray) -> float: ...
    def gradient(self, x: np.ndarray) -> np.ndarray: ...


class Quadratic:
    A: np.ndarray
    B: np.ndarray
    C: float

    def __init__(self, a: np.ndarray, b: np.ndarray, c: float):
        self.A = a
        self.B = b
        self.C = c

    def __call__(self, x: np.ndarray) -> float:
        # x^T A x + x^T B x + C
        return float(x.T @ self.A @ x + self.B @ x + self.C)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2 * self.A @ x + self.B

GRADIENT_DESCENT_LOGGING = True

def gradient_descent(x_0: np.ndarray,
                     func: BiFunc,
                     step_selector: Callable[[int, np.ndarray, np.ndarray, BiFunc], float],
                     sc: StopCondition) -> np.ndarray:
    x = x_0.copy()
    k = 0
    trajectory = [x.copy()]

    while True:
        grad = func.gradient(x)
        prev = x.copy()

        h = step_selector(k, x, grad, func)

        x = x - h * grad
        trajectory.append(x.copy())

        if GRADIENT_DESCENT_LOGGING:
            print(f'k: {k}, x: {x}, f: {func(x)}')
        if sc(x, prev) or k > MAX_ITERATION_LIMIT:
            break

        k += 1

    return np.array(trajectory)


def learning_rate_scheduling(x_0: np.ndarray,
                             func: BiFunc,
                             h: LearningRateFunc,
                             sc: StopCondition) -> np.ndarray:
    return gradient_descent(x_0, func,
                            lambda k, x, grad, func: h(k),
                            sc)


def steepest_gradient_descent_dichotomy(x_0: np.ndarray,
                              func: BiFunc,
                              eps: float,
                              sc: StopCondition) -> np.ndarray:
    def step_selector(k, x, grad, func):
        def f(h): return func(x - h * grad)
        return dichotomy(0, find_b(f), f, eps)

    return gradient_descent(x_0, func, step_selector, sc)

def steepest_gradient_descent_armijo(x_0: np.ndarray,
                              func: BiFunc,
                              eps: float,
                              sc: StopCondition) -> np.ndarray:
    def step_selector(k, x, grad, func):
        return armijo(x, func)

    return gradient_descent(x_0, func, step_selector, sc)


def find_b(func: Func) -> float:
    MAX_X = 10000
    curr_val = func(0)
    b = 1
    while (abs(b) < MAX_X and func(b) < curr_val):
        b *= 2
    return b


def dichotomy(a: float, b: float, func: Func, eps: float) -> float:
    c = (a + b) / 2
    c_val = func(c)
    for _ in range(MAX_ITERATION_LIMIT):
        if (b - a < eps):
            return c
        c = (a + b) / 2

        left = (a + c) / 2
        left_val = func(left)
        if (left_val < c_val):
            b = c
            c_val = left_val
            continue

        right = (c + b) / 2
        right_val = func(right)

        if (right_val < c_val):
            a = c
            c_val = right_val
            continue

        a = left
        b = right
    return c


def armijo(x_k: np.ndarray, func: Quadratic) -> float:
    grad = func.gradient(x_k)
    derivative: float = -float(grad @ grad.T)
    c1 = random.random()
    q = random.random()
    alpha: float = -abs(func(x_k)) / (derivative * c1)
    
    for _ in range(MAX_ITERATION_LIMIT):
        l_alpha: float = func(x_k) + c1*alpha*derivative
        if func(x_k - alpha*grad) < float(l_alpha):
            break
        alpha = q*alpha
    return float(alpha)
