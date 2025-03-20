from typing import Callable, Protocol
import numpy as np
import random

MAX_ITERATION_LIMIT = 10000

type LearningRateFunc = Callable[[int], float]
type StopCondition = Callable[[np.ndarray, np.ndarray], bool]
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


class BiFuncCallableWrapper:
    f: Callable[[float, float], float]

    def __init__(self, f: Callable[[float, float], float]):
        self.f = f

    def __call__(self, x: np.ndarray) -> float:
        return self.f(x[0], x[1])

    def gradient(self, x: np.ndarray) -> np.ndarray:
        h = 0.001
        d1 = (self.f(x[0] + h, x[1]) - self.f(x[0] - h, x[1])) / (2*h)
        d2 = (self.f(x[0], x[1] + h) - self.f(x[0], x[1] - h)) / (2*h)
        g = np.array([d1, d2])
        return g


class BiFuncStatsDecorator:
    f: BiFunc
    call_count: int = 0
    gradient_count: int = 0

    def __init__(self, f: BiFunc):
        self.f = f

    def __call__(self, x: np.ndarray):
        self.call_count += 1
        return self.f.__call__(x)

    def gradient(self, x: np.ndarray):
        self.gradient_count += 1
        return self.f.gradient(x)

    def reset(self):
        self.gradient_count = self.call_count = 0


def gradient_descent(x_0: np.ndarray,
                     func: BiFunc,
                     step_selector: Callable[[int, np.ndarray, np.ndarray, BiFunc], float],
                     sc: StopCondition) -> np.ndarray:
    x = x_0.copy()
    k = 0
    trajectory = [x.copy()]

    while True:
        grad = func.gradient(x)

        if (grad.T @ grad) < 1e-9:
            EPS = 1e-7

            def random_eps():
                return random.choice([-EPS, EPS])
            grad += np.array([random_eps(), random_eps()])

        prev = x.copy()

        h = step_selector(k, x, grad, func)
        x = x - h * grad
        trajectory.append(x.copy())

        if sc(x, prev) or k > MAX_ITERATION_LIMIT:
            break
        if False:
            print(f'k: {k}, x: {x}, f: {func(x)}')

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
                                     sc: StopCondition) -> np.ndarray:
    def step_selector(k, x, grad, func):
        return armijo(x, func, grad)

    return gradient_descent(x_0, func, step_selector, sc)


def find_b(func: Func) -> float:
    MAX_X = 10000
    curr_val = func(0)
    b = 1
    while (b < MAX_X and func(b) < curr_val):
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


def armijo(x_k: np.ndarray, func: BiFunc, grad: np.ndarray) -> float:
    derivative: float = -float(grad @ grad.T)
    c1 = random.random() * 0.8 + 0.1
    q = random.random() * 0.8 + 0.1
    alpha: float = abs(func(x_k) / max(abs(derivative * c1), 1e-9))
    for _ in range(MAX_ITERATION_LIMIT):
        l_alpha: float = func(x_k) + c1*alpha*derivative
        if func(x_k - alpha*grad) < float(l_alpha):
            break
        alpha = q*alpha
    return float(alpha)
