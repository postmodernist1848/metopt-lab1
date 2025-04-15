from typing import Callable, Protocol
import numpy as np
import random
import math
from scipy.optimize import line_search
from scipy.optimize import fmin_bfgs

MAX_ITERATION_LIMIT = 10000

type LearningRateFunc = Callable[[int], float]
type StopCondition = Callable[[np.ndarray, np.ndarray], bool]
type Func = Callable[[float], float]

class BiFunc(Protocol):
    def __call__(self, x: np.ndarray) -> float: ...
    def gradient(self, x: np.ndarray) -> np.ndarray: ...
    def hessian(self, x: np.ndarray) -> np.ndarray: ...


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

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return 2 * self.A


class BiFuncCallableWrapper:
    f: Callable[[float, float], float]
    h: float = 0.001

    def __init__(self, f: Callable[[float, float], float]):
        self.f = f

    def __call__(self, x: np.ndarray) -> float:
        return self.f(x[0], x[1])

    def gradient(self, x: np.ndarray) -> np.ndarray:
        d1 = (self.f(x[0] + self.h, x[1]) -
              self.f(x[0] - self.h, x[1])) / (2*self.h)
        d2 = (self.f(x[0], x[1] + self.h) -
              self.f(x[0], x[1] - self.h)) / (2*self.h)
        g = np.array([d1, d2])
        return g

    def hessian(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x

        # Use a central difference formula for better accuracy
        # For f_xx (second derivative with respect to x)
        f_xx = (self.f(x1 + self.h, x2) - 2 * self.f(x1, x2) +
                self.f(x1 - self.h, x2)) / (self.h * self.h)

        # For f_yy (second derivative with respect to y)
        f_yy = (self.f(x1, x2 + self.h) - 2 * self.f(x1, x2) +
                self.f(x1, x2 - self.h)) / (self.h * self.h)

        # For f_xy (mixed derivative)
        # Use the cross partial difference formula
        f_xy = (self.f(x1 + self.h, x2 + self.h) - self.f(x1 + self.h, x2 - self.h) -
                self.f(x1 - self.h, x2 + self.h) + self.f(x1 - self.h, x2 - self.h)) / (4 * self.h * self.h)

        hessian = np.array([
            [f_xx, f_xy],
            [f_xy, f_yy]
        ])

        return hessian


class NoisyWrapper:
    f: BiFunc
    factor: float

    def __init__(self, f: BiFunc, factor: float = 1.0):
        self.f = f
        self.factor = factor

    def __call__(self, x: np.ndarray):
        return self.f.__call__(x) + random.random() * self.factor

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.f.gradient(x)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return self.f.hessian(x)


class BiFuncStatsDecorator:
    f: BiFunc
    call_count: int = 0
    gradient_count: int = 0
    hessian_count: int = 0

    def __init__(self, f: BiFunc):
        self.f = f

    def __call__(self, x: np.ndarray):
        self.call_count += 1
        return self.f.__call__(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        self.gradient_count += 1
        return self.f.gradient(x)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        self.hessian_count += 1
        return self.f.hessian(x)

    def reset(self):
        self.hessian_count = self.gradient_count = self.call_count = 0


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


def constant_h(c: float) -> LearningRateFunc:
    return lambda k: c


def geometric_h() -> LearningRateFunc:
    h0 = 1
    return lambda k: h0 / 2**k


def exponential_decay(λ: float) -> LearningRateFunc:
    assert λ > 0
    h0 = 1
    return lambda k: h0 * math.exp(-λ * k)


def polynomial_decay(α: float, β: float) -> LearningRateFunc:
    assert α > 0
    assert β > 0
    return lambda k: 1/math.sqrt(k + 1) * (β * k + 1) ** -α


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


def steepest_gradient_descent_wolfe(x_0: np.ndarray,
                                    func: BiFunc,
                                    sc: StopCondition) -> np.ndarray:
    def step_selector(k, x, grad, func):
        return wolfe(x, func, grad)

    return gradient_descent(x_0, func, step_selector, sc)


def steepest_gradient_descent_scipy_wolfe(x_0: np.ndarray,
                                          func: BiFunc,
                                          sc: StopCondition) -> np.ndarray:
    def step_selector(k, x, grad, func):
        alpha = line_search(func, func.gradient, x, -grad)[0]
        return alpha if alpha != None else armijo(x, func, grad)

    return gradient_descent(x_0, func, step_selector, sc)


def bfgs(x_0: np.ndarray,
         func: BiFunc,
         eps: float) -> np.ndarray:

    k = 0
    I = np.identity(2)
    c = I
    x = x_0
    grad = func.gradient(x)
    trajectory = [x.copy()]

    while np.linalg.norm(grad) > eps and k < MAX_ITERATION_LIMIT:
        p_k = -c @ grad
        alpha = armijo(x, func, grad)

        x_next = x + alpha * p_k
        grad_next = func.gradient(x_next)

        s_k = x_next - x
        y_k = grad_next - grad

        rho_k1 = (y_k.T @ s_k)

        if abs(rho_k1) < 1e-9:
            if rho_k1 == 0:
                rho_k1 = 1e-9
            else:
                rho_k1 = np.sign(rho_k1) * 1e-9

        rho_k = 1.0 / rho_k1
        

        c1 = (I - rho_k * s_k @ y_k.T)
        c2 = (I - rho_k * y_k @ s_k.T) 
        c3 = rho_k * s_k @ s_k.T

        c = c1 @ c @ c2 + c3
        
        x = x_next
        grad = grad_next

        trajectory.append(x)

        k += 1
        print(f'k: {k}, x: {x}, f: {func(x)}')

    return np.array(trajectory)


def damped_newton_descent(x_0: np.ndarray,
                          func: BiFunc,
                          sc: StopCondition,
                          learning_rate_func: LearningRateFunc = constant_h(1),
                          ) -> np.ndarray:
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

        alpha = learning_rate_func(k)
        p = np.linalg.inv(func.hessian(x)) @ grad
        x = x - alpha * p
        trajectory.append(x.copy())

        if sc(x, prev) or k > MAX_ITERATION_LIMIT:
            break
        if False:
            print(f'k: {k}, x: {x}, f: {func(x)}')

        k += 1

    return np.array(trajectory)


def newton_descent_with_1d_search(x_0: np.ndarray,
                                  func: BiFunc,
                                  sc: StopCondition,
                                  step_selector: Callable[[int, np.ndarray, np.ndarray, BiFunc], float],
                                  ) -> np.ndarray:

    delta = 1.0
    x = x_0.copy()
    k = 0
    trajectory = [x.copy()]

    def dogleg() -> np.ndarray:
        p = -(np.linalg.inv(B) @ grad)

        if np.linalg.norm(p) > delta:
            print("c outside trust region")

            h = step_selector(k, x, grad, func)
            p_b = p
            p_u = -h * grad

            norm_pu = np.linalg.norm(p_u)

            if norm_pu >= delta:
                print("norm_pu >= delta")
                p = delta * p_u / norm_pu
            else:
                pb_pu = np.dot(p_b - p_u, p_b - p_u)
                dot_pU_pB_pU = np.dot(p_u, p_b - p_u)
                fact = dot_pU_pB_pU**2 - pb_pu * (np.dot(p_u, p_u) - delta**2)
                tau = (-dot_pU_pB_pU + math.sqrt(fact)) / pb_pu
                print("x += ", p_u + tau * (p_b - p_u))
                p = p_u + tau * (p_b - p_u)
        return p

    while True:
        grad = func.gradient(x)

        if (grad.T @ grad) < 1e-9:
            EPS = 1e-7

            def random_eps():
                return random.choice([-EPS, EPS])
            grad += np.array([random_eps(), random_eps()])

        prev = x.copy()

        B = func.hessian(x)

        p = 0
        DELTA_ITERATIONS_LIMIT = 3
        for _ in range(DELTA_ITERATIONS_LIMIT):
            p = dogleg()

            df = func(x) - func(x + p)
            dm = -(np.dot(grad, p) + 0.5 * np.dot(p, np.dot(B, p)))

            if abs(dm) < 1e-10:
                dm = 1e-10

            rho = df / dm
            print(f'rho: {rho}')

            if rho < 0.05:
                delta = 1/2 * delta
                continue

            if rho >= 3/4:
                delta *= 2
            elif rho > 1/4:
                delta = delta
            elif 0.05 < rho < 1/4:
                delta = 1/2 * delta

            break

        x = x + p

        trajectory.append(x.copy())

        if sc(x, prev) or k > MAX_ITERATION_LIMIT:
            break
        if True:
            print(f'k: {k}, x: {x}, f: {func(x)}')

        k += 1

    return np.array(trajectory)


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


def wolfe(x_k: np.ndarray, func: BiFunc, grad: np.ndarray) -> float:
    derivative: float = -float(grad @ grad.T)

    # "c1 is usually chosen to be quite small while c2 is much larger"
    c1 = random.random()*0.05 + 0.1
    c2 = random.random()*0.05 + 0.9

    alpha_left = 0
    alpha_right = 1e10
    func_x_k = func(x_k)
    alpha = 1
    for _ in range(MAX_ITERATION_LIMIT):
        if (func(x_k - alpha*grad) > func_x_k + c1*alpha*derivative):
            alpha_right = alpha
            alpha = (alpha_left + alpha_right) / 2
            continue
        wolfe_grad = func.gradient(x_k - alpha*grad)
        wolfe_derivative = -float(wolfe_grad @ grad.T)
        if (abs(wolfe_derivative) > c2*abs(derivative)):
            alpha_left = alpha
            alpha = (alpha_left + alpha_right) / 2
            continue
        break

    return float(alpha)
