from typing import Callable, Protocol
import numpy as np
import random
import math
from scipy.optimize import line_search, fmin_cg, fmin_bfgs

'''

    Learning rate functions

'''
type LearningRateFunc = Callable[[int], float]

def lr_constant(c: float) -> LearningRateFunc:
    return lambda k: c

def lr_geometric() -> LearningRateFunc:
    h0 = 1
    return lambda k: h0 / 2**k

def lr_exponential_decay(λ: float) -> LearningRateFunc:
    assert λ > 0
    h0 = 1
    return lambda k: h0 * math.exp(-λ * k)

def lr_polynomial_decay(α: float, β: float) -> LearningRateFunc:
    assert α > 0
    assert β > 0
    return lambda k: 1/math.sqrt(k + 1) * (β * k + 1) ** -α


type StopCondition = Callable[[np.ndarray, np.ndarray], bool]
type Func = Callable[[float], float]

class BiFunc(Protocol):
    def __call__(self, x: np.ndarray) -> float: ...
    def gradient(self, x: np.ndarray) -> np.ndarray: ...
    def hessian(self, x: np.ndarray) -> np.ndarray: ...
    def min(self) -> float | None: ...


def relative_x_condition() -> StopCondition:
    eps = 1e-9
    # ‖𝑥_{𝑘+1} − 𝑥_𝑘‖ < 𝜀(‖𝑥_{𝑘+1}‖ + 1)
    return lambda x, prev: bool(np.linalg.norm(x - prev) < eps * (np.linalg.norm(x) + 1))


def relative_f_condition(func: BiFunc, x_0: np.ndarray) -> StopCondition:
    eps = 1e-9
    # ‖∇𝑓(𝑥_𝑘)‖^2 < 𝜀‖∇𝑓(𝑥_0)‖^2
    return lambda x, prev: bool(np.linalg.norm(func.gradient(x) ** 2) < eps * np.linalg.norm(func.gradient(x_0)) ** 2)


MAX_ITERATION_LIMIT = 10000

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
        # print(f'k: {k}, x: {x}, f: {func(x)}')

        k += 1

    return np.array(trajectory)


def armijo_step_selector(k, x, grad, func):
    return armijo(x, func, grad)


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
    return gradient_descent(x_0, func, armijo_step_selector, sc)


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
        # print(f'k: {k}, x: {x}, f: {func(x)}')

    return np.array(trajectory)


def damped_newton_descent(x_0: np.ndarray,
                          func: BiFunc,
                          sc: StopCondition,
                          learning_rate_func: LearningRateFunc = lr_constant(0.1)
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
            very_low = 0.05,
            low = 0.25,
            high = 3/4
                                  ) -> np.ndarray:

    delta = 1.0
    x = x_0.copy()
    k = 0
    trajectory = [x.copy()]

    def dogleg() -> np.ndarray:
        p = -(np.linalg.inv(B) @ grad)

        if np.linalg.norm(p) > delta:
            h = step_selector(k, x, grad, func)
            p_b = p
            p_u = -h * grad

            norm_pu = np.linalg.norm(p_u)

            if norm_pu >= delta:
                p = delta * p_u / norm_pu
            else:
                pb_pu = np.dot(p_b - p_u, p_b - p_u)
                dot_pU_pB_pU = np.dot(p_u, p_b - p_u)
                fact = dot_pU_pB_pU**2 - pb_pu * (np.dot(p_u, p_u) - delta**2)
                tau = (-dot_pU_pB_pU + math.sqrt(fact)) / pb_pu
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
        DELTA_ITERATIONS_LIMIT = 15
        for _ in range(DELTA_ITERATIONS_LIMIT):
            p = dogleg()

            df = func(x) - func(x + p)
            dm = -(np.dot(grad, p) + 0.5 * np.dot(p, np.dot(B, p)))

            if abs(dm) < 1e-10:
                dm = 1e-10

            rho = df / dm

            if rho < very_low:
                delta = 1/2 * delta
                continue

            if rho >= high:
                delta *= 2
            elif rho >= low:
                delta = delta
            elif very_low <= rho < low:
                delta = 1/2 * delta

            break

        x = x + p

        trajectory.append(x.copy())

        if sc(x, prev) or k > MAX_ITERATION_LIMIT:
            break
        NEWTON_DESCENT_WITH_1D_SEARCH_LOGGING = False
        if NEWTON_DESCENT_WITH_1D_SEARCH_LOGGING:
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

def scipy_cg(x_0: np.ndarray, func: BiFunc) -> np.ndarray:
    trajectory = [x_0]
    fmin_cg(
        func,
        x_0,
        func.gradient,
        callback=lambda x: trajectory.append(x),
        disp=False
    )
    return np.array(trajectory)

def scipy_bfgs(x_0: np.ndarray, func: BiFunc) -> np.ndarray:
    trajectory = [x_0]
    fmin_bfgs(
        func,
        x_0,
        func.gradient,
        callback=lambda x: trajectory.append(x),
        disp=False
    )
    return np.array(trajectory)