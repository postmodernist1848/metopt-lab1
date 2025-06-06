from typing import Callable, Protocol
import numpy as np
import random
import math
from scipy.optimize import line_search, fmin_bfgs, minimize

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

def _iterative_descent(x_0: np.ndarray,
                      func: BiFunc,
                      sc: StopCondition,
                      next_point_selector: Callable[[np.ndarray, np.ndarray, BiFunc, int], np.ndarray]) -> np.ndarray:
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
        x = next_point_selector(x, grad, func, k)
        trajectory.append(x.copy())

        if sc(x, prev) or k > MAX_ITERATION_LIMIT:
            break

        k += 1

    return np.array(trajectory)

def gradient_descent(x_0: np.ndarray,
                     func: BiFunc,
                     step_selector: Callable[[int, np.ndarray, np.ndarray, BiFunc], float],
                     sc: StopCondition) -> np.ndarray:
    def next_point(x, grad, func, k):
        h = step_selector(k, x, grad, func)
        return x - h * grad
    
    return _iterative_descent(x_0, func, sc, next_point)


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
        return wolfe(x, func, grad)[0]

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
    n = len(x_0)
    I = np.identity(n)
    x = x_0
    grad = func.gradient(x)
    f_x = func(x)
    trajectory = [x.copy()]

    H0 = I * (1.0 / np.linalg.norm(grad))
    c = H0

    while np.linalg.norm(grad) > eps and k < MAX_ITERATION_LIMIT:
        p_k = -c @ grad
        
        _, x_next, f_next, grad_next = wolfe(x, func, grad, p_k, f_x)

        s_k = x_next - x
        y_k = grad_next - grad
        y_k_dot_s_k = np.dot(y_k, s_k)

        if y_k_dot_s_k <= 1e-10 * np.linalg.norm(s_k) * np.linalg.norm(y_k):
            x = x_next
            grad = grad_next
            f_x = f_next
            trajectory.append(x)
            k += 1
            continue

        rho_k = 1.0 / y_k_dot_s_k

        c1 = (I - rho_k * np.outer(s_k, y_k))
        c2 = (I - rho_k * np.outer(y_k, s_k))
        c3 = rho_k * np.outer(s_k, s_k)

        c = c1 @ c @ c2 + c3

        x = x_next
        grad = grad_next
        f_x = f_next

        trajectory.append(x)
        k += 1

    return np.array(trajectory)


def damped_newton_descent(x_0: np.ndarray,
                          func: BiFunc,
                          sc: StopCondition,
                          learning_rate_func: LearningRateFunc = lr_constant(0.1)
                          ) -> np.ndarray:
    def next_point(x, grad, func, k):
        alpha = learning_rate_func(k)
        p = np.linalg.inv(func.hessian(x)) @ grad
        return x - alpha * p
    
    return _iterative_descent(x_0, func, sc, next_point)


def _compute_dogleg_step(x: np.ndarray, grad: np.ndarray, B: np.ndarray, delta: float, step_selector: Callable[[int, np.ndarray, np.ndarray, BiFunc], float], k: int, func: BiFunc) -> tuple[np.ndarray, float]:
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

def newton_descent_with_1d_search(x_0: np.ndarray,
                                  func: BiFunc,
                                  sc: StopCondition,
                                  step_selector: Callable[[int, np.ndarray, np.ndarray, BiFunc], float],
                                  very_low = 0.05,
                                  low = 0.25,
                                  high = 3/4) -> np.ndarray:
    delta = 1.0
    
    def next_point(x, grad, func, k):
        nonlocal delta
        B = func.hessian(x)
        
        p = 0
        DELTA_ITERATIONS_LIMIT = 15
        for _ in range(DELTA_ITERATIONS_LIMIT):
            p = _compute_dogleg_step(x, grad, B, delta, step_selector, k, func)

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

        return x + p
    
    return _iterative_descent(x_0, func, sc, next_point)


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
    
    f_xk = func(x_k)
    alpha: float = abs(f_xk / max(abs(derivative * c1), 1e-9))
    
    min_alpha = 1e-10 
    
    for _ in range(MAX_ITERATION_LIMIT):
        if alpha < min_alpha:
            return min_alpha
            
        l_alpha: float = f_xk + c1*alpha*derivative
        if func(x_k - alpha*grad) < float(l_alpha):
            break
        alpha = q*alpha
    return float(alpha)

def wolfe(x_k: np.ndarray, 
          func: BiFunc, 
          grad: np.ndarray, 
          p_k: np.ndarray = None,
          f_x: float = None) -> tuple[float, np.ndarray, float, np.ndarray]:
   
    # "c1 is usually chosen to be quite small while c2 is much larger"
    c1 = random.random()*0.05 + 0.1
    c2 = random.random()*0.05 + 0.9
    
    alpha = 1.0
    g_x = np.dot(grad, p_k)
    
    x_next = x_k + alpha * p_k
    f_next = func(x_next)
    grad_next = func.gradient(x_next)
    g_next = np.dot(grad_next, p_k)
    
    if f_next <= f_x + c1 * alpha * g_x and abs(g_next) <= c2 * abs(g_x):
        return alpha, x_next, f_next, grad_next
        
    alpha_min = 0
    alpha_max = 1
    
    for _ in range(MAX_ITERATION_LIMIT):
        alpha = 0.5 * (alpha_min + alpha_max)
        x_next = x_k + alpha * p_k
        f_next = func(x_next)
        
        if f_next > f_x + c1 * alpha * g_x:
            alpha_max = alpha
            continue
            
        grad_next = func.gradient(x_next)
        g_next = np.dot(grad_next, p_k)
        
        if abs(g_next) <= c2 * abs(g_x):
            break
            
        if g_next < 0:
            alpha_min = alpha
        else:
            alpha_max = alpha
            
        if alpha_max - alpha_min < 0.1:
            break
    
    x_next = x_k + alpha * p_k
    f_next = func(x_next)
    grad_next = func.gradient(x_next)
    
    return alpha, x_next, f_next, grad_next

def scipy_cg(x_0: np.ndarray, func: BiFunc) -> np.ndarray:
    trajectory = [x_0]
    minimize(
        func,
        x_0,
        method='Newton-CG',
        jac=func.gradient,
        hess=func.hessian,
        callback=lambda x: trajectory.append(x),
        options={'disp': False}
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

    