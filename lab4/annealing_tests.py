import numpy as np
from lab4.annealing import *
from lib.funcs import *
from lab4.annealing_funcs import *
from typing import Callable
from lab4.commivoyager import commivoyager
from lib.stats import BiFuncStatsDecorator

def print_x(prefix: str, x: Vector):
    print(prefix, end=" ")
    for i in range(len(x)):
        print(x[i], end=" ")
    print()

def run_annealing_test(x0: Vector, 
                      func: BiFunc, 
                      F: BiFunc,
                      T: Callable[[float], float] = calc_temperature(1 - 1e-2),
                      P: Callable[[float, float], float] = calc_probability,
                      t_max: float = 100,
                      t_min: float = 1e-3,
                      test_name: str = "Test") -> None:
    x, iter_count = annealing(func, T, F, P, s0=x0, t_min=t_min, t_0=np.array([t_max]))
    
    print(f"\n{test_name}:")
    print_x("Оптимальная точка: ", x)
    print(f"Значение функции: {func(x)}")
    print(f"Количество итераций: {iter_count}")
    print(f'Кол-во вызовов функции: {func.call_count}')
    print(f'Кол-во вызовов градиента: {func.gradient_count}')
    print(f'Кол-во вызовов гессиана: {func.hessian_count}')
    if func.min() is not None:
        print(f"Ошибка: {abs(func(x) - func.min())}")
    return x

def commivoyager_annealing_test(x0: Vector, correct_value: float = None):
    func = BiFuncCallableWrapper(commivoyager, correct_value)
    x = run_annealing_test(x0, func, BiFuncCallableWrapper(commivoyager_F), calc_temperature(1 - 1e-4), test_name="Коммивояжер")
    commivoyager_plot(x0, x, "Initial vs Optimized path")

def lol_test(f: BiFunc):
    x0 = np.array([0, 0])
    run_annealing_test(x0, f, BiFuncCallableWrapper(lol_F), calc_temperature(1 - 1e-1), test_name="LOL")
    

def main():
    lol_test(BiFuncStatsDecorator(f4))
    # commivoyager_annealing_test(np.array([[0, 0], [2, 0], [4, 0], [0, 2], [0, 4], [-2, 0], [-4, 0], [0, -2], [0, -4], [1, 1]]), 26.14213562373095)
    # commivoyager_annealing_test(
    #     np.array(
    #         [[random.uniform(0, 200), random.uniform(0, 200)] for _ in range(100)]
    #         ),
    #     None)
    # armijo_annealing_test(f4)
    # wolfe_annealing_test(f4)
    # constraint_annealing_test(f4)

if __name__ == "__main__":
    main()

