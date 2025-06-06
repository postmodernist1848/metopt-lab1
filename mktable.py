from lib.algorithms import *
from lib.funcs import q2, f4, fsinsin, frosenbrock
from lib.stats import BiFuncStatsDecorator, print_stats
from lib.plotting import plot_methods_comparison, ensure_plot_dir
import os
import numpy as np


def main():
    funcs = [
        ("q2", BiFuncStatsDecorator(q2), [np.array((1.0, 4.0))]), # "x.T*[[0.1, 0], [0, 3]]*x"
        ("f4", BiFuncStatsDecorator(f4), [np.array((1.0, 1.0))]), # "(x**2 - 1)**2 + y**2 + 0.5 * x"
        ("sin(x) + sin(y)", BiFuncStatsDecorator(fsinsin), [np.array((1.0, 1.0))]),
        ("rosenbrock", BiFuncStatsDecorator(frosenbrock), [np.array((2.0, 2.0))])
    ]

    stop_condition = relative_x_condition()
    dichotomy_eps = 1e-9
    bfgs_eps = 1e-2

    dog_leg_optimized = {'very_low': 0.006589013504624153, 'low': 0.49597617703076274, 'high': 0.669573144357006}

    algorithms = [
        ("Gradient Descent LR const(0.1)",
            lambda x_0, func: learning_rate_scheduling(x_0, func, lr_constant(0.1), relative_f_condition(func, x_0))),
        ("Gradient Descent LR exp(0.5)",
            lambda x_0, func: learning_rate_scheduling(x_0, func, lr_exponential_decay(0.5), relative_f_condition(func, x_0))),
        ("Gradient Descent LR exp(1.27166510627676)",
            lambda x_0, func: learning_rate_scheduling(x_0, func, lr_exponential_decay(1.27166510627676), relative_f_condition(func, x_0))),
        ("Armijo Gradient Descent",
            lambda x_0, func: steepest_gradient_descent_armijo(x_0, func, stop_condition)),
        ("Dichotomy Gradient Descent",
            lambda x_0, func: steepest_gradient_descent_dichotomy(x_0, func, dichotomy_eps, stop_condition)),
        ("Scipy Wolfe Gradient Descent",
            lambda x_0, func: steepest_gradient_descent_scipy_wolfe(x_0, func, stop_condition)),
        ("Damped Newton Descent",
            lambda x_0, func: damped_newton_descent(x_0, func, relative_f_condition(func, x_0), lr_constant(0.1))),
        ("Damped Newton Descent optimized™",
            lambda x_0, func: damped_newton_descent(x_0, func, stop_condition, lr_constant(0.8725560318527655))),
        ("Dog Leg Armijo 0.05 0.25 0.75",
            lambda x_0, func: newton_descent_with_1d_search(x_0, func, stop_condition, armijo_step_selector)),
        ("Dog Leg Armijo optimized™",
            lambda x_0, func: newton_descent_with_1d_search(x_0, func, stop_condition, armijo_step_selector, **dog_leg_optimized)),
        ("BFGS",
            lambda x_0, func: bfgs(x_0, func, bfgs_eps)),
        ("Scipy Newton-CG",
            lambda x_0, func: scipy_cg(x_0, func)), 
        ("Scipy BFGS",
            lambda x_0, func: scipy_bfgs(x_0, func))   
    ]

    plot_dir = ensure_plot_dir()

    comparison_results = {}

    for func_name, func, points in funcs:
        comparison_results[func_name] = {}
        
        for algorithm_name, applier in algorithms:
            assert func.min() is not None, f"minimum value for {func_name} is not found"
            for x_0 in points:
                trajectory = applier(x_0, func)

                print_stats(
                    func, trajectory, f'{func_name} | {algorithm_name} | x0={x_0}', plot=False, comparison_results=comparison_results, func_name=func_name, algorithm_name=algorithm_name)
        
        comparison_path = os.path.join(plot_dir, f"{func_name}_comparison.png")
        plot_methods_comparison(func_name, comparison_results[func_name], comparison_path)


if __name__ == "__main__":
    main()
