from lib.algorithms import *
from lib.funcs import q1, q2, f4, fsinsin
from lib.stats import BiFuncStatsDecorator, print_stats


def main():
    xs = [np.array(x) for x in ([-1, -1], [1.0, 4.0], [1.0, 1.0])]

    funcs = [
        ("q1", BiFuncStatsDecorator(q1)), # "x.T*[[1, 0], [0, 1]]*x + [1, 1]*x - 1.4"
        ("q2", BiFuncStatsDecorator(q2)), # "x.T*[[0.1, 0], [0, 3]]*x"
        ("f4", BiFuncStatsDecorator(f4)), # "(x**2 - 1)**2 + y**2 + 0.5 * x"
        ("sin(x) + sin(y)", BiFuncStatsDecorator(fsinsin))
    ]

    stop_condition = relative_x_condition()
    dichotomy_eps = 1e-9

    algorithms = [
        ("Learning rate scheduling const(0.1)",
            lambda x_0, func: learning_rate_scheduling(x_0, func, lr_constant(0.1), stop_condition)),
        ("Learning rate scheduling exp(0.5)",
            lambda x_0, func: learning_rate_scheduling(x_0, func, lr_exponential_decay(0.5), stop_condition)),
        ("Learning rate scheduling exp(0.3)",
            lambda x_0, func: learning_rate_scheduling(x_0, func, lr_exponential_decay(0.3), stop_condition)),
        ("Armijo Gradient Descent",
            lambda x_0, func: steepest_gradient_descent_armijo(x_0, func, stop_condition)),
        ("Dichotomy Gradient Descent",
            lambda x_0, func: steepest_gradient_descent_dichotomy(x_0, func, dichotomy_eps, stop_condition)),
        ("Scipy Wolfe Gradient Descent",
            lambda x_0, func: steepest_gradient_descent_scipy_wolfe(x_0, func, stop_condition)),
        ("Damped Newton Descent",
            lambda x_0, func: damped_newton_descent(x_0, func, stop_condition, lr_constant(0.1))),
        ("Dog Leg Armijo",
            lambda x_0, func: newton_descent_with_1d_search(x_0, func, stop_condition, armijo_step_selector)),
    ]

    for algorithm_name, applier in algorithms:
        for func_name, func in funcs:
            assert func.min() is not None, f"minimum value for {func_name} is not found"
            for x_0 in xs:
                trajectory = applier(x_0, func)

                print_stats(
                    func, trajectory,  f'{func_name} | {algorithm_name} | x0={x_0}', plot=False)


if __name__ == "__main__":
    main()
