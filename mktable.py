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

    algorithms = [
        ("Learning rate scheduling const(0.1)", lambda args: learning_rate_scheduling(args['x_0'], args['func'], constant(0.1), stop_condition)),
        ("Learning rate scheduling exp(0.5)", lambda args: learning_rate_scheduling(args['x_0'], args['func'], exponential_decay(0.5), stop_condition)),
        ("Learning rate scheduling exp(0.3)", lambda args: learning_rate_scheduling(args['x_0'], args['func'], exponential_decay(0.3), stop_condition)),
        ("Armijo Gradient Descent", lambda args: steepest_gradient_descent_armijo(args['x_0'], args['func'], stop_condition)),
        ("Dichotomy Gradient Descent", lambda args: steepest_gradient_descent_dichotomy(args['x_0'], args['func'], args['dichotomy_eps'], stop_condition)),
        ("Scipy Wolfe Gradient Descent", lambda args: steepest_gradient_descent_scipy_wolfe(args['x_0'], args['func'], stop_condition)),
        ("Newton Descent with 1D Search", lambda args: newton_descent_with_1d_search(args['x_0'], args['func'], stop_condition, armijo_step_selector)),
        ("Dog Leg const(0.1)", lambda args: damped_newton_descent(args['x_0'], args['func'], stop_condition, constant(0.1)))
    ]

    eps = 1e-9
    for algorithm_name, applier in algorithms:
        for func_name, func in funcs:
            for x_0 in xs:
                args = {
                    "x_0": x_0,
                    "func": func,
                    "stop_condition": relative_x_condition(),
                    "dichotomy_eps": eps
                }

                trajectory = applier(args)
                print_stats(func, trajectory,  f'{func_name} | {algorithm_name} | x0={x_0}', plot=False)
        
if __name__ == "__main__":
    main()