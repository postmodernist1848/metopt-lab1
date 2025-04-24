from algorithms import *
from funcs import q1, q2, f4, fsinsin
from stats import BiFuncStatsDecorator, print_stats
from collections import namedtuple

def main():
    xs = [[0.0, 0.0], [1.0, 4.0], [1.0, 1.0]]
    xs = [np.array(x) for x in xs]

    h_map = {
        constant_h(0.1): "constant_h(0.1)",
        exponential_decay(0.5): "exponential_decay(0.5)",
        exponential_decay(0.3): "exponential_decay(0.3)"
    }

    h_array = h_map.keys()
    h_name_array = list(h_map.values())
    func_map = {
        BiFuncStatsDecorator(q1): {
            "name": "q1",
            "formula": "x.T*[[1, 0], [0, 1]]*x + [1, 1]*x - 1.4"
            },
        BiFuncStatsDecorator(q2): {
            "name": "q2",
            "formula": "x.T*[[0.1, 0], [0, 3]]*x"
            },
        BiFuncStatsDecorator(f4): {
            "name": "f4",
            "formula": "(x**2 - 1)**2 + y**2 + 0.5 * x"
            },
    }
    func_array = func_map.keys()

    algorithms = {
        "Learning rate scheduling": {
            "algorithm": learning_rate_scheduling,
            "applier": lambda args, h: learning_rate_scheduling(args['x_0'], args['func'], h, args['stop_condition'])
        },
        "Armijo Gradient Descent": {
            "algorithm": steepest_gradient_descent_armijo,
            "applier": lambda args: steepest_gradient_descent_armijo(args['x_0'], args['func'], args['stop_condition'])
        },
        "Dichotomy Gradient Descent": {
            "algorithm": steepest_gradient_descent_dichotomy,
            "applier": lambda args: steepest_gradient_descent_dichotomy(args['x_0'], args['func'], args['eps'], args['stop_condition'])
        },
        "Scipy Wolfe Gradient Descent": {
            "algorithm": steepest_gradient_descent_scipy_wolfe,
            "applier": lambda args: steepest_gradient_descent_scipy_wolfe(args['x_0'], args['func'], args['stop_condition'])
        },
        "Newton Descent with 1D Search": {
            "algorithm": newton_descent_with_1d_search,
            "applier": lambda args: newton_descent_with_1d_search(args['x_0'], args['func'], args['stop_condition'], armijo_step_selector)
        },
        "Dog Leg": {
            "algorithm": damped_newton_descent,
            "applier": lambda args: damped_newton_descent(args['x_0'], args['func'], args['stop_condition'], constant_h(0.1))
        }
    }

    Stat = namedtuple('Stat', 'func_calls grad_calls trajectory function_info')

    def get_stat(func, trajectory) -> Stat:
        stat: Stat = Stat(func.call_count, func.gradient_count, trajectory, func_map[func])
        func.reset()
        return stat

    eps = 1e-9
    stat_array = []

    for function in func_array:
        for x_0 in xs:
            args = {
                "x_0": x_0,
                "func": function,
                "stop_condition": relative_x_condition(),
                "eps": eps
            }
            stats = [[get_stat(function, algorithms["Learning rate scheduling"]["applier"](args, h)) for h in h_array]]
            algorithms_values = list(algorithms.values())[1:]
            for algorithm_info in algorithms_values:
                stats.append([get_stat(function, algorithm_info["applier"](args))])
            stat_array.append((x_0, function, stats))
            
    alg_name_array = list(algorithms.keys())

    def print_stat(index: int):
        h = ""
        for stat in stat_array:
            x_0 = stat[0]
            func: BiFuncStatsDecorator = stat[1]
            inner_stat_array = stat[2][index]
            for stat_index in range(len(inner_stat_array)):
                if (index == 0):
                    h = h_name_array[stat_index]
                func.call_count = inner_stat_array[stat_index].func_calls
                func.gradient_count = inner_stat_array[stat_index].grad_calls
                function_info = inner_stat_array[stat_index].function_info
                function_name = function_info["name"]
                function_formula = function_info["formula"]
                print_stats(func, inner_stat_array[stat_index].trajectory,  f'function: {function_name} | formula: {function_formula} | algorithm: {alg_name_array[index]} | x0={x_0} {h}', plot=False)
        print("-------------------------------------------------------------------")
        
    # Выводим ответ    
    for i in range(len(stat_array[0][2])):
        print_stat(i)
        
if __name__ == "__main__":
    main()