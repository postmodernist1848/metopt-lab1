from lab4.annealing import *
from lib.funcs import *
from lab4.annealing_funcs import *
from lab4.commivoyager import *

def test_genetic_algorithm(population: Vector, 
                         crossover_func: Callable[[Vector, Vector], Vector],
                         mutate_func: Callable[[Vector, float], Vector],
                         fitness_function: BiFunc,
                         tournament_size: int,
                         mutation_rate: float,
                         eps: float) -> None:
    clever_population, k = genetic(population,
                                 crossover_func,
                                 mutate_func,
                                 fitness_function,
                                 tournament_size,
                                 mutation_rate,
                                 eps,
                                 live_plotting=True)
    x = min(clever_population, key=fitness_function)
    print(x)
    print("function value:", fitness_function(x))

    true_min = fitness_function.min()
    if true_min is not None:
        print("error:", abs(fitness_function(x) - true_min))
    print("iterations:", k)
    return x

def f4_genetic_test():
    population = init_population(-10, 10, 100, 2)
    test_genetic_algorithm(population, crossover, mutate, f4, 10, 0.2, 1e-3)

def commivoyager_genetic_test(points: Vector, correct_value: float | None = None):
    x0 = np.random.permutation(points)
    
    population = init_population_commivoyager(x0, 50)
    func = BiFuncCallableWrapper(commivoyager, correct_value)
    x = test_genetic_algorithm(population, 
                         crossover_commivoyager,
                         mutate_commivoyager,
                         func, 20, 0.3, 1e-2)
    
    # commivoyager_plot(x0, x, "Initial vs Optimized path")

if __name__ == "__main__":
    # f4_genetic_test()
    # commivoyager_genetic_test(
    #     np.array([[0, 0], [2, 0], [4, 0],
    #                [0, 2], [0, 4], [-2, 0],
    #                  [-4, 0], [0, -2], [0, -4],
    #                    [1, 1]]), 26.14213562373095)
    commivoyager_genetic_test(
        np.array(
            [[random.uniform(0, 100), random.uniform(0, 100)] for _ in range(50)]
            ), None)
