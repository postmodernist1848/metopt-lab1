import numpy as np
from typing import Callable, Optional, Tuple, List
from lib.funcs import BiFunc
import random
import matplotlib.pyplot as plt

MAX_ITERATIONS = 10000
Vector = np.ndarray

def commivoyager_plot(initial_points: Vector, optimized_points: Vector, title: str = "Paths comparison"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    ax1.scatter(initial_points[:, 0], initial_points[:, 1], c='red', s=100)
    ax1.plot(initial_points[:, 0], initial_points[:, 1], 'r-', alpha=0.5)
    ax1.plot([initial_points[-1, 0], initial_points[0, 0]], 
             [initial_points[-1, 1], initial_points[0, 1]], 'r-', alpha=0.5)
    ax1.set_title("Initial path")
    ax1.grid(True)
    
    ax2.scatter(optimized_points[:, 0], optimized_points[:, 1], c='blue', s=100)
    ax2.plot(optimized_points[:, 0], optimized_points[:, 1], 'b-', alpha=0.5)
    ax2.plot([optimized_points[-1, 0], optimized_points[0, 0]], 
             [optimized_points[-1, 1], optimized_points[0, 1]], 'b-', alpha=0.5)
    ax2.set_title("Optimized path")
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.show()

def calc_temperature(γ: float) -> Callable[[Vector, int], Vector]:
    return lambda t, k: γ*t

def calc_probability(ΔE: float, t: float) -> float:
    return np.exp(-ΔE/t)

def calc_next_state(s_next: Vector, s_curr: Vector, p: float) -> Vector:
    return s_next if np.random.rand() < p else s_curr

def annealing(func: BiFunc,
              T: Callable[[Vector, int], Vector],
              F: BiFunc,
              P: Callable[[float, float], float],
              s0: Vector,
              t_min: float,
              t_0: Vector,
              max_iterations: int = 1000) -> Vector:
    s = s0
    t = t_0
    iter_count = 0
    history = [s.copy()]
    
    while t[0] > t_min:
        s_new = F(s)
        ΔE = func(s_new) - func(s)
        if ΔE <= 0:
            s = s_new
        else:
            p = P(ΔE, t[0])
            s = calc_next_state(s_new, s, p)
        t = T(t, iter_count)
        iter_count += 1
        history.append(s.copy())
    
    return s, iter_count, history

# ------------------------------------------------------------

def init_population(low: float, high: float, size: int, dimensions: int) -> Vector:
    return np.random.uniform(low=low, high=high, size=(size, dimensions))

def tournament_selection(population: Vector, fitness_function: BiFunc, tournament_size: int = 2) -> Vector:
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament = population[selected_indices]
    return min(tournament, key=fitness_function)

def crossover(parent1: Vector, parent2: Vector):
    alpha = np.random.random()
    return alpha * parent1 + (1 - alpha) * parent2

def mutate(individual: Vector, mutation_rate: float = 0.1) -> Vector:
    return individual + random.gauss(0, 1) if np.random.random() < mutation_rate else individual

def genetic(population: Vector,
             crossover: Callable[[Vector, Vector], Vector],
             mutate: Callable[[Vector, float], Vector],
             fitness_function: BiFunc, tournament_size: int, mutation_rate: float, eps: float, 
             live_plotting: bool = False, max_iterations: int = 1000) -> Tuple[Vector, int, List[Vector]]:
    assert tournament_size > 0
    assert 1 >= mutation_rate >= 0
    assert eps > 0

    population_size = len(population)
    curr_population = population
    k = 0
    history = [curr_population.copy()]

    for _ in range(max_iterations):
        # print(f"Iteration {k}")
        new_population = np.zeros_like(curr_population)
        for i in range(population_size):
            parent1 = tournament_selection(curr_population, fitness_function, tournament_size)
            parent2 = tournament_selection(curr_population, fitness_function, tournament_size)
            offspring = crossover(parent1, parent2)
            offspring = mutate(offspring, mutation_rate)
            new_population[i] = offspring
        curr_population = new_population
        history.append(curr_population.copy())
        curr_best = min(curr_population, key=fitness_function)
        k += 1

        if (fitness_function.min() is not None and fitness_function(curr_best) - fitness_function.min() < eps):
            break

    return curr_population, k, history