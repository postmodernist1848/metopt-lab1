from lab4.annealing import *
from lib.funcs import *
from lab4.annealing_funcs import *
from lab4.commivoyager import *
from typing import Dict, List, Callable, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def test_genetic_algorithm(population: Vector, 
                         crossover_func: Callable[[Vector, Vector], Vector],
                         mutate_func: Callable[[Vector, float], Vector],
                         fitness_function: BiFunc,
                         tournament_size: int,
                         mutation_rate: float,
                         eps: float,
                         test_name: str = "Test",
                         max_iterations: int = 1000) -> Tuple[Dict, List[Vector]]:
    clever_population, k, history = genetic(population,
                                 crossover_func,
                                 mutate_func,
                                 fitness_function,
                                 tournament_size,
                                 mutation_rate,
                                 eps,
                                 live_plotting=False,
                                 max_iterations=max_iterations)
    x = min(clever_population, key=fitness_function)
    
    print(f"\n{test_name}:")
    print("Optimal point:", x)
    print("Function value:", fitness_function(x))
    
    result = {
        'x': x,
        'func_value': fitness_function(x),
        'iterations': k,
        'population_size': len(population),
        'tournament_size': tournament_size,
        'mutation_rate': mutation_rate
    }
    
    true_min = fitness_function.min()
    if true_min is not None:
        error = abs(fitness_function(x) - true_min)
        print("Error:", error)
        result['error'] = error
    
    print("Iterations:", k)
    return result, history

def plot_test_results(results: Dict[str, List[Dict]], save_dir: str = "plots"):
    """Plot comparison of different test results."""
    Path(save_dir).mkdir(exist_ok=True)
    
    test_names = list(results.keys())
    metrics = ['iterations', 'error', 'func_value']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [np.mean([r[metric] for r in results[name]]) for name in test_names]
        stds = [np.std([r[metric] for r in results[name]]) for name in test_names]
        
        axes[i].bar(test_names, values, yerr=stds, capsize=5)
        axes[i].set_title(f'{metric.capitalize()} Comparison')
        axes[i].set_xticklabels(test_names, rotation=45, ha='right')
        axes[i].grid(True)
        
        for j, v in enumerate(values):
            axes[i].text(j, v, f'{v:.2e}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/genetic_comparison.png')
    plt.close()

def plot_convergence(histories: Dict[str, List[List[Vector]]], func: BiFunc, save_dir: str = "plots"):
    """Plot convergence history for different configurations."""
    Path(save_dir).mkdir(exist_ok=True)
    
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    
    for test_name, test_histories in histories.items():
        if not test_histories:
            continue
            
        plt.figure(figsize=(10, 8))
        plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(test_histories)))
        for i, history in enumerate(test_histories):
            best_points = [min(gen, key=func) for gen in history]
            best_points = np.array(best_points)
            
            plt.plot(best_points[:, 0], best_points[:, 1], 'o-', 
                    color=colors[i], alpha=0.5, markersize=2)
            
            plt.plot(best_points[0, 0], best_points[0, 1], 'o', 
                    color=colors[i], markersize=8, label=f'Start {i+1}')
            plt.plot(best_points[-1, 0], best_points[-1, 1], '*', 
                    color=colors[i], markersize=12, label=f'End {i+1}')
        
        plt.title(f'Convergence Paths - {test_name}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        plt.colorbar(plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.3))
        plt.savefig(f'{save_dir}/convergence_{test_name}.png')
        plt.close()
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
        
        for i, history in enumerate(test_histories):
            best_points = [min(gen, key=func) for gen in history]
            best_points = np.array(best_points)
            z_values = [func(p) for p in best_points]
            
            ax.plot(best_points[:, 0], best_points[:, 1], z_values, 'o-',
                   color=colors[i], alpha=0.5, markersize=2)
            
            ax.scatter(best_points[0, 0], best_points[0, 1], z_values[0],
                      color=colors[i], s=100, label=f'Start {i+1}')
            ax.scatter(best_points[-1, 0], best_points[-1, 1], z_values[-1],
                      color=colors[i], s=150, marker='*', label=f'End {i+1}')
        
        ax.set_title(f'3D Convergence Paths - {test_name}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Function Value')
        plt.colorbar(surf)
        plt.savefig(f'{save_dir}/convergence_3d_{test_name}.png')
        plt.close()
    
    plt.figure(figsize=(12, 6))
    for test_name, test_histories in histories.items():
        if not test_histories:
            continue
            
        max_iterations = max(len(h) for h in test_histories)
        avg_values = []
        std_values = []
        
        for i in range(max_iterations):
            values = []
            for history in test_histories:
                if i < len(history):
                    best_value = func(min(history[i], key=func))
                    values.append(best_value)
            avg_values.append(np.mean(values))
            std_values.append(np.std(values))
        
        iterations = range(max_iterations)
        plt.plot(iterations, avg_values, label=test_name)
        plt.fill_between(iterations, 
                        np.array(avg_values) - np.array(std_values),
                        np.array(avg_values) + np.array(std_values),
                        alpha=0.2)
    
    plt.title('Function Value Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{save_dir}/convergence_values.png')
    plt.close()

def run_multiple_tests(func: BiFunc, num_tests: int = 10):
    """Run multiple tests with different parameters."""
    results = {
        'default': [],
        'large_population': [],
        'large_tournament': []
    }
    
    histories = {
        'default': [],
        'large_population': [],
        'large_tournament': []
    }
    
    if func == frosenbrock:
        default_pop_size = 200
        large_pop_size = 300
        default_tournament = 15
        large_tournament = 25
        mutation_rate = 0.3
        eps = 1e-2
        max_iterations = 500
    else:
        default_pop_size = 300
        large_pop_size = 500
        default_tournament = 10
        large_tournament = 30
        mutation_rate = 0.2
        eps = 1e-3
        max_iterations = 1000
    
    for _ in range(num_tests):
        # Default parameters
        population = init_population(-2, 2, default_pop_size, 2)
        result, history = test_genetic_algorithm(
            population, crossover, mutate, func,
            tournament_size=default_tournament, mutation_rate=mutation_rate,
            eps=eps, test_name="Default", max_iterations=max_iterations
        )
        results['default'].append(result)
        histories['default'].append(history)
        
        # Large population
        population = init_population(-2, 2, large_pop_size, 2)
        result, history = test_genetic_algorithm(
            population, crossover, mutate, func,
            tournament_size=default_tournament, mutation_rate=mutation_rate,
            eps=eps, test_name="Large Population", max_iterations=max_iterations
        )
        results['large_population'].append(result)
        histories['large_population'].append(history)
        
        # Large tournament
        population = init_population(-2, 2, default_pop_size, 2)
        result, history = test_genetic_algorithm(
            population, crossover, mutate, func,
            tournament_size=large_tournament, mutation_rate=mutation_rate,
            eps=eps, test_name="Large Tournament", max_iterations=max_iterations
        )
        results['large_tournament'].append(result)
        histories['large_tournament'].append(history)
    
    plot_test_results(results)
    plot_convergence(histories, func)
    
    return results, histories

def main():
    results, histories = run_multiple_tests(frosenbrock, num_tests=10)
    
    print("\nSummary Statistics:")
    print("-" * 80)
    for test_name, test_results in results.items():
        print(f"\n{test_name}:")
        print(f"Average iterations: {np.mean([r['iterations'] for r in test_results]):.2f}")
        print(f"Average function value: {np.mean([r['func_value'] for r in test_results]):.2e}")
        print(f"Average error: {np.mean([r['error'] for r in test_results]):.2e}")
        print(f"Best error: {min([r['error'] for r in test_results]):.2e}")
        print(f"Worst error: {max([r['error'] for r in test_results]):.2e}")
        print(f"Standard deviation: {np.std([r['error'] for r in test_results]):.2e}")

if __name__ == "__main__":
    main()
