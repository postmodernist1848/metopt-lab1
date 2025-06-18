import numpy as np
from lab4.annealing import *
from lib.funcs import *
from lab4.annealing_funcs import *
from typing import Callable, Dict, List, Tuple
from lab4.commivoyager import commivoyager
from lib.stats import BiFuncStatsDecorator
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import random

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
                      test_name: str = "Test") -> Tuple[Dict, List[Vector]]:
    x, iter_count, history = annealing(func, T, F, P, s0=x0, t_min=t_min, t_0=np.array([t_max]))
    
    print(f"\n{test_name}:")
    print_x("Оптимальная точка: ", x)
    print(f"Значение функции: {func(x)}")
    print(f"Количество итераций: {iter_count}")
    
    result = {
        'x': x,
        'func_value': func(x),
        'iterations': iter_count,
        'func_calls': func.call_count if hasattr(func, 'call_count') else 0,
        'gradient_calls': func.gradient_count if hasattr(func, 'gradient_count') else 0,
        'hessian_calls': func.hessian_count if hasattr(func, 'hessian_count') else 0
    }
    
    if func.min() is not None:
        error = abs(func(x) - func.min())
        print(f"Ошибка: {error}")
        result['error'] = error
    
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
    plt.savefig(f'{save_dir}/annealing_comparison.png')
    plt.close()

def plot_convergence(histories: Dict[str, List[List[Vector]]], func: BiFunc, save_dir: str = "plots"):
    """Plot convergence history for different configurations."""
    Path(save_dir).mkdir(exist_ok=True)
    
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
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
            history = np.array(history)
            plt.plot(history[:, 0], history[:, 1], 'o-', 
                    color=colors[i], alpha=0.5, markersize=2)
            
            plt.plot(history[0, 0], history[0, 1], 'o', 
                    color=colors[i], markersize=8, label=f'Start {i+1}')
            plt.plot(history[-1, 0], history[-1, 1], '*', 
                    color=colors[i], markersize=12, label=f'End {i+1}')
        
        plt.title(f'Convergence Paths - {test_name}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        plt.colorbar(plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.3))
        plt.savefig(f'{save_dir}/annealing_convergence_{test_name}.png')
        plt.close()
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
        
        for i, history in enumerate(test_histories):
            history = np.array(history)
            z_values = [func(p) for p in history]
            
            ax.plot(history[:, 0], history[:, 1], z_values, 'o-',
                   color=colors[i], alpha=0.5, markersize=2)
            
            ax.scatter(history[0, 0], history[0, 1], z_values[0],
                      color=colors[i], s=100, label=f'Start {i+1}')
            ax.scatter(history[-1, 0], history[-1, 1], z_values[-1],
                      color=colors[i], s=150, marker='*', label=f'End {i+1}')
        
        ax.set_title(f'3D Convergence Paths - {test_name}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Function Value')
        plt.colorbar(surf)
        plt.savefig(f'{save_dir}/annealing_convergence_3d_{test_name}.png')
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
                    value = func(history[i])
                    values.append(value)
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
    plt.savefig(f'{save_dir}/annealing_convergence_values.png')
    plt.close()

def run_multiple_tests(func: BiFunc, num_tests: int = 10):
    results = {
        'default': [],
        'high_temp': [],
        'low_temp': [],
        'fast_cooling': []
    }
    
    histories = {
        'default': [],
        'high_temp': [],
        'low_temp': [],
        'fast_cooling': []
    }
    
    t_max = 100
    t_min = 1e-3
    cooling_rate = 1 - 1e-1
    x_range = (-2, 2)
    
    for _ in range(num_tests):
        x0 = np.random.uniform(x_range[0], x_range[1], size=2)
        result, history = run_annealing_test(
            x0, func, BiFuncCallableWrapper(random_F),
            calc_temperature(cooling_rate), calc_probability,
            t_max=t_max, t_min=t_min, test_name="Default"
        )
        results['default'].append(result)
        histories['default'].append(history)
        
        # High temperature
        x0 = np.random.uniform(x_range[0], x_range[1], size=2)
        result, history = run_annealing_test(
            x0, func, BiFuncCallableWrapper(random_F),
            calc_temperature(1 - 1e-1), calc_probability,
            t_max=t_max*2, t_min=t_min, test_name="High Temperature")
        results['high_temp'].append(result)
        histories['high_temp'].append(history)
        
        # Low temperature
        x0 = np.random.uniform(x_range[0], x_range[1], size=2)
        result, history = run_annealing_test(
            x0, func, BiFuncCallableWrapper(random_F),
            calc_temperature(cooling_rate), calc_probability,
            t_max=t_max, t_min=t_min, test_name="Low Temperature")
        results['low_temp'].append(result)
        histories['low_temp'].append(history)
        
        # Fast cooling
        x0 = np.random.uniform(x_range[0], x_range[1], size=2)
        result, history = run_annealing_test(
            x0, func, BiFuncCallableWrapper(random_F),
            calc_temperature(cooling_rate), calc_probability,
            t_max=t_max, t_min=t_min, test_name="Fast Cooling")
        results['fast_cooling'].append(result)
        histories['fast_cooling'].append(history)
    
    plot_test_results(results)
    plot_convergence(histories, func)
    
    return results, histories

def commivoyager_annealing_test(x0: np.ndarray, correct_value: float = None):
    func = BiFuncCallableWrapper(commivoyager, correct_value)
    result, _ = run_annealing_test(
        x0, func, BiFuncCallableWrapper(commivoyager_F),
        calc_temperature(1 - 1e-4), calc_probability,
        test_name="Коммивояжер"
    )
    commivoyager_plot(x0, result['x'], "Initial vs Optimized path")

def random_test(f: BiFunc):
    """Test annealing algorithm on a random initial point."""
    x0 = np.array([0, 0])
    result, _ = run_annealing_test(
        x0, f, BiFuncCallableWrapper(random_F),
        calc_temperature(1 - 1e-1), calc_probability,
        t_max=100, t_min=1e-3, test_name="Random Test"
    )

def main():
    results, histories = run_multiple_tests(frosenbrock, num_tests=10)
    
    # print("\nSummary Statistics:")
    # print("-" * 80)
    # for test_name, test_results in results.items():
    #     print(f"\n{test_name}:")
    #     print(f"Average iterations: {np.mean([r['iterations'] for r in test_results]):.2f}")
    #     print(f"Average function value: {np.mean([r['func_value'] for r in test_results]):.2e}")
    #     print(f"Average error: {np.mean([r['error'] for r in test_results]):.2e}")
    #     print(f"Best error: {min([r['error'] for r in test_results]):.2e}")
    #     print(f"Worst error: {max([r['error'] for r in test_results]):.2e}")
    #     print(f"Standard deviation: {np.std([r['error'] for r in test_results]):.2e}")
    
    x0 = np.array([[random.uniform(0, 200), random.uniform(0, 200)] for _ in range(100)])
    commivoyager_annealing_test(x0)

if __name__ == "__main__":
    main()

