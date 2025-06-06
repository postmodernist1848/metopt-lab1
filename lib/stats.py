import matplotlib.pyplot as plt
import numpy as np
from lib.algorithms import BiFunc

class BiFuncStatsDecorator:
    f: BiFunc
    call_count: int = 0
    gradient_count: int = 0
    hessian_count: int = 0

    def __init__(self, f: BiFunc):
        self.f = f

    def __call__(self, x: np.ndarray):
        self.call_count += 1
        return self.f.__call__(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        self.gradient_count += 1
        return self.f.gradient(x)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        self.hessian_count += 1
        return self.f.hessian(x)

    def reset(self):
        self.hessian_count = self.gradient_count = self.call_count = 0

    def min(self) -> float | None:
        return self.f.min()

PLOT_SIZE=3
def plot_trajectory(func: BiFunc, trajectory: np.ndarray, title=None, plot_size=PLOT_SIZE):
    # Create a meshgrid for the 3D plot
    x = np.linspace(-plot_size, plot_size, 100)
    y = np.linspace(-plot_size, plot_size, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    # Plot the 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)  # type: ignore

    # Plot the trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], [func(np.array([x, y]))
            for x, y in trajectory], color='r', marker='o')

    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], func(trajectory[-1]), color='b', label='Final point')

    if title:
        ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')  # type: ignore
    plt.show()

def print_stats(func: BiFuncStatsDecorator, trajectory: np.ndarray, title=None, plot=True, comparison_results=None, func_name=None, algorithm_name=None):
    cc, gc, hc = func.call_count, func.gradient_count, func.hessian_count
    calculated_min = func(trajectory[-1])
    print(f'title: {title}')
    print(f'Iterations: {len(trajectory) - 1}')
    print(f'x: {trajectory[-1]} f(x): {calculated_min}')
    print(f'Function evaluations: {cc}')
    print(f'Gradient evaluations: {gc}')
    print(f'Hessian evaluations: {hc}')
    min_value = func.min()
    error = abs(calculated_min - min_value)
    if min_value is not None:
        print(f'True minimum: {min_value}')
        print(f'Error: {error}')
    print()
    if plot:
        plot_trajectory(func, trajectory, title)

    comparison_results[func_name][algorithm_name] = {
        'func_evals': cc,
        'grad_evals': gc,
        'hess_evals': hc,
        'error': error,
        'iterations': len(trajectory) - 1
    }
    
    func.reset()
