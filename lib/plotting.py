import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def plot_methods_comparison(func_name, results, save_path):
    """Plot comparison of different methods for a given function."""
    plt.figure(figsize=(15, 15))
    
    # Prepare data
    methods = list(results.keys())
    func_evals = [results[m]['func_evals'] for m in methods]
    grad_evals = [results[m]['grad_evals'] for m in methods]
    hess_evals = [results[m]['hess_evals'] for m in methods]
    errors = [results[m]['error'] for m in methods]
    
    # Plot function evaluations
    plt.subplot(2, 2, 1)
    plt.bar(methods, func_evals)
    plt.title('Function evaluations')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    if max(func_evals) > 20000:
        plt.ylim(0, 20000)  # Set y-axis limit only if values exceed threshold
    
    # Plot gradient evaluations
    plt.subplot(2, 2, 2)
    plt.bar(methods, grad_evals)
    plt.title('Gradient evaluations')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    if max(grad_evals) > 3000:
        plt.ylim(0, 3000)  # Set y-axis limit only if values exceed threshold
    
    # Plot hessian evaluations
    plt.subplot(2, 2, 3)
    plt.bar(methods, hess_evals)
    plt.title('Hessian evaluations')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    
    # Plot error
    plt.subplot(2, 2, 4)
    plt.bar(methods, errors)
    plt.title('Final error')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    
    plt.suptitle(f'Comparison of methods for {func_name}')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def ensure_plot_dir():
    """Ensure plots directory exists."""
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    return plot_dir 