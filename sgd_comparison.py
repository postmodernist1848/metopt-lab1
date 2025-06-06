import sgd
import numpy as np
import time
import torch
import gc
import tracemalloc
from lib import algorithms
import matplotlib.pyplot as plt
import os


quadratic = sgd.Polynomial(3)
dataset_parameters = np.array([4, 3, 2, 1])  # y = 4 + 3x + 2x^2 + x^3
dataset = [(np.array([x]), quadratic(dataset_parameters, np.array([x])) + np.random.normal(0, 0.1)) 
          for x in np.linspace(-2, 2, 500)]

def measure_time_and_memory(func, *args, **kwargs):
    """Измеряет время выполнения, использование памяти и количество операций"""
    gc.collect()
    torch.cuda.empty_cache()
    
    tracemalloc.start()
    
    start_time = time.time()
    result, total_ops = func(*args, **kwargs)
    end_time = time.time()
    
    return result, end_time - start_time, tracemalloc.get_traced_memory()[1], total_ops

def run_experiment(config):
    """Запускает один эксперимент с заданной конфигурацией"""
    kwargs = {
        'epochs': config['epochs'],
        'batch_size': config['batch_size'],
        'lr': config['learning_rate']
    }
    if 'momentum' in config:
        kwargs['momentum'] = config['momentum']
    
    result, execution_time, memory_used, operations = measure_time_and_memory(
        sgd.sgd,
        quadratic,
        config['regularization'],
        dataset,
        **kwargs
    )
    
    ef = sgd.ErrorFunc(quadratic, config['regularization'], dataset)
    accuracy = ef(result)
    
    print(f"Config: {config['name']}")
    print(f"Accuracy (MSE): {accuracy:.6f}")
    print(f"Time: {execution_time:.3f} seconds")
    print(f"Memory used: {memory_used} B")
    print(f"Operations: {operations}")
    print(f"Batches per epoch: {len(dataset) // config['batch_size']}")
    print(f"Total batches: {(len(dataset) // config['batch_size']) * config['epochs']}")
    print("-" * 50)
    
    return {
        'name': config['name'],
        'accuracy': accuracy,
        'time': execution_time,
        'memory': memory_used,
        'operations': operations,
        'batches_per_epoch': len(dataset) // config['batch_size'],
        'total_batches': (len(dataset) // config['batch_size']) * config['epochs']
    }

configs = [
    {
        'name': 'Base (bs=32,ep=100)',
        'regularization': sgd.L1Regularization(0),
        'learning_rate': algorithms.lr_constant(0.01),
        'batch_size': 32,
        'epochs': 100
    },
    {
        'name': 'Small (bs=8,ep=400)',
        'regularization': sgd.L1Regularization(0),
        'learning_rate': algorithms.lr_constant(0.01),
        'batch_size': 8,
        'epochs': 400
    },
    {
        'name': 'Large (bs=128,ep=25)',
        'regularization': sgd.L1Regularization(0),
        'learning_rate': algorithms.lr_constant(0.01),
        'batch_size': 128,
        'epochs': 25
    },
    {
        'name': 'L1(λ=0.1) Small',
        'regularization': sgd.L1Regularization(0.1),
        'learning_rate': algorithms.lr_constant(0.01),
        'batch_size': 8,
        'epochs': 400
    },
    {
        'name': 'L1(λ=0.1) Large',
        'regularization': sgd.L1Regularization(0.1),
        'learning_rate': algorithms.lr_constant(0.01),
        'batch_size': 128,
        'epochs': 25
    },
    {
        'name': 'L2(λ=0.1) Small',
        'regularization': sgd.L2Regularization(0.1),
        'learning_rate': algorithms.lr_constant(0.01),
        'batch_size': 8,
        'epochs': 400
    },
    {
        'name': 'L2(λ=0.1) Large',
        'regularization': sgd.L2Regularization(0.1),
        'learning_rate': algorithms.lr_constant(0.01),
        'batch_size': 128,
        'epochs': 25
    },
    {
        'name': 'M(μ=0.9) Small',
        'regularization': sgd.L1Regularization(0),
        'learning_rate': algorithms.lr_constant(0.01),
        'momentum': 0.9,
        'batch_size': 8,
        'epochs': 400
    },
    {
        'name': 'M(μ=0.9) Large',
        'regularization': sgd.L1Regularization(0),
        'learning_rate': algorithms.lr_constant(0.01),
        'momentum': 0.9,
        'batch_size': 128,
        'epochs': 25
    },
    {
        'name': 'M+L1 Small',
        'regularization': sgd.L1Regularization(0.1),
        'learning_rate': algorithms.lr_constant(0.01),
        'momentum': 0.9,
        'batch_size': 8,
        'epochs': 400
    },
    {
        'name': 'M+L1 Large',
        'regularization': sgd.L1Regularization(0.1),
        'learning_rate': algorithms.lr_constant(0.01),
        'momentum': 0.9,
        'batch_size': 128,
        'epochs': 25
    },
    {
        'name': 'M+L2 Small',
        'regularization': sgd.L2Regularization(0.1),
        'learning_rate': algorithms.lr_constant(0.01),
        'momentum': 0.9,
        'batch_size': 8,
        'epochs': 400
    },
    {
        'name': 'M+L2 Large',
        'regularization': sgd.L2Regularization(0.1),
        'learning_rate': algorithms.lr_constant(0.01),
        'momentum': 0.9,
        'batch_size': 128,
        'epochs': 25
    },
    {
        'name': 'Single (bs=1)',
        'regularization': sgd.L1Regularization(0),
        'learning_rate': algorithms.lr_constant(0.01),
        'batch_size': 1,
        'epochs': 1600
    },
    {
        'name': 'Single M(μ=0.9)',
        'regularization': sgd.L1Regularization(0),
        'learning_rate': algorithms.lr_constant(0.01),
        'momentum': 0.9,
        'batch_size': 1,
        'epochs': 1600
    },
    {
        'name': 'Single L1(λ=0.1)',
        'regularization': sgd.L1Regularization(0.1),
        'learning_rate': algorithms.lr_constant(0.01),
        'batch_size': 1,
        'epochs': 1600
    },
    {
        'name': 'Full (bs=500)',
        'regularization': sgd.L1Regularization(0),
        'learning_rate': algorithms.lr_constant(0.01),
        'batch_size': len(dataset),
        'epochs': 6
    },
    {
        'name': 'Full M(μ=0.9)',
        'regularization': sgd.L1Regularization(0),
        'learning_rate': algorithms.lr_constant(0.01),
        'momentum': 0.9,
        'batch_size': len(dataset),
        'epochs': 6
    },
    {
        'name': 'Full L1(λ=0.1)',
        'regularization': sgd.L1Regularization(0.1),
        'learning_rate': algorithms.lr_constant(0.01),
        'batch_size': len(dataset),
        'epochs': 6
    },
    {
        'name': 'Medium (bs=64)',
        'regularization': sgd.L1Regularization(0),
        'learning_rate': algorithms.lr_constant(0.01),
        'batch_size': 64,
        'epochs': 50
    },
    {
        'name': 'Medium M(μ=0.9)',
        'regularization': sgd.L1Regularization(0),
        'learning_rate': algorithms.lr_constant(0.01),
        'momentum': 0.9,
        'batch_size': 64,
        'epochs': 50
    },
    {
        'name': 'Medium L1(λ=0.1)',
        'regularization': sgd.L1Regularization(0.1),
        'learning_rate': algorithms.lr_constant(0.01),
        'batch_size': 64,
        'epochs': 50
    },
    {
        'name': 'M(μ=0.5) Small',
        'regularization': sgd.L1Regularization(0),
        'learning_rate': algorithms.lr_constant(0.01),
        'momentum': 0.5,
        'batch_size': 8,
        'epochs': 400
    },
    {
        'name': 'M(μ=0.99) Small',
        'regularization': sgd.L1Regularization(0),
        'learning_rate': algorithms.lr_constant(0.01),
        'momentum': 0.99,
        'batch_size': 8,
        'epochs': 400
    },
    {
        'name': 'L1(λ=0.01) Small',
        'regularization': sgd.L1Regularization(0.01),
        'learning_rate': algorithms.lr_constant(0.01),
        'batch_size': 8,
        'epochs': 400
    },
    {
        'name': 'L2(λ=0.01) Small',
        'regularization': sgd.L2Regularization(0.01),
        'learning_rate': algorithms.lr_constant(0.01),
        'batch_size': 8,
        'epochs': 400
    }
]

results = []
for config in configs:
    results.append(run_experiment(config))

names = [r['name'] for r in results]
accuracies = [r['accuracy'] for r in results]
times = [r['time'] for r in results]
memories = [r['memory'] for r in results]
operations = [r['operations'] for r in results]

def remove_outliers(data, threshold=2):
    """Удаляет выбросы из данных, используя z-score"""
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    return np.where(z_scores > threshold, mean + threshold * std, data)

def plot_with_outliers(ax, x, y, title, ylabel, color='b', rotation=45, ylim=None):
    """Строит график с обработкой выбросов"""
    y_clean = remove_outliers(y)
    
    bars = ax.bar(x, y_clean, color=color)
    
    for i, (y_orig, y_clean) in enumerate(zip(y, y_clean)):
        if y_orig != y_clean:
            ax.plot(i, y_orig, 'r*', markersize=10)
            ax.plot([i, i], [y_clean, y_orig], 'r--', alpha=0.5)
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=rotation)
    ax.grid(True)
    
    if ylim is not None:
        ax.set_ylim(top=ylim)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2e}' if height > 1000 else f'{height:.2f}',
                ha='center', va='bottom')

dir_name = 'sgd_comparison'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

fig1 = plt.figure(figsize=(12, 6))
ax1 = plt.gca()
plot_with_outliers(ax1, names, accuracies, 'Accuracy Comparison', 'MSE', 'b', ylim=3)
plt.tight_layout(pad=3.0)
plt.savefig(f'{dir_name}/accuracy_comparison.png')

fig2 = plt.figure(figsize=(12, 6))
ax2 = plt.gca()
plot_with_outliers(ax2, names, times, 'Execution Time Comparison', 'Time (seconds)', 'r')
plt.tight_layout(pad=3.0)
plt.savefig(f'{dir_name}/time_comparison.png')

fig3 = plt.figure(figsize=(12, 6))
ax3 = plt.gca()
plot_with_outliers(ax3, names, memories, 'Memory Usage Comparison', 'Memory (B)', 'g')
plt.tight_layout(pad=3.0)
plt.savefig(f'{dir_name}/memory_comparison.png')

fig4 = plt.figure(figsize=(12, 6))
ax4 = plt.gca()
plot_with_outliers(ax4, names, operations, 'Computational Complexity Comparison', 'Total Operations', 'm')
plt.tight_layout(pad=3.0)
plt.savefig(f'{dir_name}/operations_comparison.png')

plt.show()

print("\nИтоговая таблица сравнения:")
print("-" * 120)
print(f"{'Configuration':<20} {'Accuracy':<12} {'Time (s)':<10} {'Memory (B)':<12} {'Operations':<12} {'Batches/Ep':<10} {'Total Batches':<12}")
print("-" * 120)
for r in results:
    print(f"{r['name']:<20} {r['accuracy']:<12.6f} {r['time']:<10.3f} {r['memory']:<12} {r['operations']:<12} {r['batches_per_epoch']:<10} {r['total_batches']:<12}")

print("\nАнализ выбросов:")
print("-" * 50)
for metric, values in [('Accuracy', accuracies), ('Time', times), ('Memory', memories), ('Operations', operations)]:
    mean = np.mean(values)
    std = np.std(values)
    outliers = [v for v in values if abs(v - mean) > 2 * std]
    if outliers:
        print(f"\n{metric}:")
        print(f"Среднее: {mean:.2e}")
        print(f"Стд. откл.: {std:.2e}")
        print(f"Выбросы: {[f'{v:.2e}' for v in outliers]}") 