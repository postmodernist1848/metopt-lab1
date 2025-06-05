import sgd
import numpy as np
import time
import torch
import gc
import tracemalloc
from lib import algorithms
import matplotlib.pyplot as plt
from memory_profiler import profile

quadratic = sgd.Polynomial(3)
dataset_parameters = np.array([4, 3, 2, 1])  # y = 4 + 3x + 2x^2 + x^3
dataset = [(np.array([x]), quadratic(dataset_parameters, np.array([x])) + np.random.normal(0, 0.1)) for x in np.linspace(-2, 2, 500)]

def measure_time_and_memory(func, *args, **kwargs):
    gc.collect()
    torch.cuda.empty_cache()
    
    tracemalloc.start()
    
    start_time = time.time()
    result, total_ops = func(*args, **kwargs)
    end_time = time.time()
    
    return result, end_time - start_time, tracemalloc.get_traced_memory()[1], total_ops


batch_sizes = [2 * i for i in range(1, len(dataset) // 2)]
epochs = 100
lr = algorithms.lr_constant(0.01)
reg = sgd.L1Regularization(0)

results = {
    'accuracy': [],
    'time': [],
    'memory': [],
    'operations': []
}

for batch_size in batch_sizes:
    result, execution_time, memory_used, operations = measure_time_and_memory(
        sgd.sgd,
        quadratic,
        reg,
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )
    
    ef = sgd.ErrorFunc(quadratic, reg, dataset)
    accuracy = ef(result)
    
    results['accuracy'].append(accuracy)
    results['time'].append(execution_time)
    results['memory'].append(memory_used)
    results['operations'].append(operations)
    
    print(f"Batch size: {batch_size}")
    print(f"Accuracy (MSE): {accuracy:.6f}")
    print(f"Time: {execution_time:.3f} seconds")
    print(f"Memory used: {memory_used} B")
    print(f"Operations: {operations}")
    print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot(batch_sizes, results['accuracy'], 'b-o')
axes[0, 0].set_xlabel('Batch Size')
axes[0, 0].set_ylabel('MSE')
axes[0, 0].set_title('Accuracy vs Batch Size')
axes[0, 0].grid(True)

axes[0, 1].plot(batch_sizes, results['time'], 'r-o')
axes[0, 1].set_xlabel('Batch Size')
axes[0, 1].set_ylabel('Time (seconds)')
axes[0, 1].set_title('Execution Time vs Batch Size')
axes[0, 1].grid(True)

axes[1, 0].plot(batch_sizes, [m for m in results['memory']], 'g-o')
axes[1, 0].set_xlabel('Batch Size')
axes[1, 0].set_ylabel('Memory (B)')
axes[1, 0].set_title('Memory Usage vs Batch Size')
axes[1, 0].grid(True)

ops_per_epoch = [ops / epochs for ops in results['operations']]
axes[1, 1].plot(batch_sizes, ops_per_epoch, 'm-o')
axes[1, 1].set_xlabel('Batch Size')
axes[1, 1].set_ylabel('Operations per Epoch')
axes[1, 1].set_title('Operations per Epoch vs Batch Size')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('batch_size_comparison.png')
plt.show()