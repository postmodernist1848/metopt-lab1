import torch
import torch.nn
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sgd import Polynomial, L1Regularization, L2Regularization, sgd
from lib.algorithms import lr_constant
from typing import List, Tuple, Dict, Any, Optional
import json
from dataclasses import dataclass
import os

dir_name = 'torch_sgd_comparison'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

@dataclass
class TestConfig:
    """Configuration for a single test run."""
    name: str
    learning_rate: float
    n_epochs: int
    batch_size: int
    momentum: float = 0.0
    reg_type: str = 'none'
    reg_lambda: float = 0.0

class TestResults:
    """Container for test results."""
    def __init__(self, losses: List[float], time: float, final_loss: float, 
                 parameters: List[float], total_ops: Optional[int] = None):
        self.losses = losses
        self.time = time
        self.final_loss = final_loss
        self.parameters = parameters
        self.total_ops = total_ops
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            'losses': self.losses,
            'time': self.time,
            'final_loss': self.final_loss,
            'parameters': self.parameters,
            'total_ops': self.total_ops
        }

def generate_data(n_samples: int = 1000, noise_level: float = 0.1) -> Tuple[List[Tuple[np.ndarray, float]], np.ndarray]:
    """Generate synthetic data for testing."""
    model = Polynomial(3)
    true_params = np.array([1.0, 2.0, 3.0, 4.0])
    
    x_points = np.linspace(-2, 2, n_samples)
    dataset = [(np.array([x]), model(true_params, np.array([x])) + np.random.normal(0, noise_level)) 
              for x in x_points]
    
    return dataset, true_params

def get_regularization(reg_type: str, reg_lambda: float) -> L1Regularization:
    """Get regularization object based on type."""
    match reg_type:
        case 'none':
            return L1Regularization(0.0)
        case 'l1':
            return L1Regularization(reg_lambda)
        case 'l2':
            return L2Regularization(reg_lambda)
        case _:
            raise ValueError(f"Unknown regularization type: {reg_type}")

def custom_sgd_test(dataset: List[Tuple[np.ndarray, float]], config: TestConfig) -> TestResults:
    """Test custom SGD implementation."""
    model = Polynomial(3)
    reg = get_regularization(config.reg_type, config.reg_lambda)
    
    start_time = time()
    
    w, total_ops = sgd(
        model,
        reg,
        dataset,
        epochs=config.n_epochs,
        batch_size=config.batch_size,
        lr=lr_constant(config.learning_rate),
        momentum=config.momentum
    )
    
    end_time = time()
    
    losses = []
    for epoch in range(config.n_epochs):
        epoch_loss = 0
        for x, y in dataset:
            y_pred = model(w, x)
            epoch_loss += (y_pred - y) ** 2
        losses.append(epoch_loss / len(dataset))
    
    return TestResults(
        losses=losses,
        time=end_time - start_time,
        final_loss=losses[-1],
        parameters=w.tolist(),
        total_ops=total_ops
    )

def torch_sgd_test(dataset: List[Tuple[np.ndarray, float]], config: TestConfig) -> TestResults:
    """Test PyTorch's SGD implementation."""
    X = torch.tensor([x[0] for x in dataset], dtype=torch.float32)
    y = torch.tensor([x[1] for x in dataset], dtype=torch.float32)
    
    X_poly = torch.cat([X**i for i in range(4)], dim=1)
    
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    criterion = torch.nn.MSELoss()
    
    match config.reg_type:
        case 'l1':
            def regularization(model):
                return config.reg_lambda * sum(p.abs().sum() for p in model.parameters())
        case 'l2':
            def regularization(model):
                return config.reg_lambda * sum(p.pow(2).sum() for p in model.parameters())
        case _:
            def regularization(model):
                return 0
    
    start_time = time()
    losses = []
    
    for epoch in range(config.n_epochs):
        epoch_loss = 0
        for i in range(0, len(dataset), config.batch_size):
            batch_X = X_poly[i:i+config.batch_size]
            batch_y = y[i:i+config.batch_size]
            
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y.unsqueeze(1)) + regularization(model)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_X)
        
        losses.append(epoch_loss / len(dataset))
    
    end_time = time()
    
    return TestResults(
        losses=losses,
        time=end_time - start_time,
        final_loss=losses[-1],
        parameters=model.weight.detach().numpy().flatten().tolist()
    )

def plot_comparison(results: Dict[str, TestResults], config: TestConfig):
    """Plot comparison of losses and execution times for a single test."""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(results['custom'].losses, label='Custom SGD')
    plt.plot(results['torch'].losses, label='PyTorch SGD')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Comparison\n{config.name}')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    times = [results['custom'].time, results['torch'].time]
    labels = ['Custom SGD', 'PyTorch SGD']
    plt.bar(labels, times)
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time Comparison')
    
    plt.subplot(1, 3, 3)
    x = np.arange(4)
    width = 0.35
    plt.bar(x - width/2, results['custom'].parameters, width, label='Custom SGD')
    plt.bar(x + width/2, results['torch'].parameters, width, label='PyTorch SGD')
    plt.xlabel('Parameter Index')
    plt.ylabel('Value')
    plt.title('Parameter Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{dir_name}/torch_sgd_comparison_{config.name.replace(" ", "_")}.png')
    plt.close()

def plot_summary_comparison(all_results: Dict[str, Dict[str, Dict[str, Any]]]):
    """Create a summary comparison plot of all tests."""
    configs = list(all_results.keys())
    
    custom_times = [all_results[cfg]['custom']['time'] for cfg in configs]
    torch_times = [all_results[cfg]['torch']['time'] for cfg in configs]
    custom_losses = [all_results[cfg]['custom']['final_loss'] for cfg in configs]
    torch_losses = [all_results[cfg]['torch']['final_loss'] for cfg in configs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(configs))
    width = 0.35
    
    ax1.bar(x - width/2, custom_times, width, label='Custom SGD', color='blue', alpha=0.7)
    ax1.bar(x + width/2, torch_times, width, label='PyTorch SGD', color='red', alpha=0.7)
    
    ax1.set_xlabel('Test Configuration')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.legend()
    
    for i, v in enumerate(custom_times):
        ax1.text(i - width/2, v, f'{v:.2f}s', ha='center', va='bottom')
    for i, v in enumerate(torch_times):
        ax1.text(i + width/2, v, f'{v:.2f}s', ha='center', va='bottom')
    
    ax2.bar(x - width/2, custom_losses, width, label='Custom SGD', color='blue', alpha=0.7)
    ax2.bar(x + width/2, torch_losses, width, label='PyTorch SGD', color='red', alpha=0.7)
    
    ax2.set_xlabel('Test Configuration')
    ax2.set_ylabel('Final Loss')
    ax2.set_title('Accuracy Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim(top=3.0)
    
    for i, v in enumerate(custom_losses):
        ax2.text(i - width/2, v, f'{v:.2e}', ha='center', va='bottom')
    for i, v in enumerate(torch_losses):
        ax2.text(i + width/2, v, f'{v:.2e}', ha='center', va='bottom')
    
    plt.tight_layout()
    

    plt.savefig(f'{dir_name}/summary_comparison.png')
    
    plt.show()

def run_test(config: TestConfig) -> Dict[str, TestResults]:
    """Run a single test configuration."""
    dataset, true_params = generate_data()
    
    custom_results = custom_sgd_test(dataset, config)
    torch_results = torch_sgd_test(dataset, config)
    
    results = {
        'custom': custom_results,
        'torch': torch_results
    }
    
    plot_comparison(results, config)
    
    return results

def main():
    """Main function to run all tests."""
    test_configs = [
        TestConfig('base', 0.01, 100, 32),
        TestConfig('small_batch', 0.01, 100, 8),
        TestConfig('large_batch', 0.01, 100, 128),
        TestConfig('high_lr', 0.1, 100, 32),
        TestConfig('low_lr', 0.001, 100, 32),
        TestConfig('momentum', 0.01, 100, 32, momentum=0.9),
        TestConfig('l1_reg', 0.01, 100, 32, reg_type='l1', reg_lambda=0.1),
        TestConfig('l2_reg', 0.01, 100, 32, reg_type='l2', reg_lambda=0.1),
        TestConfig('momentum_l1', 0.01, 100, 32, momentum=0.9, reg_type='l1', reg_lambda=0.1)
    ]
    
    all_results = {}
    
    for config in test_configs:
        print(f"\nRunning test: {config.name}")
        results = run_test(config)
        all_results[config.name] = {
            'custom': results['custom'].to_dict(),
            'torch': results['torch'].to_dict()
        }
        
        print(f"Custom SGD final loss: {results['custom'].final_loss:.6f}")
        print(f"PyTorch SGD final loss: {results['torch'].final_loss:.6f}")
        print(f"Custom SGD execution time: {results['custom'].time:.2f} seconds")
        print(f"PyTorch SGD execution time: {results['torch'].time:.2f} seconds")
        if results['custom'].total_ops is not None:
            print(f"Total operations in custom SGD: {results['custom'].total_ops}")
    
    with open(f'{dir_name}/sgd_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    plot_summary_comparison(all_results)

if __name__ == "__main__":
    main() 