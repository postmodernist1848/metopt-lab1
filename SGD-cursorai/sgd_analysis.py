import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import torch
from sgd_implementation import train_polynomial_regression, predict
from lib.algorithms import lr_constant, lr_geometric, lr_exponential_decay, lr_polynomial_decay
from torch_implementation import create_model, get_optimizers, train_model, predict as torch_predict
from adagrad import train_adagrad, predict as adagrad_predict

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create plots directory if it doesn't exist
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_training_history(losses, weights, gradients, val_losses=None, title=None):
    """Plot training history including losses, weights, and gradients."""
    print(f"\nPlotting history for {title}:")
    print(f"Losses length: {len(losses)}")
    print(f"First few losses: {[float(x) for x in losses[:5]]}")
    print(f"Last few losses: {[float(x) for x in losses[-5:]]}")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot losses with moving average
    epochs = range(len(losses))
    ax1.plot(epochs, losses, label='Training Loss', alpha=0.5)
    
    # Plot validation losses if provided
    if val_losses is not None:
        print(f"Val losses length: {len(val_losses)}")
        print(f"First few val losses: {[float(x) for x in val_losses[:5]]}")
        print(f"Last few val losses: {[float(x) for x in val_losses[-5:]]}")
        ax1.plot(epochs, val_losses, label='Validation Loss', alpha=0.5, color='red')
    
    # Calculate and plot moving average
    window_size = 5
    if len(losses) >= window_size:
        moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(losses)), moving_avg, 
                label=f'Moving average (window={window_size})', 
                color='green', linewidth=2)
    
    ax1.set_title('Training and Validation Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Add annotations for key points
    min_loss_idx = np.argmin(losses)
    ax1.annotate(f'Min train loss: {float(losses[min_loss_idx]):.4f}',
                xy=(min_loss_idx, losses[min_loss_idx]),
                xytext=(min_loss_idx+5, losses[min_loss_idx]),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    if val_losses is not None:
        min_val_loss_idx = np.argmin(val_losses)
        ax1.annotate(f'Min val loss: {float(val_losses[min_val_loss_idx]):.4f}',
                    xy=(min_val_loss_idx, val_losses[min_val_loss_idx]),
                    xytext=(min_val_loss_idx+5, val_losses[min_val_loss_idx]),
                    arrowprops=dict(facecolor='red', shrink=0.05))
    
    # Plot weight norms
    print(f"Weights length: {len(weights)}")
    print(f"First few weights: {[float(x) for x in weights[:5]]}")
    print(f"Last few weights: {[float(x) for x in weights[-5:]]}")
    ax2.plot(epochs, weights, label='Weight norm', color='green')
    ax2.set_title('Weight Norms Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Weight Norm')
    ax2.legend()
    ax2.grid(True)
    
    # Plot gradient norms with moving average
    print(f"Gradients length: {len(gradients)}")
    print(f"First few gradients: {[float(x) for x in gradients[:5]]}")
    print(f"Last few gradients: {[float(x) for x in gradients[-5:]]}")
    ax3.plot(epochs, gradients, label='Gradient norm', alpha=0.5, color='purple')
    
    # Calculate and plot moving average for gradients
    if len(gradients) >= window_size:
        grad_moving_avg = np.convolve(gradients, np.ones(window_size)/window_size, mode='valid')
        ax3.plot(range(window_size-1, len(gradients)), grad_moving_avg,
                label=f'Moving average (window={window_size})',
                color='orange', linewidth=2)
    
    ax3.set_title('Gradient Norms Over Epochs')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Gradient Norm')
    ax3.legend()
    ax3.grid(True)
    
    # Add statistics to the plot
    stats_text = f"""
    Statistics:
    Final Train Loss: {float(losses[-1]):.4f}
    Min Train Loss: {float(min(losses)):.4f}
    Final Weight Norm: {float(weights[-1]):.4f}
    Final Gradient Norm: {float(gradients[-1]):.4f}
    """
    if val_losses is not None:
        stats_text += f"\nFinal Val Loss: {float(val_losses[-1]):.4f}\nMin Val Loss: {float(min(val_losses)):.4f}"
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot in the plots directory
    if title:
        plot_path = os.path.join(PLOTS_DIR, f'{title}_history.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")

def generate_data(regression_type: str = 'polynomial', n_samples: int = 1000, n_features: int = 2):
    """Generate data for regression."""
    if regression_type == 'polynomial':
        # Generate polynomial data
        X = np.linspace(-10, 10, n_samples).reshape(-1, 1)
        y = 2 * X**2 + 3 * X + 1 + 0.1 * np.random.randn(n_samples, 1)
    else:  # multivariate
        # Generate multivariate data
        X = np.random.randn(n_samples, n_features)
        # Generate random coefficients
        coef = np.random.randn(n_features, 1)
        # Generate target with noise
        y = X @ coef + 0.1 * np.random.randn(n_samples, 1)
    
    return X, y

def run_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    reg_name: str,
    reg_params: dict,
    lr_name: str,
    lr_func: callable,
    use_torch: bool,
    regression_type: str,
    n_features: int
):
    """Run a single experiment with given parameters."""
    print(f"\nTesting with {reg_name} regularization, "
          f"{lr_name} learning rate, batch_size={batch_size}")
    
    # Get optimizers configuration
    if use_torch:
        input_dim = n_features if regression_type == 'multivariate' else 1
        model = create_model(degree=2, input_dim=input_dim)
        optimizers = get_optimizers(model)
    else:
        optimizers = {
            'SGD': {'momentum': 0.0, 'nesterov': False},
            'SGD_Momentum': {'momentum': 0.9, 'nesterov': False},
            'SGD_Nesterov': {'momentum': 0.9, 'nesterov': True},
            'AdaGrad': {}  # AdaGrad doesn't need additional parameters
        }
    
    # Run experiments for each optimizer
    for opt_name, opt_params in optimizers.items():
        # Train model
        if use_torch:
            history, norm_params = train_model(
                model, opt_params, X_train, y_train, X_val, y_val,
                batch_size=batch_size, epochs=100, patience=10, min_delta=1e-4,
                lr_func=lr_func, max_grad_norm=1.0
            )
        elif opt_name == 'AdaGrad':
            history, weights, norm_params = train_adagrad(
                X_train, y_train,
                degree=2,
                batch_size=batch_size,
                learning_rate=0.01,
                epsilon=1e-8,
                **reg_params,
                lr_func=lr_func,
                X_val=X_val,
                y_val=y_val,
                epochs=100,
                patience=10,
                min_delta=1e-4,
                max_grad_norm=1.0
            )
        else:
            history, weights, norm_params = train_polynomial_regression(
                X_train, y_train,
                degree=2,
                batch_size=batch_size,
                learning_rate=0.001,
                **opt_params,
                **reg_params,
                lr_func=lr_func,
                X_val=X_val,
                y_val=y_val,
                epochs=100,
                patience=10,
                min_delta=1e-4,
                max_grad_norm=1.0
            )
        
        # Plot training history
        plot_training_history(
            history['loss'],
            history['weights_norm'],
            history['gradients_norm'],
            history.get('val_loss'),
            f"{regression_type}_{opt_name}_{reg_name}_{lr_name}_batch_{batch_size}"
        )
        
        # Test prediction on test set
        if use_torch:
            y_pred = torch_predict(model, X_test, norm_params)
        elif opt_name == 'AdaGrad':
            y_pred = adagrad_predict(X_test, weights, degree=2, normalization_params=norm_params)
        else:
            y_pred = predict(X_test, weights, degree=2, normalization_params=norm_params)
        
        test_mse = np.mean((y_test - y_pred) ** 2)
        print(f"Test MSE: {test_mse:.4f}")

def main():
    # Choose regression type
    regression_type = 'polynomial'
    n_features = 3  # number of features for multivariate regression
    
    # Generate data
    X, y = generate_data(regression_type=regression_type, n_features=n_features)
    
    # Split into train, validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")
    
    # Test different batch sizes
    batch_sizes = [8, 32, 64, len(X_train)]
    
    # Test different learning rate functions
    lr_functions = {
        'constant': lr_constant(0.001),  # Reduced learning rate
        'geometric': lr_geometric(),
        'exponential': lr_exponential_decay(0.01)  # Reduced learning rate
    }
    
    # Test different regularization parameters
    regularization_configs = {
        'no_reg': {'l1_lambda': 0.0, 'l2_lambda': 0.0},
        'l1': {'l1_lambda': 0.0001, 'l2_lambda': 0.0},  # Reduced regularization
        'l2': {'l1_lambda': 0.0, 'l2_lambda': 0.0001},  # Reduced regularization
        'elastic': {'l1_lambda': 0.00005, 'l2_lambda': 0.00005}  # Reduced regularization
    }
    
    # Choose implementation (True for PyTorch, False for our implementation)
    use_torch = False
    
    # Run experiments
    for batch_size in batch_sizes:
        for reg_name, reg_params in regularization_configs.items():
            for lr_name, lr_func in lr_functions.items():
                run_experiment(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    batch_size, reg_name, reg_params, lr_name, lr_func,
                    use_torch, regression_type, n_features
                )

if __name__ == "__main__":
    main() 