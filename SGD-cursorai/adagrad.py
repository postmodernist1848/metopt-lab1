"""
Implementation of AdaGrad optimizer.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from lib.algorithms import LearningRateFunc

def normalize_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Normalize input data and target values."""
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    y_mean = y.mean()
    y_std = y.std()
    
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std
    
    return X_norm, y_norm, X_mean, X_std, y_mean, y_std

def generate_polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Generate polynomial features up to given degree."""
    n_samples = X.shape[0]
    X_poly = np.ones((n_samples, 1))
    
    for d in range(1, degree + 1):
        X_poly = np.hstack((X_poly, X ** d))
    
    return X_poly

def compute_gradients(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, 
                     weights: np.ndarray, l1_lambda: float = 0.0, 
                     l2_lambda: float = 0.0) -> np.ndarray:
    """Compute gradients with L1 and L2 regularization."""
    n_samples = X.shape[0]
    error = y_pred - y
    gradients = 2 * X.T @ error / n_samples
    
    # Add regularization gradients
    if l1_lambda > 0:
        gradients += l1_lambda * np.sign(weights)
    if l2_lambda > 0:
        gradients += 2 * l2_lambda * weights
    
    return gradients

def compute_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                weights: np.ndarray, l1_lambda: float = 0.0, 
                l2_lambda: float = 0.0) -> float:
    """Compute MSE loss with L1 and L2 regularization."""
    mse_loss = np.mean((y_true - y_pred) ** 2)
    
    # Add regularization terms
    reg_loss = 0
    if l1_lambda > 0:
        reg_loss += l1_lambda * np.sum(np.abs(weights))
    if l2_lambda > 0:
        reg_loss += l2_lambda * np.sum(weights ** 2)
    
    return mse_loss + reg_loss

def adagrad_step(params: np.ndarray, grads: np.ndarray, 
                learning_rate: float = 0.01,
                epsilon: float = 1e-8,
                squared_grads: Optional[np.ndarray] = None,
                max_grad_norm: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Update parameters using AdaGrad algorithm."""
    if squared_grads is None:
        squared_grads = np.zeros_like(params)
    
    # Gradient clipping
    grad_norm = np.linalg.norm(grads)
    if grad_norm > max_grad_norm:
        grads = grads * (max_grad_norm / grad_norm)
    
    # Update squared gradients
    squared_grads += grads ** 2
    
    # Compute adaptive learning rate
    adaptive_lr = learning_rate / (np.sqrt(squared_grads) + epsilon)
    
    # Update parameters
    params = params - adaptive_lr * grads
    
    return params, squared_grads

def train_adagrad(
    X: np.ndarray, y: np.ndarray,
    degree: int = 2,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    epsilon: float = 1e-8,
    l1_lambda: float = 0.0,
    l2_lambda: float = 0.0,
    lr_func: Optional[LearningRateFunc] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 100,
    patience: int = 10,
    min_delta: float = 1e-4,
    max_grad_norm: float = 1.0
) -> Tuple[Dict[str, List[float]], np.ndarray, Tuple[np.ndarray, np.ndarray, float, float]]:
    """Train polynomial regression model using AdaGrad."""
    # Normalize data
    X_norm, y_norm, X_mean, X_std, y_mean, y_std = normalize_data(X, y)
    if X_val is not None:
        X_val_norm = (X_val - X_mean) / X_std
        y_val_norm = (y_val - y_mean) / y_std
    
    # Generate polynomial features
    X_poly = generate_polynomial_features(X_norm, degree)
    if X_val is not None:
        X_val_poly = generate_polynomial_features(X_val_norm, degree)
    
    # Initialize weights and squared gradients
    n_features = X_poly.shape[1]
    weights = np.random.randn(n_features, 1) * 0.01  # Smaller initial weights
    squared_grads = None
    
    # Initialize history
    history = {
        'loss': [],
        'val_loss': [],
        'weights_norm': [],
        'gradients_norm': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Get learning rate
        lr = lr_func(epoch) if lr_func is not None else learning_rate
        history['learning_rate'].append(lr)
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X_shuffled = X_poly[indices]
        y_shuffled = y_norm[indices]
        
        # Mini-batch training
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(X), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            # Forward pass
            y_pred = batch_X @ weights
            
            # Compute gradients
            gradients = compute_gradients(batch_X, batch_y, y_pred, weights, 
                                       l1_lambda, l2_lambda)
            
            # Update weights using AdaGrad
            weights, squared_grads = adagrad_step(weights, gradients, lr, epsilon,
                                                squared_grads, max_grad_norm)
            
            # Compute loss
            batch_loss = compute_loss(batch_y, y_pred, weights, 
                                    l1_lambda, l2_lambda)
            total_loss += batch_loss
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        # Validation
        if X_val is not None:
            val_pred = X_val_poly @ weights
            val_loss = compute_loss(y_val_norm, val_pred, weights, 
                                  l1_lambda, l2_lambda)
        else:
            val_loss = None
        
        # Update history
        history['loss'].append(avg_loss)
        if val_loss is not None:
            history['val_loss'].append(val_loss)
        history['weights_norm'].append(np.linalg.norm(weights))
        history['gradients_norm'].append(np.linalg.norm(gradients))
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Train Loss: {avg_loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Weight Norm: {history['weights_norm'][-1]:.4f}")
            print(f"  Gradient Norm: {history['gradients_norm'][-1]:.4f}")
        
        # Early stopping
        if val_loss is not None:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
    
    print("\nTraining finished:")
    print(f"Final train loss: {history['loss'][-1]:.4f}")
    if val_loss is not None:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"History lengths: {len(history['loss'])} epochs")
    
    return history, weights, (X_mean, X_std, y_mean, y_std)

def predict(X: np.ndarray, weights: np.ndarray, degree: int, 
           normalization_params: Tuple[np.ndarray, np.ndarray, float, float]) -> np.ndarray:
    """Make predictions using trained weights."""
    X_mean, X_std, y_mean, y_std = normalization_params
    
    # Normalize input
    X_norm = (X - X_mean) / X_std
    
    # Generate polynomial features
    X_poly = generate_polynomial_features(X_norm, degree)
    
    # Make prediction
    y_pred_norm = X_poly @ weights
    
    # Denormalize prediction
    y_pred = y_pred_norm * y_std + y_mean
    
    return y_pred 