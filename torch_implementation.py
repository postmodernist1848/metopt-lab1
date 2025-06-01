"""
PyTorch implementation of polynomial regression with various optimizers.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from lib.algorithms import LearningRateFunc

def generate_polynomial_features(x: torch.Tensor, degree: int, input_dim: int) -> torch.Tensor:
    """Generate polynomial features for input tensor."""
    from itertools import combinations_with_replacement
    features = []
    
    # Add constant term
    features.append(torch.ones(x.shape[0], 1))
    
    # Add linear terms
    for i in range(input_dim):
        features.append(x[:, i:i+1])
    
    # Add higher degree terms
    for d in range(2, degree + 1):
        for combo in combinations_with_replacement(range(input_dim), d):
            term = torch.ones(x.shape[0], 1)
            for idx in combo:
                term *= x[:, idx:idx+1]
            features.append(term)
    
    return torch.cat(features, dim=1)

def count_polynomial_features(degree: int, input_dim: int) -> int:
    """Count number of polynomial features for given degree and input dimension."""
    from itertools import combinations_with_replacement
    count = 0
    for d in range(degree + 1):
        # Count combinations with replacement for each degree
        count += len(list(combinations_with_replacement(range(input_dim), d)))
    return count

def normalize_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Normalize input data and target values."""
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    y_mean = y.mean()
    y_std = y.std()
    
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std
    
    return X_norm, y_norm, X_mean, X_std, y_mean, y_std

def create_model(degree: int = 2, input_dim: int = 1) -> torch.nn.Module:
    """Create a polynomial regression model."""
    class PolynomialModel(torch.nn.Module):
        def __init__(self, degree: int, input_dim: int):
            super().__init__()
            self.degree = degree
            self.input_dim = input_dim
            self.n_features = count_polynomial_features(degree, input_dim)
            self.linear = torch.nn.Linear(self.n_features, 1)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_poly = generate_polynomial_features(x, self.degree, self.input_dim)
            return self.linear(x_poly)
    
    return PolynomialModel(degree, input_dim)

def get_optimizers(model: torch.nn.Module) -> Dict[str, torch.optim.Optimizer]:
    """Get dictionary of PyTorch optimizers for the model."""
    return {
        'SGD': torch.optim.SGD(model.parameters(), lr=0.001),
        'SGD_Momentum': torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9),
        'SGD_Nesterov': torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True),
        'Adam': torch.optim.Adam(model.parameters(), lr=0.001),
        'RMSprop': torch.optim.RMSprop(model.parameters(), lr=0.001),
        'Adagrad': torch.optim.Adagrad(model.parameters(), lr=0.001)
    }

def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    lr_func: Optional[LearningRateFunc] = None,
    epochs: int = 100,
    patience: int = 10,
    min_delta: float = 1e-4,
    max_grad_norm: float = 1.0
) -> Tuple[Dict[str, List[float]], Tuple[np.ndarray, np.ndarray, float, float]]:
    """Train PyTorch model with early stopping."""
    print("\nStarting PyTorch training:")
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Batch size: {batch_size}")
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")
    
    model.train()
    criterion = torch.nn.MSELoss()
    
    # Normalize data
    X_train_norm, y_train_norm, X_mean, X_std, y_mean, y_std = normalize_data(X_train, y_train)
    X_val_norm = (X_val - X_mean) / X_std
    y_val_norm = (y_val - y_mean) / y_std
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_norm)
    y_train_tensor = torch.FloatTensor(y_train_norm)
    X_val_tensor = torch.FloatTensor(X_val_norm)
    y_val_tensor = torch.FloatTensor(y_val_norm)
    
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
        # Update learning rate if function is provided
        if lr_func is not None:
            lr = lr_func(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            history['learning_rate'].append(lr)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Learning rate: {lr}")
        
        # Training
        model.train()
        total_loss = 0
        n_batches = 0
        
        # Shuffle data
        indices = torch.randperm(len(X_train))
        X_train_shuffled = X_train_tensor[indices]
        y_train_shuffled = y_train_tensor[indices]
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Check for invalid values
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss value detected: {loss.item()}")
                continue
                
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if n_batches > 0:
            avg_loss = total_loss / n_batches
        else:
            avg_loss = float('inf')
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        # Calculate metrics
        weights_norm = torch.norm(torch.cat([p.flatten() for p in model.parameters()])).item()
        gradients_norm = torch.norm(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])).item()
        
        # Update history
        history['loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['weights_norm'].append(weights_norm)
        history['gradients_norm'].append(gradients_norm)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Train Loss: {avg_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Weight Norm: {weights_norm:.4f}")
            print(f"  Gradient Norm: {gradients_norm:.4f}")
        
        # Early stopping
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
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"History lengths: {len(history['loss'])} epochs")
    
    return history, (X_mean, X_std, y_mean, y_std)

def predict(
    model: torch.nn.Module,
    X: np.ndarray,
    normalization_params: Tuple[np.ndarray, np.ndarray, float, float]
) -> np.ndarray:
    """Make predictions using trained model."""
    X_mean, X_std, y_mean, y_std = normalization_params
    
    # Normalize input
    X_norm = (X - X_mean) / X_std
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_norm)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(X_tensor).numpy()
    
    # Denormalize prediction
    y_pred = y_pred_norm * y_std + y_mean
    
    return y_pred 