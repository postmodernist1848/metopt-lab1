from typing import List, Protocol, Tuple
import numpy as np
import torch

type Dataset = List[Tuple[np.ndarray, float]]

class Model(Protocol):
    def __call__(self, parameters, x) -> float: ...
    def gradient(self, parameters, x) -> np.ndarray: ...
    def n_parameters(self) -> int: ...

class Regularization(Protocol):
    def __call__(self, parameters: np.ndarray) -> float: ...
    def gradient(self, parameters: np.ndarray) -> np.ndarray: ...

class L1Regularization(Regularization):
    def __init__(self, λ: float = 1.0):
        assert λ >= 0, "Regularization parameter λ must be non-negative."
        self.λ = λ

    def __call__(self, parameters: np.ndarray) -> float:
        return self.λ * np.sum(np.abs(parameters))

    def gradient(self, parameters: np.ndarray) -> np.ndarray:
        return self.λ * np.sign(parameters)

class L2Regularization(Regularization):
    def __init__(self, λ: float = 1.0):
        assert λ >= 0, "Regularization parameter λ must be non-negative."
        self.λ = λ

    def __call__(self, parameters: np.ndarray) -> float:
        return self.λ * np.sum(parameters**2)

    def gradient(self, parameters: np.ndarray) -> np.ndarray:
        return self.λ * 2 * parameters

class ElasticRegularization(Regularization):
    def __init__(self, λ1: float = 1.0, λ2: float = 1.0):
        self.l1 = L1Regularization(λ1)
        self.l2 = L2Regularization(λ2)

    def __call__(self, parameters: np.ndarray) -> float:
        return self.l1(parameters) + self.l2(parameters)

    def gradient(self, parameters: np.ndarray) -> np.ndarray:
        return self.l1.gradient(parameters) + self.l2.gradient(parameters)

class Polynomial(Model):
    def __init__(self, deg: int):
        self.deg = deg

    def __call__(self, parameters: np.ndarray, x: np.ndarray) -> float:
        assert len(x) == 1, "Polynomial model only supports one-dimensional input."
        assert len(parameters) == self.deg + 1, "Parameters length must match polynomial degree + 1."
        return sum(parameters[i] * x[0]**i for i in range(self.deg + 1))

    def gradient(self, parameters, x: np.ndarray) -> np.ndarray:
        assert len(x) == 1, "Polynomial model only supports one-dimensional input."
        assert len(parameters) == self.deg + 1, "Parameters length must match polynomial degree + 1."
        return np.array([x[0]**i for i in range(self.deg + 1)])
    
    def n_parameters(self) -> int:
        return self.deg + 1

class ErrorFunc:
    def __init__(self, model: Model, reg: Regularization, dataset: Dataset):
        self.model = model
        self.reg = reg
        self.dataset = dataset

    def __call__(self, parameters: np.ndarray) -> float:
        return sum((self.model(parameters, x) - y)**2 for x, y in self.dataset) / len(self.dataset) + self.reg(parameters)
    
    def gradient(self, parameters: np.ndarray, indices) -> np.ndarray:
        '''Calculate stochastic gradient for given indices.'''

        if len(indices) == 0:
            raise ValueError("Indices list cannot be empty.")

        result = np.zeros_like(parameters)
        for i in indices:
            x, y = self.dataset[i]
            grad = 2 * (self.model(parameters, x) - y) * self.model.gradient(parameters, x)
            result += grad
        
        return result / len(indices) + self.reg.gradient(parameters)

def sgd(m: Model, reg: Regularization, d: Dataset, epochs: int, batch_size: int, lr: float):
    # assign zero vector to w of size m.n_parameters()
    w = np.zeros(m.n_parameters())
    ef = ErrorFunc(m, reg, d)
    for k in range(epochs):
        indices = np.random.choice(len(d), batch_size, replace=False)
        grad = ef.gradient(w, indices)
        w -= lr * grad

        # print(f"Epoch {k}, Parameters: {w}, Loss: {ef(w)}, grad: {grad}")

    return w

def torch_sgd(dataset: Dataset, degree: int = 2, epochs: int = 100, batch_size: int = 1, lr: float = 0.01) -> np.ndarray:
    # Convert dataset to tensors
    X = torch.tensor([x[0] for x in dataset], dtype=torch.float32)
    y = torch.tensor([x[1] for x in dataset], dtype=torch.float32)
    
    # Create polynomial features
    X_poly = torch.cat([X**i for i in range(degree + 1)], dim=1)
    
    # Initialize model
    model = torch.nn.Linear(degree + 1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    for _ in range(epochs):
        # Shuffle data
        indices = torch.randperm(len(dataset))
        X_shuffled = X_poly[indices]
        y_shuffled = y[indices]
        
        # Mini-batch training
        for i in range(0, len(dataset), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
    
    return model.weight.detach().numpy().flatten()