from lib.algorithms import *

class Quadratic:
    A: np.ndarray
    B: np.ndarray
    C: float

    def __init__(self, a: np.ndarray, b: np.ndarray, c: float):
        self.A = a
        self.B = b
        self.C = c

    def __call__(self, x: np.ndarray) -> float:
        # x^T A x + x^T B x + C
        return float(x.T @ self.A @ x + self.B @ x + self.C)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2 * self.A @ x + self.B

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return 2 * self.A

    def min(self) -> float | None:
        # x = -1/2 * A^(-1) * B
        try:
            x = -0.5 * np.linalg.inv(self.A) @ self.B
            return self(x)
        except np.linalg.LinAlgError:
            return None


class BiFuncCallableWrapper:
    f: Callable[[float, float], float]
    min_value: float | None = None
    h: float = 0.001

    def __init__(self, f: Callable[[float, float], float], min_value: float | None = None):
        self.f = f
        self.min_value = min_value

    def __call__(self, x: np.ndarray) -> float:
        return self.f(x[0], x[1])

    def gradient(self, x: np.ndarray) -> np.ndarray:
        d1 = (self.f(x[0] + self.h, x[1]) -
              self.f(x[0] - self.h, x[1])) / (2*self.h)
        d2 = (self.f(x[0], x[1] + self.h) -
              self.f(x[0], x[1] - self.h)) / (2*self.h)
        g = np.array([d1, d2])
        return g

    def hessian(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x

        # Use a central difference formula for better accuracy
        # For f_xx (second derivative with respect to x)
        f_xx = (self.f(x1 + self.h, x2) - 2 * self.f(x1, x2) +
                self.f(x1 - self.h, x2)) / (self.h * self.h)

        # For f_yy (second derivative with respect to y)
        f_yy = (self.f(x1, x2 + self.h) - 2 * self.f(x1, x2) +
                self.f(x1, x2 - self.h)) / (self.h * self.h)

        # For f_xy (mixed derivative)
        # Use the cross partial difference formula
        f_xy = (self.f(x1 + self.h, x2 + self.h) - self.f(x1 + self.h, x2 - self.h) -
                self.f(x1 - self.h, x2 + self.h) + self.f(x1 - self.h, x2 - self.h)) / (4 * self.h * self.h)

        hessian = np.array([
            [f_xx, f_xy],
            [f_xy, f_yy]
        ])

        return hessian
    
    def min(self) -> float | None:
        return self.min_value


class NoisyWrapper:
    f: BiFunc
    factor: float

    def __init__(self, f: BiFunc, factor: float = 1.0):
        self.f = f
        self.factor = factor

    def __call__(self, x: np.ndarray):
        return self.f.__call__(x) + random.random() * self.factor

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.f.gradient(x)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return self.f.hessian(x)

    def min(self) -> float | None:
        return self.f.min()


q1 = Quadratic(
    np.array([[1, 0], [0, 1]]),
    np.array([1, 1]),
    -1.4
)

q2 = Quadratic(
    np.array([[0.1, 0], [0, 3]]),
    np.array([0, 0]),
    0
)  # Dichotomy is much better

q3 = Quadratic(
    np.array([[4, 0], [0, 1]]),
    np.array([-1, 2]),
    1.5
)


def mf1(x, y):
    term1 = np.sin(x) * np.cos(y)
    term2 = -1.0 * np.exp(-(x**2 + y**2)/10)
    term3 = 0.1 * (x**2 + y**2)
    return term1 + term2 + term3


def mf2(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def mf3(x, y):
    term2 = -1.0 * np.exp(-(x**2 + y**2)/10)
    return term2


def mf4(x, y):
    return (x**2 - 1)**2 + y**2 + 0.5 * x


def mf5(x, y):
    return (x**2 - 1)**2 + y**2


def mf6(x, y):
    norm_squared = x**2 + y**2
    term1 = ((norm_squared - 2)**2)**(1/8)
    term2 = 0.5 * (0.5 * norm_squared + (x + y))
    return term1 + term2 + 0.5

def opp(x, y):
    return 0.65 * (x - 21) ** 2 + 1.1*(y+37)**2 -7 * x + 12 * y - 15

def opp2(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def opp3(x, y):
    term1 = np.sin(10*(x**2 + y**2)) / 10 
    term2 = (x**2 + y**2)/4 
    term3 = np.cos(5*x)*np.sin(5*y)/5
    return term1 + term2 + term3

f1 = BiFuncCallableWrapper(mf1)
f2 = BiFuncCallableWrapper(mf2)
f3 = BiFuncCallableWrapper(mf3)
f4 = BiFuncCallableWrapper(mf4, -0.514753641275705599276576050856482912233727798097409)
f5 = BiFuncCallableWrapper(mf5, -0.5)
f6 = BiFuncCallableWrapper(mf6, 0)
fopp = BiFuncCallableWrapper(opp, -657.573)
fopp2 = BiFuncCallableWrapper(opp2, 0) # 1
fopp3 = BiFuncCallableWrapper(opp3, -0.119789)
fsinsin = BiFuncCallableWrapper(lambda x, y: math.sin(x) + math.sin(y), -2) # 2