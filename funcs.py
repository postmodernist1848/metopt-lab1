from main import *

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
f4 = BiFuncCallableWrapper(mf4)  # TOP 1
f5 = BiFuncCallableWrapper(mf5)
f6 = BiFuncCallableWrapper(mf6)
fopp = BiFuncCallableWrapper(opp)
fopp2 = BiFuncCallableWrapper(opp2)
fopp3 = BiFuncCallableWrapper(opp3)