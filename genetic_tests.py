from annealing import *
from lib.funcs import *
from annealing_funcs import *

def init_population_commivoyager(x: Vector, n_permutations: int) -> Vector:
    return np.array([np.random.permutation(x) for _ in range(n_permutations)])

def crossover_commivoyager(parent1: Vector, parent2: Vector) -> Vector:
    n = len(parent1)
    prefix_index = np.random.randint(0, n)
    child = [] 
    child[:prefix_index] = parent1[:prefix_index]
    prefix_set = set(tuple(x) for x in child)
    
    remaining = []
    for x in parent2:
        if tuple(x) not in prefix_set:
            remaining.append(x)
    child[prefix_index:] = remaining
    return child

def build_paths(path: dict) -> list:
    paths = []
    used_points = set()
    
    for start_point in path:
        if start_point in used_points:
            continue
            
        current_path = [start_point]
        used_points.add(start_point)
        current_point = start_point
        
        while True:
            next_points = path[current_point]
            if not next_points:
                break
                
            next_point = None
            for point in next_points:
                if point not in used_points:
                    next_point = point
                    break
                    
            if next_point is None:
                break
                
            current_path.append(next_point)
            used_points.add(next_point)
            current_point = next_point
            
        if len(current_path) > 1:
            paths.append(current_path)
    return paths

def build_path_dict(distance: list) -> dict:
    path = {}
    for _, points in distance:
        key1 = points[0]
        key2 = points[1]
        if key1 not in path:
            path[key1] = []
        path[key1].append(key2)
        if key2 not in path:
            path[key2] = []
        path[key2].append(key1)
    return path

def get_distance(parent: Vector) -> list:
    distance = []
    n = len(parent)
    prev_point = parent[0]
    for i in range(1, n + 1):
        point = parent[i % n]
        distance.append((np.linalg.norm(prev_point - point), [tuple(prev_point), tuple(point)]))
        prev_point = point

    distance.sort(key=lambda x: x[0])
    return distance

def crossover_commivoyager_smart(parent1: Vector, parent2: Vector) -> Vector:
    n = len(parent1)
    alpha = np.random.randint(0, n)

    distance = get_distance(parent1)
    distance = distance[:alpha]

    path = build_path_dict(distance)
    paths = build_paths(path)

    merged_path = []
    for path in paths:
        merged_path.extend(path)

    merged_path_set = set(merged_path)
    diff = []
    
    for point in parent2:
        if tuple(point) not in merged_path_set:
            diff.append(point)
            
    merged_path.extend(diff)
    return merged_path

def mutate_commivoyager(individual: Vector, mutation_rate: float = 0.1) -> Vector:
    if np.random.random() < mutation_rate:
        size = len(individual)
        i, j = np.random.choice(size, 2, replace=False)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

def test_genetic_algorithm(population: Vector, 
                         crossover_func: Callable[[Vector, Vector], Vector],
                         mutate_func: Callable[[Vector, float], Vector],
                         fitness_function: BiFunc,
                         tournament_size: int,
                         mutation_rate: float,
                         eps: float) -> None:
    clever_population, k = genetic(population,
                                 crossover_func,
                                 mutate_func,
                                 fitness_function,
                                 tournament_size,
                                 mutation_rate,
                                 eps)
    x = min(clever_population, key=fitness_function)
    print(x)
    print("function value:", fitness_function(x))
    if fitness_function.min() is not None:
        print("error:", abs(fitness_function(x) - fitness_function.min()))
    print("iterations:", k)

def f4_genetic_test():
    population = init_population(-10, 10, 100, 2)
    func = f4
    test_genetic_algorithm(population, crossover, mutate, func, 10, 0.2, 1e-3)

def commivoyager_genetic_test(points: Vector, correct_value: float = None):
    x0 = np.random.permutation(points)
    
    population = init_population_commivoyager(x0, 100)
    func = BiFuncCallableWrapper(commivoyager, correct_value)
    test_genetic_algorithm(population, 
                         crossover_commivoyager_smart,
                         mutate_commivoyager,
                         func, 60, 0.1, 1e-2)
    
    x = min(population, key=func)
    commivoyager_plot(x0, x, "Initial vs Optimized path")

if __name__ == "__main__":
    # f4_genetic_test()
    commivoyager_genetic_test(
        np.array([[0, 0], [2, 0], [4, 0],
                   [0, 2], [0, 4], [-2, 0],
                     [-4, 0], [0, -2], [0, -4],
                       [1, 1]]), 26.14213562373095)
