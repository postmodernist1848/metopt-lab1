import numpy as np
from lab4.annealing import Vector

def commivoyager(x: Vector) -> float:
    n = len(x)
    res = 0
    for i in range(n):
        res += np.linalg.norm(x[(i+1) % n] - x[i])
    return res

def init_population_commivoyager(x: Vector, n_permutations: int) -> Vector:
    return np.array([np.random.permutation(x) for _ in range(n_permutations)])

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
            if current_point not in path:
                break
                
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
        key1 = tuple(points[0])
        key2 = tuple(points[1])
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
    if n == 0:
        return distance
    prev_point = parent[0]
    for i in range(1, n + 1):
        point = parent[i % n]
        distance.append((np.linalg.norm(prev_point - point), [tuple(prev_point), tuple(point)]))
        prev_point = point

    distance.sort(key=lambda x: x[0])
    return distance

def add_key_with_values(path: dict, key: int, value: list):
    if key not in path:
        path[key] = []
    for val in value:
        path[key].append(val)

def check_path_exists(paths: list, key: int, value: int) -> bool:
    for path in paths:
        if key in path and value in path:
            return False
    return True

def get_deg(path: dict, key: int) -> int:
    return len(path[key])

def update_paths(path: dict, paths: list, parent: Vector) -> list:
    distance = get_distance(parent)
    path_p2 = build_path_dict(distance)

    for key, value in path_p2.items():
        deg_key = get_deg(path_p2, key)

        if ((key in path) and (deg_key == 2)):
            continue

        if ((key in path) and (deg_key == 2) and (value[0] not in path) and (value[1] not in path)):
            continue

        if ((key in path) and (deg_key == 2) and (value[0] in path) and (value[1] in path) and (get_deg(path, value[0]) == 2) and (get_deg(path, value[1]) == 2)):
            continue

        if ((key in path) and (deg_key == 2)):
            values_with_deg_1 = [val for val in value if get_deg(path_p2, val) == 1]
            for val in values_with_deg_1:
                if (val in path[key]): # if key - val exist
                    continue
                add_key_with_values(path, key, value)

        if ((key not in path) and (deg_key == 1) and (value[0] in path) and (get_deg(path, value[0]) == 2)):
            continue

        if ((key not in path) and (deg_key == 2) and (value[0] in path) and (value[1] in path)):
            continue
        
        if ((key not in path) and (deg_key == 1) and ((value[0] not in path) or (value[0] in path and get_deg(path, value[0]) == 1))):
            add_key_with_values(path, key, value)

        if ((key not in path) and (deg_key ==  2) and (value[0] not in path) and (value[1] not in path)):
            add_key_with_values(path, key, value)

        if ((key in path) and (deg_key == 1) and (value[0] not in path)):
            add_key_with_values(path, key, value)

        if ((key in path) and (deg_key == 1) and (value[0] in path) and (get_deg(path, value[0]) == 1)):
            if (check_path_exists(paths, key, value[0])):
                add_key_with_values(path, key, value)
            continue

        paths = build_paths(path)
    return paths

def crossover_commivoyager(parent1: Vector, parent2: Vector) -> Vector:
    n = len(parent1)
    alpha = np.random.randint(0, n)

    distance = get_distance(parent1)
    distance = distance[:alpha]
    path = build_path_dict(distance)
    paths = build_paths(path)
    
    paths = update_paths(path, paths, parent2)

    merged_path = []
    for path in paths:
        merged_path.extend(path)

    distance_p2 = get_distance(parent2)
    path_p2 = build_path_dict(distance_p2)
    paths_p2 = build_paths(path_p2)
    merged_path_p2 = []
    for path in paths_p2:
        merged_path_p2.extend(path)
    
    merged_path_set = set(merged_path)
    merged_path_p2_set = set(merged_path_p2)
    diff = merged_path_p2_set - merged_path_set

    only_in_p2 = []
    for point in merged_path_p2:
        if tuple(point) in diff:
            only_in_p2.append(point)

    merged_path.extend(only_in_p2)
    return merged_path

def mutate_commivoyager(individual: Vector, mutation_rate: float = 0.1) -> Vector:
    if np.random.random() < mutation_rate:
        size = len(individual)
        i, j = np.random.choice(size, 2, replace=False)
        individual[i], individual[j] = individual[j], individual[i]
    return individual