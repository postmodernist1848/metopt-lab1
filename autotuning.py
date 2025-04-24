import os
import sys
from lib.funcs import q1, q2, f4
import lib.algorithms as algorithms
import optuna
import numpy as np
import pathlib

class OptunaTime:
    SEC = 1
    MIN = 60 * SEC
    HOUR = 60 * MIN


def clean(objectives):
    for objective in objectives:
        try:
            os.remove(f'studies/{objective.__name__}.db')
        except OSError:
            pass

        try:
            os.remove(f'studies/{objective.__name__}.db-journal')
        except OSError:
            pass


def optimize(objectives):
    for objective in objectives:
        study_name = objective.__name__

        pathlib.Path("studies/").mkdir(parents=True, exist_ok=True)
        storage_name = f"sqlite:///studies/{study_name}.db"

        study = optuna.create_study(study_name=study_name, storage=storage_name, directions=[
                                    'minimize', 'minimize'], load_if_exists=True)

        if not study.best_trials:
            study.optimize(objective, n_trials=200, timeout=OptunaTime.MIN *
                           10, n_jobs=-1, show_progress_bar=True)

        print(study_name)
        for trial in study.best_trials:
            print(trial.number)
            print(trial.params)
            print(trial.values)


def main():
    optuna.logging.set_verbosity(optuna.logging.WARN)

    x_0s = list(map(np.array, ([-10, 10], [4, 1], [1, 3])))
    funcs = [q1, q2, f4]

    def damped_newton_constant_learning_rate(trial: optuna.Trial):
        α = trial.suggest_float('λ', 0, 1)

        values = []
        iterations = []

        for func in funcs:
            for x_0 in x_0s:
                sc = algorithms.relative_x_condition()
                trajectory = algorithms.damped_newton_descent(
                    x_0, func, sc, algorithms.constant_h(α))
                x = trajectory[-1]
                values.append(func(x))
                iterations.append(len(trajectory))

        return sum(values), sum(iterations)

    def dogleg_armijo(trial: optuna.Trial):
        very_low = trial.suggest_float('very_low', 0, 0.5)
        low = trial.suggest_float('low', very_low, 1)
        high = trial.suggest_float('high', low, 1)

        values = []
        iterations = []

        for func in funcs:
            for x_0 in x_0s:
                sc = algorithms.relative_x_condition()
                trajectory = algorithms.newton_descent_with_1d_search(x_0, func, sc,
                                                                      algorithms.armijo_step_selector,
                                                                      very_low, low, high)
                x = trajectory[-1]
                values.append(func(x))
                iterations.append(len(trajectory))

        return sum(values), sum(iterations)

    def learning_rate_scheduling_exponential_decay(trial: optuna.Trial):
        λ = trial.suggest_float('λ', 1e-10, 10, log=True)

        values = []
        iterations = []

        for func in funcs:
            for x_0 in x_0s:
                sc = algorithms.relative_x_condition()
                trajectory = algorithms.learning_rate_scheduling(x_0, func, algorithms.exponential_decay(λ), sc)
                x = trajectory[-1]
                values.append(func(x))
                iterations.append(len(trajectory))

        return sum(values), sum(iterations)

    objectives = [damped_newton_constant_learning_rate, dogleg_armijo, learning_rate_scheduling_exponential_decay]

    # clean subcommand removes all study files
    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        clean(objectives)
        return

    optimize(objectives)


if __name__ == "__main__":
    main()
