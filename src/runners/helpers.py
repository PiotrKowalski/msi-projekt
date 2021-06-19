from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score
from src.constants import FilterTypes


def run_kfold(runner: Any, filter_type: FilterTypes) -> (np.ndarray, np.ndarray):
    """
    Runs k folds using skf from runner
    :param runner: 
    :param filter_type: 
    :return: 
    """
    for fold_id, (train, test) in enumerate(runner.skf.split(runner.X, runner.y)):

        X = runner.filter_x(filter_type, train)

        runner.fit(X[train], runner.y[train])
        y_pred = runner.predict(X[test])
        runner.scores[fold_id] = accuracy_score(runner.y[test], y_pred)


    mean, std = runner.calculate_mean_and_std()

    return mean, std
