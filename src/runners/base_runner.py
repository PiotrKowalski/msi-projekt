from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from deslib.base import BaseDS
from deslib.des import KNORAU
from sklearn import datasets, clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from src.constants import *
from src.filters.chi2 import Chi2Filter
from src.filters.pca import PCAFilter
from sklearn.ensemble import AdaBoostClassifier


class BaseRunner(ABC):

    def __init__(self):
        """
        Init of runner. Now create synthetic data.
        """
        super(BaseRunner, self).__init__()
        self.X, self.y = datasets.make_classification(**data_set_values)

        self.skf = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        self.scores = np.zeros(N_SPLITS)

    @abstractmethod
    def run_classifier(self, filter_type: FilterTypes) -> (np.ndarray, np.ndarray):
        """
        Run classifier using filter_type to get mean and std
        :param filter_type:
        :return:
        """
        pass

    def _filter_x(self, filter_type: FilterTypes, train):
        """
        Switch case for filtering x
        :param filter_type:
        :param train:
        :return:
        """
        return {
            FilterTypes.NoFilter: self.X,
            FilterTypes.PCA: PCAFilter().filter_x(self.X, train=train),
            FilterTypes.chi2: Chi2Filter().filter_x(self.X, y=self.y),

        }[filter_type]

    def calculate_mean_and_std(self) -> (np.ndarray, np.ndarray):
        """
        Calculate mean and std using list of scores
        :return: Returns means and std
        """
        mean = round(np.mean(self.scores), 3)
        std = round(np.std(self.scores), 3)

        return mean, std

