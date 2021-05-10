import copy

import numpy as np
from sklearn.metrics import accuracy_score

from constants import ClassifierTypes, FilterTypes
from src.runners.ada_boost import AdaBoostRunner
from src.runners.knora_u import KNORAURunner
from src.runners.knora_e import KNORAERunner


def get_mean_and_std(classifier_type: ClassifierTypes, filter_type: FilterTypes) -> (np.ndarray, np.ndarray):
    """
    Return mean and standard deviation of given classifier and filter
    :param classifier_type:
    :param filter_type:
    :return: mean and standard deviation
    """

    return {
        ClassifierTypes.DESkNN: NotImplementedError,
        ClassifierTypes.KNORAU: KNORAURunner().run_classifier(filter_type),
        ClassifierTypes.KNORAE: KNORAERunner().run_classifier(filter_type),
        ClassifierTypes.ADABoost: AdaBoostRunner().run_classifier(filter_type)

    }.get(classifier_type, ValueError)

