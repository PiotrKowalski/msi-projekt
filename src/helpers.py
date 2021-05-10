import copy
import logging

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
        ClassifierTypes.DESkNN: NotImplemented,
        ClassifierTypes.KNORAU: KNORAURunner().run_classifier(filter_type),
        ClassifierTypes.KNORAE: KNORAERunner().run_classifier(filter_type),
        ClassifierTypes.ADABoost: AdaBoostRunner().run_classifier(filter_type)

    }.get(classifier_type, ValueError)


def get_filled_table():
    """
    Create and fill table with mean and standard deviation using
    classifiers in ClassifiersTypes and filter in FilterTypes
    :return: 2d list filled with means and standard deviations
    """
    result_table = [[None for _ in range(len(FilterTypes) + 1)] for _ in range(len(ClassifierTypes) + 1)]

    for clf_id, clf_name in enumerate(ClassifierTypes):
        result_table[clf_id + 1][0] = str(clf_name).removeprefix('ClassifierTypes.')

        for filter_id, filter_name in enumerate(FilterTypes):
            result_table[0][filter_id + 1] = str(filter_name).removeprefix('FilterTypes.')

            try:
                result_table[clf_id + 1][filter_id + 1] = (get_mean_and_std(clf_name, filter_name))
            except NotImplemented:
                logging.info(f"Not implemented: {clf_name, filter_name}")

    return result_table
