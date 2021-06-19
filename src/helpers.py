import logging
import numpy as np
from constants import ClassifierTypes, FilterTypes
from src.runners.ada_boost import AdaBoostRunner
from src.runners.knora_u import KNORAURunner
from src.runners.knora_e import KNORAERunner
from src.runners.desknn import DESKNNRunner
import copy
import matplotlib.pyplot as plt
import pandas as pd

def get_mean_and_std(classifier_type: ClassifierTypes, filter_type: FilterTypes) -> (np.ndarray, np.ndarray):
    """
    Return mean and standard deviation of given classifier and filter
    :param classifier_type:
    :param filter_type:
    :return: mean and standard deviation
    """

    return {
        ClassifierTypes.DESkNN: DESKNNRunner().run_classifier(filter_type),
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
                result_table[clf_id + 1][filter_id + 1] = get_mean_and_std(clf_name, filter_name)
            except NotImplemented:
                logging.info(f"Not implemented: {clf_name, filter_name}")

    return result_table


def transform_row_data_to_row_mean(row):
    transformed = copy.deepcopy(row)

    for i, data in enumerate(transformed[1:]):
        transformed[i + 1] = data[0]

    return transformed


def transform_row_data_to_row_std(row):
    transformed = copy.deepcopy(row)

    for i, data in enumerate(transformed[1:]):
        transformed[i + 1] = data[1]

    return transformed


def save_figs(result_table):
    filters_labels = result_table[0]
    filters_labels[0] = "Klasyfikatory"

    desknn = transform_row_data_to_row_mean(result_table[1])
    knorau = transform_row_data_to_row_mean(result_table[2])
    knorae = transform_row_data_to_row_mean(result_table[3])
    adaboost = transform_row_data_to_row_mean(result_table[4])

    desknn_std = transform_row_data_to_row_std(result_table[1])
    knorau_std = transform_row_data_to_row_std(result_table[2])
    knorae_std = transform_row_data_to_row_std(result_table[3])
    adaboost_std = transform_row_data_to_row_std(result_table[4])

    # print(filters_labels)
    # create data
    df = pd.DataFrame([desknn, knorau, knorae, adaboost],
                      columns=filters_labels)

    errors = pd.DataFrame([desknn_std, knorau_std, knorae_std, adaboost_std],
                          columns=filters_labels)

    # view data

    # plot grouped bar chart
    fig = df.plot(x='Klasyfikatory',
                  kind='bar',
                  stacked=False,
                  ylabel="Średnia wartość metryki",
                  title='Średnia wartość metryki jakości accuracy score',
                  rot=0,
                  )
    fig.figure.savefig('srednia.png')

    plt.show()

    fig2 = errors.plot(x='Klasyfikatory',
                       kind='bar',
                       stacked=False,
                       ylabel="Odchylenie standardowe metryki",
                       title='Odchylenie standardowe metryki jakości accuracy score',
                       rot=0,
                       )
    fig2.figure.savefig('errors.png')

    plt.show()

