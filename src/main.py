import numpy as np
from sklearn import datasets, clone
from deslib.des.knora_u import KNORAU
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from constants import *
from src.helpers import get_mean_and_std
import logging

if __name__ == '__main__':

    for clf_id, clf_name in enumerate(ClassifierTypes):

        for filter_id, filter_name in enumerate(FilterTypes):

            try:
                print(get_mean_and_std(clf_name, filter_name), f'{clf_name, filter_name}')
            except NotImplementedError:
                logging.log(f"Not implemented: {clf_name, filter_name}")
