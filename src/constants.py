from enum import Enum, IntEnum

RANDOM_STATE = 1
N_SPLITS = 5

data_set_values = {
    'n_samples': 1000,
    'n_features': 20,
    'n_informative': 1,
    'n_repeated': 0,
    'n_redundant': 0,
    'flip_y': .05,
    'random_state': RANDOM_STATE,
    'n_clusters_per_class': 1
}


class ClassifierTypes(IntEnum):
    """
    Classifiers types
    """
    DESkNN = 1
    KNORAU = 2
    KNORAE = 3
    ADABoost = 4


class FilterTypes(IntEnum):
    """
    Filter types
    """
    NoFilter = 1
    PCA = 2
    chi2 = 3
