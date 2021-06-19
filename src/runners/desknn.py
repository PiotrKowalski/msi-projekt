import numpy as np
from deslib.des import DESKNN as des
from src.runners.base_runner import BaseRunner
from src.constants import *
from src.runners.helpers import run_kfold
from src.classifiers.desknn import DESKNN




class DESKNNRunner(BaseRunner, des):

    def run_classifier(self, filter_type: FilterTypes) -> (np.ndarray, np.ndarray):
        return run_kfold(self, filter_type)
