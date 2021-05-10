import numpy as np
from deslib.des import KNORAU
from src.runners.helpers import run_kfold
from src.runners.base_runner import BaseRunner
from src.constants import *


class KNORAURunner(BaseRunner, KNORAU):

    def run_classifier(self, filter_type: FilterTypes) -> (np.ndarray, np.ndarray):
        return run_kfold(self, filter_type)

