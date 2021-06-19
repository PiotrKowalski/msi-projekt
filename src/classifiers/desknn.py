import numpy as np
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaseEnsemble, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import (check_X_y, check_is_fitted, check_array,
                                      check_random_state)
from deslib.util.diversity import negative_double_fault, Q_statistic, \
    ratio_errors, compute_pairwise_diversity
from deslib.base import BaseDS


class DESKNN(BaseDS):

    def __init__(self, pool_classifiers=None, k=7, DFP=False, with_IH=False,
                 safe_k=None, IH_rate=0.30, pct_accuracy=0.5,
                 pct_diversity=0.3, more_diverse=True, metric='DF',
                 random_state=None, knn_classifier='knn', knne=False,
                 DSEL_perc=0.5, n_jobs=-1):

        super(DESKNN, self).__init__(pool_classifiers=pool_classifiers,
                                     k=k,
                                     DFP=DFP,
                                     with_IH=with_IH,
                                     safe_k=safe_k,
                                     IH_rate=IH_rate,
                                     random_state=random_state,
                                     knn_classifier=knn_classifier,
                                     knne=knne,
                                     DSEL_perc=DSEL_perc,
                                     n_jobs=n_jobs)

        self.metric = metric
        self.pct_accuracy = pct_accuracy
        self.pct_diversity = pct_diversity
        self.more_diverse = more_diverse

    def fit(self, X, y):
        super(DESKNN, self).fit(X, y)

        self.N_ = int(self.n_classifiers_ * self.pct_accuracy)

        self.J_ = int(np.ceil(self.n_classifiers_ * self.pct_diversity))

        if self.metric not in ['DF', 'Q', 'ratio']:
            raise ValueError(
                'Diversity metric must be one of the following values:'
                ' "DF", "Q" or "Ratio"')

        if self.N_ <= 0 or self.J_ <= 0:
            raise ValueError("The values of N_ and J_ should be higher than 0"
                             "N_ = {}, J_= {} ".format(self.N_, self.J_))
        if self.N_ < self.J_:
            raise ValueError(
                "The value of N_ should be greater or equals than J_"
                "N_ = {}, J_= {} ".format(self.N_, self.J_))

        if self.metric == 'DF':
            self.diversity_func_ = negative_double_fault
        elif self.metric == 'Q':
            self.diversity_func_ = Q_statistic
        else:
            self.diversity_func_ = ratio_errors

        return self

    def predict(self, X):
        super(DESKNN, self).predict(X)

    #     # sprawdzenie do wywolany zostal fit
    #     check_is_fitted(self)
    #     # sprawdzenie wejscia
    #     X = check_array(X)
