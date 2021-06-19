from typing import Any
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

from src.filters.base_filter import BaseFilter


class Chi2Filter(BaseFilter):
    def filter_x(self, x, **kwargs) -> Any:
        y = kwargs.get("y")
        minmaxscaler = MinMaxScaler()
        minmaxscaler.fit(x)
        scaled_x = minmaxscaler.transform(x)

        chi2_x = SelectKBest(chi2, k=2).fit_transform(scaled_x, y)
        # pca = PCA()
        # pca.fit(x[train])
        # pca_x = pca.transform(x)

        return chi2_x
