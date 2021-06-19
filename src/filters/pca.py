from typing import Any
from sklearn.decomposition import PCA
from src.filters.base_filter import BaseFilter


class PCAFilter(BaseFilter):
    def filter_x(self, x, **kwargs) -> Any:
        train = kwargs.get("train")
        pca = PCA()
        pca.fit(x[train])
        pca_x = pca.transform(x)
        return pca_x
