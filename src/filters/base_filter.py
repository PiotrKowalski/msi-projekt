from abc import ABC, abstractmethod


class BaseFilter(ABC):
    pass

    @abstractmethod
    def filter_x(self, x, **kwargs):
        pass