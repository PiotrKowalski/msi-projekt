from abc import ABC, abstractmethod
from typing import Any


class BaseFilter(ABC):
    pass

    @abstractmethod
    def filter_x(self, x, **kwargs) -> Any:
        pass