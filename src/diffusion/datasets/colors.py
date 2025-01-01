from enum import Enum
from typing import Tuple


class Color(Enum):
    RED = (1.0, 0.0, 0.0)
    GREEN = (0.0, 1.0, 0.0)
    BLUE = (0.0, 0.0, 1.0)
    YELLOW = (1.0, 1.0, 0.0)
    PURPLE = (0.8, 0.0, 0.8)
    ORANGE = (1.0, 0.65, 0.0)
    WHITE = (1.0, 1.0, 1.0)

    @property
    def rgb(self) -> Tuple[float, float, float]:
        return self.value

    @property
    def name(self) -> str:
        return self._name_
