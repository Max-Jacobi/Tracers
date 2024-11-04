from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Interpolator(ABC):
    coord_keys: tuple[str, ...]
    data_keys: tuple[str, ...]
    data: Any

    def load_data(
        self,
        t_span: tuple[float, float],
    ):
        pass


    @staticmethod
    @abstractmethod
    def interpolate_velocities(
        time: float,
        coords: np.ndarray,
        data: Any,
    ) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def interpolate_data(
        time: float,
        coords: np.ndarray,
        data: Any,
    ) -> np.ndarray:
        pass
