from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate._ivp.ivp import OdeResult

from .Tracer import Tracer

class Seeds(ABC):
    coord_keys: tuple[str, ...]
    n_seeds: int

    @abstractmethod
    def get_seeds(self) -> tuple[np.ndarray, ...]:
        """
        Return tuple of n+1 arrays.
        First n arrays are the coordinates.
        Last array are the start time.
        """
        pass
