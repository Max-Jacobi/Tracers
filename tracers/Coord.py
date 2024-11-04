from typing import Iterable

import numpy as np
import numpy.typing as npt

Mask = np.ndarray[bool]

class Coord:
    def __init__(self,
                 keys: Iterable[str],
                 n_points: int,
        ) -> None:
        self.coord_keys = set(keys)
        self.n_points = n_points
        self.key_map = {key: idx for idx, key in enumerate(self.coord_keys)}
        self.array = np.empty(self.n_points * len(keys))


    def _get_slice(self, key: str) -> slice:
        if key not in self.key_map:
            raise KeyError(f"Key '{key}' not found.")
        idx = self.key_map[key]
        start = idx * self.n_points
        end = start + self.n_points
        return slice(start, end)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.array[self._get_slice(key)]

    def __setitem__(self, key: str, value: npt.ArrayLike) -> None:
        if len(value) != self.n_points:
            raise ValueError(f"Coordinates must have length {self.n_points}, not {len(value)}")
        self.array[self._get_slice(key)] = value

    def mask(self, mask: Mask) -> 'Coord':
        assert len(mask) == self.n_points
        if mask.all():
            return self.copy()
        coord = Coord(self.coord_keys, self.n_points - np.sum(~mask))
        for key in self.coord_keys:
            coord[key] = self[key][mask]
        return coord

    def mask_set(self, mask: Mask, coord: 'Coord') -> None:
        assert len(mask) == self.n_points
        assert coord.n_points == np.sum(mask)
        for key in self.coord_keys:
            self[key][mask] = coord[key]

    def copy(self) -> 'Coord':
        new = Coord(self.coord_keys, self.n_points)
        new.array[:] = self.array.copy()
        return new

    def __repr__(self) -> str:
        return f"Coord(keys={self.keys}, m={self.n_points})"

    def keys(self) -> Iterable[str]:
        return self.coord_keys

def
