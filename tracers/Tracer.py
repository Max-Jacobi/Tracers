from typing import Iterable, Iterator
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult
from collections.abc import  Mapping

class Tracer(Mapping):
    def __init__(
        self,
        pos_start: Iterable[float],
        t_start: float,
        index: int,
        coord_keys: tuple[str, ...],
        data_keys: tuple[str, ...],
    ):
        self.pos_start = np.array(pos_start)
        self.t_start = t_start
        self.id = index
        self.coord_keys = coord_keys
        self.data_keys = data_keys
        self.message = ""

        self.pos = self.pos_start
        self.dt: (None | float) = None
        self.started = False
        self.finished = False
        self.failed = False

        self.trajectory = {key: np.array([x_ini])
                           for key, x_ini in zip(coord_keys, pos_start)}
        self.trajectory['time'] = np.array([t_start])
        for key in data_keys:
            self.trajectory[key] = np.array([])

    def update_trajectory(self, sol: (OdeResult | None)) -> None:
        if sol is None:
            return

        if sol.success and len(sol.t) > 0:
            self.pos = sol.y[:, -1]
            for ii, tv in enumerate(sol.t_events):
                if len(tv) > 0:
                    self.finished = True
                    self.message = f"event {ii} reached"
                    break
        elif not sol.success:
            self.failed = True
            self.finished = True
            self.message = sol.message

        for i in range(2, len(sol.t)):
            self.dt = sol.t[-i] - sol.t[-i-1]
            if self.dt != 0:
                break
        else:
            self.dt = None

        if self.dt > 0:
            new = sol.t > self.trajectory['time'][-1]
        else:
            self.dt *= -1
            new = sol.t < self.trajectory['time'][-1]
        self.trajectory['time'] = np.append(self.trajectory['time'], sol.t[new])

        for ix, key in enumerate(self.coord_keys):
            self.trajectory[key] = np.append(self.trajectory[key], sol.y[ix][new])


    def handle_error(self, error: Exception) -> None:
        self.failed = True
        self.finished = True
        self.message = str(error)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.trajectory[key]

    def __iter__(self) -> Iterator:
        return self.trajectory.__iter__()

    def __len__(self) -> int:
        return len(self.trajectory)

    def __repr__(self) -> str:
        if self.failed:
            status = "failed"
        elif self.finished:
            status = "finished"
        elif self.started:
            status = "running"
        else:
            status = "not started"
        return f"Tracer {self.id}: {status}"

    def __str__(self) -> str:
        return f"Tracer {self.id}"
