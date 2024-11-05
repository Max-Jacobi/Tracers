from typing import Callable, Any, Iterable, Iterator, Any, Sequence
from collections.abc import  Mapping
from abc import ABC, abstractmethod
from multiprocessing import Pool

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from tqdm import tqdm

def do_parallel(
    func: Callable,
    args: (Sequence | np.ndarray),
    n_cpu: int,
    verbose: bool = False,
    **kwargs
    ):
    kwargs["total"] = len(args)
    kwargs["disable"] = not verbose
    kwargs["ncols"] = 0
    kwargs["file"] = stdout

    if n_cpu == 1:
        return list(tqdm(map(func, args), **kwargs))
    with Pool(n_cpu) as pool:
        return list(tqdm(pool.imap_unordered(func, args), **kwargs))

class Tracer(Mapping):
    def __init__(
        self,
        pos_start: Iterable[float],
        t_start: float,
        index: int,
        coord_keys: tuple[str, ...],
        data_keys: tuple[str, ...],
        props: None | dict[str, Any] = None,
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

        self.props = dict() if props is None else props

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
            return

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

    def output_to_ascii(self, filebase: str) -> str:
        keys = ['time', *self.coord_keys, *self.data_keys]

        props = "; ".join((f"{key}={val}" for key, val in self.props.items()))
        legend = "8s" + " 12.6f" * (len(keys) - 1)
        legend = legend.format(*keys)
        header = f"{props}\nmessage: {self.message}\n{legend}"

        filename = f"{filebase}_{self.id}.dat"

        tsort = np.argsort(self.trajectory['time'])
        data = np.column_stack([self.trajectory[key][tsort] for key in keys])

        np.savetxt(filename, data, header=header)
        return filename


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

class Tracers:

    def __init__(
        self,
        seeds: dict[str, Any],
        interpolator: Interpolator,
        n_cpu: int = 1,
        verbose: bool = False,
        end_conditions: list[Callable[[Tracer], Tracer]] = [],
        **kwargs,
    ):
        self.coord_keys = interpolator.coord_keys
        self.data_keys = interpolator.data_keys
        self.interpolator = interpolator
        self.seeds = seeds
        self.n_cpu = n_cpu
        self.verbose = verbose
        self.end_conditions = end_conditions
        self.kwargs = kwargs

        assert all(key in self.seeds.keys() for key in self.coord_keys), \
            f"Seed coordinates do not match Tracers coordinates {self.coord_keys}"
        self.n_tracers = len(seeds[self.coord_keys[0]])

        self.last_time: None | float = None

        self.init_tracers()

    def init_tracers(self):
        time = self.seeds['time']
        coords = [self.seeds[key] for key in self.coord_keys]
        props = {key: self.seeds[key] for key in self.seeds
                 if key not in self.coord_keys and key != 'time'}

        self.tracers =  [Tracer(pos, t, ii, self.coord_keys, self.data_keys, props=props)
                         for ii, (*pos, t) in enumerate(zip(*coords, time))]

    @staticmethod
    def _integrate_inner(
        args: tuple[Tracer, tuple[float, float], Interpolator, dict]
    ) -> Tracer:
        tracer, t_span, interpolator, kwargs = args

        if min(*t_span) < tracer.t_start <= max(*t_span):
            tracer.started = True
            t_span = (tracer.t_start, t_span[1])

        if (not tracer.started) or tracer.finished:
            return tracer

        try:
            sol = solve_ivp(
                interpolator.interpolate_velocities,
                t_span=t_span,
                first_step=tracer.dt,
                y0=tracer.pos,
                args=(interpolator.data,),
                **kwargs
            )
            tracer.update_trajectory(sol)
        except InterpolationError as er:
            tracer.handle_error(er)
        return tracer

    def integrate_step(
    self,
        t_span: tuple[float, float],
    ) -> None:

        args = [(tracer, t_span, self.interpolator, self.kwargs) for tracer in self.tracers]
        self.tracers = do_parallel(
            func=self._integrate_inner,
            args=args,
            desc=f"Integrating t = {t_span[0]:.0f} - {t_span[1]:.0f}",
            unit='tracer',
            n_cpu=self.n_cpu,
            verbose=self.verbose,
        )

    @staticmethod
    def _interpolate_inner(
        args: tuple[Tracer, Interpolator, tuple[float, float]]
    ) -> Tracer:
        tracer, interpolator, t_span = args
        for tt, *pos in zip(*(tracer.trajectory[key]
                              for key in ("time", *interpolator.coord_keys))):
            if not (min(*t_span) <= tt <= max(*t_span)):
                continue

            try:
                data = interpolator.interpolate_data(tt, np.array(pos), interpolator.data)
                for key, value in zip(interpolator.data_keys, data):
                    tracer.trajectory[key] = np.append(tracer.trajectory[key], value)
            except InterpolationError as er:
                for key in tracer.trajectory.keys():
                    tracer.trajectory[key] = np.append(tracer.trajectory[key], np.nan)

        return tracer

    def interpolate_step(
        self,
        t_span: tuple[float, float],
    ):
        args = [(tracer, self.interpolator, t_span) for tracer in self.tracers]
        self.tracers = do_parallel(
            func=self._interpolate_inner,
            args=args,
            desc=f"Interpolating t = {t_span[0]:.0f} - {t_span[1]:.0f}",
            unit='tracer',
            n_cpu=self.n_cpu,
            verbose=self.verbose,
        )


    def check_end_conditions(self):

        for ii, end_condition in enumerate(self.end_conditions):
            #self.tracers = list(map(end_condition, self.tracers))
            self.tracers = do_parallel(
                func=end_condition,
                args=self.tracers,
                desc=f"Checking condition {ii+1}/{len(self.end_condition)}",
                unit='tracer',
                n_cpu=self.n_cpu,
                verbose=self.verbose,
            )

        if self.verbose:
            running = sum(tr.started and not tr.finished
                          for tr in self.tracers)
            finished = sum(tr.finished and not tr.failed
                           for tr in self.tracers)
            failed = sum(tr.failed for tr in self.tracers)
            print(f"running: {running}, finished: {finished}, failed: {failed}")


    def take_step(self, t_span: tuple[float, float]) -> None:
        if (self.last_time is not None and
            not (min(*t_span) <= self.last_time <= max(*t_span))):
            raise ValueError(
                f"Gap between last step {self.last_time} and current step {t_span}"
            )
        self.interpolator.load_data(t_span)
        self.integrate_step(t_span)
        self.interpolate_step(t_span)
        self.check_end_conditions()

        self.last_time = t_span[1]


    def __iter__(self):
        return self.tracers.__iter__()

    def  __len__(self):
        return len(self.tracers)

class InterpolationError(Exception):
    pass
