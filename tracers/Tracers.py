from typing import Callable, Iterable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from .utils import do_parallel
from .Interpolator import Interpolator
from .Tracer import Tracer

class Tracers:

    def __init__(
        self,
        seeds: dict[str, np.ndarray],
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
        self.tracers =  [Tracer(pos, t, ii, self.coord_keys, self.data_keys)
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

class InterpolationError(Exception):
    pass
