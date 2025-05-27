################################################################################
from typing import Iterable
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Pool
import os

import numpy as np
import h5py as h5
from numba import njit
from scipy.interpolate import RegularGridInterpolator, interp1d

from . import Tracers, Interpolator, do_parallel

################################################################################

def _rotjac(theta, phi):
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    return np.array([
        [st*cp, st*sp,  ct,], #r
        [ct*cp, ct*sp, -st,], #th
        [  -sp,    cp,   0,], #ph
    ])

_2pi = 2*np.pi
def _unravel(th: float, ph: float) -> tuple[float, float]:
    if th<0:
        th = -th
        ph += np.pi
    assert (th < _2pi)
    if th>np.pi:
      th = - th + _2pi
      ph += np.pi
    ph = ph%_2pi
    return th, ph

def _fill_with_ghosts(buf: np.ndarray, h5f: h5.File, key: str):
    buf[1:-1, 1:-1] = h5f[key][:]
    buf[1:-1,  0] = buf[1:-1, -2]
    buf[1:-1, -1] = buf[1:-1,  1]
    buf[ 0, 1:-1] = buf[ 1, -2:0:-1]
    buf[-1, 1:-1] = buf[-2, -2:0:-1]
    buf[ :,  0] = buf[:, -2]
    buf[ :, -1] = buf[:,  1]

class SphericalGridInterpolator(RegularGridInterpolator):
    def __call__(self, xi, method=None, *, nu=None):
        r, th, ph = xi.T
        th, ph = _unravel(th, ph)
        return super().__call__(np.array([r, th, ph]).T, method=method, nu=nu)

################################################################################

class SurfaceFile:
    def __init__(
        self,
        filename: str,
        keys: Iterable[str],
    ):
        self.filename = filename
        self.keys = tuple(keys)
        self.shared_memory = {}

        with h5.File(filename, 'r') as f:
            self.time = float(f['coordinates/00/T'][:])

            n_r = len(f['coordinates'].keys())
            r = np.array([float(f[f'coordinates/{ir:02d}/R'][:]) for ir in range(n_r)])
            th =  np.array(f['coordinates/00/th'][:])
            ph =  np.array(f['coordinates/00/ph'][:])
            th = np.concatenate(([-th[0]], th, [th[0]+np.pi]))
            ph = np.concatenate(([-ph[0]], ph, [ph[0]+2*np.pi]))
            self.grid = (r, th, ph)
            n_th = len(th)
            n_ph = len(ph)
            self.shape = (n_r, n_th, n_ph)
            mem_size = 8*int(np.prod(self.shape))

            for key in self.keys:
                grp = key.split('.')[0]
                if key in self.shared_memory:
                    continue
                shm = SharedMemory(create=True, size=mem_size)
                self.shared_memory[key] = shm.name
                #print(f'Copying {key} to {shm.name}')
                data: np.ndarray = np.ndarray(self.shape, dtype=float, buffer=shm.buf)
                for ir in range(n_r):
                    _fill_with_ghosts(data[ir], f, f'fields/{ir:02d}/{grp}/{key}')

    def interpolate(self, x: np.ndarray, keys: tuple[str, ...]) -> np.ndarray:
        assert self.grid[0][0] < x[0] and self.grid[0][-1] > x[-1], \
          f"r({x[0]}) out of bounds ({self.grid[0][0]}:{self.grid[0][-1]})"

        res = np.empty(len(keys))
        for ii, key in enumerate(keys):
            #print(f'Reading {key} from {self.shared_memory[key]}')
            shm = SharedMemory(name=self.shared_memory[key])
            ar: np.ndarray = np.ndarray(self.shape, dtype=float, buffer=shm.buf)
            try:
                res[ii] = SphericalGridInterpolator(self.grid, ar)(x, method='linear')
            except ValueError:
                print(f"Error: r_grid:({self.grid[0][0]}-{self.grid[0][-1]}), r={x[0]}")
                res[ii] = 0
        return res

    def free_shared_memory(self):
        for key, name in self.shared_memory.items():
            try:
                shm = SharedMemory(name=name)
            except FileNotFoundError:
                print(f"Could not find shared memory for {key}")
                continue
            shm.close()
            shm.unlink()
        self.shared_memory = {}

    def __repr__(self):
        return f"SurfaceFile({self.filename})"

    def __str__(self):
        return self.filename

################################################################################

class SurfaceInterpolator(Interpolator):
    coord_keys = ('r', 'th', 'ph')
    vel_keys = ("tracer.hydro.aux.V_u_x",
                "tracer.hydro.aux.V_u_y",
                "tracer.hydro.aux.V_u_z")
    data: tuple

    def __init__(
        self,
        path: str,
        data_keys: Iterable[str],
        t_int_order: str = 'linear',
        n_cpu: int = 1,
        verbose: bool = False,
        every: int = 1,
    ):

        self.path = path
        self.data_keys = tuple(data_keys)
        self.n_cpu = n_cpu
        self.verbose = verbose
        implemented = ('linear', 'cubic')
        if t_int_order not in implemented:
            raise ValueError(f"Time interpolation order {t_int_order} not implemented.")
        self.t_int_order = t_int_order


        files: list[str] = []
        times: list[float] = []

        for ff in os.scandir(path):
            if not (ff.is_file() and 'surface' in ff.name):
                continue

            with h5.File(ff.path, 'r') as f:
                files.append(ff.path)
                times.append(float(f['coordinates/00/T'][:]))

        self.files = np.array(files)
        self.times = np.array(times)

        if len(self.files) == 0:
            raise ValueError(f'No files found in {path}')

        isort = np.argsort(self.times)
        self.files = self.files[isort]
        self.times = self.times[isort]
        self.times = self.times[::every]
        self.files = self.files[::every]

        self.surfaces: list[SurfaceFile] = list()

        self.data = (self.surfaces, self.t_int_order, self.data_keys)

    @staticmethod
    def interpolate(
        time: float,
        coords: np.ndarray,
        data: tuple,
        mode: str,
    ) -> np.ndarray:

        files, order, keys  = data
        if mode == 'velocity':
            keys = SurfaceInterpolator.vel_keys

        times = np.array([f.time for f in files])
        i_t = np.searchsorted(times, time, side='right')

        if order == 'linear':
            il = i_t -1
            ir = i_t + 1
        elif order == 'cubic':
            il = i_t - 2
            ir = i_t + 2
        else:
            raise ValueError(f"Time interpolation order {order} not implemented.")

        if ir == len(files)+1:
            ir -= 1
            il -= 1
        if il == -1:
            il += 1
            ir += 1

        res = [f.interpolate(coords, keys=keys) for f in files[il:ir]]

        if mode == 'velocity':
            jac = _rotjac(*coords[1:])
            res = [jac@vel/coords[0] for vel in res]

        if order == 'linear':
            return res[0] + (time - times[il])*(res[1] - res[0])/(times[il+1] - times[il])
        else:
            return interp1d(times[il:ir], res, axis=0, kind=order)(time)

    @staticmethod
    def interpolate_velocities(
        time: float,
        coords: np.ndarray,
        data: tuple,
    ) -> np.ndarray:
        return SurfaceInterpolator.interpolate(time, coords, data, 'velocity')

    @staticmethod
    def interpolate_data(
        time: float,
        coords: np.ndarray,
        data: tuple,
    ) -> np.ndarray:
        return SurfaceInterpolator.interpolate(time, coords, data, 'data')

    def load_file(self, ff):
        return SurfaceFile(ff, self.vel_keys+self.data_keys)

    def load_data(
        self,
        t_span: tuple[float, float],
    ):

        self.free_shared_memory()

        i_s = self.times.searchsorted(min(t_span), side='right') - 1
        i_e = self.times.searchsorted(max(t_span), side='right')

        if self.t_int_order == 'linear':
            pass
        elif self.t_int_order == 'cubic':
            i_s -= 1
            i_e += 1
            if i_s < 0:
                i_s = 0

        files = self.files[i_s:i_e]

        self.surfaces[:] = do_parallel(
            self.load_file,
            files,
            n_cpu=self.n_cpu,
            desc=f"Loading files for t = {t_span[0]} - {t_span[1]}",
            unit="file",
            verbose=self.verbose,
        )
        self.surfaces.sort(key=lambda f: f.time)

    def free_shared_memory(self):
        for surface in self.surfaces:
            surface.free_shared_memory()

################################################################################

class SurfaceTracers(Tracers):
    interpolator: SurfaceInterpolator
    vel_keys: tuple[str, ...] = ('velr', 'velth', 'velph')

    def __init__(
        self,
        path: str,
        data_keys: Iterable[str],
        n_cpu: int = 1,
        reverse: bool = True,
        files_per_step: int = 20,
        t_int_order: str = 'linear',
        every: int = 1,
        verbose: bool = False,
        **kwargs,
    ):

        self.path = path
        self.data_keys = tuple(data_keys)
        interpolator = SurfaceInterpolator(
            self.path,
            data_keys=data_keys,
            t_int_order=t_int_order,
            n_cpu=n_cpu,
            verbose=verbose,
            every=every,
        )


        self.files_per_step = files_per_step
        if t_int_order == 'linear':
            self.step = files_per_step - 1
        elif t_int_order == 'cubic':
            self.step = files_per_step - 2
        else:
            raise ValueError(f"Time interpolation order {t_int_order} not implemented.")

        super().__init__(
            interpolator=interpolator,
            n_cpu=n_cpu,
            verbose=verbose,
            **kwargs,
        )
        if reverse:
            self.times = self.interpolator.times[::-1]
            max_t = self.seeds['time'].max()
            max_t = self.times[self.times > max_t][-1]
            self.times = self.times[self.times <= max_t]
        else:
            self.times = self.interpolator.times
            min_time = self.seeds['time'].min()
            min_time = self.times[self.times < min_time][-1]
            self.times = self.times[self.times >= min_time]


    def integrate(self):
        start_times = self.times[:-self.step:self.step]
        start_times = np.append(start_times, self.times[-self.step])
        end_times = self.times[self.step::self.step]
        end_times = np.append(end_times, self.times[-1])

        try:
            for t_span in zip(start_times, end_times):
                self.take_step(t_span)
                if all(tr.finished for tr in self.tracers):
                    break
        finally:
            self.interpolator.free_shared_memory()

    def __del__(self):
        if hasattr(self, "interpolator"):
            self.interpolator.free_shared_memory()
