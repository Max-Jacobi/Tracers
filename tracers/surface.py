from typing import Iterable
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Pool
import os

import numpy as np
import h5py as h5
from numba import njit
from scipy.interpolate import RegularGridInterpolator, interp1d

from . import Tracers, Interpolator, do_parallel


def _fill_with_ghosts(buf: np.ndarray, h5f: h5.File, key: str, ng: int = 1):
    nth, nphi = h5f[key].shape
    half = nphi // 2
    buf[ng:-ng, ng:-ng] = h5f[key][:]
    for ig in range(ng):
        ign = -ig-1
        buf[ ig, ng:-ng] = np.roll(buf[ ng+ ig, ng:-ng], half)
        buf[ign, ng:-ng] = np.roll(buf[-ng+ign, ng:-ng], half)
        buf[:,  ig] = buf[:, -ng+ign]
        buf[:, ign] = buf[:,  ng+ig ]

_2pi = 2*np.pi

class SphericalGridInterpolator(RegularGridInterpolator):
    def __call__(self, xi, method=None, *, nu=None):
        x, y, z = xi
        r = np.sqrt(x*x + y*y + z*z)
        th = np.arccos(z/r)
        ph = (np.arctan2(y, x)+_2pi)%_2pi
        return super().__call__(np.array([r, th, ph]).T, method=method, nu=nu)

class SurfaceFile:
    def __init__(
        self,
        filename: str,
        shm_names: dict[str, str],
        grid: tuple[np.ndarray, np.ndarray, np.ndarray],
        shape: tuple[int, int, int],
    ):
        self.filename = filename
        self.keys = tuple(shm_names.keys())
        self.shared_memory = shm_names
        self.grid = grid
        self.shape = shape

        with h5.File(filename, 'r') as f:
            self.time = float(f['coordinates/00/T'][:])
            for key in self.keys:
                # print(f'Copying {key} to {self.shared_memory[key]}', flush=True)
                shm = SharedMemory(name=self.shared_memory[key])
                grp = key.split('.')[0]
                data: np.ndarray = np.ndarray(self.shape, dtype=float, buffer=shm.buf)
                for ir in range(self.shape[0]):
                    _fill_with_ghosts(data[ir], f, f'fields/{ir:02d}/{grp}/{key}')

    def interpolate(self, x: np.ndarray, keys: tuple[str, ...]) -> np.ndarray:
        r = np.sqrt(np.sum(x*x))
        if self.grid[0][0] > r or self.grid[0][-1] < r:
            return np.zeros(len(keys))

        res = np.empty(len(keys))
        for ii, key in enumerate(keys):
            # print(f'Reading {key} from {self.shared_memory[key]}', flush=True)
            shm = SharedMemory(name=self.shared_memory[key])
            ar: np.ndarray = np.ndarray(self.shape, dtype=float, buffer=shm.buf)
            res[ii] = SphericalGridInterpolator(self.grid, ar)(x, method='linear')
        return res

    def __repr__(self):
        return f"SurfaceFile({self.filename})"

    def __str__(self):
        return self.filename

class SurfaceInterpolator(Interpolator):
    coord_keys = ('x', 'y', 'z')
    vel_keys = ("tracer.hydro.aux.V_u_x",
                "tracer.hydro.aux.V_u_y",
                "tracer.hydro.aux.V_u_z")
    data: tuple
    grid: tuple[np.ndarray, np.ndarray, np.ndarray]
    shape: tuple[int, int, int]

    def __init__(
        self,
        path: str,
        data_keys: Iterable[str],
        t_int_order: str = 'linear',
        n_cpu: int = 1,
        files_per_step: int = 2,
        verbose: bool = False,
        every: int = 1,
    ):

        self.path = path
        self.data_keys = tuple(data_keys)
        self.n_cpu = n_cpu
        self.files_per_step = files_per_step
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
        if every > 1:
            last_t = self.times[-1]
            last_f = self.files[-1]
            self.times = self.times[::every]
            self.files = self.files[::every]
            if self.times[-1] != last_t:
                self.times = np.append(self.times, last_t)
                self.files = np.append(self.files, last_f)

        self.surfaces: list[SurfaceFile] = list()

        self.data = (self.surfaces, self.t_int_order, self.data_keys)

        self.allocate_shm()

    def allocate_shm(self, ng: int = 1):
        filename = self.files[0]
        with h5.File(filename, 'r') as f:
            n_r = len(f['coordinates'].keys())
            r = np.array([float(f[f'coordinates/{ir:02d}/R'][:]) for ir in range(n_r)])
            th =  np.array(f['coordinates/00/th'][:])
            ph =  np.array(f['coordinates/00/ph'][:])
            dth = th[0]*2
            dph = ph[0]*2
            th = np.concatenate((
                -dth*np.arange(ng)-dth/2,
                th,
                dth*np.arange(ng)+dth/2+2*np.pi,
                ))
            ph = np.concatenate(([-ph[0]], ph, [ph[0]+2*np.pi]))
            self.grid = (r, th, ph)
            n_th = len(th)
            n_ph = len(ph)
            if n_ph%2 != 0:
                raise ValueError(f"Files have unqual number of points in phi direction (n_phi={n_ph})!")
            self.shape = (n_r, n_th, n_ph)
            mem_size = 8*int(np.prod(self.shape))
            keys = np.unique(self.vel_keys+self.data_keys)
        print(f"Allocating shared memory for {self.files_per_step} files and {len(keys)} grid functions "
              f"using {len(keys)*self.files_per_step*mem_size/1024**3:.2f}GB of memory.", flush=True)
        self.shared_memory =  [{key: SharedMemory(create=True, size=mem_size).name
                                for key in keys} for _ in range(self.files_per_step)]

    def load_file(self, args: tuple[str, dict[str, str]]):
        return SurfaceFile(*args, self.grid, self.shape)

    def load_data(
        self,
        t_span: tuple[float, float],
    ):
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
            list(zip(files, self.shared_memory)),
            n_cpu=self.n_cpu,
            desc=f"Loading files for t = {t_span[0]} - {t_span[1]}",
            unit="file",
            verbose=self.verbose,
        )
        self.surfaces.sort(key=lambda f: f.time)

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
        vel =  SurfaceInterpolator.interpolate(time, coords, data, 'velocity')
        return vel

    @staticmethod
    def interpolate_data(
        time: float,
        coords: np.ndarray,
        data: tuple,
    ) -> np.ndarray:
        return SurfaceInterpolator.interpolate(time, coords, data, 'data')

    def free_shared_memory(self):
        for file_shm in self.shared_memory:
            for key, name in file_shm.items():
                try:
                    shm = SharedMemory(name=name)
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    # print(f"Could not find shared memory for {key}")
                    ...
        self.shared_memory = []

class SurfaceTracers(Tracers):
    interpolator: SurfaceInterpolator
    vel_keys: tuple[str, ...] = ('velr', 'velth', 'velph')

    def __init__(
        self,
        path: str,
        data_keys: Iterable[str],
        n_cpu: int = 1,
        reverse: bool = True,
        files_per_step: int = 2,
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
            files_per_step = files_per_step,
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
            max_t = self.times[self.times >= max_t][-1]
            self.times = self.times[self.times <= max_t]
        else:
            self.times = self.interpolator.times
            min_time = self.seeds['time'].min()
            min_time = self.times[self.times <= min_time][-1]
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
