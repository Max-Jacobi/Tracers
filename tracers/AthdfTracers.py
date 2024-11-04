################################################################################
from typing import Iterable
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Pool
import os

import numpy as np
import h5py as h5
from numba import njit
from scipy.interpolate import RegularGridInterpolator, interp1d

from .import Tracers, Interpolator
from .Tracers import InterpolationError
from .utils import do_parallel

################################################################################

class AthdfFile:
    def __init__(
        self,
        filename: str,
        keys: Iterable[str],
    ):

        self.filename = filename
        self.keys = tuple(keys)
        self.shared_memory = {}

        with h5.File(filename, 'r') as f:
            self.time = float(f.attrs['Time'][()])

            vars_in_file = list(f.attrs['VariableNames'][:].astype(str))
            nmb: int = int(f.attrs["NumMeshBlocks"])
            mb_size: np.ndarray = f.attrs['MeshBlockSize'][:].astype(int)
            mb_size = mb_size[:2]
            mem_size = 8*nmb*int(np.prod(mb_size))

            # assume all grid functions are in the first dataset
            dset = f.attrs['DatasetNames'][0].decode('utf-8')
            self.full_shape = (nmb, *mb_size)
            self.shape = tuple(mb_size)

            for key in self.keys:
                if key in self.shared_memory:
                    continue
                shm = SharedMemory(create=True, size=mem_size)
                self.shared_memory[key] = shm.name
                data: np.ndarray = np.ndarray(self.full_shape, dtype=float, buffer=shm.buf)
                j = vars_in_file.index(key)
                data[:] = f[dset][j, :, 0]
                # data[:] = f[dset][j]

            self.x1 = np.array(f['x1v'][:])
            self.x2 = np.array(f['x2v'][:])
            # x3 = f['x2v'][:]
            o1 = self.x1[:, 0]
            o2 = self.x2[:, 0]
            # o3 = x3[:, 0]
            d1 = self.x1[:, 1] - o1
            d2 = self.x2[:, 1] - o2
            # d3 = x3[:, 1] - o3
            # self.origins = np.array([o3, o2, o1]).T
            # self.ih = 1/np.array([d3, d2, d1]).T
            self.origins = np.array([o2, o1]).T
            self.ih = 1/np.array([d2, d1]).T

    def get_mb_data(
        self,
        x: np.ndarray
        ) -> int:
        """
        Get the mesh block data for a given point.
        Returns origin, inverse cell widths, and mesh block index.
        """
        diff = (x[None, :] - self.origins)*self.ih
        mask = np.all(diff >= 0, axis=1)
        mask = mask & np.all(diff < np.array(self.shape)[None, :] - 1, axis=1)
        if sum(mask) == 0:
            raise InterpolationError(f"Point {x} is out of bounds.")
        idx = np.where(mask)[0][0]
        # return self.origins[idx], self.ih[idx], idx
        return idx

    def interpolate(self, x: np.ndarray, keys: tuple[str, ...]) -> np.ndarray:
        # o, ih, imb = self.get_mb_data(x)
        imb = self.get_mb_data(x)
        xx = (self.x2[imb], self.x1[imb])
        # data = np.empty((len(keys), *self.shape))
        res = np.empty(len(keys))
        for ii, key in enumerate(keys):
            shm = SharedMemory(name=self.shared_memory[key])
            ar: np.ndarray = np.ndarray(self.full_shape, dtype=float, buffer=shm.buf)
            res[ii] = RegularGridInterpolator(xx, ar[imb])(x, method='linear')
        return res

    def free_shared_memory(self):
        for key, name in self.shared_memory.items():
            try:
                shm = SharedMemory(name=name)
            except FileNotFoundError:
                print(f"Could not find shared memory for {key} in section {ii}")
                continue
            shm.close()
            shm.unlink()
        self.shared_memory = {}


    def __repr__(self):
        return f"AthdfFile({self.filename})"

    def __str__(self):
        return self.filename

################################################################################

class AthdfInterpolator(Interpolator):
    # coord_keys = ('x3', 'x2', 'x1')
    # vel_keys = ('vel3', 'vel2', 'vel1')
    coord_keys = ('x2', 'x1')
    vel_keys = ('vel2', 'vel1')
    data: tuple

    def __init__(
        self,
        path: str,
        data_keys: Iterable[str],
        t_int_order: str = 'linear',
        n_cpus: int = 1,
        verbose: bool = False,
        every: int = 1,
    ):

        self.path = path
        self.data_keys = tuple(data_keys)
        self.n_cpus = n_cpus
        self.verbose = verbose
        implemented = ('linear', 'cubic')
        if t_int_order not in implemented:
            raise ValueError(f"Time interpolation order {t_int_order} not implemented.")
        self.t_int_order = t_int_order


        files: list[str] = []
        times: list[float] = []

        for ff in os.scandir(path):
            if not (ff.is_file() and ff.name.endswith('.athdf')):
                continue
            with h5.File(ff.path, 'r') as f:
                if not all(key in f.attrs['VariableNames'][:].astype(str)
                           for key in self.vel_keys+self.data_keys):
                    continue
                files.append(ff.path)
                times.append(float(f.attrs['Time'][()]))

        if len(files) == 0:
            raise ValueError(f'No files containing all keys found in {path}')

        self.files = np.array(files)
        self.times = np.array(times)
        isort = np.argsort(self.times)
        self.files = self.files[isort]
        self.times = self.times[isort]
        self.times = self.times[::every]
        self.files = self.files[::every]


        self.athdfs: list[AthdfFile] = list()

        self.data = (self.athdfs, self.t_int_order, self.data_keys)

    @staticmethod
    def interpolate(
        time: float,
        coords: np.ndarray,
        data: tuple,
        mode: str,
    ) -> np.ndarray:

        files, order, keys  = data
        if mode == 'velocity':
            keys = AthdfInterpolator.vel_keys

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
        return AthdfInterpolator.interpolate(time, coords, data, 'velocity')

    @staticmethod
    def interpolate_data(
        time: float,
        coords: np.ndarray,
        data: tuple,
    ) -> np.ndarray:
        return AthdfInterpolator.interpolate(time, coords, data, 'data')

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

        # def load_file(ff):
        #     return AthdfFile(ff, self.vel_keys+self.data_keys)

        # self.athdfs = do_parallel(
        #     load_file,
        #     files,
        #     n_cpu=self.n_cpus,
        #     message=f"Loading files for t = {t_span[0]} - {t_span[1]}",
        #     verbose=self.verbose,
        # )
        self.athdfs[:] = [AthdfFile(ff, self.vel_keys+self.data_keys) for ff in files]
        self.athdfs.sort(key=lambda f: f.time)

    def free_shared_memory(self):
        for athdf in self.athdfs:
            athdf.free_shared_memory()

################################################################################

class AthdfTracers(Tracers):
    interpolator: AthdfInterpolator
    # vel_keys: tuple[str, ...] = ('vel3', 'vel2', 'vel1')
    vel_keys: tuple[str, ...] = ('vel2', 'vel1')

    def __init__(
        self,
        path: str,
        data_keys: Iterable[str],
        n_cpu: int = 1,
        reverse: bool = True,
        files_per_step: int = 20,
        t_int_order: str = 'linear',
        every: int = 1,
        **kwargs,
    ):

        self.path = path
        self.data_keys = tuple(data_keys)
        interpolator = AthdfInterpolator(
            self.path,
            data_keys=data_keys,
            t_int_order=t_int_order,
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
        self.interpolator.free_shared_memory()
