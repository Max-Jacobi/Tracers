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

class AthdfFile:
    def __init__(
        self,
        filename: str,
        keys: Iterable[str],
        coordinates: str = "",
    ):

        self.filename = filename
        self.keys = tuple(keys)
        self.coordinates = coordinates
        self.shared_memory = {}

        with h5.File(filename, 'r') as f:
            self.time = float(f.attrs['Time'][()])

            vars_in_file = list(f.attrs['VariableNames'][:].astype(str))
            nmb: int = int(f.attrs["NumMeshBlocks"])
            mb_size: np.ndarray = f.attrs['MeshBlockSize'][:].astype(int)
            mem_size = 8*nmb*int(np.prod(mb_size))

            # assume all grid functions are in the first dataset
            dset = f.attrs['DatasetNames'][0].decode('utf-8')
            self.full_shape = (nmb, *mb_size)
            self.shape = tuple(mb_size)

            self.x1 = np.array(f['x1v'][:])
            self.x2 = np.array(f['x2v'][:])
            self.x3 = np.array(f['x3v'][:])
            o1 = self.x1[:, 1]
            o2 = self.x2[:, 1]
            o3 = self.x3[:, 1]
            e1 = self.x1[:, -2]
            e2 = self.x2[:, -2]
            e3 = self.x3[:, -2]

            self.origins = np.array([o3, o2, o1])
            self.ends = np.array([e3, e2, e1])


            for key in self.keys:
                if key in self.shared_memory:
                    continue
                shm = SharedMemory(create=True, size=mem_size)
                self.shared_memory[key] = shm.name
                data: np.ndarray = np.ndarray(self.full_shape, dtype=float, buffer=shm.buf)
                j = vars_in_file.index(key)
                data[:] = f[dset][j]

    def get_mb_data(
        self,
        x: np.ndarray
        ) -> int:
        """
        Get the mesh block data for a given point.
        Returns origin, inverse cell widths, and mesh block index.
        """
        mask = np.all(
                (x[:, None] >= self.origins) &
                (x[:, None] <= self.ends),
                axis=0)
        if sum(mask) == 0:
            return -1
        idx = np.where(mask)[0][0]
        return idx

    def interpolate(self, x: np.ndarray, key: str) -> float:
        if 'bitant' in self.coordinates:
            if (mirror_z := x[0] < 0):
                x[0] = -x[0]

        if "spherical" in self.coordinates:
            # convert to coordinates coordinates
            r = np.linalg.norm(x)
            theta = np.arccos(x[0]/r)
            phi = np.arctan2(x[1], x[2])
            if phi < 0: phi = phi + np.pi*2
            x = np.array([phi, theta, r])

        imb = self.get_mb_data(x)
        if imb == -1:
            return np.zeros_like(x)

        xx = (self.x3[imb], self.x2[imb], self.x1[imb])
        assert self.x3[imb][0] < x[0] < self.x3[imb][-1], f"z({x[0]}) out of bounds({self.x3[imb][0]}:{self.x3[imb][-1]})"
        assert self.x2[imb][0] < x[1] < self.x2[imb][-1], f"z({x[1]}) out of bounds({self.x2[imb][0]}:{self.x2[imb][-1]})"
        assert self.x1[imb][0] < x[2] < self.x1[imb][-1], f"z({x[2]}) out of bounds({self.x1[imb][0]}:{self.x1[imb][-1]})"

        res = np.empty(len(keys))
        for ii, key in enumerate(keys):
            shm = SharedMemory(name=self.shared_memory[key])
            ar: np.ndarray = np.ndarray(self.full_shape, dtype=float, buffer=shm.buf)
            res[ii] = RegularGridInterpolator(xx, ar[imb])(x, method='linear')
        if 'spherical' in self.coordinates and all(key.startswith('vel') for key in keys):
            # convert back to cartesian
            jac = _rotjac(theta, phi)
            res = jac @ res
        if 'bitant' in self.coordinates:
            if mirror_z:
                res[0] = -res[0]
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
        return f"AthdfFile({self.filename})"

    def __str__(self):
        return self.filename

################################################################################

class AthdfInterpolator(Interpolator):
    coord_keys = ('x3', 'x2', 'x1')
    vel_keys = ('vel3', 'vel2', 'vel1')
    data: tuple

    def __init__(
        self,
        path: str,
        data_keys: Iterable[str],
        t_int_order: str = 'linear',
        n_cpus: int = 1,
        verbose: bool = False,
        every: int = 1,
        coordinates: str = "",
    ):

        self.path = path
        self.data_keys = tuple(data_keys)
        self.n_cpus = n_cpus
        self.verbose = verbose
        self.coordinates = coordinates
        implemented = ('linear', 'cubic')
        if t_int_order not in implemented:
            raise ValueError(f"Time interpolation order {t_int_order} not implemented.")
        self.t_int_order = t_int_order

        all_keys = self.vel_keys+self.data_keys
        files: dict[str, list[str]] = {key: [] for key in all_keys}
        times: dict[str, list[float]] = {key: [] for key in all_keys}

        for ff in os.scandir(path):
            if not (ff.is_file() and ff.name.endswith('.athdf')):
                continue

            with h5.File(ff.path, 'r') as f:
                file_keys = f.attrs['VariableNames'][:].astype(str)
                for key in all_keys:
                    if key not in file_keys: continue
                    files[key].append(ff.path)
                    times[key].append(float(f.attrs['Time'][()]))

        self.files = {k: np.array(v) for k, v in files.items()}
        self.times = {k: np.array(v) for k, v in times.items()}

        for key in all_keys:
            if key not in self.files:
                raise ValueError(f'{key} not found in any files in {path}')

            isort = np.argsort(self.times[key])
            self.files[key] = self.files[key][isort]
            self.times[key] = self.times[key][isort]
            self.times[key] = self.times[key][::every]
            self.files[key] = self.files[key][::every]

        self.athdfs: dict[str, list[AthdfFile]] = {}
        self.allocated_athdfs: list[AthdfFile] = []
        self.data = (self.athdfs, self.t_int_order, self.data_keys)

    @staticmethod
    def interpolate(
        time: float,
        coords: np.ndarray,
        data: tuple,
        mode: str,
    ) -> np.ndarray:

        file_dict, order, keys  = data
        if mode == 'velocity':
            keys = AthdfInterpolator.vel_keys

        ret = np.zeros(len(keys))
        for ik, key in enumerate(keys):
            files = file_dict[key]
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
                ret[ik] = res[0] + (time - times[il])*(res[1] - res[0])/(times[il+1] - times[il])
            else:
                ret[ik] = interp1d(times[il:ir], res, axis=0, kind=order)(time)
        return ret

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

        files_keys = {}
        for key in self.vel_keys+self.data_keys:
            i_s = self.times[key].searchsorted(min(t_span), side='right') - 1
            i_e = self.times[key].searchsorted(max(t_span), side='right')
            if self.t_int_order == 'cubic':
                i_s -= 1
                i_e += 1
                i_s = max(i_s, 0)

            for ff in self.files[key][i_s:i_e]:
                if ff not in files_keys:
                    files_keys[ff] = []
                files_keys[ff].append(key)

        def load_file(fk):
            ff, keys = fk
            return keys, AthdfFile(ff, keys, self.coordinates)

        keys_file = do_parallel(
            load_file,
            files_keys.items(),
            n_cpu=self.n_cpus,
            desc=f"Loading files for t = {t_span[0]} - {t_span[1]}",
            unit="file",
            verbose=self.verbose,
        )

        for keys, athdf in keys_file:
            self.allocated_athdfs.append(athdf)
            for key in keys:
                if key not in self.athdfs:
                    self.athdfs[key] = []
                self.athdfs[key].append(athdf)
        for key in self.athdfs:
            self.athdfs[key].sort(key=lambda f: f.time)

    def free_shared_memory(self):
        for athdf in self.allocated_athdfs:
            athdf.free_shared_memory()
        self.allocated_athdfs = []
        self.athdfs = {}

################################################################################

class AthdfTracers(Tracers):
    interpolator: AthdfInterpolator
    vel_keys: tuple[str, ...] = ('vel3', 'vel2', 'vel1')

    def __init__(
        self,
        path: str,
        data_keys: Iterable[str],
        n_cpu: int = 1,
        reverse: bool = True,
        files_per_step: int = 20,
        t_int_order: str = 'linear',
        coordinates: str = "",
        every: int = 1,
        verbose: bool = False,
        **kwargs,
    ):

        self.path = path
        self.data_keys = tuple(data_keys)
        self.coordinates = coordinates
        interpolator = AthdfInterpolator(
            self.path,
            data_keys=data_keys,
            t_int_order=t_int_order,
            every=every,
            coordinates=coordinates,
            verbose=verbose,
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


def _rotjac(theta, phi):
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    return np.array([
        [0, -st, ct],
        [cp, ct*sp, st*sp],
        [-sp, ct*cp, st*cp],
    ])
