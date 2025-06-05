################################################################################
from multiprocessing import shared_memory
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
        [0, -st, ct],
        [cp, ct*sp, st*sp],
        [-sp, ct*cp, st*cp],
    ])

class AthdfFile:
    def __init__(
        self,
        filename: str,
        shm_names: dict[str, str],
        bitant: bool = False,
        spherical: bool = False,
    ):

        self.filename = filename
        self.shared_memory = shm_names
        self.keys = tuple(self.shared_memory.keys())
        self.bitant = bitant
        self.spherical = spherical

        with h5.File(filename, 'r') as f:
            self.time = float(f.attrs['Time'][()])

            vars_in_file = list(f.attrs['VariableNames'][:].astype(str))
            nmb: int = int(f.attrs["NumMeshBlocks"])
            mb_size: np.ndarray = f.attrs['MeshBlockSize'][:].astype(int)

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
                shm = SharedMemory(name=self.shared_memory[key])
                data: np.ndarray = np.ndarray(self.full_shape, dtype=float, buffer=shm.buf)
                j = vars_in_file.index(key)
                data[:] = f[dset][j]

    def get_mb_data(
        self,
        x: np.ndarray
        ) -> int:
        """
        Get the mesh block data for a given point.
        Returns index of meshblock
        """
        mask = np.all(
                (x[:, None] >= self.origins) &
                (x[:, None] <= self.ends),
                axis=0)
        if sum(mask) == 0:
            return -1
        idx = np.where(mask)[0][0]
        return idx

    def interpolate(self, x: np.ndarray, keys: tuple[str, ...]) -> np.ndarray:
        if (mirror_z := (x[0] < 0) and self.bitant):
            x[0] = -x[0]
        if self.spherical:
            r = np.linalg.norm(x)
            theta = np.arccos(x[0]/r)
            phi = np.arctan2(x[1], x[2])
            if phi < 0: phi = phi + np.pi*2
            x = np.array([phi, theta, r])

        imb = self.get_mb_data(x)
        if imb == -1:
            return np.zeros_like(x)

        xx = (self.x3[imb], self.x2[imb], self.x1[imb])
        assert self.x3[imb][0] < x[0] < self.x3[imb][-1], \
          f"z({x[0]}) out of bounds({self.x3[imb][0]}:{self.x3[imb][-1]})"
        assert self.x2[imb][0] < x[1] < self.x2[imb][-1], \
          f"z({x[1]}) out of bounds({self.x2[imb][0]}:{self.x2[imb][-1]})"
        assert self.x1[imb][0] < x[2] < self.x1[imb][-1], \
          f"z({x[2]}) out of bounds({self.x1[imb][0]}:{self.x1[imb][-1]})"

        res = np.empty(len(keys))
        for ii, key in enumerate(keys):
            shm = SharedMemory(name=self.shared_memory[key])
            ar: np.ndarray = np.ndarray(self.full_shape, dtype=float, buffer=shm.buf)
            res[ii] = RegularGridInterpolator(xx, ar[imb])(x, method='linear')

        if self.spherical:
            jac = _rotjac(theta, phi)
            res = jac @ res
        if mirror_z:
            res[0] = -res[0]
        return res

    def __repr__(self):
        return f"AthdfFile({self.filename})"

    def __str__(self):
        return self.filename

class AthdfInterpolator(Interpolator):
    coord_keys = ('x3', 'x2', 'x1')
    vel_keys = ('vel3', 'vel2', 'vel1')

    def parse_files(self, path: str, every: int = 1) -> tuple[np.ndarray, np.ndarray, int]:
        '''
        returns an array of filnames and an array of the corresponding times
        and the maximum memory size in bytes that one gridfunction needs
        '''

        _files: list[str] = []
        _times: list[float] = []

        mem_size = 0
        for ff in os.scandir(path):
            if not (ff.is_file() and ff.name.endswith('.athdf')):
                continue

            with h5.File(ff.path, 'r') as f:
                if not all(key in f.attrs['VariableNames'][:].astype(str)
                           for key in self.vel_keys+self.data_keys):
                    continue
                _files.append(ff.path)
                _times.append(float(f.attrs['Time'][()]))
                nmb: int = int(f.attrs["NumMeshBlocks"])
                mb_size: np.ndarray = f.attrs['MeshBlockSize'][:].astype(int)
                mem_size = max(mem_size, 8*nmb*int(np.prod(mb_size)))

        files = np.array(_files)
        times = np.array(_times)

        isort = np.argsort(times)
        files = files[isort]
        times = times[isort]
        if every > 1:
            last_t =times[-1]
            last_f =files[-1]
            times =times[::every]
            files =files[::every]
            if times[-1] != last_t:
               times = np.append(_times, last_t)
               files = np.append(_files, last_f)

        return files,times, mem_size

    def __init__(self, bitant: bool = False, spherical: bool = False, *args, **kwargs):
        self.bitant = bitant
        self.spherical = spherical
        super.__init__(*args, **kwargs)

    def load_file(self, args: tuple[str, dict[str, str]]) -> AthdfFile:
        return AthdfFile(*args, bitant=self.bitant, spherical=self.spherical)

class AthdfTracers(Tracers):
    interpolator_class = AthdfInterpolator
