################################################################################
import os
from multiprocessing.shared_memory import SharedMemory
from itertools import repeat
from functools import reduce

import numpy as np
import h5py as h5
from scipy.interpolate import RegularGridInterpolator

from .from_file import File, FileInterpolator, FileTracers
from .utils import do_parallel

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

class AthdfFile(File):
    coord_keys = ('x3', 'x2', 'x1')
    vel_keys = ('vel3', 'vel2', 'vel1')
    kwargs = ("bitant", "spherical", "gr", "mass")

    def __init__(
        self,
        filenames: dict[str, tuple[str, ...]],
        shm_names: dict[str, str],
        bitant: bool = False,
        spherical: bool = False,
        gr: bool = False,
        mass: float = 0,
    ):

        self.filenames = filenames
        self.shared_memory = shm_names
        self.bitant = bitant
        self.spherical = spherical

        for filename, keys in self.filenames.items():
            with h5.File(filename, 'r') as f:
                self.time = float(f.attrs['Time'][()])

                vars_in_file = list(f.attrs['VariableNames'][:].astype(str))
                nmb: int = int(f.attrs["NumMeshBlocks"])
                mb_size: np.ndarray = f.attrs['MeshBlockSize'][:].astype(int)

                # assume all grid functions are in the first dataset
                dset = f.attrs['DatasetNames'][0].decode('utf-8')
                self.full_shape = (nmb, *mb_size)
                self.shape = tuple(mb_size)

                self.grid = np.array([f[f'{key}v'][:] for key in self.coord_keys])
                self.grid = self.grid.transpose((1, 0, 2))
                # check for mirrored theta ghost zones at the poles
                if self.spherical:
                    for imb, coord in enumerate(self.grid):
                        if (ghost_mask := np.diff(coord[1, :6]) <= 0).any():
                            self.grid[imb, 1, :5][ghost_mask] *= -1
                        if (ghost_mask := np.diff(coord[1, -6:]) <= 0).any():
                            self.grid[imb, 1, -5:][ghost_mask] = 2*np.pi - coord[1, -5:][ghost_mask]
                self.origins = np.array([x[:, 1] for x in self.grid])
                self.ends = np.array([x[:, -2] for x in self.grid])

                for key in keys:
                    shm = SharedMemory(name=self.shared_memory[key])
                    data: np.ndarray = np.ndarray(self.full_shape, dtype=float, buffer=shm.buf)
                    j = vars_in_file.index(key)
                    data[:] = f[dset][j]

        if gr:
            self.transform_Wv(mass)

    def transform_Wv(self, mass: float):
        shared_mem = [SharedMemory(name=self.shared_memory[key]) for key in self.vel_keys]
        vel = [np.ndarray(self.full_shape, dtype=float, buffer=shm.buf) for shm in shared_mem]
        for imb, coords in enumerate(self.grid):
            coords = np.meshgrid(*coords, indexing='ij')
            if self.spherical:
                _, th, rr = coords
                r2 = rr*rr
                # unravel ghost zone theta
                sth = np.abs(np.sin(th))

                gam = [sth*sth*r2, r2, (1-2*mass/rr)**-1]
            else:
                rr = np.sqrt(sum(x*x for x in coords))
                gam = 3*[(1-2*mass/rr)**-1]

            W = np.sqrt(1 + sum(u[imb]*u[imb]*g for u, g in zip(vel, gam)))
            for i in range(3):
                vel[i][imb] /= W

            # give angular velocities in units length/time
            if self.spherical:
                vel[0][imb] *= rr*sth
                vel[1][imb] *= rr

    def get_mb_index(
        self,
        x: np.ndarray
        ) -> int:
        """
        Returns index of meshblock that contains x
        """
        mask = np.all(
                (x >= self.origins) &
                (x <= self.ends),
                axis=1)
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

        imb = self.get_mb_index(x)
        if imb == -1:
            # out of bounds
            return np.zeros_like(x)

        grid = self.grid[imb]

        res = np.empty(len(keys))
        for ii, key in enumerate(keys):
            shm = SharedMemory(name=self.shared_memory[key])
            ar: np.ndarray = np.ndarray(self.full_shape, dtype=float, buffer=shm.buf)
            res[ii] = RegularGridInterpolator(
                grid, ar[imb],
                bounds_error=False, fill_value=None
                )(x, method='linear')

        if self.spherical and keys==AthdfInterpolator.vel_keys:
            jac = _rotjac(theta, phi)
            res = jac @ res
        if mirror_z:
            res[0] = -res[0]
        return res

def read_time(
    path_keys: tuple[str, tuple[str]]
    ) -> tuple[dict[float, dict[str, tuple[str, ...]]], int]:
    fpath, keys = path_keys
    with h5.File(fpath, 'r') as f:
        file_keys = f.attrs['VariableNames'][:].astype(str)
        avail_keys = tuple(key for key in keys if key in file_keys)
        if not any(avail_keys):
            return dict(), 0
        time = float(f.attrs['Time'][()])
        nmb: int = int(f.attrs["NumMeshBlocks"])
        mb_size: np.ndarray = f.attrs['MeshBlockSize'][:].astype(int)
        mem_size = 8*nmb*int(np.prod(mb_size))
    return {time: {fpath: avail_keys}}, mem_size

class AthdfInterpolator(FileInterpolator):
    coord_keys = ('x3', 'x2', 'x1')
    vel_keys = ('vel3', 'vel2', 'vel1')
    file_class = AthdfFile

    def __init__(self, *args, **kwargs):
        self.file_args = {key: kwargs.pop(key)
                          for key in AthdfFile.kwargs
                          if key in kwargs}
        super().__init__(*args, **kwargs)

    def parse_files(self, path: str) -> tuple[dict, int]:
        '''
        returns a nested dictoary containing the file times, the file names and
        the contained keys as well as the maximum memory size in bytes that one
        gridfunction needs.
        '''

        fnames = [ff.path for ff in os.scandir(path) if ff.name.endswith('.athdf')]

        res = do_parallel(
            read_time,
            list(zip(fnames, repeat(self.vel_keys+self.data_keys))),
            desc="Parsing file times",
            unit="files",
            **self.do_parallel_kw
            )

        def combine_dict(do, dn):
            dn = dn[0]
            for t in dn:
                if t in do:
                    do[t] = {**do[t], **dn[t]}
                else:
                    do[t] = dn[t]
            return do


        file_dict = reduce(combine_dict, res, {})
        mem_size = max(r[1] for r in res)

        return file_dict, mem_size

class AthdfTracers(FileTracers):
    interpolator_class = AthdfInterpolator

    def __init__(self, *args, **kwargs):
        self.interp_kwargs = {key: kwargs.pop(key)
                              for key in AthdfFile.kwargs
                              if key in kwargs}
        super().__init__(*args, **kwargs)
