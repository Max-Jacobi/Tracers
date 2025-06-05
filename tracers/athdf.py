################################################################################
from multiprocessing.shared_memory import SharedMemory
import os

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
    def __init__(
        self,
        filename: str,
        shm_names: dict[str, str],
        coord_keys: tuple[str],
        bitant: bool = False,
        spherical: bool = False,
    ):

        self.filename = filename
        self.shared_memory = shm_names
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

            self.grid = np.array([f[f'{key}v'][:] for key in coord_keys]).T
            self.origins = np.array([x[1] for x in self.grid])
            self.ends = np.array([x[-2] for x in self.grid])

            for key, shm_name in self.shared_memory.items():
                shm = SharedMemory(name=shm_name)
                data: np.ndarray = np.ndarray(self.full_shape, dtype=float, buffer=shm.buf)
                j = vars_in_file.index(key)
                data[:] = f[dset][j]

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

        xx = (self.x3[imb], self.x2[imb], self.x1[imb])

        res = np.empty(len(keys))
        for ii, key in enumerate(keys):
            shm = SharedMemory(name=self.shared_memory[key])
            ar: np.ndarray = np.ndarray(self.full_shape, dtype=float, buffer=shm.buf)
            res[ii] = RegularGridInterpolator(xx, ar[imb])(x, method='linear')

        if self.spherical and keys==AthdfInterpolator.vel_keys:
            jac = _rotjac(theta, phi)
            res = jac @ res
        if mirror_z:
            res[0] = -res[0]
        return res

class AthdfInterpolator(FileInterpolator):
    coord_keys = ('x3', 'x2', 'x1')
    vel_keys = ('vel3', 'vel2', 'vel1')
    file_class = AthdfFile

    def __init__(self, *args, bitant: bool = False, spherical: bool = False, **kwargs):
        self.file_args = dict(bitant=bitant, spherical=spherical)
        super().__init__(*args, **kwargs)

    def parse_files(self, path: str) -> tuple[np.ndarray, np.ndarray, int]:
        '''
        returns an array of filnames and an array of the corresponding times
        and the maximum memory size in bytes that one gridfunction needs
        '''

        files: list[str] = []
        times: list[float] = []

        mem_size = 0

        fnames = [ff.path for ff in os.scandir(path) if ff.name.endswith('.athdf')]
        def read_time(fpath):
            nonlocal mem_size
            with h5.File(fpath, 'r') as f:
                if not all(key in f.attrs['VariableNames'][:].astype(str)
                           for key in self.vel_keys+self.data_keys):
                    return
                files.append(fpath)
                times.append(float(f.attrs['Time'][()]))
                nmb: int = int(f.attrs["NumMeshBlocks"])
                mb_size: np.ndarray = f.attrs['MeshBlockSize'][:].astype(int)
                mem_size = max(mem_size, 8*nmb*int(np.prod(mb_size)))

        do_parallel(
            read_time, fnames,
            n_cpu=1,
            desc="Parsing file times",
            unit="files",
            verbose=self.verbose,
            )

        return np.array(files), np.array(times), mem_size

class AthdfTracers(FileTracers):
    interpolator_class = AthdfInterpolator

    def __init__(self, *args, bitant: bool = False, spherical: bool = False, **kwargs):
        self.interp_kwargs = dict(bitant=bitant, spherical=spherical)
        super().__init__(*args, **kwargs)
