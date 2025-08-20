################################################################################
import os
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import h5py as h5
from scipy.interpolate import RegularGridInterpolator

from .from_file import File, FileInterpolator, FileTracers
from .utils import do_parallel
from .tracers import InterpolationError

################################################################################

class FLASHFile(File):
    def __init__(
        self,
        filename: str,
        shm_names: dict[str, str],
        dim: int = 2,
    ):
        self.filename = filename
        self.shared_memory = shm_names

        with h5.File(filename, 'r') as hf:
            rscalars = {k.strip(): float(v) for k, v in dict(hf['real scalars']).items()}
            iscalars = {k.strip(): int(v) for k, v in dict(hf['integer scalars']).items()}
            self.time = rscalars[b'time']
            self.shape = (iscalars[b'nxb'], iscalars[b'nyb'], iscalars[b'nzb'])[:dim]
            mb_mask = np.array(hf['node type'][:] == 1, bool)
            nmb = mb_mask.sum()
            self.full_shape = (nmb, *self.shape)
            bb = np.array(hf['bounding box'][mb_mask, :dim])
            self.gb_l = bb[:, :, 0]
            self.gb_r = bb[:, :, 1]
            self.grid = np.array([[np.linspace(o, e, n+1)
                      for o, e, n in zip(self.gb_l[imb], self.gb_r[imb], self.shape)]
                      for imb in range(nmb)])
            self.grid = (self.grid[:, :, 1:] + self.grid[:, :, :-1])/2

            self.origins = np.array([x[0] for x in self.grid])
            self.ends = np.array([x[-1] for x in self.grid])

            for key, shm_name in self.shared_memory.items():
                shm = SharedMemory(name=shm_name)
                data: np.ndarray = np.ndarray(self.full_shape, dtype=float, buffer=shm.buf)
                data[:] = np.squeeze(hf[key][mb_mask, :dim])

    def get_mb_index(
        self,
        x: np.ndarray
        ) -> int:
        """
        Returns index of meshblock that contains x
        """
        mask = np.all(
                (x >= self.gb_l) &
                (x <= self.gb_r),
                axis=1)
        if sum(mask) == 0:
            raise InterpolationError("Out of bounds in interpolation")
        idx = np.where(mask)[0][0]
        return idx


    def interpolate(self, x: np.ndarray, keys: tuple[str, ...]) -> np.ndarray:
        imb = self.get_mb_index(x)

        grid = self.grid[imb]

        res = np.empty(len(keys))
        for ii, key in enumerate(keys):
            shm = SharedMemory(name=self.shared_memory[key])
            ar: np.ndarray = np.ndarray(self.full_shape, dtype=float, buffer=shm.buf)

            res[ii] = RegularGridInterpolator(
                grid, ar[imb],
                bounds_error=False, fill_value=None
                )(x, method='nearest')

        return res

class FLASHInterpolator(FileInterpolator):
    file_class = FLASHFile

    def __init__(self, *args, coords: str = '2d', **kwargs):
        if coords == "2d":
            self.coord_keys = ("x", "y")
            self.vel_keys = ("velx", "vely")
        elif coords == "3d":
            self.coord_keys = ("x", "y", "z")
            self.vel_keys = ("velx", "vely","velz")
        else:
            raise ValueError(f"coordinates {coords} not implemented. Only '2d' and '3d'")
        super().__init__(*args, **kwargs)
        self.dim = len(self.coord_keys)
        self.file_args = {"dim": self.dim}


    def parse_files(self, path: str) -> tuple[np.ndarray, np.ndarray, int]:
        '''
        returns an array of filnames and an array of the corresponding times
        and the maximum memory size in bytes that one gridfunction needs
        '''

        files: list[str] = []
        times: list[float] = []

        mem_size = 0

        fnames = [ff.path for ff in os.scandir(path)]
        def read_time(fpath):
            nonlocal mem_size
            files.append(fpath)
            with h5.File(fpath, 'r') as hf:
                rscalars = {k.strip(): float(v) for k, v in dict(hf['real scalars']).items()}
                iscalars = {k.strip(): int(v) for k, v in dict(hf['integer scalars']).items()}
                times.append(rscalars[b'time'])
                mb_size = (iscalars[b'nxb'], iscalars[b'nyb'], iscalars[b'nzb'])[:self.dim]
                mb_mask = np.array(hf['node type'][:] == 1, bool)
                nmb = mb_mask.sum()
                mem_size = max(mem_size, 8*nmb*int(np.prod(mb_size)))

        do_parallel(
            read_time, fnames,
            n_cpu=1,
            desc="Parsing file times",
            unit="files",
            verbose=self.verbose,
            outf=self.outf,
            )

        return np.array(files), np.array(times), mem_size

class FLASHTracers(FileTracers):
    interpolator_class = FLASHInterpolator

    def __init__(self, *args, coords: str = '2d', **kwargs):
        self.interp_kwargs = {'coords': coords}
        super().__init__(*args, **kwargs)
