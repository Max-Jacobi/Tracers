from typing import Iterable
from multiprocessing.shared_memory import SharedMemory
import os, signal, sys

import numpy as np
import h5py as h5
from scipy.interpolate import RegularGridInterpolator

from . import Tracers, Interpolator
from .from_file import File, FileInterpolator, FileTracers
from .utils import do_parallel

_2pi = 2*np.pi

def _fill_with_ghosts(buf: np.ndarray, h5f: h5.File, key: str, ng: int = 1, ev: np.ndarray = np.ones(3, dtype=int)):
    ar = np.array(h5f[key][:])
    nphi = ar.shape[1]
    buf[ng:-ng, ng:-ng] = ar[::ev[1], ::ev[2]]
    for ig in range(ng):
        ign = -ig-1
        buf[ ig, ng:-ng] = np.roll(ar[ ig, ::ev[2]], nphi//2)
        buf[ign, ng:-ng] = np.roll(ar[ign, ::ev[2]], nphi//2)
        buf[ng:-ng,  ig] = ar[::ev[1], ign]
        buf[ng:-ng, ign] = ar[::ev[1],  ig]
        buf[ig, ig] = ar[ig, ig]
        buf[ign, ig] = ar[ign, ig]
        buf[ig, ign] = ar[ig, ign]
        buf[ign, ign] = ar[ign, ign]

class SphericalGridInterpolator(RegularGridInterpolator):
    def __call__(self, xi, method=None, *, nu=None):
        x, y, z = xi
        r = np.sqrt(x*x + y*y + z*z)
        th = np.arccos(z/r)
        ph = (np.arctan2(y, x)+_2pi)%_2pi
        return super().__call__(np.array([r, th, ph]).T, method=method, nu=nu)

class SurfaceFile(File):
    def __init__(
        self,
        filename: str,
        shm_names: dict[str, str],
        grid: tuple[np.ndarray, np.ndarray, np.ndarray],
        every: np.ndarray = np.ones(3, dtype=int),
    ):
        self.filename = filename
        self.keys = tuple(shm_names.keys())
        self.shared_memory = shm_names
        self.grid = grid
        self.shape = tuple(len(g) for g in self.grid)

        with h5.File(filename, 'r') as f:
            self.time = float(f['coordinates/00/T'][:])
            for key in self.keys:
                # print(f'Copying {key} to {self.shared_memory[key]}', flush=True)
                shm = SharedMemory(name=self.shared_memory[key])
                grp = key.split('.')[0]
                data: np.ndarray = np.ndarray(self.shape, dtype=float, buffer=shm.buf)
                for ir in range(self.shape[0]):
                    _fill_with_ghosts(data[ir], f, f'fields/{ir*every[0]:02d}/{grp}/{key}', ev=every)

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

class SurfaceInterpolator(FileInterpolator):
    coord_keys = ('x', 'y', 'z')
    vel_keys = ("tracer.hydro.aux.V_u_x",
                "tracer.hydro.aux.V_u_y",
                "tracer.hydro.aux.V_u_z")

    def parse_files(self, path: str, every: int = 1) -> tuple[np.ndarray, np.ndarray, int]:
        _files = []
        _times = []
        fnames = [ff.path for ff in os.scandir(path) if 'surface' in ff.name]

        def read_time(fpath):
            with h5.File(fpath, 'r') as f:
                _files.append(fpath)
                _times.append(float(f['coordinates/00/T'][:]))

        do_parallel(
            read_time, fnames,
            n_cpu=1,
            desc="Parsing file times",
            unit="files",
            verbose=self.verbose,
            )

        files = np.array(_files)
        times = np.array(_times)

        self.grid = self.load_grid(files[0], every=self.every_grid)
        self.shape = tuple(len(g) for g in self.grid)

        if len(files) == 0:
            raise ValueError(f'No files found in {path}')

        isort = np.argsort(times)
        files = files[isort]
        times = times[isort]
        if every > 1:
            last_t = times[-1]
            last_f = files[-1]
            times = times[::every]
            files = files[::every]
            if times[-1] != last_t:
                times = np.append(times, last_t)
                files = np.append(files, last_f)
        return files, times, int(np.prod(self.shape))*8

    def load_grid(self, filename, every: np.ndarray | None) -> tuple[np.ndarray, ...]:
        """
        Loads the grid
        """
        if every is None:
            every = np.ones(3)
        with h5.File(filename, 'r') as f:
            n_r = len(f['coordinates'].keys())
            r = np.array([float(f[f'coordinates/{ir:02d}/R'][:]) for ir in range(n_r)])
            th =  np.array(f['coordinates/00/th'][:])
            ph =  np.array(f['coordinates/00/ph'][:])
            if (n_ph := len(ph))%2 != 0:
                raise ValueError(f"Files have unqual number of points in phi direction (n_phi={n_ph})!")
            r = r[::every[0]]
            th = np.concatenate(([-th[0]], th[::every[1]], [th[0]+2*np.pi]))
            ph = np.concatenate(([-ph[0]], ph[::every[2]], [ph[0]+2*np.pi]))
        return (r, th, ph)

    def load_file(self, args: tuple[str, dict[str, str]]) -> SurfaceFile:
        return SurfaceFile(*args, grid=self.grid, every=self.every_grid)

class SurfaceTracers(FileTracers):
    interpolator_class = SurfaceInterpolator
