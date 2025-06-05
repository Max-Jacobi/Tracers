from typing import Iterable
from multiprocessing.shared_memory import SharedMemory
import os, signal
from abc import ABC, abstractmethod

import numpy as np
import h5py as h5
from scipy.interpolate import interp1d

from . import Tracers, Interpolator, do_parallel
from .utils import cleanup_pool

class File(ABC):
    time: float

    @abstractmethod
    def __init__(
        self,
        filename: str,
        shm_names: dict[str, str],
        *args, **kwargs
        ):
        '''
        loads all keys in the file into the shared memories given by shm_names
        '''
        ...

    @abstractmethod
    def interpolate(self, x: np.ndarray, keys: tuple[str, ...]) -> np.ndarray:
        '''
        Interpolates keys in keys to position x
        '''
        ...


class FileInterpolator(Interpolator, ABC):
    coord_keys: tuple[str, ...]
    vel_keys: tuple[str, ...]
    data: tuple
    file_class: type

    def __init__(
        self,
        path: str,
        data_keys: Iterable[str],
        t_int_order: str = 'linear',
        n_cpu: int = 1,
        files_per_step: int | None = None,
        use_shared_memory: float | None = None,
        verbose: bool = False,
        every_time: int = 1,
        every_grid: int | np.ndarray = 1,
    ):
        self.path = path
        self.data_keys = tuple(data_keys)
        self.n_cpu = n_cpu
        self.verbose = verbose
        self.dim = len(self.vel_keys)
        implemented = ('linear', 'cubic')
        if t_int_order not in implemented:
            raise ValueError(f"Time interpolation order {t_int_order} not implemented.")
        self.t_int_order = t_int_order
        if isinstance(every_grid, int):
            every_grid = np.ones(self.dim, int)*every_grid
        self.every_grid = every_grid

        self.filenames, self.times = self.parse_files(path, every=every_time)
        self.max_size = 8*int(max(np.prod(sh) for sh in self.shapes))

        self.files: list[File] = list()
        self.data = (self.files, self.t_int_order, self.data_keys, self.vel_keys)

        self.allocate_shm(files_per_step, use_shared_memory)

    @abstractmethod
    def parse_files(self, path: str, every: int = 1) -> tuple[np.ndarray, np.ndarray, int]:
        '''
        returns an array of filnames and an array of the corresponding times
        and the maximum memory size in bytes that one gridfunction needs
        '''
        ...


    def allocate_shm(
        self,
        files_per_step: int|None,
        req_mem: float|None,
        ):
        keys = np.unique(self.vel_keys+self.data_keys)

        st = os.statvfs("/dev/shm")
        free_mem = st.f_bavail * st.f_frsize

        if files_per_step is not None:
            ...
        elif req_mem is not None:
            req_mem = req_mem*1024**2
            files_per_step = int(req_mem/len(keys)/self.max_size)
        else:
            raise RuntimeError("Must either supply use_shared_memory or files_per_step argument")
        self.req_mem = len(keys)*files_per_step*self.max_size
        self.files_per_step = files_per_step

        print(f"Allocating shared memory for {self.files_per_step} files and {len(keys)} grid functions "
              f"using {self.req_mem/1024**self.dim:.2f}GB of memory.", flush=True)
        if free_mem < 1.2*req_mem:
            raise RuntimeError(f"This configuration would request to much memory. Free: {free_mem/1024**self.dim}GB")

        # Ensure shared mempry cleanup on SIGTERM
        def handler(signum, frame):
            os.write(2, b"SIGTERM received\n")
            os.write(2, b" Cleaning up shared memory\n")
            self.free_shared_memory()
            os.write(2, b" Done\n")
            os._exit(1)

        signal.signal(signal.SIGTERM, handler)

        self.shared_memory =  [{key: SharedMemory(create=True, size=self.max_size).name
                                for key in keys} for _ in range(self.files_per_step)]

    @abstractmethod
    def load_file(self, args: tuple[str, dict[str, str]]) -> File:
        ...

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

        files = self.filenames[i_s:i_e]

        self.files[:] = do_parallel(
            self.load_file,
            list(zip(files, self.shared_memory)),
            n_cpu=self.n_cpu,
            desc=f"Loading files for t = {t_span[0]} - {t_span[1]}",
            unit="file",
            verbose=self.verbose,
        )
        self.files.sort(key=lambda f: f.time)

    @staticmethod
    def interpolate(
        time: float,
        coords: np.ndarray,
        data: tuple,
        mode: str,
    ) -> np.ndarray:
        files, order, data_keys, vel_keys  = data
        if mode == 'velocity':
            keys = vel_keys
        else:
            keys = data_keys

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
        ...

    @staticmethod
    def interpolate_data(
        time: float,
        coords: np.ndarray,
        data: tuple,
    ) -> np.ndarray:
        ...

    def free_shared_memory(self):
        for file_shm in self.shared_memory:
            for key, name in file_shm.items():
                try:
                    shm = SharedMemory(name=name)
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    print(f"Could not find shared memory for {key}: {name}")

        self.shared_memory = []

class FileTracers(Tracers, ABC):
    interpolator_class: type
    interpolator: FileInterpolator

    def __init__(
        self,
        path: str,
        data_keys: Iterable[str],
        n_cpu: int = 1,
        reverse: bool = True,
        files_per_step: int | None = None,
        use_shared_memory: float | None = None,
        t_int_order: str = 'linear',
        every_time: int = 1,
        every_grid: int | np.ndarray = 1,
        verbose: bool = False,
        **kwargs,
    ):

        self.path = path
        self.data_keys = tuple(data_keys)
        interpolator = self.interpolator_class(
            self.path,
            data_keys=data_keys,
            t_int_order=t_int_order,
            n_cpu=n_cpu,
            verbose=verbose,
            files_per_step = files_per_step,
            use_shared_memory = use_shared_memory,
            every_time=every_time,
            every_grid=every_grid,
        )

        self.files_per_step = interpolator.files_per_step
        if t_int_order == 'linear':
            self.step = self.files_per_step - 1
        elif t_int_order == 'cubic':
            self.step = self.files_per_step - 2
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
            print("Cleaning up", flush=True)
            self.interpolator.free_shared_memory()

    def __del__(self):
        if hasattr(self, "interpolator"):
            self.interpolator.free_shared_memory()
