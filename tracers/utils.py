from typing import Callable
from time import time
from multiprocessing import Pool
from sys import stdout
from tqdm import tqdm
import atexit

_pool = None
_n_cpu = None

def _get_pool(n_cpu):
    """
    Lazily initialize (or reuse) a module‐level Pool of size n_cpu.
    """
    global _pool, _n_cpu
    if _pool is None:
        # print(f"Spawning pool with {n_cpu} workers")
        _pool = Pool(n_cpu)
        _n_cpu = n_cpu
        atexit.register(cleanup_pool)
    elif _n_cpu != n_cpu:
        raise RuntimeError(f"Tried to get pool with {n_cpu} cpus "
                           f"but we only have one with {_n_cpu}!")
    return _pool

def cleanup_pool():
    global _pool
    if _pool is not None:
        _pool.close()
        _pool.join()
        _pool = None

class TimeoutError(Exception):
    pass

class Timeout:
    t_start: (None | float)

    def __init__(self, func: Callable, timeout: float):
        self.func = func
        self.timeout = timeout
        self.t_start = None

    def __call__(self, *args, **kwargs):
        if self.t_start is None:
            self.t_start = time()
        if (t := time() - self.t_start) > self.timeout:
            raise TimeoutError(f"Tracer timed out after {t:.1f}s")
        return self.func(*args, **kwargs)

def do_parallel(
    func,
    args,
    n_cpu,
    timeout: float = -1.0,
    verbose: bool = False,
    **kwargs
):
    kwargs["total"]   = len(args)
    kwargs["disable"] = not verbose
    kwargs["ncols"]   = 0
    kwargs["file"]    = stdout

    if timeout > 0:
        func = Timeout(func, timeout)
    if n_cpu == 1:
        return list(tqdm(map(func, args), **kwargs))
    pool = _get_pool(n_cpu)
    return list(tqdm(pool.imap_unordered(func, args), **kwargs))
