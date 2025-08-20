from multiprocessing import Pool
from sys import stdout
from tqdm import tqdm
import functools
import concurrent.futures
import atexit

_pool = None
_n_cpu = None

def _get_pool(n_cpu):
    """
    Lazily initialize (or reuse) a module‐level Pool of size n_cpu.
    """
    global _pool, _n_cpu
    if _pool is None:
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

def timeout(seconds=10., error_message="Function call timed out"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not seconds > 0:
                return func(*args, **kwargs)
            # Use a ThreadPoolExecutor with max_workers=1
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    # Wait up to 'seconds' for completion
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError:
                    # Cancel the future (best‐effort)
                    future.cancel()
                    raise TimeoutError(error_message)
        return wrapper
    return decorator

def do_parallel(
    func,
    args,
    n_cpu,
    verbose: bool = False,
    **kwargs
):
    kwargs.setdefault("total", len(args))
    kwargs.setdefault("disable", not verbose)
    kwargs.setdefault("ncols", 0)
    kwargs.setdefault("file", stdout)

    if n_cpu == 1:
        return list(tqdm(map(func, args), **kwargs))
    pool = _get_pool(n_cpu)
    return list(tqdm(pool.imap_unordered(func, args), **kwargs))
