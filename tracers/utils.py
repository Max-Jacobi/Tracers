from tqdm import tqdm
from multiprocessing import Pool

from typing import Callable, Sequence
import numpy as np


def do_parallel(
    func: Callable,
    args: (Sequence | np.ndarray),
    n_cpu: int,
    verbose: bool = False,
    **kwargs
    ):
    kwargs["total"] = len(args)
    kwargs["disable"] = not verbose
    kwargs["ncols"] = 0

    if n_cpu == 1:
        return list(tqdm(map(func, args), **kwargs))
    with Pool(n_cpu) as pool:
        return list(tqdm(pool.imap_unordered(func, args), **kwargs))
