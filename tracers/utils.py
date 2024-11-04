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
    if n_cpu > 1:
        with Pool(n_cpu) as pool:
            ret = list(tqdm(
                pool.imap_unordered(func, args),
                total=len(args),
                disable=not verbose,
                ncols=0,
                **kwargs
            ))
    else:
        ret = list(tqdm(
            map(func, args),
            total=len(args),
            disable=not verbose,
            ncols=0,
            **kwargs
        ))
    return ret
