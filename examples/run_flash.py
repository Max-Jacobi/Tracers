################################################################################
import os
from h5py import File
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from tqdm import tqdm

from tracers.utils import do_parallel
from tracers.flash import FLASHTracers
from matplotlib.animation import FuncAnimation

################################################################################

time = 0.4
sq_rmax = 4e14 # rmax = 2e7cm
x = np.linspace(0, 1e8, 30)
y = np.linspace(-0.8e8, 1.7e8, 30)
x, y = np.meshgrid(x, y)
mask = (x*x + y*y) > sq_rmax
x, y = x[mask], y[mask]

seeds = dict(x=x, y=y, time=np.ones_like(x)*0.4)

# domain bounderies
def out_of_bounds_inner(t: float, x: np.ndarray, *_) -> float:
    return (x*x).sum() - sq_rmax
out_of_bounds_inner.direction = -1
out_of_bounds_inner.terminal = True

def out_of_time(t: float, *_) -> float:
    return t
out_of_time.direction = -1
out_of_time.terminal = True

def check_temp(tr):
    if np.any(tr.trajectory['temp'] > 10):
        tr.finished = tr.failed = True
        tr.message = "Temperature reached 10GK"
    return tr

################################################################################


path = '/home/mjacobi/Documents/Projects/tracers/S20_WHW_tetralith'
trs = FLASHTracers(
    path=path,
    coords='2d',
    data_keys=[
        'dens',
        'ye',
        'temp',
        ],
    seeds=seeds,
    events=[out_of_bounds_inner, out_of_time],
    end_conditions=[check_temp,],
    n_cpu=4,
    verbose=True,
    t_int_order='linear',
    use_shared_memory=2500,
    timeout=5,
    max_step=0.001,
    )

trs.integrate()

################################################################################


outdir = 'tracers_test'
def output(tr):
    tr.output_to_ascii(f"{outdir}/tracer_")

do_parallel(
    output,
    trs,
    n_cpu=1,
    desc=f"Outputting to {outdir}",
    unit="tracers",
    verbose=True,
)
print("Done ::)")
