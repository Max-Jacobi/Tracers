################################################################################
import os
import sys
# import signal
import numpy as np

from tracers.surface import SurfaceTracers
from tracers import do_parallel, Tracer
sys.path.append("/home/ho54hof/repos/Tracers/tracers")

################################################################################

path = "../combine"

rmin = 300
rmax = 1000
t_end = 2400
t_start = 7800
max_dt = 50
n_cpu = 48
avail_mem = 52000
#t_eval = np.arange(t_start, t_end-1, -max_dt)

nr = 32
nphi = 32
ntheta = 16

rand_seed = 42
outdir = f"trajectories"

if not os.path.isdir(outdir):
    os.mkdir(outdir)

################################################################################

# setup of seeds

rr = np.linspace(rmin**3, rmax**3, nr+1)**(1/3)
phi = np.linspace(0, 2*np.pi, nphi+1)[:-1]
cost = np.linspace(-1, 1, ntheta+1)[:-1]

dr = np.diff(rr)
dphi = phi[1] - phi[0]
dct = cost[1] - cost[0]

rr = rr[:-1]
rr += dr/2
cost += dct/2

rr, phi, cost = np.meshgrid(rr, phi, cost, indexing='ij')

dvol = rr * dr[:, None, None]*dphi

rng = np.random.default_rng(rand_seed)
rr += rng.uniform(-0.5, 0.5, size=rr.shape)*dr[:, None, None]
phi += rng.uniform(-0.5, 0.5, size=phi.shape)*dphi
cost += rng.uniform(-0.5, 0.5, size=cost.shape)*dct

rr = rr.flatten()
phi = phi.flatten()
cost = cost.flatten()
dvol = dvol.flatten()
time = np.ones_like(rr)*t_start

sint = np.sqrt(1-cost*cost)
cosp = np.cos(phi)
sinp = np.sin(phi)

seeds = dict(
    x=rr*cosp*sint,
    y=rr*sinp*sint,
    z=rr*cost,
    time=time,
    dvol=dvol,
)

################################################################################


# domain bounderies
rmin2 = 150**2
def out_of_bounds_inner(t: float, x: np.ndarray, *_) -> float:
    r2 = np.sum(x*x)
    return r2 - rmin2
out_of_bounds_inner.direction = -1
out_of_bounds_inner.terminal = True

rmax2 = rmax*rmax
def out_of_bounds_outer(t: float, x: np.ndarray, *_) -> float:
    r2 = np.sum(x*x)
    return rmax2 - r2
out_of_bounds_outer.direction = -1
out_of_bounds_outer.terminal = True

def out_of_time(t: float, *_) -> float:
    return t - t_end
out_of_time.direction = -1
out_of_time.terminal = True

################################################################################


trs = SurfaceTracers(
    path,
    data_keys=[
        'tracer.hydro.aux.T',
        'tracer.hydro.aux.hu_t',
        'tracer.hydro.aux.s',
        'tracer.hydro.aux.u_t',
        'tracer.hydro.prim.rho',
        'tracer.passive_scalars.r_0',
        'tracer.hydro.aux.V_u_x',
        'tracer.hydro.aux.V_u_y',
        'tracer.hydro.aux.V_u_z',
        ],
    seeds=seeds,
    n_cpu=n_cpu,
    verbose=True,
    t_int_order='linear',
    events=[out_of_bounds_inner, out_of_bounds_outer, out_of_time],
    #t_eval=t_eval,
    max_step=max_dt,
    timeout=60,
    avail_memory=avail_mem,

)

# def handle_sigterm(signum, frame):
#     print("Received signal {}".format(signum))
#     trs.interpolator.free_shared_memory()
#     print("Cleanup finished, exiting.")
#     sys.exit(0)
# 
# signal.signal(signal.SIGTERM, handle_sigterm)

trs.integrate()

################################################################################


def output(tr):
    tr.trajectory = {k.split('.')[-1]: val for k, val in tr.trajectory.items()}
    tr.data_keys = tuple([k.split('.')[-1] for k in tr.data_keys])
    if 'r_0' in tr.trajectory:
        tr.trajectory['ye'] = tr.trajectory['r_0']
        dk = list(tr.data_keys)
        dk.remove('r_0')
        dk.append('ye')
        tr.data_keys = tuple(dk)
    if 'rho' in tr.trajectory:
        tr.props['rho0'] = tr['rho'][0]
        tr.props['mass'] = tr['rho'][0]*tr.props['dvol']
    tr.output_to_ascii(f"{outdir}/tracer_")

do_parallel(
    output,
    trs,
    n_cpu=1,
    desc=f"Outputting to {outdir}",
    unit="tracer",
    verbose=True,
)
print("Done ::)")
