################################################################################
import sys
import numpy as np

from tracers.surface import SurfaceTracers
from tracers import do_parallel, Tracer

################################################################################

path = "athena/runs/SFHo_VLR_nps/output-0000"

rmin = 200
rmax = 300
time = 70

nr = 10
nphi = 6
ntheta = 4
n_cpu = 4
n_files = 2 #max(2, n_cpu)

################################################################################

# setup of seeds

rr = np.linspace(rmin**3, rmax**3, nr+1)**(1/3)
phi = np.linspace(0, 2*np.pi, nphi+1)[:-1]
cost = np.linspace(0, 1, ntheta+1)[:-1]

dr = np.diff(rr)
dphi = phi[1] - phi[0]
dct = cost[1] - cost[0]

rr = rr[:-1]
rr += dr/2
cost += dct/2

rr, phi, cost = np.meshgrid(rr, phi, cost, indexing='ij')

dvol = rr**2 * dr[:, None, None]*dct*dphi

rr += np.random.uniform(-0.5, 0.5, size=rr.shape)*dr[:, None, None]
phi += np.random.uniform(-0.5, 0.5, size=phi.shape)*dphi
cost += np.random.uniform(-0.5, 0.5, size=cost.shape)*dct

rr = rr.flatten()
phi = phi.flatten()
theta = np.acos(cost).flatten()
dvol = dvol.flatten()
time = np.ones_like(rr)*time

sint = np.sqrt(1 - cost**2)
seeds = dict(
    r=rr,
    th=theta,
    ph=phi,
    time=time,
    dvol=dvol,
)

################################################################################

# domain bounderies
def oob(t: float, x: np.ndarray, *_) -> float:
    r = np.sqrt(np.sum(x**2))
    return r - 150
oob.direction = -1
oob.terminal = True

def oot(t: float, *_) -> float:
    return t-10

oot.direction = -1
oot.terminal = True

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
    files_per_step=2,
    events=[oob, oot],
)

trs.integrate()

################################################################################


def output(tr):
    tr.trajectory = {k.split('.')[-1]: val for k, val in tr.trajectory.items()}
    tr.data_keys = tuple([k.split('.')[-1] for k in tr.data_keys])
    tr.trajectory['ye'] = tr.trajectory['r_0']
    tr.props['rho0'] = tr['rho'][0]
    tr.props['mass'] = tr['rho'][0]*tr.props['dvol']
    tr.output_to_ascii("trajectories/tracer_")

do_parallel(
    output,
    trs,
    n_cpu=n_cpu,
    desc='Outputting',
    unit='tracer',
    verbose=True,
)
print("Done ::)")
