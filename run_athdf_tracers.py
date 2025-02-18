################################################################################
import sys
import numpy as np

sys.path.append("/home/ho54hof/repos/Tracers")
from tracers.athdf import AthdfTracers
from tracers import do_parallel, Tracer

################################################################################

path = "files"

rmin = 500
rmax = 1.5e5
time = 2e5

nr = 12
nphi = 36
ntheta = 9
n_cpu = 36
#nr = 1
#nphi = 2
#ntheta = 2
#n_cpu = 1

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

################################################################################

rr += np.random.uniform(-0.5, 0.5, size=rr.shape)*dr[:, None, None]
phi += np.random.uniform(-0.5, 0.5, size=phi.shape)*dphi
cost += np.random.uniform(-0.5, 0.5, size=cost.shape)*dct

rr = rr.flatten()
phi = phi.flatten()
cost = cost.flatten()
dvol = dvol.flatten()
time = np.ones_like(rr)*time

sint = np.sqrt(1 - cost**2)
seeds = dict(
    x3=rr*cost,
    x2=rr*np.sin(phi)*sint,
    x1=rr*np.cos(phi)*sint,
    time=time,
    dvol=dvol,
)

################################################################################

# domain bounderies
def oob(t: float, x: np.ndarray, *_) -> float:
    r = np.sqrt(np.sum(x**2))
    return r - 400
oob.direction = -1
oob.terminal = True

def oot(t: float, *_) -> float:
    return t

oot.direction = -1
oot.terminal = True

def check_flag(tr: Tracer) -> Tracer:
    if np.any(tr.trajectory['rFlag'] < .1):
        tr.finished = tr.failed = True
        tr.message = "rFlag < 0.1"
    return tr

################################################################################

trs = AthdfTracers(
    path,
    data_keys=['rho', 'rYE', 'rENT', 'rAbar'],
    seeds=seeds,
    n_cpu=n_cpu,
    verbose=True,
    t_int_order='linear',
    files_per_step=n_cpu*2,
    end_conditions=[check_flag,],
    coordinates='spherical bitant',
    events=[oob, oot],
)

trs.integrate()


def output(tr):
    tr.props['rho0'] = tr['rho'][0]
    tr.props['mass'] = tr['rho'][0]*tr.props['dvol']
    tr.trajectory['rYE'] *= 1.2
    tr.trajectory['rENT'] *= 1000
    tr.trajectory['rAbar'] *= 500
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
