################################################################################
import os
import sys
import atexit
from time import sleep
import numpy as np

from tracers.athdf import AthdfTracers
from tracers import do_parallel, Tracer

################################################################################

path = "new_files"

n_batches = 3

r_min = 500
r_max = 2e3

nr = 5 # per batch
nphi = 36
ntheta = 9

t_start = 2e3
t_end = 0
r_end = 300

max_dt = 100
atmo_cut = 0.9
rand_seed = 42
mem_mb = 6e3 # 6GB
verbose = True

################################################################################

# batch configuration

try:
    i_pr = int(os.environ["SLURM_PROCID"])
    n_pr = int(os.environ["SLURM_NPROCS"])
    n_cpu = int(os.environ["SLURM_CPUS_PER_TASK"])
    if i_pr == 0:
        print(f"Detected SLURM setup: NPROCS={n_pr} CPUS_PER_TASK={n_cpu}", flush=True)
except KeyError:
    i_pr = 0
    n_pr = 1
    n_cpu = 1

if len(sys.argv) > 1:
    batch_id = int(sys.argv[1])
else:
    batch_id = 0


if n_pr > 1:
    outf = open(f"tracer_out_{i_pr+batch_id*n_pr}.txt", "w")
    def close_out():
        outf.flush()
        outf.close()
    atexit.register(close_out)
else:
    outf = sys.stdout

if (i_pr == 0) and (n_pr > 1):
    print(f"Running batch {batch_id}", flush=True)
print(f"Running batch {batch_id}", flush=True, file=outf)

sleep(2)

r_batch = np.linspace(r_min**3, r_max**3, n_batches+1)**(1/3)
r_min = r_batch[batch_id]
r_max = r_batch[batch_id+1]

################################################################################

# setup of seeds

time = np.array([t_start])
dt = 0

rr = np.linspace(r_min**3, r_max**3, nr+1)**(1/3)
dr = np.diff(rr)
rr = rr[:-1]
rr += dr/2

phi = np.linspace(0, 2*np.pi, nphi+1)[:-1]
dphi = phi[1] - phi[0]
phi += dphi/2

cost = np.linspace(0, 1, ntheta+1)[:-1]
dct = cost[1] - cost[0]
cost += dct/2

# time first so tracers with similar start times are started on the same rank
time, rr, phi, cost = np.meshgrid(time, rr, phi, cost, indexing='ij')
dr = dr[None, :, None, None]

dvol = rr**2 * dr*dct*dphi
if dt>0:
    dvol *= dt

################################################################################

rng = np.random.default_rng(rand_seed)

time += rng.uniform(-0.5, 0.5, size=time.shape)*dt
rr += rng.uniform(-0.5, 0.5, size=rr.shape)*dr
phi += rng.uniform(-0.5, 0.5, size=phi.shape)*dphi
cost += rng.uniform(-0.5, 0.5, size=cost.shape)*dct

time = time.flatten()
rr = rr.flatten()
phi = phi.flatten()
cost = cost.flatten()
dvol = dvol.flatten()

sint = np.sqrt(1 - cost**2)
seeds = dict(
    x3=rr*cost,
    x2=rr*np.sin(phi)*sint,
    x1=rr*np.cos(phi)*sint,
    time=time,
    dvol=dvol,
)

n_tr = len(rr)
i_off = np.array_split(np.arange(n_tr), n_pr)[i_pr][0]
i_end = np.array_split(np.arange(n_tr), n_pr)[i_pr][-1]
print(f"Calculating tracers {i_off}-{i_end} on rank {i_pr}", flush=True)
if n_pr > 1:
    perm = rng.permutation(n_tr)
    seeds = {k: np.array_split(v[perm], n_pr)[i_pr] for k, v in seeds.items()}

################################################################################

# domain bounderies
def oob(t: float, x: np.ndarray, *_) -> float:
    r = np.sqrt(np.sum(x**2))
    return r - r_end
oob.direction = -1
oob.terminal = True

def oot(t: float, *_) -> float:
    return t - t_end

oot.direction = -1
oot.terminal = True

def check_flag(tr: Tracer) -> Tracer:
    if np.any(tr.trajectory['rFlag'] < atmo_cut):
        tr.finished = tr.failed = True
        tr.message = f"rFlag < {atmo_cut}"
    return tr

################################################################################

trs = AthdfTracers(
    path,
    data_keys=['rho', 'rYE', 'rENT', 'rAbar', 'rFlag', 'Temperature', "h", "u_t", "user_out_var3"],
    seeds=seeds,
    n_cpu=n_cpu,
    verbose=verbose,
    max_step=max_dt,
    t_int_order='linear',
    end_conditions=[check_flag,],
    use_shared_memory=mem_mb,
    spherical=True,
    bitant=True,
    events=[oob, oot],
    index_offset=i_off+batch_id*n_tr,
    outf=outf,
)

trs.integrate()


def output(tr):
    tr.props['rho0'] = tr['rho'][0]
    tr.props['mass'] = tr['rho'][0]*tr.props['dvol']

    #unscramble input
    if "user_out_var3" in tr.trajectory:
        tr.trajectory['Edot'] = tr.trajectory['Temperature']
        tr.trajectory['Temperature'] = tr.trajectory['h']
        tr.trajectory['h'] = tr.trajectory['u_t']
        tr.trajectory['u_t'] = tr.trajectory['user_out_var3']
        del tr.trajectory['user_out_var3']

    tr.output_to_ascii("trajectories/tracer_")

do_parallel(
    output,
    trs,
    n_cpu=1,
    desc='Outputting',
    unit='tracer',
    verbose=verbose,
    file=outf,
)
print(f"Done ::)", flush=True, file=outf)
print(f"Proc {i_pr} done", flush=True)
