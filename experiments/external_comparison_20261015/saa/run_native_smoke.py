"""Bounded numerical smoke of the authors' drone model, not paper replication.

Original model/constraints/objective/SCP are executed from the MIT upstream source.
Only compatibility substitutions and counts stated in native_smoke.json differ.
"""
from pathlib import Path
import os, sys, json, re, hashlib, time, platform
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
BASE = Path(__file__).resolve().parent
DRONE = BASE / 'upstream' / 'drone'
sys.path.insert(0, str(DRONE))
import numpy as np
import jax, osqp
from jax import config
config.update('jax_enable_x64', True)
import drone_params
drone_params.M = 8
source = (DRONE / 'drone_risk.py').read_text()
source = source.split('# -----------------------------------------\n# sample uncertain parameters')[0]
source = source.replace('from jax.config import config', 'from jax import config')
source = re.sub(r'(\b[A-Za-z_]\w*)\.to_py\(\)', r'np.asarray(\1)', source)
source = source.replace('linsys_solver="qdldl",', '')
source = source.replace("rc('text', usetex=True)", "rc('text', usetex=False)")
(BASE / 'native_compat_model.py').write_text(source, encoding='utf-8')
ns = {'__name__': 'native_compat_model', '__file__': str(DRONE/'drone_risk.py')}
exec(compile(source, str(BASE/'native_compat_model.py'), 'exec'), ns)
from drone_utils import sample_uncertain_parameters
np.random.seed(0)
dw, masses, qs = sample_uncertain_parameters('saa', M=8)
model = ns['Model'](drone_params.S, dw, masses, qs, method='saa', alpha=.1)
start = time.perf_counter()
us = model.initial_guess_us_mat()
model.define_problem(us, verbose=False)
setup = time.perf_counter()-start
trace = []
for iteration in range(20):
    ts = time.perf_counter()
    model.update_problem(us, scp_iter=iteration, verbose=False)
    new_us, threshold = model.solve(verbose=False)
    status = model.res.info.status
    rec = {'iteration': iteration, 'status': status,
           'seconds': time.perf_counter()-ts,
           'slack': float(model.res.x[-2]),
           'control_change': float(np.linalg.norm(np.asarray(new_us)-np.asarray(us)))}
    trace.append(rec)
    print(json.dumps(rec), flush=True)
    if status not in ('solved', 'solved inaccurate') or not np.all(np.isfinite(new_us)):
        break
    us = new_us
xs = np.asarray(model.us_to_state_trajectories(us))
constraints = np.array([model.obstacle_avoidance_constraints(x, q) for x,q in zip(xs,qs)])
losses = np.max(constraints, axis=(1,2))
tail_alpha = .1
thresholds = np.r_[losses.min()-1, losses]
avar = min(float(t + np.maximum(losses-t, 0).mean()/tail_alpha) for t in thresholds)
np.savez_compressed(BASE/'native_smoke_paths.npz', controls=np.asarray(us), states=xs,
                    obstacle_matrices=np.asarray(qs), sampled_losses=losses)
report = {'classification': 'reduced-count official drone numerical smoke, not full paper replication',
          'upstream_commit': '59b0ad029508c357b31960cc2088d973531a5f7c', 'license':'MIT',
          'original_counts': {'samples':50,'repeats_per_alpha':30,'alphas':[.05,.1,.2,.3],'scp_iterations':60},
          'actual_counts': {'samples':8,'repeats':1,'alpha':.1,'scp_iterations':len(trace)},
          'unchanged': ['S=20','T=50','nonlinear six-state dynamics','uncertain mass and obstacle shape','Brownian disturbance','full-horizon AVaR epigraph','OSQP SCP','control-squared objective'],
          'compatibility_changes':['jax.config import location','Array.to_py -> numpy.asarray','remove legacy OSQP linsys_solver string; default direct solver','headless/no plots'],
          'task_note':'Obstacle avoidance in original code uses x[:2] and elliptic cylinders; this is not the common 3D-box task.',
          'versions':{'python':platform.python_version(),'jax':jax.__version__,'osqp':osqp.__version__,'numpy':np.__version__},
          'setup_including_JIT_seconds':setup,'total_seconds':time.perf_counter()-start,
          'trace':trace,'final_sample_AVaR':avar,
          'mean_final_state_error':float(np.linalg.norm(xs[:,-1,:].mean(axis=0)-np.asarray(drone_params.x_final))),
          'finite':bool(np.isfinite(xs).all()),
          'source_sha256':hashlib.sha256((DRONE/'drone_risk.py').read_bytes()).hexdigest()}
(BASE/'native_smoke.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
print(json.dumps({k:report[k] for k in ['total_seconds','final_sample_AVaR','mean_final_state_error','finite']}), flush=True)
