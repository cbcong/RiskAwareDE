"""Run the inspected official notebook headlessly and check our 3D adaptation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PPoly
from scipy.stats import binom
import cma

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vp-sto"))
from adapter import plan, binomial_threshold, bezier_polyline, collision_bits
from vpsto.obf import OBF


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def official_demo(out):
    source = HERE / "vp-sto/examples/2D_collision_avoidance_set_final_position.ipynb"
    book = json.loads(source.read_text(encoding="utf-8"))
    ns = {"__name__": "__official_demo__"}
    original = cma.CMAEvolutionStrategy
    def seeded(mean, sigma, options):
        options = dict(options, seed=9, verb_log=0)
        return original(mean, sigma, options)
    t0 = time.perf_counter()
    with patch("cma.CMAEvolutionStrategy", seeded):
        for idx in range(6):
            cell = book["cells"][idx]
            if cell["cell_type"] != "code":
                continue
            exec(compile("".join(cell["source"]), f"official_notebook_cell_{idx}", "exec"), ns)
            if idx == 3:
                ns["opt"].verbose = False
    elapsed = time.perf_counter()-t0
    pos, vel, acc = ns["pos"], ns["vel"], ns["acc"]
    result = {
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "changes": ["Headless execution of code cells 0-5", "CMA random seed fixed at 9", "Progress printing disabled"],
        "seconds": elapsed, "native_duration": float(ns["sol"].T_best),
        "native_cost": float(ns["sol"].c_best),
        "sampled_polyline_intersection_length": float(ns["env"].getTrajDist(pos)),
        "endpoint_max_error": float(max(np.abs(pos[0]-ns["q0"]).max(), np.abs(pos[-1]-ns["qd"]).max())),
        "max_abs_velocity": np.abs(vel).max(axis=0),
        "max_abs_acceleration": np.abs(acc).max(axis=0),
        "note": "Official example collision metric is its sampled polyline; not a continuous-curve certificate.",
    }
    np.savez_compressed(out / "official_2d_demo.npz", pos=pos, vel=vel, acc=acc,
                        weights=ns["sol"].w_best, duration=ns["sol"].T_best)
    fig, ax = plt.subplots(figsize=(5,5))
    from matplotlib.patches import Polygon
    for poly in ns["env"].poly_list:
        ax.add_patch(Polygon(poly, color="#b6bcc5"))
    ax.plot(pos[:,0], pos[:,1], color="#1561d8", lw=2)
    ax.scatter(pos[[0,-1],0], pos[[0,-1],1], c=["green","red"])
    ax.set(xlim=(0,.5),ylim=(0,.5),aspect="equal")
    fig.savefig(out / "official_2d_demo.png",dpi=180)
    plt.close(fig)
    return result


def validate_math():
    checks = []
    for n in [16, 50, 100, 1000]:
        for eta in [.05,.1,.175,.4]:
            k = binomial_threshold(n, eta, .05)
            assert k < 0 or binom.cdf(k,n,eta) <= .05
            assert k >= n or binom.cdf(k+1,n,eta) > .05
    checks.append("Binomial threshold maximality over 16 sample/risk combinations")
    boxes = np.array([[[[.4,-.1,-.1],[.6,.1,.1]]]])
    assert collision_bits(np.array([[0,0,0],[1,0,0]]), boxes)[0]
    assert not collision_bits(np.array([[0,.2,0],[1,.2,0]]), boxes)[0]
    assert collision_bits(np.array([[0,.1,0],[1,.1,0]]), boxes)[0]
    checks.append("Continuous segment crossing, parallel miss, and tangent contact")
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--budget", type=float, default=3.)
    args = parser.parse_args()
    out = HERE / "validation"
    out.mkdir(exist_ok=True)
    result = {"math_checks": validate_math()}
    if not args.skip_official:
        result["official_demo"] = official_demo(out)
    scene = {
        "start": [1,1,1], "goal": [19,19,8],
        "bounds": {"minimum": [0,0,0], "maximum": [20,20,10]},
        "obstacles": [
            {"minimum": [7,6,0], "maximum": [10,10,7]},
            {"minimum": [12,11,0], "maximum": [15,15,8]},
        ],
    }
    planned = plan(scene["start"],scene["goal"],scene["obstacles"],scene["bounds"],seed=9,budget_seconds=args.budget)
    coef, breaks = planned["coefficients"], planned["breaks"]
    native = PPoly(coef, breaks)
    obf = OBF(3)
    obf.setup_task(np.diff(breaks))
    phases = np.linspace(0,1,503)
    from_basis = (obf.get_Phi(phases) @ planned["native_weights"]).reshape(-1,3)
    error = float(np.abs(native(phases)-from_basis).max())
    assert error < 1e-9
    endpoint_error = float(max(np.abs(native(0)-scene["start"]).max(), np.abs(native(1)-scene["goal"]).max()))
    assert endpoint_error < 1e-10
    from adapter import as_boxes
    rng = np.random.default_rng(91019)
    boxes = as_boxes(scene["obstacles"])
    worlds = boxes[None] + rng.normal(0,.6,(10000,len(boxes),1,3))
    risk = collision_bits(planned["points"],worlds,planned["stats"]["export_curve_error_bound"]).mean()
    result["demo_3d"] = {"scene": scene, "stats": planned["stats"],
                         "success": planned["success"], "independent_mc_samples": 10000,
                         "independent_cfr": float(1-risk),
                         "polynomial_vs_official_basis_max_error": error,
                         "endpoint_max_error": endpoint_error,
                         "seed_isolated_from_training": True}
    np.savez_compressed(out/"demo_3d.npz",coefficients=coef,breaks=breaks,
                        points=planned["points"],weights=planned["native_weights"],duration=planned["duration"])
    (out/"validation.json").write_text(json.dumps(result,indent=2,default=json_default),encoding="utf-8")
    print(json.dumps(result,default=json_default,indent=2))


if __name__ == "__main__":
    main()
