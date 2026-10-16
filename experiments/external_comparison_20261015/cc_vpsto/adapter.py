"""CC-VPSTO-3D geometric adaptation, not an official CC-VPSTO release.

Retains official VP-STO OBF trajectory basis/smoothness preconditioning and
active CMA-ES. Implements IJRR 2026 Eq. (10), joint collision counts, and
Sec. 5.2's discontinuous penalty. Objective here is spatial arc length.
The fixed optimization bank is reused, so padding is heuristic, not a
post-selection safety certificate. All uncertainty draws are joint worlds.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binom
import cma

sys.path.insert(0, str(Path(__file__).parent / "vp-sto"))
_err_before_import = np.geterr()
from vpsto.vptraj import VPTraj
from vpsto.obf import OBF
np.seterr(**_err_before_import)  # upstream import changes global error handling


def binomial_threshold(n_samples: int, eta: float, beta: float) -> int:
    """Largest k with Pr(Binomial(N,eta)<=k)<=beta; -1 means infeasible."""
    if n_samples < 1 or not 0 < eta < 1 or not 0 < beta < 1:
        raise ValueError("N>=1 and eta,beta strictly between zero and one required")
    acceptable = np.flatnonzero(binom.cdf(np.arange(n_samples + 1), n_samples, eta) <= beta)
    return int(acceptable[-1]) if len(acceptable) else -1


def as_boxes(boxes):
    if isinstance(boxes, (list, tuple)) and boxes and isinstance(boxes[0], dict):
        return np.array([[b["minimum"], b["maximum"]] for b in boxes], dtype=float)
    return np.asarray(boxes, dtype=float).reshape(-1, 2, 3)


def as_bounds(bounds):
    if isinstance(bounds, dict):
        return np.array([bounds["minimum"], bounds["maximum"]], dtype=float)
    return np.asarray(bounds, dtype=float).reshape(2, 3)


def bezier_polyline(coefficients, breaks, epsilon=0.01):
    """Cubic PPoly -> chords with a provable pointwise Euclidean error bound.

    The linear chord is represented by cubic Bezier controls. The maximum
    control-vector difference bounds the entire cubic difference by convexity.
    Split by de Casteljau until every segment has bound <= epsilon.
    """
    coefficients = np.asarray(coefficients)
    output = []
    worst = 0.0
    for i, h in enumerate(np.diff(breaks)):
        c3, c2, c1, c0 = coefficients[:, i, :]
        p3 = ((c3 * h + c2) * h + c1) * h + c0
        d3 = (3 * c3 * h + 2 * c2) * h + c1
        controls = np.array([c0, c0 + h * c1 / 3, p3 - h * d3 / 3, p3])
        stack = [(controls, 0)]
        while stack:
            b, depth = stack.pop()
            chord = np.array([b[0], (2*b[0]+b[3])/3, (b[0]+2*b[3])/3, b[3]])
            err = float(np.linalg.norm(b - chord, axis=1).max())
            if err <= epsilon:
                if not output:
                    output.append(b[0])
                output.append(b[3])
                worst = max(worst, err)
            else:
                if depth >= 24:
                    raise RuntimeError("Cubic subdivision depth exceeded")
                a = (b[:-1] + b[1:]) / 2
                q = (a[:-1] + a[1:]) / 2
                m = (q[0] + q[1]) / 2
                stack.append((np.array([m, q[1], a[2], b[3]]), depth+1))
                stack.append((np.array([b[0], a[0], q[0], m]), depth+1))
    return np.array(output), worst


def collision_bits(points, shifted_boxes, inflation=0.0):
    """Exact closed segment/AABB intersection, vectorized over independent worlds.

    shifted_boxes shape (worlds, obstacles, 2, 3). A world is counted once.
    Inflation covers the certified curve/chord discrepancy conservatively.
    """
    points = np.asarray(points, dtype=float)
    worlds = np.asarray(shifted_boxes, dtype=float)
    n = len(worlds)
    if worlds.shape[1] == 0:
        return np.zeros(n, dtype=bool)
    lo = worlds[:, :, 0] - inflation
    hi = worlds[:, :, 1] + inflation
    hit = np.zeros(n, dtype=bool)
    for offset in range(0, len(points)-1, 12):
        p = points[offset:min(offset+12, len(points)-1)]
        d = points[offset+1:min(offset+13, len(points))] - p
        parallel = np.abs(d) < 1e-14
        inv = np.divide(1.0, d, out=np.zeros_like(d), where=~parallel)
        a = (lo[:, :, None, :] - p[None, None, :, :]) * inv
        b = (hi[:, :, None, :] - p[None, None, :, :]) * inv
        enter = np.where(parallel, -np.inf, np.minimum(a, b)).max(axis=-1)
        leave = np.where(parallel, np.inf, np.maximum(a, b)).min(axis=-1)
        outside = np.any(parallel & ((p[None, None] < lo[:, :, None]) |
                                      (p[None, None] > hi[:, :, None])), axis=-1)
        hits = (np.maximum(enter, 0) <= np.minimum(leave, 1)) & ~outside
        hit |= np.any(hits, axis=(1, 2))
        if hit.all():
            break
    return hit


def _cubic_coefficients(weights, phi_knots, dphi_knots, n_via):
    q = (phi_knots @ weights).reshape(n_via+1, 3)
    v = (dphi_knots @ weights).reshape(n_via+1, 3)
    h = 1/n_via
    c3 = (2*q[:-1] - 2*q[1:] + h*(v[:-1]+v[1:])) / h**3
    c2 = (-3*q[:-1] + 3*q[1:] - h*(2*v[:-1]+v[1:])) / h**2
    return np.array([c3, c2, v[:-1], q[:-1]])


def plan(start, goal, boxes, bounds, seed=0, budget_seconds=1.0, sigma=0.6,
         n_samples=100, eta=0.175, beta=0.05, n_via=5, pop_size=25,
         collision_epsilon=0.02, export_epsilon=0.001, sigma_init=None,
         max_generations=1000, velocity_limit=2.0, acceleration_limit=2.0):
    """Plan one point-robot path under independent Gaussian box translations.

    n_via is the number of spline pieces (n_via-1 free via points).
    Time includes basis construction, sampling, optimization and export.
    Stops between candidate evaluations; a single candidate/export may overshoot.
    Returns normalized-phase PPoly coefficients (4,pieces,3), breaks and stats.
    """
    begin = time.perf_counter()
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    boxes = as_boxes(boxes)
    bounds = as_bounds(bounds)
    if budget_seconds <= 0 or n_via < 2:
        raise ValueError("Positive budget and at least two spline pieces required")
    k_thresh = binomial_threshold(n_samples, eta, beta)
    if k_thresh < 0:
        raise ValueError("Sample count too small for requested binomial padding")
    rng_bank = np.random.default_rng(np.random.SeedSequence([int(seed), 47713]))
    rng_opt = np.random.default_rng(np.random.SeedSequence([int(seed), 47717]))
    shifts = rng_bank.normal(0, sigma, size=(n_samples, len(boxes), 1, 3))
    worlds = boxes[None, :, :, :] + shifts
    trajectory = VPTraj(3, 100, n_via,
                        np.full(3, velocity_limit), np.full(3, acceleration_limit))
    ddphi_b = np.concatenate((trajectory.ddPhi[:, :3], trajectory.ddPhi[:, -9:]), axis=1)
    mu_p = -trajectory.ddPhi_p @ ddphi_b @ np.concatenate((start, goal, np.zeros(6)))
    smooth_chol = trajectory.S_chol
    obf = OBF(3)
    breaks = np.linspace(0, 1, n_via+1)
    obf.setup_task(np.diff(breaks))
    phi_knots, dphi_knots = obf.get_Phi(breaks), obf.get_dPhi(breaks)
    scale = float(np.max(bounds[1]-bounds[0]))
    sigma_init = float(3*scale if sigma_init is None else sigma_init)
    # Feasible lengths are bounded by the variation of bounded cubic pieces;
    # this barrier is several orders of magnitude larger than workspace scale.
    penalty_min, penalty_step = 1e5 * scale, 1e3 * scale
    es = cma.CMAEvolutionStrategy(np.zeros(3*(n_via-1)), sigma_init, {
        "verbose": -9, "verb_log": 0, "CMA_active": True, "popsize": pop_size,
        "CMA_diagonal": False, "tolfun": 1e-6,
        "randn": lambda *shape: rng_opt.standard_normal(shape),
        "seed": max(1, int(seed) % (2**31-1)),
    })
    best = None
    history = []
    count, generations = 0, 0

    def evaluate(x):
        nonlocal best, count
        p = mu_p + smooth_chol @ x
        w = np.concatenate((start, p, goal, np.zeros(6)))
        coef = _cubic_coefficients(w, phi_knots, dphi_knots, n_via)
        chords, err = bezier_polyline(coef, breaks, collision_epsilon)
        length = float(np.linalg.norm(np.diff(chords, axis=0), axis=1).sum())
        k = int(collision_bits(chords, worlds, err).sum())
        nominal = bool(collision_bits(chords, boxes[None], err)[0])
        # Extrema of each cubic coordinate give exact workspace containment.
        outside = 0.0
        for j, h in enumerate(np.diff(breaks)):
            for axis in range(3):
                c = coef[:, j, axis]
                roots = np.roots([3*c[0], 2*c[1], c[2]])
                t = [0.0, h] + [float(r.real) for r in roots
                                if abs(r.imag) < 1e-10 and 0 < r.real < h]
                values = np.polyval(c, t)
                outside += max(0.0, bounds[0, axis]-values.min())
                outside += max(0.0, values.max()-bounds[1, axis])
        boundary_fail = outside > 1e-9
        # Sec. 5.2 exact count penalty; deterministic constraints are separate.
        risk_penalty = penalty_min + penalty_step*(k-k_thresh-1) if k > k_thresh else 0.0
        hard_penalty = (penalty_min + penalty_step*(int(nominal)+outside)) if nominal or boundary_fail else 0.0
        cost = length + risk_penalty + hard_penalty
        count += 1
        rec = {"x": x.copy(), "p": p, "weights": w, "coefficients": coef,
               "length": length, "count": k, "nominal_collision": nominal,
               "boundary_violation": float(outside), "cost": float(cost)}
        if best is None or cost < best["cost"]:
            best = rec
            history.append({"seconds": time.perf_counter()-begin, "evaluations": count,
                            "cost": cost, "length": length, "collision_count": k})
        return cost

    evaluate(np.zeros(3*(n_via-1)))
    while generations < max_generations and not es.stop():
        if time.perf_counter()-begin >= budget_seconds:
            break
        population = es.ask()
        costs = []
        for x in population:
            if time.perf_counter()-begin >= budget_seconds:
                break
            costs.append(evaluate(np.asarray(x)))
        if len(costs) < len(population):
            break
        es.tell(population, costs)
        generations += 1
    optimize_finished = time.perf_counter()
    points, export_error = bezier_polyline(best["coefficients"], breaks, export_epsilon)
    with np.errstate(invalid="ignore"):
        duration = float(trajectory.get_min_duration(best["p"], start, np.zeros(3), goal, np.zeros(3))[0])
    elapsed = time.perf_counter()-begin
    surrogate_ok = best["count"] <= k_thresh
    deterministic_ok = not best["nominal_collision"] and best["boundary_violation"] <= 1e-9
    return {
        "method": "CC-VPSTO-3D geometric adaptation",
        "points": points, "coefficients": best["coefficients"], "breaks": breaks,
        "coefficient_parameter": "normalized_phase_0_to_1",
        "duration": duration, "native_weights": best["weights"],
        "success": bool(surrogate_ok and deterministic_ok),
        "stats": {"wall_seconds": elapsed, "optimization_seconds": optimize_finished-begin,
                  "budget_seconds": budget_seconds, "budget_overrun_seconds": max(0.0, elapsed-budget_seconds),
                  "objective_evaluations": count, "complete_generations": generations,
                  "training_collision_count": best["count"], "n_samples": n_samples,
                  "eta": eta, "beta": beta, "k_threshold": k_thresh,
                  "training_surrogate_feasible": bool(surrogate_ok),
                  "nominal_collision": best["nominal_collision"],
                  "boundary_violation": best["boundary_violation"],
                  "approximate_length": float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
                  "optimization_curve_error": collision_epsilon,
                  "export_curve_error_bound": export_error,
                  "seed": int(seed), "n_via": n_via, "sigma_init": sigma_init,
                  "pop_size": pop_size, "sigma": sigma,
                  "safety_certificate": False,
                  "status": "surrogate_feasible" if surrogate_ok and deterministic_ok else "no_feasible_candidate",
                  "history": history},
    }
