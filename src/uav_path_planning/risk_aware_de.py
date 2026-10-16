"""Chance-style, feasibility-first DE candidate for robust 3-D planning.

This module is deliberately separate from :mod:`uad_de`.  UAD changes DE
parameters from a nominal geometric risk signal; this candidate changes the
selection relation itself.  Its default candidate is the chance-budgeted
selection relation described below; strict lexicographic and nominal controls
are retained for ablation.

The default ``risk_budgeted`` relation is a chance-constraint gate: nominal
feasibility is checked first, then candidates violating ``chance_alpha`` are
ordered by violation, while candidates that both satisfy the gate are ordered
only by nominal objective.  This preserves the nominal DE objective after the
risk budget has been met.  The chance term is an empirical collision
probability under a *frozen* bank of Gaussian obstacle translations.  The bank
is generated once per optimisation run and shared by every candidate, so it
does not introduce decision-dependent randomness.  It is a synthetic
robustness proxy, not a calibrated probability of safety in a real
environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Sequence

from .continuous_planner import ContinuousPathProblem
from .de import DEConfig
from .metrics import PathMetrics
from .models import Environment3D, ObstacleBox, Path3D


def _wilson_upper_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    """One-sided Wilson upper bound for a Bernoulli rate.

    The optimizer uses this only as a conservative gate.  It is intentionally
    kept independent of SciPy so benchmark runs have no hidden dependency.
    ``confidence`` is the coverage level of the two-sided Wilson interval;
    the upper endpoint is used as a one-sided conservative score.
    """
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("invalid Bernoulli count")
    # 95% normal quantile.  The value is fixed to keep results reproducible
    # across platforms and to avoid a statistical-library dependency.
    if confidence != 0.95:
        raise ValueError("only 0.95 Wilson confidence is supported")
    z = 1.959963984540054
    n = float(trials)
    p = float(successes) / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return min(1.0, center + radius)


@dataclass(frozen=True)
class RiskAwareDEConfig:
    """Configuration for the independent risk-aware DE candidate."""

    base: DEConfig = field(default_factory=DEConfig)
    perturbation_sigma: float = 0.0
    # Optional multi-scale uncertainty bank.  When provided, one frozen bank
    # is generated for each sigma and the configured aggregation rule combines
    # their empirical rates.  The default ``None`` preserves the single-scale
    # protocol used by earlier checkpoints.
    perturbation_sigmas: tuple[float, ...] | None = None
    perturbation_samples: int = 8
    risk_aggregation: str = "pooled"
    chance_alpha: float = 0.10
    # Optional risk-budget continuation.  ``fixed`` is the locked protocol
    # used by the original method.  ``linear`` starts from ``risk_gate_start``
    # and contracts to ``chance_alpha`` as the evaluation budget is consumed,
    # including local-refinement evaluations.  Generation progress is also
    # respected when the generation limit terminates a run before its budget.
    # The final population is re-ranked at ``chance_alpha`` before return.
    risk_gate_schedule: str = "fixed"
    risk_gate_start: float | None = None
    perturbation_seed: int | None = None
    selection_mode: str = "risk_budgeted"
    # ``nominal_guarded`` additionally limits the relative nominal cost of a
    # risk improvement, for both mutation and local refinement.  This mode
    # uses nominal_tolerance in place of the historical mutation-only
    # risk_tradeoff guard below.
    nominal_tolerance: float = 0.0
    # A bounded risk trade-off: among nominally feasible candidates, allow
    # mutation risk improvements to win only when their normalized objective
    # increase stays below this fraction.  Zero is an active zero-increase
    # guard, NOT a disabled guard.  Historical local refinement uses its own
    # chance-class guard and does not consult this parameter.  This preserves
    # the fixed-gate checkpoints; use nominal_guarded for a guard on both
    # update types.  None explicitly disables this mutation cost cap for its
    # ablation, without disabling nominal feasibility or risk-key selection.
    risk_tradeoff: float | None = 0.0
    # ``rand1`` preserves the original v1 search.  ``current_to_pbest1`` is
    # the JADE-style current-to-pbest/1 mutation, included as an explicitly
    # ablatable search option rather than a claim of a new mutation rule.
    mutation_strategy: str = "rand1"
    p_best_rate: float = 0.20
    # ``risk_first`` reproduces the locked p-best ordering.  ``risk_diverse``
    # forms a two-elite p-best pool: one half is selected by nominal length
    # among risk-feasible paths and the other half by lowest observed risk.
    # This keeps a short feasible basin available without removing the safe
    # basin from the current-to-pbest/1 donor pool.
    pbest_selection: str = "risk_first"
    # Fraction of the p-best pool reserved for nominal-length elites when
    # ``pbest_selection='risk_diverse'``.  The remainder is filled by the
    # safest feasible/ranked individuals.
    pbest_length_fraction: float = 0.50
    archive_size: int | None = None
    # Optional success-history adaptation of F and CR.  ``fixed`` preserves
    # the locked protocol.  ``risk_success`` adapts the next generation's
    # parameters from accepted trials, weighting risk reductions in addition
    # to nominal objective improvements.  It changes no evaluation budget.
    parameter_adaptation: str = "fixed"
    adaptation_learning_rate: float = 0.10
    # Optional short-circuit: nominally infeasible candidates already lose
    # before the chance relation is consulted, so their perturbed checks can
    # be skipped.  The reported internal_collision_checks then reflects the
    # actual checks, making this an auditable runtime ablation.
    skip_risk_for_nominal_infeasible: bool = False
    # Optional conservative local refinement.  A candidate is perturbed
    # toward the straight start-goal interpolation only when its empirical
    # risk class is unchanged or improved.  This is an ablatable mechanism,
    # not a hidden post-processing step.
    local_refinement: bool = False
    local_refinement_probability: float = 0.15
    local_refinement_scale: float = 0.08
    # Used only by adaptive_partial_shortcut. Each population slot owns a
    # contraction scale, initially local_refinement_scale, retained even
    # when mutation replaces its path. Local inadmissibility shrinks it;
    # accepted local proposals expand it up to the initial scale. No inner
    # backtracking evaluations or additional random draws are performed.
    local_refinement_min_scale: float = 0.05
    local_refinement_shrink: float = 0.5
    local_refinement_grow: float = 1.2
    # ``direct`` contracts waypoints toward the start-goal chord.  ``shortcut``
    # replaces one waypoint by a point on its neighboring chord; by the
    # triangle inequality this cannot increase nominal polyline length before
    # the normal risk-aware acceptance relation is applied.  ``greedy_shortcut``
    # chooses the waypoint with the largest local detour before applying the
    # same chord replacement. ``partial_shortcut`` moves the selected waypoint
    # a fixed fraction ``local_refinement_scale`` toward the chord point;
    # this fraction lies in (0, 1] and preserves the length-nonincreasing
    # property by convexity, while retaining more of the incumbent detour.
    # ``detour_weighted_partial`` samples the waypoint in proportion to its
    # removable local detour, whereas ``greedy_partial`` selects the largest
    # detour. Both use the same partial contraction and risk acceptance as
    # ``partial_shortcut``. These proposals use geometry only, without extra
    # collision checks or objective evaluations.
    # ``safe_shortcut`` uses the largest-detour
    # shortcut and additionally requires the empirical perturbation risk not
    # to increase during local refinement. ``mixed`` samples either operator
    # for a budget-neutral ablation.
    local_refinement_mode: str = "direct"

    def __post_init__(self) -> None:
        if not math.isfinite(self.perturbation_sigma) or self.perturbation_sigma < 0.0:
            raise ValueError("perturbation_sigma must be finite and non-negative")
        if self.perturbation_sigmas is not None:
            if not self.perturbation_sigmas:
                raise ValueError("perturbation_sigmas must be non-empty when provided")
            if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in self.perturbation_sigmas):
                raise ValueError("perturbation_sigmas must contain finite non-negative values")
        if isinstance(self.perturbation_samples, bool) or not isinstance(self.perturbation_samples, int) or self.perturbation_samples < 1:
            raise ValueError("perturbation_samples must be a positive integer")
        if self.risk_aggregation not in {"pooled", "max"}:
            raise ValueError("risk_aggregation must be 'pooled' or 'max'")
        if not math.isfinite(self.chance_alpha) or not 0.0 <= self.chance_alpha <= 1.0:
            raise ValueError("chance_alpha must lie in [0, 1]")
        if self.risk_gate_schedule not in {"fixed", "linear"}:
            raise ValueError("risk_gate_schedule must be 'fixed' or 'linear'")
        if self.risk_gate_start is not None and (
            not math.isfinite(self.risk_gate_start) or not 0.0 <= self.risk_gate_start <= 1.0
        ):
            raise ValueError("risk_gate_start must lie in [0, 1] or be None")
        if self.risk_gate_schedule == "linear" and self.risk_gate_start is not None and self.risk_gate_start < self.chance_alpha:
            raise ValueError("linear risk_gate_start must be at least chance_alpha")
        if self.perturbation_seed is not None and (
            isinstance(self.perturbation_seed, bool) or not isinstance(self.perturbation_seed, int)
        ):
            raise ValueError("perturbation_seed must be an integer or None")
        if self.selection_mode not in {"risk_budgeted", "ucb_budgeted", "nominal_guarded", "lexicographic", "lexicographic_risk", "nominal", "risk_budgeted_tiebreak"}:
            raise ValueError("selection_mode must be 'risk_budgeted', 'ucb_budgeted', 'nominal_guarded', 'lexicographic', 'lexicographic_risk', 'nominal', or 'risk_budgeted_tiebreak'")
        if (self.selection_mode == "ucb_budgeted" and self.risk_aggregation == "pooled"
                and self.perturbation_sigmas is not None and len(self.perturbation_sigmas) > 1):
            raise ValueError("ucb_budgeted cannot pool correlated multi-scale samples; use risk_aggregation='max'")
        if self.risk_tradeoff is not None and (not math.isfinite(self.risk_tradeoff) or self.risk_tradeoff < 0.0):
            raise ValueError("risk_tradeoff must be finite and non-negative or None")
        if not math.isfinite(self.nominal_tolerance) or self.nominal_tolerance < 0.0:
            raise ValueError("nominal_tolerance must be finite and non-negative")
        if not isinstance(self.mutation_strategy, str) or self.mutation_strategy not in {"rand1", "current_to_pbest1"}:
            raise ValueError("mutation_strategy must be 'rand1' or 'current_to_pbest1'")
        if not isinstance(self.p_best_rate, (int, float)) or isinstance(self.p_best_rate, bool):
            raise ValueError("p_best_rate must be a finite number in (0, 1]")
        if not math.isfinite(float(self.p_best_rate)) or not 0.0 < float(self.p_best_rate) <= 1.0:
            raise ValueError("p_best_rate must be a finite number in (0, 1]")
        if self.pbest_selection not in {"risk_first", "risk_diverse"}:
            raise ValueError("pbest_selection must be 'risk_first' or 'risk_diverse'")
        if (not math.isfinite(self.pbest_length_fraction)
                or not 0.0 <= self.pbest_length_fraction <= 1.0):
            raise ValueError("pbest_length_fraction must lie in [0, 1]")
        if self.archive_size is not None and (
            isinstance(self.archive_size, bool)
            or not isinstance(self.archive_size, int)
            or self.archive_size < 0
        ):
            raise ValueError("archive_size must be a non-negative integer or None")
        if self.parameter_adaptation not in {"fixed", "risk_success"}:
            raise ValueError("parameter_adaptation must be 'fixed' or 'risk_success'")
        if (not math.isfinite(self.adaptation_learning_rate)
                or not 0.0 < self.adaptation_learning_rate <= 1.0):
            raise ValueError("adaptation_learning_rate must lie in (0, 1]")
        if not isinstance(self.skip_risk_for_nominal_infeasible, bool):
            raise ValueError("skip_risk_for_nominal_infeasible must be a boolean")
        if not 0.0 <= self.local_refinement_probability <= 1.0:
            raise ValueError("local_refinement_probability must be in [0, 1]")
        if not math.isfinite(self.local_refinement_scale) or self.local_refinement_scale <= 0.0:
            raise ValueError("local_refinement_scale must be finite and positive")
        if self.local_refinement_mode not in {"direct", "shortcut", "partial_shortcut", "greedy_shortcut", "safe_shortcut", "mixed", "detour_weighted_partial", "greedy_partial", "adaptive_partial_shortcut"}:
            raise ValueError("unsupported local_refinement_mode")
        if self.local_refinement_mode in {"partial_shortcut", "detour_weighted_partial", "greedy_partial", "adaptive_partial_shortcut"} and self.local_refinement_scale > 1.0:
            raise ValueError("partial refinement local_refinement_scale must lie in (0, 1]")
        if self.local_refinement_mode == "adaptive_partial_shortcut":
            if not math.isfinite(self.local_refinement_min_scale) or not 0.0 < self.local_refinement_min_scale <= self.local_refinement_scale:
                raise ValueError("adaptive partial minimum scale must lie in (0, initial scale]")
            if not math.isfinite(self.local_refinement_shrink) or not 0.0 < self.local_refinement_shrink < 1.0:
                raise ValueError("adaptive partial shrink must lie in (0, 1)")
            if not math.isfinite(self.local_refinement_grow) or self.local_refinement_grow <= 1.0:
                raise ValueError("adaptive partial grow must exceed 1")


@dataclass(frozen=True)
class RiskEvaluation:
    nominal_feasible: bool
    perturbed_collision_rate: float
    chance_violation: float
    objective: float
    key: tuple[float, ...]
    # Wilson upper endpoint used as a search score, not a safety certificate
    # for an adaptively selected path.  Pooled multi-scale samples share CRN
    # draws, so their recorded endpoint has no binomial-coverage claim.
    risk_upper_bound: float = math.nan


@dataclass
class RiskAwareDEResult:
    best_vector: list[float]
    best_fitness: float
    history: list[float]
    evaluations: int
    internal_collision_checks: int
    best_evaluation: RiskEvaluation
    # The scalar objective is retained for compatibility with the benchmark
    # schema, but the tuple is the actual selection criterion.  A nominal
    # length/objective trace alone can therefore appear to increase while the
    # selection key improves through feasibility or chance risk.
    key_history: list[tuple[float, ...]] = field(default_factory=list)
    # Gate used to form each recorded population's selection key. This keeps
    # annealed runs auditable and makes final retargeting explicit.
    gate_history: list[float] = field(default_factory=list)
    # Generation-level parameter traces make the optional adaptation auditable
    # and support ablation tables without reconstructing RNG state.
    mutation_factor_history: list[float] = field(default_factory=list)
    crossover_rate_history: list[float] = field(default_factory=list)
    # Adaptive-partial-only feedback diagnostics. Historical modes keep
    # empty diagnostics, preserving their numerical results and RNG stream.
    local_refinement_stats: dict[str, float | int] = field(default_factory=dict)
    local_refinement_events: list[dict[str, float | int | bool]] = field(default_factory=list)


def _shifted_environments(
    problem: ContinuousPathProblem,
    config: RiskAwareDEConfig,
    sigma: float | None = None,
) -> list[Environment3D]:
    """Create one deterministic common-random-number perturbation bank."""

    # Keep environment uncertainty independent from the DE stream.  A fixed
    # explicit seed is recommended for paired experiments; the deterministic
    # fallback is only a convenience for one-off runs.
    planner_seed = config.base.seed if config.base.seed is not None else 0
    seed = config.perturbation_seed if config.perturbation_seed is not None else (planner_seed ^ 0x5EEDC0DE)
    rng = random.Random(seed)
    output: list[Environment3D] = []
    sigma_value = config.perturbation_sigma if sigma is None else float(sigma)
    for _ in range(config.perturbation_samples):
        obstacles = []
        for obstacle in problem.environment.obstacles:
            shift = tuple(rng.gauss(0.0, sigma_value) for _ in range(3))
            obstacles.append(ObstacleBox(
                tuple(a + d for a, d in zip(obstacle.minimum, shift)),
                tuple(b + d for b, d in zip(obstacle.maximum, shift)),
            ))
        output.append(Environment3D(problem.environment.bounds, obstacles))
    return output


def _better(left: RiskEvaluation, right: RiskEvaluation) -> bool:
    return left.key < right.key


class RiskAwareDE:
    """A risk-aware DE optimizer with auditable chance-budgeted selection.

    Each optimizer evaluation performs one nominal metric calculation and
    ``perturbation_samples`` collision checks.  ``internal_collision_checks``
    is returned explicitly so a paper can report the extra robustness cost
    instead of hiding it inside the usual DE evaluation count.  The default
    ``rand1`` mutation is the v1 behavior.  ``current_to_pbest1`` borrows the
    JADE current-to-pbest/1 rule and uses a generation-frozen historical
    archive for its second differential vector; both options use the fixed
    ``mutation_factor`` and ``crossover_rate`` from :class:`DEConfig`.
    """

    def __init__(self, bounds: Sequence[tuple[float, float]], problem: ContinuousPathProblem, config: RiskAwareDEConfig | None = None):
        if not bounds:
            raise ValueError("at least one decision variable is required")
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        if any(not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi for lo, hi in self.bounds):
            raise ValueError("every bound must be finite and lower < upper")
        self.problem = problem
        self.config = config or RiskAwareDEConfig()
        sigmas = self.config.perturbation_sigmas
        if sigmas is None:
            sigmas = (self.config.perturbation_sigma,)
        # Keep banks separate so ``max`` aggregation protects every declared
        # noise scale instead of hiding a high-noise failure in a pooled mean.
        # Banks share standardized Gaussian draws across scales (common
        # random numbers); they are fixed for the lifetime of this optimizer.
        # They must not be described as independent uncertainty samples.
        self._perturbed_banks = [
            _shifted_environments(problem, self.config, sigma=value)
            for value in sigmas
        ]
        self._perturbed = [environment for bank in self._perturbed_banks for environment in bank]
        self._internal_collision_checks = 0

    def _gate_alpha(self, completed: float, total: float) -> float:
        """Return the gate at a normalized continuation progress.

        The schedule changes only the ordering relation; collision outcomes
        themselves are always computed against the same frozen perturbation
        bank.  Zero progress is the initial population; full progress uses
        the locked final ``chance_alpha``.
        """
        if self.config.risk_gate_schedule == "fixed":
            return self.config.chance_alpha
        start = self.config.risk_gate_start
        if start is None:
            # A linear schedule with no explicit start is intentionally a
            # no-op.  This keeps the opt-in flag safe while making the start
            # value explicit in all reported experiments that anneal.
            start = self.config.chance_alpha
        denominator = max(1.0, float(total))
        progress = min(1.0, max(0.0, float(completed) / denominator))
        if progress >= 1.0:
            return self.config.chance_alpha
        if progress <= 0.0:
            return start
        return start + progress * (self.config.chance_alpha - start)

    def _search_gate(self, evaluations: int, budget: int, generation: int) -> float:
        """Gate for the evaluation whose inclusive count is ``evaluations``.

        Local search consumes the same budget as mutation.  Advancing only
        once per generation leaves a relaxed gate active at budget exhaustion.
        The generation fraction additionally handles a configuration whose
        generation cap prevents it from consuming its entire NFE allowance.
        """
        base = self.config.base
        nfe_progress = (evaluations - base.population_size) / max(1, budget - base.population_size)
        generation_progress = (generation + 1) / max(1, base.generations)
        return self._gate_alpha(max(nfe_progress, generation_progress), 1.0)

    def _retarget(self, evaluation: RiskEvaluation, chance_alpha: float) -> RiskEvaluation:
        """Rebuild a cached evaluation under a new risk gate.

        The expensive nominal and perturbed collision measurements are reused;
        only the derived violation/key changes.  This is required for an
        annealed gate because parent records from the preceding generation
        were ranked under a different alpha.
        """
        rate = evaluation.perturbed_collision_rate
        violation = max(0.0, rate - chance_alpha)
        ucb_violation = max(0.0, evaluation.risk_upper_bound - chance_alpha)
        if self.config.selection_mode in {"risk_budgeted", "nominal_guarded"}:
            key = (
                0.0 if evaluation.nominal_feasible else 1.0,
                0.0 if violation <= 0.0 else 1.0,
                violation if violation > 0.0 else 0.0,
                evaluation.objective,
            )
        elif self.config.selection_mode == "ucb_budgeted":
            key = (
                0.0 if evaluation.nominal_feasible else 1.0,
                0.0 if ucb_violation <= 0.0 else 1.0,
                ucb_violation if ucb_violation > 0.0 else 0.0,
                evaluation.objective,
            )
        elif self.config.selection_mode == "lexicographic":
            key = (0.0 if evaluation.nominal_feasible else 1.0, violation, evaluation.objective)
        elif self.config.selection_mode == "lexicographic_risk":
            key = (0.0 if evaluation.nominal_feasible else 1.0, violation, rate, evaluation.objective)
        elif self.config.selection_mode == "risk_budgeted_tiebreak":
            # A deterministic risk tie-break after the chance budget is met.
            # This differs from ``risk_budgeted`` only among candidates that
            # are both within the budget, where the observed rate is used as
            # a safety tie-break before nominal objective.  It exposes a
            # specific safety-efficiency mechanism for ablation.
            key = (
                0.0 if evaluation.nominal_feasible else 1.0,
                0.0 if violation <= 0.0 else 1.0,
                violation if violation > 0.0 else 0.0,
                rate if violation <= 0.0 else 0.0,
                evaluation.objective,
            )
        else:
            key = (0.0 if evaluation.nominal_feasible else 1.0, 0.0, evaluation.objective)
        return RiskEvaluation(
            evaluation.nominal_feasible, rate, violation, evaluation.objective, key,
            evaluation.risk_upper_bound,
        )

    def _evaluate(self, vector: Sequence[float], chance_alpha: float | None = None) -> RiskEvaluation:
        gate = self.config.chance_alpha if chance_alpha is None else float(chance_alpha)
        if not math.isfinite(gate) or not 0.0 <= gate <= 1.0:
            raise ValueError("chance_alpha gate must lie in [0, 1]")
        path = self.problem.decode(vector)
        metrics: PathMetrics = self.problem.evaluate(path)
        if self.config.skip_risk_for_nominal_infeasible and not metrics.feasible:
            # A skipped risk calculation is a conservative sentinel, not a
            # measurement.  Use the same key constructor as measured paths so
            # cached and freshly evaluated records agree even at alpha=1.
            measured = RiskEvaluation(False, 1.0, 0.0, float(metrics.objective), (), 1.0)
            return self._retarget(measured, gate)
        bank_counts: list[tuple[int, int]] = []
        for bank in self._perturbed_banks:
            failures = sum(
                path.collision_count(environment, step=self.problem.segment_step, safety_margin=self.problem.safety_margin) > 0
                for environment in bank
            )
            bank_counts.append((failures, len(bank)))
        self._internal_collision_checks += sum(trials for _, trials in bank_counts)
        if self.config.risk_aggregation == "max":
            # Every scale must meet the declared empirical gate.  Per-bank
            # Wilson endpoints are marginal uncertainty scores, not a joint
            # safety certificate over all scales.
            rate = max(failures / trials for failures, trials in bank_counts)
            ucb = max(_wilson_upper_bound(failures, trials) for failures, trials in bank_counts)
        else:
            failures = sum(failures for failures, _ in bank_counts)
            trials = sum(trials for _, trials in bank_counts)
            rate = failures / trials
            # Correlated multi-scale pooling produces a heuristic score only;
            # config validation prohibits this combination for ucb_budgeted.
            ucb = _wilson_upper_bound(failures, trials)
        measured = RiskEvaluation(metrics.feasible, rate, 0.0, float(metrics.objective), (), ucb)
        return self._retarget(measured, gate)

    def _accept_trial(
        self, candidate: RiskEvaluation, incumbent: RiskEvaluation, *, refinement: bool = False,
    ) -> bool:
        """Apply the common ordering and the explicitly chosen cost guard.

        The historical risk_tradeoff control is mutation-only.  In contrast,
        nominal_guarded uses nominal_tolerance for every update, so its local
        refinement cannot bypass the advertised nominal cost bound.
        """
        if not _better(candidate, incumbent):
            return False
        if (candidate.nominal_feasible and incumbent.nominal_feasible
                and candidate.objective > incumbent.objective):
            relative_increase = (candidate.objective - incumbent.objective) / max(abs(incumbent.objective), 1e-12)
            if self.config.selection_mode == "nominal_guarded":
                return (candidate.chance_violation < incumbent.chance_violation
                        and relative_increase <= self.config.nominal_tolerance)
            if (not refinement and self.config.risk_tradeoff is not None
                    and candidate.chance_violation < incumbent.chance_violation):
                return relative_increase <= self.config.risk_tradeoff
        return True

    def _refine_toward_direct(self, vector: Sequence[float], rng: random.Random, *, scale: float | None = None) -> list[float]:
        """Generate a small, reproducible direct-path contraction trial.

        The trial remains in the box bounds.  Selection still goes through
        the same risk-aware comparator, so this operation cannot bypass the
        nominal or chance constraints.
        """
        if not self.config.local_refinement:
            return list(vector)
        if self.config.local_refinement_mode == "adaptive_partial_shortcut":
            return self._refine_shortcut(vector, rng, scale=scale)
        if self.config.local_refinement_mode in {"shortcut", "partial_shortcut", "greedy_shortcut", "safe_shortcut", "detour_weighted_partial", "greedy_partial"}:
            return self._refine_shortcut(vector, rng)
        if self.config.local_refinement_mode == "mixed" and rng.random() < 0.5:
            return self._refine_shortcut(vector, rng)
        start, goal = self.problem.start, self.problem.goal
        refined: list[float] = []
        for index, value in enumerate(vector):
            axis = index % 3
            waypoint_index = index // 3 + 1
            fraction = waypoint_index / (self.problem.intermediate_waypoints + 1)
            direct = start[axis] + fraction * (goal[axis] - start[axis])
            contraction = self.config.local_refinement_scale * rng.random()
            candidate = value + contraction * (direct - value)
            lo, hi = self.bounds[index]
            refined.append(max(lo, min(hi, candidate)))
        return refined

    def _refine_shortcut(self, vector: Sequence[float], rng: random.Random, *, scale: float | None = None) -> list[float]:
        """Move one waypoint toward a point on its neighboring chord.

        The generated polyline segment is no longer than the two original
        segments by the triangle inequality.  Obstacle and chance constraints
        are still evaluated by the ordinary acceptance relation, so geometric
        shortening cannot silently trade away safety.  For partial_shortcut,
        f(p) = ||a-p|| + ||p-b|| is convex, so replacing p by
        (1-scale)*p + scale*q, with q on [a,b], also cannot increase f.
        All path waypoints use the same convex environment box.
        """
        waypoint_count = self.problem.intermediate_waypoints
        if waypoint_count <= 0:
            return list(vector)
        if self.config.local_refinement_mode in {"greedy_shortcut", "safe_shortcut", "detour_weighted_partial", "greedy_partial"}:
            # Measure removable local detours. Moving a waypoint toward a
            # point on its adjacent chord is length-monotone by the triangle
            # inequality, while the ordinary risk comparator remains the
            # authority for acceptance under obstacles and uncertainty.
            detours: list[tuple[float, int]] = []
            for candidate_index in range(waypoint_count):
                previous_point = self.problem.start if candidate_index == 0 else tuple(
                    vector[3 * (candidate_index - 1):3 * candidate_index]
                )
                next_point = self.problem.goal if candidate_index == waypoint_count - 1 else tuple(
                    vector[3 * (candidate_index + 1):3 * (candidate_index + 2)]
                )
                point = tuple(vector[3 * candidate_index:3 * (candidate_index + 1)])
                detour = (
                    math.dist(previous_point, point)
                    + math.dist(point, next_point)
                    - math.dist(previous_point, next_point)
                )
                detours.append((detour, candidate_index))
            if self.config.local_refinement_mode == "detour_weighted_partial":
                # A small geometric detour has little potential shortening.
                # Weighting spends more attempts on high-potential sites
                # while leaving alternatives to a blocked largest detour.
                # Negative round-off at exactly straight sites is clipped.
                weights = [max(0.0, detour) for detour, _ in detours]
                if sum(weights) > 0.0:
                    index = rng.choices(range(waypoint_count), weights=weights, k=1)[0]
                else:
                    index = rng.randrange(waypoint_count)
            else:
                index = max(detours)[1]
        else:
            index = rng.randrange(waypoint_count)
        previous = self.problem.start if index == 0 else tuple(vector[3 * (index - 1):3 * index])
        following = self.problem.goal if index == waypoint_count - 1 else tuple(vector[3 * (index + 1):3 * (index + 2)])
        # A small random offset avoids collapsing every waypoint to its exact
        # midpoint while retaining the no-longer-than-chord guarantee.
        fraction = 0.5 + 0.1 * (rng.random() - 0.5)
        replacement = tuple(a + fraction * (b - a) for a, b in zip(previous, following))
        contraction = self.config.local_refinement_scale if scale is None else scale
        if self.config.local_refinement_mode in {"partial_shortcut", "detour_weighted_partial", "greedy_partial", "adaptive_partial_shortcut"} and contraction < 1.0:
            # Fixed contraction introduces no RNG draws or extra evaluations.
            # scale=1 deliberately uses the original chord point bit-for-bit.
            point = vector[3 * index:3 * index + 3]
            replacement = tuple(p + contraction * (q - p) for p, q in zip(point, replacement))
        output = list(vector)
        output[3 * index:3 * index + 3] = replacement
        return [max(lo, min(hi, value)) for value, (lo, hi) in zip(output, self.bounds)]

    def _feedback_scale(self, scale: float, *, admissible: bool, accepted: bool) -> float:
        """Update a slot's scale from an already evaluated local proposal."""
        if not admissible:
            return max(self.config.local_refinement_min_scale, scale * self.config.local_refinement_shrink)
        if accepted:
            return min(self.config.local_refinement_scale, scale * self.config.local_refinement_grow)
        return scale

    def _refinement_allowed(
        self, candidate: RiskEvaluation, incumbent: RiskEvaluation,
        chance_alpha: float | None = None,
    ) -> bool:
        """Return whether a contraction trial may replace its incumbent.

        Direct-path contraction is useful for nominal length, but it is not a
        safety proof.  Keep the operation conservative and auditable: it may
        only be considered when nominal feasibility is preserved and the
        chance-constraint class does not worsen.  The normal selection key is
        still applied afterwards, so this guard cannot turn a worse objective
        into an accepted update.
        """
        if not candidate.nominal_feasible or not incumbent.nominal_feasible:
            return candidate.nominal_feasible and not incumbent.nominal_feasible
        if self.config.selection_mode == "nominal":
            # The matched-cost control still measures risk, but uncertainty
            # must not affect any selection or refinement decision.
            return True
        alpha = self.config.chance_alpha if chance_alpha is None else float(chance_alpha)
        candidate_in_budget = candidate.perturbed_collision_rate <= alpha
        incumbent_in_budget = incumbent.perturbed_collision_rate <= alpha
        if incumbent_in_budget and not candidate_in_budget:
            return False
        if not incumbent_in_budget and candidate_in_budget:
            return True
        candidate_violation = max(0.0, candidate.perturbed_collision_rate - alpha)
        incumbent_violation = max(0.0, incumbent.perturbed_collision_rate - alpha)
        if self.config.local_refinement_mode == "safe_shortcut":
            # All evaluations use the same frozen perturbation bank, so an
            # empirical-rate comparison is paired and deterministic.  The
            # strict monotone guard prevents nominal contraction from paying
            # for a higher observed risk merely because both rates lie below
            # alpha.  The normal comparator still decides objective ties.
            return candidate.perturbed_collision_rate <= incumbent.perturbed_collision_rate + 1e-15
        return candidate_violation <= incumbent_violation + 1e-15

    def _pbest_indices(
        self, records: Sequence[RiskEvaluation], size: int, p_count: int, gate_alpha: float,
    ) -> list[int]:
        """Return the p-best donor pool for the current generation.

        The risk-diverse option is a genuine search mechanism: it retains two
        complementary elites (shortest paths that meet the current risk gate
        and safest paths) instead of collapsing all p-best donors onto the
        lexicographically safest basin.  It changes donor selection only; the
        ordinary chance-budgeted acceptance relation remains authoritative.
        """
        ranked = sorted(range(size), key=lambda index: records[index].key)
        if self.config.selection_mode == "nominal" or self.config.pbest_selection == "risk_first":
            return ranked[:p_count]
        feasible = [
            index for index in range(size)
            if records[index].nominal_feasible
            and records[index].perturbed_collision_rate <= gate_alpha
        ]
        by_length = sorted(feasible, key=lambda index: (records[index].objective, records[index].perturbed_collision_rate))
        by_risk = sorted(
            range(size),
            key=lambda index: (
                0 if records[index].nominal_feasible else 1,
                records[index].perturbed_collision_rate,
                records[index].objective,
            ),
        )
        left_count = min(
            len(by_length),
            max(1, int(math.ceil(p_count * self.config.pbest_length_fraction)),
            ) if by_length and self.config.pbest_length_fraction > 0.0 else 0,
        )
        pool: list[int] = list(by_length[:left_count])
        for index in by_risk:
            if index not in pool:
                pool.append(index)
            if len(pool) >= p_count:
                break
        for index in ranked:
            if index not in pool:
                pool.append(index)
            if len(pool) >= p_count:
                break
        return pool[:p_count]

    @staticmethod
    def _sample_adaptive_parameters(
        rng: random.Random, mean_f: float, mean_cr: float,
    ) -> tuple[float, float]:
        """Sample bounded DE parameters for one trial.

        A truncated Cauchy perturbation for ``F`` preserves the heavy-tailed
        exploration used by SHADE-style success history, while a clipped
        Gaussian perturbation for ``CR`` keeps crossover rates near their
        learned mean.  The bounds are part of the optimizer contract.
        """
        for _ in range(16):
            factor = mean_f + 0.10 * math.tan(math.pi * (rng.random() - 0.5))
            if factor > 0.0 and math.isfinite(factor):
                break
        else:
            factor = mean_f
        factor = min(2.0, max(1e-6, factor))
        crossover = min(1.0, max(0.0, rng.gauss(mean_cr, 0.10)))
        return factor, crossover

    def _success_weight(self, parent: RiskEvaluation, trial: RiskEvaluation) -> float:
        """Return a bounded, auditable weight for parameter adaptation.

        Risk reduction is deliberately represented separately from objective
        improvement.  A very large penalty objective cannot dominate the
        update, and an accepted objective-only step still contributes weight.
        """
        objective_gain = max(0.0, parent.objective - trial.objective) / max(abs(parent.objective), 1e-12)
        risk_gain = (0.0 if self.config.selection_mode == "nominal"
                     else max(0.0, parent.perturbed_collision_rate - trial.perturbed_collision_rate))
        return 1.0 + min(1.0, objective_gain) + risk_gain

    def _initial_population(self, rng: random.Random, size: int, dim: int) -> list[list[float]]:
        """Create the initial population; subclasses may add domain seeds.

        The default implementation preserves the historical random draw order
        exactly.  This narrow hook allows separately registered search
        candidates without duplicating the risk-selection implementation.
        """
        return [[rng.uniform(lo, hi) for lo, hi in self.bounds] for _ in range(size)]

    def _create_trial(
        self, rng: random.Random, parents: Sequence[Sequence[float]],
        frozen_archive: Sequence[Sequence[float]], index: int, mutation_factor: float,
        crossover_rate: float, pbest_pool: Sequence[int] | None,
    ) -> list[float]:
        """Generate one bounded DE trial; subclasses may change the proposal."""
        size, dim = len(parents), len(self.bounds)
        target = parents[index]
        if self.config.mutation_strategy == "rand1":
            choices = [j for j in range(size) if j != index]
            a_idx, b_idx, c_idx = rng.sample(choices, 3)
            mutant = [
                parents[a_idx][d] + mutation_factor * (parents[b_idx][d] - parents[c_idx][d])
                for d in range(dim)
            ]
        else:
            if not pbest_pool:
                raise ValueError("current_to_pbest1 requires a non-empty pbest pool")
            pbest = parents[rng.choice(pbest_pool)]
            r1_candidates = [j for j in range(size) if j != index]
            r1_idx = rng.choice(r1_candidates)
            donor_pool = [parents[j] for j in range(size) if j != index and j != r1_idx] + list(frozen_archive)
            r2 = rng.choice(donor_pool)
            mutant = [
                target[d] + mutation_factor * (pbest[d] - target[d])
                + mutation_factor * (parents[r1_idx][d] - r2[d])
                for d in range(dim)
            ]
        mutant = [max(lo, min(hi, value)) for value, (lo, hi) in zip(mutant, self.bounds)]
        forced = rng.randrange(dim)
        return [mutant[d] if rng.random() < crossover_rate or d == forced else target[d]
                for d in range(dim)]

    def optimize(self) -> RiskAwareDEResult:
        cfg = self.config.base
        rng = random.Random(cfg.seed)
        self._internal_collision_checks = 0
        size, dim = cfg.population_size, len(self.bounds)
        population = self._initial_population(rng, size, dim)
        if len(population) != size or any(len(vector) != dim for vector in population):
            raise ValueError("initial-population hook returned an invalid shape")
        total_generations = cfg.generations
        records = [self._evaluate(vector, self._gate_alpha(0, total_generations)) for vector in population]
        evaluations = size
        best_index = min(range(size), key=lambda i: records[i].key)
        best_vector = list(population[best_index])
        best_eval = records[best_index]
        history = [best_eval.objective]
        key_history = [best_eval.key]
        gate_history = [self._gate_alpha(0, total_generations)]
        mean_f = float(cfg.mutation_factor)
        mean_cr = float(cfg.crossover_rate)
        mutation_factor_history = [mean_f]
        crossover_rate_history = [mean_cr]
        budget = cfg.max_evaluations if cfg.max_evaluations is not None else size * (cfg.generations + 1)
        archive: list[list[float]] = []
        archive_limit = size if self.config.archive_size is None else self.config.archive_size
        adaptive_partial = self.config.local_refinement_mode == "adaptive_partial_shortcut"
        local_scales = [self.config.local_refinement_scale] * size if adaptive_partial else []
        refinement_events: list[dict[str, float | int | bool]] = []
        for generation in range(cfg.generations):
            if evaluations >= budget:
                break
            gate_alpha = self._search_gate(evaluations + 1, budget, generation)
            parents = [list(vector) for vector in population]
            # Parent records may have been ranked under the preceding gate.
            # Retargeting reuses their measured rates and avoids extra risk
            # checks while restoring a generation-consistent ordering.
            parent_records = [self._retarget(record, gate_alpha) for record in records]
            records = list(parent_records)
            # Both the parent population and the historical archive are
            # frozen for this generation.  Successful parents are appended
            # only after all targets have sampled their donors.
            frozen_archive = [list(vector) for vector in archive]
            generation_archive: list[list[float]] = []
            successful_f: list[float] = []
            successful_cr: list[float] = []
            successful_weights: list[float] = []
            if self.config.mutation_strategy == "current_to_pbest1":
                p_count = max(1, math.ceil(self.config.p_best_rate * size))
                pbest_pool = self._pbest_indices(parent_records, size, p_count, gate_alpha)
            for i, target in enumerate(parents):
                if evaluations >= budget:
                    break
                if self.config.parameter_adaptation == "risk_success":
                    trial_f, trial_cr = self._sample_adaptive_parameters(rng, mean_f, mean_cr)
                else:
                    trial_f, trial_cr = cfg.mutation_factor, cfg.crossover_rate
                trial = self._create_trial(
                    rng, parents, frozen_archive, i, trial_f, trial_cr,
                    pbest_pool if self.config.mutation_strategy == "current_to_pbest1" else None,
                )
                gate_alpha = self._search_gate(evaluations + 1, budget, generation)
                trial_eval = self._evaluate(trial, gate_alpha)
                evaluations += 1
                parent_eval = self._retarget(parent_records[i], gate_alpha)
                records[i] = parent_eval
                accepted = self._accept_trial(trial_eval, parent_eval)
                if accepted:
                    if self.config.parameter_adaptation == "risk_success":
                        successful_f.append(trial_f)
                        successful_cr.append(trial_cr)
                        successful_weights.append(self._success_weight(parent_eval, trial_eval))
                    if self.config.mutation_strategy == "current_to_pbest1":
                        generation_archive.append(list(target))
                    population[i], records[i] = trial, trial_eval
                # One extra objective evaluation is deliberately budgeted and
                # recorded when the optional local refinement is enabled.
                if (self.config.local_refinement
                        and evaluations < budget
                        and rng.random() < self.config.local_refinement_probability):
                    if adaptive_partial:
                        refined = self._refine_toward_direct(population[i], rng, scale=local_scales[i])
                    else:
                        refined = self._refine_toward_direct(population[i], rng)
                    gate_alpha = self._search_gate(evaluations + 1, budget, generation)
                    refined_eval = self._evaluate(refined, gate_alpha)
                    evaluations += 1
                    records[i] = self._retarget(records[i], gate_alpha)
                    allowed = self._refinement_allowed(refined_eval, records[i], gate_alpha)
                    refinement_accepted = allowed and self._accept_trial(refined_eval, records[i], refinement=True)
                    if adaptive_partial:
                        admissible = refined_eval.nominal_feasible and allowed
                        next_scale = self._feedback_scale(local_scales[i], admissible=admissible, accepted=refinement_accepted)
                        # Pure geometry for diagnostics, not another nominal
                        # objective evaluation or collision check.
                        length_gain = self.problem.decode(population[i]).length - self.problem.decode(refined).length
                        refinement_events.append({
                            "nfe": evaluations, "generation": generation, "slot": i,
                            "scale": local_scales[i], "next_scale": next_scale,
                            "nominal_feasible": refined_eval.nominal_feasible,
                            "risk_guard_pass": allowed,
                            "in_risk_budget": refined_eval.perturbed_collision_rate <= gate_alpha,
                            "accepted": refinement_accepted, "proposed_length_gain": length_gain,
                            "accepted_length_gain": length_gain if refinement_accepted else 0.0,
                        })
                        local_scales[i] = next_scale
                    if refinement_accepted:
                        if self.config.mutation_strategy == "current_to_pbest1":
                            generation_archive.append(list(population[i]))
                        population[i], records[i] = refined, refined_eval
            if self.config.mutation_strategy == "current_to_pbest1" and archive_limit is not None:
                archive.extend(generation_archive)
                if archive_limit == 0:
                    archive.clear()
                elif len(archive) > archive_limit:
                    # JADE-style archive truncation, using the optimizer RNG
                    # so repeated seeds remain exactly reproducible.
                    archive = [list(vector) for vector in rng.sample(archive, archive_limit)]
            if self.config.parameter_adaptation == "risk_success" and successful_weights:
                weight_sum = sum(successful_weights)
                mean_f_success = sum(w * f * f for w, f in zip(successful_weights, successful_f)) / max(
                    sum(w * f for w, f in zip(successful_weights, successful_f)), 1e-12
                )
                mean_cr_success = sum(w * cr for w, cr in zip(successful_weights, successful_cr)) / weight_sum
                learning_rate = self.config.adaptation_learning_rate
                mean_f = (1.0 - learning_rate) * mean_f + learning_rate * mean_f_success
                mean_cr = (1.0 - learning_rate) * mean_cr + learning_rate * mean_cr_success
                mean_f = min(2.0, max(1e-6, mean_f))
                mean_cr = min(1.0, max(0.0, mean_cr))
            # A moving gate can reorder old candidates without evaluating
            # them again.  Every recorded incumbent must be ranked under the
            # same gate as its key, rather than a stale best from a looser gate.
            records = [self._retarget(record, gate_alpha) for record in records]
            best_index = min(range(size), key=lambda i: records[i].key)
            best_vector, best_eval = list(population[best_index]), records[best_index]
            history.append(best_eval.objective)
            key_history.append(best_eval.key)
            gate_history.append(gate_alpha)
            mutation_factor_history.append(mean_f)
            crossover_rate_history.append(mean_cr)
        # Rank under the final gate; this does not guarantee that any member
        # satisfies it.  best_evaluation retains any remaining violation.
        records = [self._retarget(record, self.config.chance_alpha) for record in records]
        final_index = min(range(size), key=lambda i: records[i].key)
        best_vector = list(population[final_index])
        best_eval = records[final_index]
        # The returned key always corresponds to the locked final alpha, even
        # when an explicit budget stopped before the final scheduled gate.
        if key_history:
            key_history[-1] = best_eval.key
            history[-1] = best_eval.objective
        if gate_history:
            gate_history[-1] = self.config.chance_alpha
        else:
            gate_history.append(self.config.chance_alpha)
        refinement_stats: dict[str, float | int] = {}
        if adaptive_partial:
            refinement_stats = {
                "proposals": len(refinement_events),
                "nominal_feasible": sum(bool(event["nominal_feasible"]) for event in refinement_events),
                "risk_guard_pass": sum(bool(event["risk_guard_pass"]) for event in refinement_events),
                "in_risk_budget": sum(bool(event["in_risk_budget"]) for event in refinement_events),
                "accepted": sum(bool(event["accepted"]) for event in refinement_events),
                "nominal_rejected": sum(not event["nominal_feasible"] for event in refinement_events),
                "risk_guard_rejected_after_nominal": sum(bool(event["nominal_feasible"]) and not event["risk_guard_pass"] for event in refinement_events),
                "accepted_length_gain_sum": sum(float(event["accepted_length_gain"]) for event in refinement_events),
                "final_scale_min": min(local_scales), "final_scale_max": max(local_scales),
                "final_scale_mean": sum(local_scales) / len(local_scales),
            }
        return RiskAwareDEResult(
            best_vector=best_vector,
            best_fitness=best_eval.objective,
            history=history,
            evaluations=evaluations,
            internal_collision_checks=self._internal_collision_checks,
            best_evaluation=best_eval,
            key_history=key_history,
            gate_history=gate_history,
            mutation_factor_history=mutation_factor_history,
            crossover_rate_history=crossover_rate_history,
            local_refinement_stats=refinement_stats,
            local_refinement_events=refinement_events,
        )


def plan_continuous_path_risk_aware(
    problem: ContinuousPathProblem, config: RiskAwareDEConfig | None = None,
) -> tuple[Path3D, RiskAwareDEResult]:
    """Plan with chance-style risk and return the auditable robust result."""

    config = config or RiskAwareDEConfig()
    if not problem.environment.is_free(problem.start, problem.safety_margin) or not problem.environment.is_free(problem.goal, problem.safety_margin):
        raise ValueError("start and goal must be collision-free and within bounds")
    if not problem.bounds:
        optimizer = RiskAwareDE([(0.0, 1.0)], problem, config)
        result = optimizer._evaluate([], config.chance_alpha)
        raw = RiskAwareDEResult(
            [], result.objective, [result.objective], 1, optimizer._internal_collision_checks, result,
            key_history=[result.key],
            gate_history=[config.chance_alpha],
            mutation_factor_history=[config.base.mutation_factor],
            crossover_rate_history=[config.base.crossover_rate],
        )
        return problem.decode([]), raw
    optimizer = RiskAwareDE(problem.bounds, problem, config)
    raw = optimizer.optimize()
    return problem.decode(raw.best_vector), raw


__all__ = ["RiskAwareDEConfig", "RiskEvaluation", "RiskAwareDEResult", "RiskAwareDE", "plan_continuous_path_risk_aware"]
