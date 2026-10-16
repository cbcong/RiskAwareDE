"""Shared result types and a lightweight planner protocol.

The experiment runner can use :class:`PlannerResult` for grid planners,
continuous metaheuristics, and later online replanners without having to know
the implementation details of each algorithm.  The result intentionally keeps
raw history and metadata so that every comparison can be audited after a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from .models import Environment3D, Path3D


@dataclass
class PlannerResult:
    """Outcome of one planning call.

    ``cost`` is the objective value used by the planner.  It may include
    penalties, so use ``path.length`` for a geometric length comparison.
    ``history`` stores the best objective after each search iteration (or
    expansion batch for a graph search).  ``metadata`` is deliberately a plain
    dictionary so future experiments can add counters without changing this
    public structure.
    """

    path: Path3D | None
    cost: float
    success: bool
    evaluations: int = 0
    runtime_seconds: float = 0.0
    history: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> float | None:
        """Geometric path length, or ``None`` when no path was found."""

        return None if self.path is None else self.path.length

    @property
    def collision_count(self) -> int | None:
        """Number of colliding segments when a path is available.

        Collision counting needs an environment and is therefore intentionally
        not performed here.  Planners that use an environment record the count
        in ``metadata`` under ``"collision_count"``.
        """

        value = self.metadata.get("collision_count")
        return int(value) if value is not None else None


@runtime_checkable
class Planner(Protocol):
    """Minimal structural interface shared by baseline planners.

    Concrete planners may require either an occupancy grid or a continuous
    environment.  They expose these through keyword arguments so a batch
    runner can call all planners through one small adapter.
    """

    name: str

    def plan(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        *,
        environment: Environment3D | None = None,
        grid: Any | None = None,
    ) -> PlannerResult:
        ...

