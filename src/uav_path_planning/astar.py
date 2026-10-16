"""A* baseline for 3-D occupancy grids.

This implementation is intentionally dependency free.  It supports 6-neighbor
motion by default and optional 26-neighbor motion for experiments.  Grid cells
are interpreted as Cartesian coordinates; callers that use a physical voxel
size can scale the returned waypoints afterwards or pass scaled points to a
metric evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time
from typing import Sequence

from .grid import Cell3D, GridMap3D
from .models import Environment3D, Path3D
from .planning import PlannerResult


def _cell(value: Sequence[float]) -> Cell3D:
    if len(value) != 3:
        raise ValueError("a grid point must have exactly three coordinates")
    result = tuple(int(round(float(x))) for x in value)
    if any(abs(float(x) - y) > 1e-9 for x, y in zip(value, result)):
        raise ValueError("grid start and goal must be integer cell coordinates")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class AStarConfig:
    """Configuration for :class:`GridAStarPlanner`."""

    include_diagonal: bool = False
    safety_margin_cells: int = 0
    safety_margin: float = 0.0
    world_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cell_size: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if self.safety_margin_cells < 0:
            raise ValueError("safety_margin_cells cannot be negative")
        if not math.isfinite(self.safety_margin) or self.safety_margin < 0:
            raise ValueError("safety_margin must be finite and non-negative")
        if len(self.world_origin) != 3 or not all(math.isfinite(float(value)) for value in self.world_origin):
            raise ValueError("world_origin must contain three finite coordinates")
        if len(self.cell_size) != 3 or not all(math.isfinite(float(value)) and float(value) > 0 for value in self.cell_size):
            raise ValueError("cell_size must contain three positive finite values")


class GridAStarPlanner:
    """Optimal A* search on a finite :class:`GridMap3D`."""

    name = "astar3d"

    def __init__(self, grid: GridMap3D | None = None, config: AStarConfig | None = None):
        self.grid = grid
        self.config = config or AStarConfig()

    def _blocked(self, grid: GridMap3D, cell: Cell3D, environment: Environment3D | None = None) -> bool:
        if grid.is_free(cell):
            margin = self.config.safety_margin_cells
            if margin > 0:
                x, y, z = cell
                for dx in range(-margin, margin + 1):
                    for dy in range(-margin, margin + 1):
                        for dz in range(-margin, margin + 1):
                            if (dx, dy, dz) != (0, 0, 0) and not grid.is_free((x + dx, y + dy, z + dz)):
                                return True
            if environment is not None and not environment.is_free(self._world_point(cell), self.config.safety_margin):
                return True
            return False
        return True

    def _edge_free(self, grid: GridMap3D, current: Cell3D, neighbor: Cell3D, environment: Environment3D | None) -> bool:
        """Reject diagonal corner cutting and continuous obstacle crossings."""
        delta = tuple(b - a for a, b in zip(current, neighbor))
        changed = [axis for axis, value in enumerate(delta) if value]
        if len(changed) > 1:
            # Every proper axis-aligned substep must be free. This prevents a
            # 2-D or 3-D diagonal from squeezing through touching corners.
            from itertools import combinations
            for count in range(1, len(changed)):
                for subset in combinations(changed, count):
                    intermediate = list(current)
                    for axis in subset:
                        intermediate[axis] += delta[axis]
                    if self._blocked(grid, tuple(intermediate), environment):
                        return False
        if environment is not None:
            return environment.segment_is_free(
                self._world_point(current),
                self._world_point(neighbor),
                safety_margin=self.config.safety_margin,
            )
        return True

    def _world_point(self, cell: Cell3D) -> tuple[float, float, float]:
        return tuple(origin + float(index) * scale for origin, index, scale in zip(
            self.config.world_origin, cell, self.config.cell_size
        ))  # type: ignore[return-value]

    @staticmethod
    def _heuristic(a: Cell3D, b: Cell3D) -> float:
        return math.dist(a, b)

    def plan(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        *,
        environment: Environment3D | None = None,
        grid: GridMap3D | None = None,
    ) -> PlannerResult:
        grid = grid or self.grid
        if grid is None:
            raise ValueError("a GridMap3D must be provided to GridAStarPlanner")
        source, target = _cell(start), _cell(goal)
        if not grid.in_bounds(source) or not grid.in_bounds(target):
            raise ValueError("start and goal must lie inside the grid")
        if self._blocked(grid, source, environment) or self._blocked(grid, target, environment):
            raise ValueError("start and goal must be free cells")

        started = time.perf_counter()
        open_heap: list[tuple[float, int, Cell3D]] = []
        counter = 0
        heapq.heappush(open_heap, (self._heuristic(source, target), counter, source))
        came_from: dict[Cell3D, Cell3D] = {}
        g_score: dict[Cell3D, float] = {source: 0.0}
        closed: set[Cell3D] = set()
        expansions = 0
        history: list[float] = []

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            expansions += 1
            history.append(g_score[current] + self._heuristic(current, target))
            if current == target:
                cells = [current]
                while cells[-1] != source:
                    cells.append(came_from[cells[-1]])
                cells.reverse()
                path = Path3D([tuple(float(v) for v in c) for c in cells])
                return PlannerResult(
                    path=path,
                    cost=g_score[target],
                    success=True,
                    evaluations=expansions,
                    runtime_seconds=time.perf_counter() - started,
                    history=history,
                    metadata={"algorithm": self.name, "expanded": expansions, "grid_size": grid.size},
                )

            for neighbor in grid.neighbors(current, include_diagonal=self.config.include_diagonal):
                if neighbor in closed or self._blocked(grid, neighbor, environment):
                    continue
                if not self._edge_free(grid, current, neighbor, environment):
                    continue
                tentative = g_score[current] + math.dist(current, neighbor)
                if tentative >= g_score.get(neighbor, math.inf):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                counter += 1
                heapq.heappush(open_heap, (tentative + self._heuristic(neighbor, target), counter, neighbor))

        return PlannerResult(
            path=None,
            cost=math.inf,
            success=False,
            evaluations=expansions,
            runtime_seconds=time.perf_counter() - started,
            history=history,
            metadata={"algorithm": self.name, "expanded": expansions, "grid_size": grid.size},
        )


def astar_3d(
    grid: GridMap3D,
    start: Sequence[float],
    goal: Sequence[float],
    config: AStarConfig | None = None,
) -> PlannerResult:
    """Functional convenience wrapper used by experiment scripts."""

    return GridAStarPlanner(grid, config).plan(start, goal)
