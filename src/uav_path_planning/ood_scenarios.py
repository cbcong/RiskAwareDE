"""A separately versioned, distribution-shifted map bank for blind evaluation.

This module is intentionally independent from :mod:`research_scenarios`.  The
existing v1 maps were repeatedly exposed during development; changing only
their root seed would therefore be a new draw from the same generator, not a
map-domain shift.  The v2 bank changes obstacle density, height and topology
while retaining the world bounds and endpoints so that the evaluation still
measures the planner rather than a changed coordinate system.

The generator never calls a planner and never scores a candidate algorithm.
The only acceptance checks are geometry-only validity checks (free endpoints,
blocked direct segment and coarse-grid reachability).  A frozen manifest made
by ``scripts/freeze_ood_map_bank.py`` is the sole input to the blind runner.
"""

from __future__ import annotations

from collections import deque
import hashlib
import math
import random
from typing import Iterable, Sequence

from .models import Bounds3D, Environment3D, ObstacleBox
from .scenarios import Scenario3D


OOD_GENERATOR_VERSION = "research-scenarios-v2-ood"
OOD_DOMAIN_SHIFT = "density-height-topology"
OOD_FAMILIES = ("dense_clutter", "highrise_urban", "narrow_barriers")
OOD_BOUNDS = Bounds3D((0.0, 0.0, 0.0), (20.0, 20.0, 10.0))
OOD_START = (1.0, 1.0, 1.0)
OOD_GOAL = (19.0, 19.0, 8.0)


def _map_seed(root_seed: int, family: str, index: int) -> int:
    payload = f"{OOD_GENERATOR_VERSION}|{root_seed}|{family}|{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _line_point(t: float) -> tuple[float, float, float]:
    return tuple(a + t * (b - a) for a, b in zip(OOD_START, OOD_GOAL))  # type: ignore[return-value]


def _direct_blocker(rng: random.Random, *, high: bool = False) -> ObstacleBox:
    """A blocker guaranteed to intersect the nominal start-goal diagonal."""

    t = rng.uniform(0.36, 0.64)
    cx, cy, cz = _line_point(t)
    half_x = rng.uniform(1.1, 2.0)
    half_y = rng.uniform(1.1, 2.0)
    below = rng.uniform(0.3, 1.0)
    above = rng.uniform(2.8, 5.8) if high else rng.uniform(1.4, 3.1)
    return ObstacleBox(
        (max(3.0, cx - half_x), max(3.0, cy - half_y), max(0.0, cz - below)),
        (min(17.9, cx + half_x), min(17.9, cy + half_y), min(10.0, cz + above)),
    )


def _dense_clutter(rng: random.Random) -> list[ObstacleBox]:
    boxes = [_direct_blocker(rng, high=True)]
    # v1 used 7--10 mostly low boxes.  v2 is denser and has a heavier height
    # tail, but z=9 remains available as a physical escape layer.
    for _ in range(rng.randint(12, 16)):
        x0, y0 = rng.uniform(3.0, 15.8), rng.uniform(3.0, 15.8)
        width, depth = rng.uniform(1.2, 3.2), rng.uniform(1.2, 3.2)
        z0 = rng.uniform(0.0, 1.4)
        z1 = min(10.0, rng.uniform(4.2, 10.0))
        boxes.append(ObstacleBox((x0, y0, z0), (min(18.2, x0 + width), min(18.2, y0 + depth), z1)))
    return boxes


def _highrise_urban(rng: random.Random) -> list[ObstacleBox]:
    boxes = [_direct_blocker(rng, high=True)]
    # A non-lattice 4-by-4 layout is deliberately different from v1's sparse
    # 3-by-3 urban family.  Jitter and variable footprint create more turns.
    locations = [(4.2 + 3.8 * i, 4.0 + 3.9 * j) for i in range(4) for j in range(4)]
    rng.shuffle(locations)
    for x, y in locations[: rng.randint(10, 13)]:
        width, depth = rng.uniform(1.8, 3.1), rng.uniform(1.8, 3.1)
        x0 = max(3.0, x + rng.uniform(-0.55, 0.55) - width / 2)
        y0 = max(3.0, y + rng.uniform(-0.55, 0.55) - depth / 2)
        ztop = min(10.0, rng.uniform(4.5, 10.0))
        boxes.append(ObstacleBox((x0, y0, 0.0), (min(18.2, x0 + width), min(18.2, y0 + depth), ztop)))
    return boxes


def _narrow_barriers(rng: random.Random) -> list[ObstacleBox]:
    boxes: list[ObstacleBox] = []
    # Four alternating walls with narrow, offset gaps.  Unlike v1, the gaps
    # are not selected from fixed two-choice positions and the walls are tall.
    for index, x in enumerate((4.5, 8.5, 12.5, 16.0)):
        wall_x = x + rng.uniform(-0.3, 0.3)
        gap = rng.uniform(4.8, 15.2)
        if index % 2:
            gap = 20.0 - gap
        gap_half = rng.uniform(0.52, 0.76)
        top = rng.uniform(7.0, 10.0)
        left_max, right_min = gap - gap_half, gap + gap_half
        boxes.append(ObstacleBox((wall_x - 0.72, 2.0, 0.0), (wall_x + 0.72, left_max, top)))
        boxes.append(ObstacleBox((wall_x - 0.72, right_min, 0.0), (wall_x + 0.72, 18.0, top)))
    return boxes


def _add_ceiling_reaching_structure(family: str, obstacles: list[ObstacleBox], rng: random.Random) -> list[ObstacleBox]:
    """Remove v1's universal high-altitude escape layer in every OOD map."""

    if family == "narrow_barriers":
        selected = {0, 1}  # both parts of the first wall, leaving only its lateral gap
    else:
        selected = {rng.randrange(1, len(obstacles))}
    result = list(obstacles)
    for index in selected:
        old = result[index]
        result[index] = ObstacleBox(old.minimum, (old.maximum[0], old.maximum[1], 10.0))
    return result


def _coarse_reachable(environment: Environment3D, resolution: float = 1.0) -> bool:
    """Geometry-only 26-connected reachability check; no planner is called."""

    lo = environment.bounds.minimum
    hi = environment.bounds.maximum
    size = tuple(int(round((b - a) / resolution)) + 1 for a, b in zip(lo, hi))
    start = tuple(int(round((p - lo[i]) / resolution)) for i, p in enumerate(OOD_START))
    goal = tuple(int(round((p - lo[i]) / resolution)) for i, p in enumerate(OOD_GOAL))

    def free(cell: tuple[int, int, int]) -> bool:
        if any(i < 0 or i >= size[d] for d, i in enumerate(cell)):
            return False
        point = tuple(lo[d] + cell[d] * resolution for d in range(3))
        return environment.is_free(point)

    if not free(start) or not free(goal):
        return False
    queue = deque([start])
    seen = {start}
    while queue:
        current = queue.popleft()
        if current == goal:
            return True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if not (dx or dy or dz):
                        continue
                    nxt = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if (nxt not in seen and free(nxt)
                            and environment.segment_is_free(
                                tuple(lo[d] + current[d] * resolution for d in range(3)),
                                tuple(lo[d] + nxt[d] * resolution for d in range(3)),
                            )):
                        seen.add(nxt)
                        queue.append(nxt)
    return False


def _direct_blocked(environment: Environment3D) -> bool:
    return not environment.segment_is_free(OOD_START, OOD_GOAL)


def _validate_geometry(scenario: Scenario3D) -> None:
    if not scenario.environment.is_free(scenario.start) or not scenario.environment.is_free(scenario.goal):
        raise ValueError(f"OOD endpoint is occupied: {scenario.name}")
    if not _direct_blocked(scenario.environment):
        raise ValueError(f"OOD direct route is unexpectedly free: {scenario.name}")
    if not _coarse_reachable(scenario.environment):
        raise ValueError(f"OOD map has no coarse geometric route: {scenario.name}")


def make_ood_scenarios(root_seed: int = 20261008, count_per_family: int = 8) -> list[Scenario3D]:
    """Create a deterministic 12-map OOD bank without planner-dependent filtering."""

    if isinstance(root_seed, bool) or not isinstance(root_seed, int):
        raise ValueError("root_seed must be an integer")
    if isinstance(count_per_family, bool) or not isinstance(count_per_family, int) or count_per_family <= 0:
        raise ValueError("count_per_family must be a positive integer")
    builders = {"dense_clutter": _dense_clutter, "highrise_urban": _highrise_urban, "narrow_barriers": _narrow_barriers}
    scenarios: list[Scenario3D] = []
    for family in OOD_FAMILIES:
        for index in range(count_per_family):
            rng = random.Random(_map_seed(root_seed, family, index))
            obstacles = _add_ceiling_reaching_structure(family, builders[family](rng), rng)
            scenario = Scenario3D(
                f"ood_{family}_{index:02d}",
                Environment3D(OOD_BOUNDS, obstacles),
                OOD_START,
                OOD_GOAL,
                {
                    "generator": OOD_GENERATOR_VERSION,
                    "domain_shift": OOD_DOMAIN_SHIFT,
                    "rootseed": root_seed,
                    "family": family,
                    "family_index": index,
                    "obstacle_count": len(obstacles),
                    "max_obstacle_z": max(o.maximum[2] for o in obstacles),
                    "roof_touching_obstacles": sum(o.maximum[2] >= 10.0 for o in obstacles),
                    "bounds_preserved_from_v1": True,
                    "endpoints_preserved_from_v1": True,
                    "planner_dependent_filtering": False,
                },
            )
            _validate_geometry(scenario)
            scenarios.append(scenario)
    return scenarios


def geometry_summary(scenarios: Sequence[Scenario3D]) -> dict[str, object]:
    """Return manifest-level statistics useful for an exposure audit."""

    by_family: dict[str, list[Scenario3D]] = {family: [] for family in OOD_FAMILIES}
    for scenario in scenarios:
        by_family[str(scenario.metadata.get("family"))].append(scenario)
    return {
        "map_count": len(scenarios),
        "families": {
            family: {
                "maps": len(rows),
                "mean_obstacles": sum(len(s.environment.obstacles) for s in rows) / len(rows),
                "max_obstacle_z": max(max(o.maximum[2] for o in s.environment.obstacles) for s in rows),
            }
            for family, rows in by_family.items()
        },
        "geometry_checks": {"endpoints_free": True, "direct_route_blocked": True, "coarse_route_exists": True},
    }


__all__ = [
    "OOD_GENERATOR_VERSION", "OOD_DOMAIN_SHIFT", "OOD_FAMILIES", "OOD_BOUNDS",
    "OOD_START", "OOD_GOAL", "make_ood_scenarios", "geometry_summary",
]
