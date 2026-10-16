"""Scenario construction and JSON I/O for reproducible path-planning experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from pathlib import Path
from typing import Any, Iterable

from .models import Bounds3D, Environment3D, ObstacleBox, Point3D


@dataclass
class Scenario3D:
    """A named planning case with fixed start/goal and environment metadata."""

    name: str
    environment: Environment3D
    start: Point3D
    goal: Point3D
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.start = tuple(float(x) for x in self.start)  # type: ignore[assignment]
        self.goal = tuple(float(x) for x in self.goal)  # type: ignore[assignment]
        if not self.environment.is_free(self.start) or not self.environment.is_free(self.goal):
            raise ValueError("scenario start and goal must be free and in bounds")
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bounds": {"minimum": list(self.environment.bounds.minimum), "maximum": list(self.environment.bounds.maximum)},
            "obstacles": [{"minimum": list(o.minimum), "maximum": list(o.maximum)} for o in self.environment.obstacles],
            "start": list(self.start),
            "goal": list(self.goal),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario3D":
        bounds = data["bounds"]
        env = Environment3D(
            Bounds3D(bounds["minimum"], bounds["maximum"]),
            [ObstacleBox(o["minimum"], o["maximum"]) for o in data.get("obstacles", [])],
        )
        return cls(str(data["name"]), env, data["start"], data["goal"], data.get("metadata"))


def load_scenario(path: str | Path) -> Scenario3D:
    with Path(path).open("r", encoding="utf-8") as fh:
        return Scenario3D.from_dict(json.load(fh))


def save_scenario(scenario: Scenario3D, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(scenario.to_dict(), fh, indent=2)


def make_benchmark_scenarios(seed: int = 7) -> list[Scenario3D]:
    """Return deterministic lightweight scenarios for algorithm comparisons."""
    rng = random.Random(seed)
    bounds = Bounds3D((0, 0, 0), (20, 20, 10))
    cases: list[Scenario3D] = []
    cases.append(Scenario3D("open", Environment3D(bounds), (1, 1, 1), (19, 19, 8), {"family": "static"}))
    cases.append(Scenario3D("central_block", Environment3D(bounds, [ObstacleBox((8, 6, 0), (12, 14, 7))]), (1, 1, 1), (19, 19, 8), {"family": "static"}))
    cases.append(Scenario3D("corridor", Environment3D(bounds, [ObstacleBox((4, 0, 0), (6, 15, 8)), ObstacleBox((14, 5, 2), (16, 20, 10))]), (1, 2, 1), (19, 18, 8), {"family": "static"}))
    for i in range(5):
        obstacles: list[ObstacleBox] = []
        for _ in range(5 + i):
            x, y, z = (rng.uniform(3, 16), rng.uniform(3, 16), rng.uniform(1, 5))
            sx, sy, sz = (rng.uniform(1, 2.5), rng.uniform(1, 2.5), rng.uniform(1, 3))
            lo, hi = (x, y, z), (min(18.0, x + sx), min(18.0, y + sy), min(9.0, z + sz))
            obstacles.append(ObstacleBox(lo, hi))
        # Keep generated endpoints valid; obstacles are sampled away from corners.
        cases.append(Scenario3D(f"random_{i:02d}", Environment3D(bounds, obstacles), (1, 1, 1), (19, 19, 8), {"family": "random", "seed": seed + i}))
    return cases

