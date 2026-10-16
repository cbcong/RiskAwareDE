"""Independent synthetic scenario bank for research evaluation.

The original benchmark scenarios are useful for development, but they should
not be reused as a held-out test set after repeated algorithm development.  This
module creates a separate, deterministic bank of 3-D maps.  It deliberately
contains no planner calls or algorithm-dependent filtering: every map is
constructed geometrically with a free high-altitude corridor and a blocked
start-to-goal straight segment.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Sequence

from .models import Bounds3D, Environment3D, ObstacleBox, Path3D
from .scenarios import Scenario3D


RESEARCH_GENERATOR_VERSION = "research-scenarios-v1"
RESEARCH_FAMILIES = ("clutter", "urban", "barriers")
# The historical split is retained for reproducibility of v1--v5 outputs.
LEGACY_RESEARCH_SPLIT_PROTOCOL = "legacy_12_6_6"
# The confirmatory protocol uses all 24 maps with a larger, family-stratified
# held-out test set and six validation maps for more reliable model selection.
RESEARCH_24_SPLIT_PROTOCOL = "research24_6_6_12"
# Compatibility option retained for the first 24-map draft.  It is not the
# default because four validation maps are too few for stable selection.
RESEARCH_24_8_4_12_SPLIT_PROTOCOL = "research24_8_4_12"
RESEARCH_SPLIT_PROTOCOLS = (
    LEGACY_RESEARCH_SPLIT_PROTOCOL,
    RESEARCH_24_SPLIT_PROTOCOL,
    RESEARCH_24_8_4_12_SPLIT_PROTOCOL,
)
RESEARCH_BOUNDS = Bounds3D((0.0, 0.0, 0.0), (20.0, 20.0, 10.0))
RESEARCH_START = (1.0, 1.0, 1.0)
RESEARCH_GOAL = (19.0, 19.0, 8.0)


@dataclass(frozen=True)
class ResearchScenarioSplit:
    """Family-stratified map split with a small mapping-compatible API."""

    train: tuple[Scenario3D, ...]
    validation: tuple[Scenario3D, ...]
    test: tuple[Scenario3D, ...]

    def __getitem__(self, key: str) -> tuple[Scenario3D, ...]:
        if key not in {"train", "validation", "test"}:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        return key in {"train", "validation", "test"}

    def keys(self) -> tuple[str, str, str]:
        return ("train", "validation", "test")

    def values(self) -> tuple[tuple[Scenario3D, ...], tuple[Scenario3D, ...], tuple[Scenario3D, ...]]:
        return (self.train, self.validation, self.test)

    def items(self) -> tuple[tuple[str, tuple[Scenario3D, ...]], ...]:
        return tuple(zip(self.keys(), self.values()))


def _validate_seed(seed: int, name: str = "seed") -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{name} must be an integer")


def _map_seed(rootseed: int, family: str, index: int) -> int:
    """Derive a process-stable 32-bit seed without using Python's hash()."""

    payload = f"{RESEARCH_GENERATOR_VERSION}|{rootseed}|{family}|{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _line_point(t: float) -> tuple[float, float, float]:
    return tuple(a + t * (b - a) for a, b in zip(RESEARCH_START, RESEARCH_GOAL))  # type: ignore[return-value]


def _line_blocker(rng: random.Random) -> ObstacleBox:
    """Create a varied box that intersects the nominal straight segment."""

    t = rng.uniform(0.35, 0.65)
    cx, cy, cz = _line_point(t)
    half_x = rng.uniform(1.0, 1.8)
    half_y = rng.uniform(1.0, 1.8)
    below = rng.uniform(1.0, 1.8)
    above = rng.uniform(1.0, 1.8)
    return ObstacleBox(
        (max(3.2, cx - half_x), max(3.2, cy - half_y), max(0.2, cz - below)),
        (min(17.8, cx + half_x), min(17.8, cy + half_y), min(7.0, cz + above)),
    )


def _random_low_box(rng: random.Random, *, top_range: tuple[float, float] = (2.0, 6.7)) -> ObstacleBox:
    """Sample a box away from the endpoint columns and below the high corridor."""

    x0 = rng.uniform(3.4, 15.8)
    y0 = rng.uniform(3.4, 15.8)
    width = rng.uniform(1.0, 2.8)
    depth = rng.uniform(1.0, 2.8)
    z0 = rng.uniform(0.2, 1.8)
    z1 = min(7.0, rng.uniform(*top_range))
    if z1 <= z0 + 0.2:
        z1 = min(7.0, z0 + 1.0)
    return ObstacleBox(
        (x0, y0, z0),
        (min(18.0, x0 + width), min(18.0, y0 + depth), z1),
    )


def _make_clutter(rng: random.Random) -> list[ObstacleBox]:
    boxes = [_line_blocker(rng)]
    # Eight to eleven low irregular blocks give moderate occupancy while the
    # forced blocker makes the direct route non-trivial in every realization.
    for _ in range(rng.randint(7, 10)):
        boxes.append(_random_low_box(rng))
    return boxes


def _make_urban(rng: random.Random) -> list[ObstacleBox]:
    boxes = [_line_blocker(rng)]
    # Buildings are jittered around a sparse 3-by-3 lattice.  Heights vary but
    # never reach z=7, leaving a consistent high-altitude planning option.
    lattice = [(x, y) for x in (5.0, 10.0, 15.0) for y in (5.0, 10.0, 15.0)]
    rng.shuffle(lattice)
    for x, y in lattice[: rng.randint(6, 8)]:
        jitter_x, jitter_y = rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6)
        width, depth = rng.uniform(1.7, 2.8), rng.uniform(1.7, 2.8)
        x0, y0 = max(3.2, x + jitter_x - width / 2), max(3.2, y + jitter_y - depth / 2)
        ztop = rng.uniform(3.0, 6.8)
        boxes.append(ObstacleBox((x0, y0, 0.0), (min(18.0, x0 + width), min(18.0, y0 + depth), ztop)))
    return boxes


def _make_barriers(rng: random.Random) -> tuple[list[ObstacleBox], list[float]]:
    """Make three perpendicular slabs, each with a non-sealing y gap."""

    boxes: list[ObstacleBox] = []
    gaps: list[float] = []
    # Gap choices intentionally stay away from y=x at the corresponding wall,
    # so the low straight segment crosses a slab.  Their alternating positions
    # require a planner to change lateral direction between barriers.
    choices = ((11.0, 13.0), (5.0, 7.0), (12.0, 14.0))
    for x, gap_options in zip((5.0, 10.0, 15.0), choices):
        wall_x = x + rng.uniform(-0.35, 0.35)
        gap = rng.choice(gap_options) + rng.uniform(-0.35, 0.35)
        gap_half = rng.uniform(1.0, 1.35)
        top = rng.uniform(5.2, 6.9)
        y_min, y_max = 2.0, 18.0
        left_max, right_min = gap - gap_half, gap + gap_half
        boxes.append(ObstacleBox((wall_x - 0.65, y_min, 0.0), (wall_x + 0.65, left_max, top)))
        boxes.append(ObstacleBox((wall_x - 0.65, right_min, 0.0), (wall_x + 0.65, y_max, top)))
        gaps.append(gap)
    return boxes, gaps


def _build_scenario(rootseed: int, family: str, index: int) -> Scenario3D:
    mapseed = _map_seed(rootseed, family, index)
    rng = random.Random(mapseed)
    if family == "clutter":
        obstacles = _make_clutter(rng)
        family_metadata = {}
    elif family == "urban":
        obstacles = _make_urban(rng)
        family_metadata = {}
    elif family == "barriers":
        obstacles, gaps = _make_barriers(rng)
        family_metadata = {"barrier_gap_centers_y": [round(value, 6) for value in gaps]}
    else:  # Internal guard; public validation catches this earlier.
        raise ValueError(f"unknown research scenario family: {family}")

    metadata = {
        "generator": RESEARCH_GENERATOR_VERSION,
        "rootseed": rootseed,
        "mapseed": mapseed,
        "family": family,
        "family_index": index,
        "obstacle_count": len(obstacles),
        "straight_line_blocked_by_construction": True,
        "high_corridor_max_obstacle_z": 7.0,
        **family_metadata,
    }
    return Scenario3D(
        f"research_{family}_{index:02d}",
        Environment3D(RESEARCH_BOUNDS, obstacles),
        RESEARCH_START,
        RESEARCH_GOAL,
        metadata,
    )


def make_research_scenarios(seed: int, count_per_family: int = 8) -> list[Scenario3D]:
    """Generate independent clutter, urban and barriers maps.

    The output order is family-major and index-minor.  Map seeds are derived
    with SHA-256 from ``(seed, family, index)`` and are therefore stable across
    Python processes and machines.  No planner is called and no map is rejected
    based on a planner result.
    """

    _validate_seed(seed)
    if isinstance(count_per_family, bool) or not isinstance(count_per_family, int) or count_per_family <= 0:
        raise ValueError("count_per_family must be a positive integer")
    return [
        _build_scenario(seed, family, index)
        for family in RESEARCH_FAMILIES
        for index in range(count_per_family)
    ]


def split_research_scenarios(
    scenarios: Sequence[Scenario3D], *, train_fraction: float = 0.5, validation_fraction: float = 0.25,
    protocol: str | None = None,
) -> ResearchScenarioSplit:
    """Return a deterministic, family-stratified train/validation/test split.

    Splitting is by complete map, never by obstacle or perturbation.  Maps are
    sorted by their stored family index/name within each family.  The default
    (``protocol=None``) preserves the historical 12/6/6 split used by earlier
    records.  ``protocol='research24_6_6_12'`` is the confirmatory protocol:
    it requires exactly 24 maps and assigns 2/2/4 maps to every family
    (train/validation/test), for 6/6/12 overall.  The
    ``research24_8_4_12`` protocol is retained for compatibility with an
    earlier exploratory run.  At least three maps per family are required for
    the fraction-based legacy split.
    """

    if not scenarios:
        raise ValueError("scenarios must be non-empty")
    if protocol is not None and protocol not in RESEARCH_SPLIT_PROTOCOLS:
        raise ValueError(f"unknown research split protocol: {protocol!r}")
    if protocol in {RESEARCH_24_SPLIT_PROTOCOL, RESEARCH_24_8_4_12_SPLIT_PROTOCOL}:
        expected_total = len(RESEARCH_FAMILIES) * 8
        if len(scenarios) != expected_total:
            raise ValueError(
                f"{protocol} requires exactly {expected_total} maps"
            )
    if not math.isfinite(train_fraction) or not math.isfinite(validation_fraction):
        raise ValueError("split fractions must be finite")
    if train_fraction <= 0.0 or validation_fraction <= 0.0 or train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must be positive and leave test data")

    groups: dict[str, list[Scenario3D]] = {family: [] for family in RESEARCH_FAMILIES}
    names: set[str] = set()
    for scenario in scenarios:
        if scenario.name in names:
            raise ValueError(f"duplicate scenario name: {scenario.name}")
        names.add(scenario.name)
        metadata = scenario.metadata or {}
        family = metadata.get("family")
        if family not in groups:
            raise ValueError("every scenario must carry a recognized research family")
        if metadata.get("generator") != RESEARCH_GENERATOR_VERSION:
            raise ValueError("split_research_scenarios accepts only research-scenario maps")
        groups[family].append(scenario)

    split: dict[str, list[Scenario3D]] = {"train": [], "validation": [], "test": []}
    # The test quotas are fixed at four per family.  This makes the family
    # means directly comparable and prevents test-family imbalance from being
    # hidden by an aggregate average.
    confirmatory_quotas = (
        {family: (2, 2, 4) for family in RESEARCH_FAMILIES}
        if protocol == RESEARCH_24_SPLIT_PROTOCOL
        else {"clutter": (3, 1, 4), "urban": (3, 1, 4), "barriers": (2, 2, 4)}
    )
    for family in RESEARCH_FAMILIES:
        items = sorted(groups[family], key=lambda item: (int((item.metadata or {}).get("family_index", 10**9)), item.name))
        if len(items) < 3:
            raise ValueError(f"family {family!r} needs at least three maps for a split")
        if protocol in {RESEARCH_24_SPLIT_PROTOCOL, RESEARCH_24_8_4_12_SPLIT_PROTOCOL}:
            if len(items) != 8:
                raise ValueError(
                    f"{protocol} requires exactly eight maps in family {family!r}"
                )
            train_count, validation_count, test_count = confirmatory_quotas[family]
        else:
            train_count = max(1, int(len(items) * train_fraction))
            validation_count = max(1, int(len(items) * validation_fraction))
            if train_count + validation_count >= len(items):
                validation_count = max(1, len(items) - train_count - 1)
            test_count = len(items) - train_count - validation_count
        split["train"].extend(items[:train_count])
        split["validation"].extend(items[train_count:train_count + validation_count])
        split["test"].extend(items[train_count + validation_count:train_count + validation_count + test_count])
    return ResearchScenarioSplit(tuple(split["train"]), tuple(split["validation"]), tuple(split["test"]))


def split_research_scenarios_24(scenarios: Sequence[Scenario3D]) -> ResearchScenarioSplit:
    """Apply the confirmatory 24-map/6-6-12 protocol.

    This named wrapper is intentionally explicit in experiment scripts and
    notebooks.  It leaves :func:`split_research_scenarios`' historical default
    unchanged, so prior v4/v5 result directories remain reproducible.
    """

    return split_research_scenarios(scenarios, protocol=RESEARCH_24_SPLIT_PROTOCOL)


def high_corridor_path() -> Path3D:
    """Return a geometry-only high-altitude sanity path for this map bank."""

    return Path3D([RESEARCH_START, (1.0, 1.0, 8.0), (19.0, 19.0, 8.0), RESEARCH_GOAL])


__all__ = [
    "LEGACY_RESEARCH_SPLIT_PROTOCOL", "RESEARCH_24_SPLIT_PROTOCOL", "RESEARCH_24_8_4_12_SPLIT_PROTOCOL",
    "RESEARCH_SPLIT_PROTOCOLS",
    "RESEARCH_BOUNDS", "RESEARCH_FAMILIES", "RESEARCH_GENERATOR_VERSION", "RESEARCH_GOAL", "RESEARCH_START",
    "ResearchScenarioSplit",
    "high_corridor_path", "make_research_scenarios", "split_research_scenarios", "split_research_scenarios_24",
]
