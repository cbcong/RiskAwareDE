"""A small occupancy-grid representation for discrete planning baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .models import Point3D


Cell3D = tuple[int, int, int]


@dataclass
class GridMap3D:
    """Finite 3-D occupancy grid.

    Coordinates are integer cell indices in ``[0, size[i])``.  This class is
    intentionally focused on storage and collision queries; A* or sampling
    planners can be added later without changing the data model.
    """

    size: tuple[int, int, int]
    occupied: set[Cell3D] = field(default_factory=set)

    def __post_init__(self) -> None:
        if len(self.size) != 3 or any(n <= 0 for n in self.size):
            raise ValueError("size must contain three positive integers")
        self.occupied = {self._validate_cell(c) for c in self.occupied}

    def _validate_cell(self, cell: Iterable[int]) -> Cell3D:
        values = tuple(int(x) for x in cell)
        if len(values) != 3 or any(i < 0 or i >= n for i, n in zip(values, self.size)):
            raise ValueError(f"cell {values} is outside grid {self.size}")
        return values  # type: ignore[return-value]

    def in_bounds(self, cell: Iterable[int]) -> bool:
        values = tuple(cell)
        return len(values) == 3 and all(0 <= i < n for i, n in zip(values, self.size))

    def is_free(self, cell: Iterable[int]) -> bool:
        values = tuple(int(x) for x in cell)
        return self.in_bounds(values) and values not in self.occupied

    def set_occupied(self, cell: Iterable[int], occupied: bool = True) -> None:
        values = self._validate_cell(cell)
        if occupied:
            self.occupied.add(values)
        else:
            self.occupied.discard(values)

    def neighbors(self, cell: Iterable[int], include_diagonal: bool = False) -> list[Cell3D]:
        c = self._validate_cell(cell)
        offsets = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if (dx, dy, dz) == (0, 0, 0):
                        continue
                    if include_diagonal or sum(v != 0 for v in (dx, dy, dz)) == 1:
                        offsets.append((dx, dy, dz))
        return [n for n in ((c[0] + dx, c[1] + dy, c[2] + dz) for dx, dy, dz in offsets) if self.is_free(n)]

    @classmethod
    def empty(cls, size: tuple[int, int, int]) -> "GridMap3D":
        return cls(size=size)
