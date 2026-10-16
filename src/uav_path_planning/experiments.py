"""Reproducible experiment records and serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .metrics import PathMetrics


@dataclass
class ExperimentRecord:
    algorithm: str
    scenario: str
    seed: int
    runtime_seconds: float
    evaluations: int
    metrics: PathMetrics
    status: str = "ok"
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRecord":
        """Reconstruct a record written by :func:`save_records`.

        JSON is the canonical interchange format because it preserves the
        nested planner metadata.  The method also accepts rows read from the
        flattened CSV writer, which is useful for lightweight reports.
        """

        raw_metrics = data.get("metrics")
        if isinstance(raw_metrics, dict):
            metrics_data = raw_metrics
        else:
            metric_keys = set(PathMetrics.__dataclass_fields__)
            metrics_data = {key: data[key] for key in metric_keys if key in data}
        metrics = PathMetrics(
            length=float(metrics_data["length"]),
            collision_segments=int(float(metrics_data["collision_segments"])),
            collision_samples=int(float(metrics_data["collision_samples"])),
            min_obstacle_distance=float(metrics_data["min_obstacle_distance"]),
            smoothness=float(metrics_data["smoothness"]),
            altitude_violation=float(metrics_data["altitude_violation"]),
            feasible=(
                metrics_data["feasible"]
                if isinstance(metrics_data["feasible"], bool)
                else str(metrics_data["feasible"]).strip().lower() in {"1", "true", "yes"}
            ),
            objective=float(metrics_data["objective"]),
        )
        config = data.get("config", {})
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except json.JSONDecodeError:
                config = {"raw": config}
        return cls(
            algorithm=str(data["algorithm"]),
            scenario=str(data["scenario"]),
            seed=int(float(data["seed"])),
            runtime_seconds=float(data["runtime_seconds"]),
            evaluations=int(float(data["evaluations"])),
            metrics=metrics,
            status=str(data.get("status", "ok")),
            config=config if isinstance(config, dict) else {},
        )


def save_records(records: Iterable[ExperimentRecord], path: str | Path) -> None:
    rows = [record.to_dict() for record in records]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".json":
        target.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return
    if target.suffix.lower() != ".csv":
        raise ValueError("records path must end with .json or .csv")
    fields = ["algorithm", "scenario", "seed", "runtime_seconds", "evaluations", "status", *PathMetrics.__dataclass_fields__.keys()]
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {k: row[k] for k in fields if k in row}
            flat.update(row["metrics"])
            writer.writerow(flat)


def load_records(path: str | Path) -> list[ExperimentRecord]:
    """Load records from JSON or the flattened CSV representation."""

    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("records JSON must contain a list")
        return [ExperimentRecord.from_dict(dict(row)) for row in payload]
    if source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            return [ExperimentRecord.from_dict(dict(row)) for row in csv.DictReader(handle)]
    raise ValueError("records path must end with .json or .csv")
