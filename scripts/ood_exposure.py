"""Append-only exposure events and conservative OOD evidence classification."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def exposure_state(bank):
    bank = Path(bank)
    manifest = read_json(bank / "manifest.json")
    ledger_path = bank / "EXPOSURE_LEDGER.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else dict(manifest.get("exposure_ledger", {}))
    events_path = bank / "EXPOSURE_EVENTS.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()] if events_path.exists() else []
    viewed = set(ledger.get("viewed_maps", []))
    exposed = bool(ledger.get("outcomes_exposed") or ledger.get("used_for_parameter_selection"))
    for event in events:
        if event.get("event") in ("results_viewed", "evaluation_started", "pilot_reclassification"):
            exposed = True
            viewed.update(event.get("maps", []))
    purpose = manifest.get("purpose", "pilot")
    # Legacy manifests were created before explicit purpose/lock semantics and
    # can never retrospectively acquire blind status.
    blind_eligible = purpose == "blind" and not exposed and bool(manifest.get("method_lock_sha256"))
    return {**ledger, "purpose": purpose, "outcomes_exposed": exposed,
            "viewed_maps": sorted(viewed), "blind_eligible": blind_eligible,
            "current_evidence_role": "prospective_blind" if blind_eligible else "pilot_descriptive_only",
            "event_count": len(events), "events": events}


def append_exposure(bank, event, maps=(), output_path=None, note=None):
    """Append history; refresh the convenience ledger without rewriting history."""
    bank = Path(bank)
    payload = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": event,
               "maps": list(maps), "output_path": str(output_path) if output_path else None,
               "note": note}
    with (bank / "EXPOSURE_EVENTS.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    state = exposure_state(bank)
    state.pop("events", None)
    path = bank / "EXPOSURE_LEDGER.json"
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return state
