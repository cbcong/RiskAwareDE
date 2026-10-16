"""Freeze a selected scientific method before generating a blind OOD bank.

This command records content hashes; it never claims the development gate was
passed.  That gate is a separate launch decision.  Any later scientific code or
configuration change invalidates this lock and requires a new untouched bank.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_source_hashes(repo):
    repo = Path(repo).resolve()
    paths = sorted(list((repo / "src").rglob("*.py")) + list((repo / "scripts").glob("*.py")))
    return {path.relative_to(repo).as_posix(): file_digest(path) for path in paths}


def _configuration_names(protocol, profile):
    names = {protocol["study_config"], protocol["main_method"]["config_file"]}
    if profile == "manuscript":
        # The matched controls and reuse provenance are configured separately
        # from the original seven-category defaults.
        names.update({"configs/paper_evidence_study.json", "configs/main_method_lock.json"})
    return names


def create_method_lock(repo, protocol_profile="full"):
    from seven_category_protocol import build_protocol
    if protocol_profile not in ("full", "manuscript"):
        raise ValueError("protocol_profile must be full or manuscript")
    repo = Path(repo).resolve()
    protocol = build_protocol(repo, profile=protocol_profile)
    configs = _configuration_names(protocol, protocol_profile)
    return {"schema": "ood-method-lock-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_at_lock": str(repo), "protocol_profile": protocol_profile, "protocol": protocol,
            "protocol_sha256": canonical_digest(protocol), "source_sha256": scientific_source_hashes(repo),
            "config_sha256": {name: file_digest(repo / name) for name in sorted(configs)},
            "note": "Method/content freeze only; no assertion that the independent development gate passed."}


def validate_method_lock(path, repo):
    from seven_category_protocol import build_protocol
    path, repo = Path(path), Path(repo).resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("schema") != "ood-method-lock-v1":
        raise ValueError("method-lock must use ood-method-lock-v1 and freeze protocol/source/config")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict) or canonical_digest(protocol) != payload.get("protocol_sha256"):
        raise ValueError("method-lock protocol hash mismatch")
    if payload.get("source_sha256") != scientific_source_hashes(repo):
        raise ValueError("scientific source changed after method lock")
    # Legacy full locks predate the explicit profile field and remain valid.
    profile = payload.get("protocol_profile", "full")
    if profile not in ("full", "manuscript"):
        raise ValueError("Unsupported method-lock protocol profile")
    current_protocol = build_protocol(repo, profile=profile)
    if current_protocol != protocol:
        raise ValueError("scientific protocol changed after method lock")
    expected_configs = _configuration_names(protocol, profile)
    configs = payload.get("config_sha256", {})
    if set(configs) != expected_configs or any(file_digest(repo / name) != value for name, value in configs.items()):
        raise ValueError("scientific configuration changed after method lock")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("full", "manuscript"), default="full",
                        help="Use manuscript for matched soft_beta050 E0--E6; full retains the legacy method")
    args = parser.parse_args()
    payload = create_method_lock(ROOT, protocol_profile=args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"method_lock": str(args.output.resolve()), "sha256": file_digest(args.output), "planners_run": 0}))


if __name__ == "__main__":
    main()
