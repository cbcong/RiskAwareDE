"""Redraw completed experiments, or the saved manuscript snapshot with --published.

Examples:
    python run_figures.py --published
    python run_figures.py --published --figure 7
    python run_figures.py --published --check

Without --published, this retains the experiment pipeline's figures launcher.
The published mode reads paper_results and writes only to a separate output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

from reproducibility.launcher import bootstrap, main as launch_main
from reproducibility.fixed_environment import check as check_environment, subprocess_environment

ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_snapshot(snapshot: Path, figure: str = "all") -> dict:
    """Verify portable saved evidence before allowing it to supply a figure.

    Optional raw run evidence is not needed here. Every file listed in
    snapshot.json is required, and every input used by the selected figure
    must be listed; this prevents a damaged or incomplete snapshot from
    silently falling back to older data/plot_inputs files.
    """
    from plotting.render import FIGURES

    snapshot = snapshot.resolve()
    manifest_file = snapshot / "snapshot.json"
    if not manifest_file.is_file():
        raise RuntimeError("Missing paper_results/snapshot.json. Restore the published-results bundle first.")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    if (manifest.get("schema") != 1 or
            manifest.get("role") != "read_only_published_evidence" or
            not manifest.get("manuscript_version")):
        raise RuntimeError("Unrecognized published snapshot manifest.")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("The published snapshot has no tracked files.")
    chosen = list(FIGURES) if figure == "all" else [figure]
    if any(key not in FIGURES for key in chosen):
        raise ValueError(f"Unknown figure: {figure}")
    needed = {name for key in chosen for name in FIGURES[key]["inputs"]}
    required = {"paper_inputs/manifest.json", *("paper_inputs/" + name for name in needed)}
    missing = required - files.keys()
    if missing:
        raise RuntimeError("Untracked published plotting inputs: " + ", ".join(sorted(missing)))
    for relative, expected in files.items():
        relative_path = PurePosixPath(relative)
        if (not relative or "\\" in relative or ":" in relative or
                relative_path.is_absolute() or ".." in relative_path.parts or
                relative_path.as_posix() != relative):
            raise RuntimeError(f"Snapshot path must be relative and portable: {relative}")
        candidate = (snapshot / relative_path).resolve()
        if not candidate.is_relative_to(snapshot) or not candidate.is_file():
            raise RuntimeError(f"Missing or invalid snapshot file: {relative}")
        if (candidate.stat().st_size != expected.get("bytes") or
                _sha256(candidate) != expected.get("sha256")):
            raise RuntimeError(f"Published snapshot file changed: {relative}")
    input_manifest = json.loads((snapshot / "paper_inputs/manifest.json").read_text(encoding="utf-8-sig"))
    for name in needed:
        expected = input_manifest.get("inputs", {}).get(name, {}).get("sha256")
        if expected != files["paper_inputs/" + name]["sha256"]:
            raise RuntimeError(f"Published input manifests disagree: {name}")
    return manifest


def main() -> int:
    if "--published" not in sys.argv[1:]:
        return launch_main("figures")
    forwarded = bootstrap()
    if forwarded is not None:
        return forwarded
    from plotting.render import FIGURES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published", action="store_true", required=True)
    parser.add_argument("--figure", choices=["all", *FIGURES], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/published_figures")
    parser.add_argument("--check", action="store_true", help="Verify environment and saved results without drawing")
    args = parser.parse_args()
    snapshot = ROOT / "paper_results"
    output = args.output.resolve()
    if output == snapshot.resolve() or output.is_relative_to(snapshot.resolve()):
        parser.error("--output must be outside the read-only paper_results directory")
    check_environment()
    manifest = verify_snapshot(snapshot, args.figure)
    print(f"Verified saved manuscript {manifest['manuscript_version']} results ({len(manifest['files'])} files).", flush=True)
    if args.check:
        return 0
    result = subprocess.run(
        [sys.executable, str(ROOT / "plotting/render.py"), "--input", str(snapshot / "paper_inputs"),
         "--figure", args.figure, "--output", str(output)],
        cwd=ROOT, env=subprocess_environment(),
    )
    if result.returncode:
        return result.returncode
    verify_snapshot(snapshot, args.figure)
    print(f"Published figures redrawn from saved results: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
