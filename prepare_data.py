"""Restore the exact frozen data payload without installing or changing code.

Download the published reference data, import a data ZIP with --archive,
or use --from-directory to import previously downloaded data. Every file is
verified against the repository's integrity manifest.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parent
MANIFEST = "provenance/data_bundle.json"
DATA_PREFIXES = (
    "data/reference/",
    "experiments/external_comparison_20261015/runs/",
    "experiments/external_comparison_20261015/analysis/",
    "experiments/external_comparison_20261015/followup_v260/stress_results/",
    "experiments/external_comparison_20261015/followup_v260/budget_analysis/",
    "paper_results/",
)
DATA_EXTENSIONS = {".json", ".csv", ".tsv", ".npz", ".npy", ".png", ".pdf",
                   ".svg", ".txt", ".md", ".tex", ".bib", ".log"}


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe payload path: {value!r}")
    path = PurePosixPath(value)
    parts = value.split("/")
    if path.is_absolute() or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Unsafe payload path: {value!r}")
    if any(part.rstrip(" .") != part or re.search(r'[<>|?*\x00-\x1f]', part)
           or part.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL",
                    *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
           for part in parts):
        raise ValueError(f"Unsafe Windows payload path: {value!r}")
    if not value.startswith(DATA_PREFIXES) or path.suffix.lower() not in DATA_EXTENSIONS:
        raise ValueError(f"Payload may contain approved data only, never source code: {value}")
    return path


def load_manifest(root):
    path = Path(root) / MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != 1 or not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError("Invalid or empty data bundle manifest")
    folded = set()
    for relative, item in manifest["files"].items():
        safe_relative(relative)
        if relative.casefold() in folded:
            raise ValueError(f"Duplicate case-insensitive payload path: {relative}")
        folded.add(relative.casefold())
        if (not isinstance(item, dict) or not isinstance(item.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                or type(item.get("bytes")) is not int or item["bytes"] < 0):
            raise ValueError(f"Invalid file metadata: {relative}")
    return manifest


def checked_path(root, relative):
    safe_relative(relative)
    root = Path(root).resolve()
    path = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise ValueError(f"Redirected payload path is not permitted: {relative}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Payload path escapes destination: {relative}")
    return path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_file(path, item):
    return path.is_file() and path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]


def inspect_payload(root, manifest):
    missing, changed = [], []
    for relative, item in manifest["files"].items():
        path = checked_path(root, relative)
        if not path.exists():
            missing.append(relative)
        elif not verify_file(path, item):
            changed.append(relative)
    return missing, changed


def check(root=ROOT):
    """Validate required data before an entry point opens frozen records."""
    manifest = load_manifest(root)
    missing, changed = inspect_payload(root, manifest)
    if changed:
        raise ValueError("Frozen data differ; no files were overwritten: " + ", ".join(changed[:5]))
    if missing:
        raise ValueError(f"Missing {len(missing)} frozen data files. Run python prepare_data.py or python prepare_data.py --archive <data.zip> before setup or experiments.")
    return manifest


@contextmanager
def import_lock(root):
    path = Path(root) / ".data_import.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError("A data import lock exists. Close other imports; if an earlier import was interrupted, remove .data_import.lock and retry.") from exc
    try:
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def stage_stream(stream, destination, item, relative):
    destination.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    total = 0
    with destination.open("wb") as target:
        while chunk := stream.read(1024 * 1024):
            total += len(chunk)
            if total > item["bytes"]:
                raise ValueError(f"Unexpected payload size: {relative}")
            hasher.update(chunk)
            target.write(chunk)
    if total != item["bytes"] or hasher.hexdigest() != item["sha256"]:
        raise ValueError(f"Payload checksum mismatch: {relative}")


def stage_archive(archive, staging, files):
    with zipfile.ZipFile(archive) as bundle:
        entries = {}
        for info in bundle.infolist():
            if info.is_dir():
                continue
            safe_relative(info.filename)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or info.filename.casefold() in entries:
                raise ValueError(f"Duplicate or symbolic-link ZIP member: {info.filename}")
            entries[info.filename.casefold()] = info
            if info.filename not in files:
                raise ValueError(f"Unexpected ZIP member (use the data-only archive): {info.filename}")
        if len(entries) != len(files):
            raise ValueError("Data archive is incomplete; its paths must match the manifest exactly")
        for relative, item in files.items():
            info = entries.get(relative.casefold())
            if info is None or info.filename != relative or info.file_size != item["bytes"]:
                raise ValueError(f"Missing or wrong-size ZIP member: {relative}")
            with bundle.open(info) as stream:
                stage_stream(stream, Path(staging) / relative, item, relative)


def commit_file(source, destination):
    try:
        # Atomic and no replacement on NTFS, the usual Windows filesystem.
        os.link(source, destination)
    except FileExistsError:
        raise
    except OSError:
        # Removable exFAT/FAT volumes may not support hard links. Exclusive
        # creation still prevents replacement of a file supplied by the user.
        target = destination.open("xb")
        try:
            with target, source.open("rb") as stream:
                shutil.copyfileobj(stream, target, length=1024 * 1024)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise


def restore(root, manifest, *, full=None, archive=None):
    """Verify all input bytes before committing any missing destination file."""
    root = Path(root).resolve()
    if (full is None) == (archive is None):
        raise ValueError("Choose exactly one input: data directory or data archive")
    with import_lock(root):
        missing, changed = inspect_payload(root, manifest)
        if changed:
            raise ValueError("Existing files differ; refusing to overwrite: " + ", ".join(changed[:5]))
        if not missing:
            return 0
        with tempfile.TemporaryDirectory(prefix=".data-import-", dir=root) as temporary:
            staging = Path(temporary)
            if full is not None:
                for relative, item in manifest["files"].items():
                    source = checked_path(full, relative)
                    if not source.is_file():
                        raise ValueError(f"Source directory is missing required data: {relative}")
                    with source.open("rb") as stream:
                        stage_stream(stream, staging / relative, item, relative)
            else:
                stage_archive(archive, staging, manifest["files"])
            # Fail before committing if the destination changed during staging.
            _, changed = inspect_payload(root, manifest)
            if changed:
                raise ValueError("Destination changed during import: " + ", ".join(changed[:5]))
            for relative in missing:
                destination = checked_path(root, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if verify_file(destination, manifest["files"][relative]):
                        continue
                    raise ValueError(f"Destination appeared during import: {relative}")
                commit_file(staging / relative, destination)
        return len(missing)


def download(url, target):
    if urlsplit(url).scheme != "https" or not urlsplit(url).netloc:
        raise ValueError("Data download requires an explicit HTTPS URL")
    with urlopen(url, timeout=60) as response:
        if urlsplit(response.geturl()).scheme != "https":
            raise ValueError("Refusing a download redirected away from HTTPS")
        with Path(target).open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--from-directory", dest="from_full", metavar="DIRECTORY", type=Path, help="Existing directory containing the matching reference data")
    sources.add_argument("--from-full", dest="from_full", type=Path, help=argparse.SUPPRESS)
    sources.add_argument("--archive", type=Path, help="Matching data-only ZIP (paths relative to repository root)")
    sources.add_argument("--url", help="HTTPS URL of the matching data-only ZIP")
    sources.add_argument("--check", action="store_true", help="Only validate the local data; never download")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(ROOT)
        missing, changed = inspect_payload(ROOT, manifest)
        if changed:
            raise ValueError("Existing data differ; refusing to overwrite: " + ", ".join(changed[:5]))
        if not missing:
            print(f"Data verified: {len(manifest['files'])} files; all checksums match.")
            return 0
        if args.check:
            print(f"Missing {len(missing)} of {len(manifest['files'])} frozen data files.", file=sys.stderr)
            print("Run prepare_data.py or prepare_data.py --archive <data.zip>.", file=sys.stderr)
            return 1
        url = args.url or manifest.get("download_url")
        if args.from_full is not None:
            count = restore(ROOT, manifest, full=args.from_full)
        elif args.archive is not None:
            count = restore(ROOT, manifest, archive=args.archive)
        elif url:
            with tempfile.TemporaryDirectory(prefix="riskawarede-download-") as temporary:
                archive = Path(temporary) / "data.zip"
                print("Downloading the published data bundle; all hashes will be checked before import.", flush=True)
                download(url, archive)
                count = restore(ROOT, manifest, archive=archive)
        else:
            raise ValueError("No data download URL is configured. Use --archive <data.zip> or --from-directory <data-directory>.")
        print(f"Imported {count} files. Frozen data verified; continue with setup_environment.py.")
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"Data preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
