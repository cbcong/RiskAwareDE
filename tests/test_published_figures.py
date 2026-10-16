"""Saved paper figures must use portable, verified inputs, never old fallbacks."""
import hashlib
import json

import pytest

from run_figures import verify_snapshot


def make_snapshot(tmp_path):
    inputs = tmp_path / "paper_inputs"
    inputs.mkdir()
    hashes = {}
    for extension in ("pdf", "png", "svg"):
        name = "fig01_framework." + extension
        data = ("saved framework " + extension).encode()
        (inputs / name).write_bytes(data)
        hashes[name] = {"sha256": hashlib.sha256(data).hexdigest()}
    (inputs / "manifest.json").write_text(json.dumps({"inputs": hashes}), encoding="utf-8")
    files = {"paper_inputs/" + p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                       "bytes": p.stat().st_size} for p in inputs.iterdir()}
    manifest = {"schema": 1, "manuscript_version": "3.1.0", "files": files,
                "role": "read_only_published_evidence"}
    (tmp_path / "snapshot.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_snapshot_is_relocatable_and_does_not_require_raw_runs(tmp_path):
    manifest = make_snapshot(tmp_path)
    assert verify_snapshot(tmp_path, "1") == manifest
    assert not (tmp_path / "run_evidence").exists()


def test_snapshot_rejects_changed_inputs(tmp_path):
    make_snapshot(tmp_path)
    (tmp_path / "paper_inputs/fig01_framework.pdf").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Published snapshot file changed"):
        verify_snapshot(tmp_path, "1")


def test_snapshot_rejects_untracked_inputs(tmp_path):
    manifest = make_snapshot(tmp_path)
    del manifest["files"]["paper_inputs/fig01_framework.pdf"]
    (tmp_path / "snapshot.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Untracked published plotting inputs"):
        verify_snapshot(tmp_path, "1")


@pytest.mark.parametrize("relative", ["../escape", "C:/escape", "/escape", "paper_inputs\\escape"])
def test_snapshot_rejects_nonportable_paths(tmp_path, relative):
    manifest = make_snapshot(tmp_path)
    manifest["files"][relative] = {"sha256": "", "bytes": 0}
    (tmp_path / "snapshot.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="relative and portable"):
        verify_snapshot(tmp_path, "1")
