"""Data provisioning never trusts archive names or replaces existing records."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

import prepare_data as data


@pytest.fixture
def payload(tmp_path):
    root, full = tmp_path / "github", tmp_path / "full"
    root.mkdir()
    full.mkdir()
    contents = {"data/reference/e/results/a.json": b'{"seed": 1}\n',
                "data/reference/p6/results/b.json": b'{"seed": 2}\n'}
    manifest = {"schema": 1, "download_url": None, "files": {
        name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
        for name, value in contents.items()}}
    for name, value in contents.items():
        path = full / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    (root / "provenance").mkdir()
    (root / data.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return root, full, contents, manifest


def test_full_import_then_check_and_idempotent_repeat(payload):
    root, full, contents, manifest = payload
    with pytest.raises(ValueError, match="Missing 2 frozen data"):
        data.check(root)
    assert data.restore(root, manifest, full=full) == 2
    assert data.check(root) == manifest
    assert data.restore(root, manifest, full=full) == 0
    for name, value in contents.items():
        assert (root / name).read_bytes() == value


def test_zip_import_validates_exact_manifest_paths(payload, tmp_path):
    root, _, contents, manifest = payload
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, value in contents.items():
            bundle.writestr(name, value)
    assert data.restore(root, manifest, archive=archive) == 2
    data.check(root)


def test_bad_final_source_prevents_all_destination_writes(payload):
    root, full, contents, manifest = payload
    (full / list(contents)[-1]).write_bytes(b"wrong")
    with pytest.raises(ValueError, match="Payload checksum mismatch"):
        data.restore(root, manifest, full=full)
    assert all(not (root / name).exists() for name in contents)
    assert not (root / ".data_import.lock").exists()


def test_existing_different_result_is_never_overwritten(payload):
    root, full, contents, manifest = payload
    existing = root / next(iter(contents))
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"a result belonging to someone else")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        data.restore(root, manifest, full=full)
    assert existing.read_bytes() == b"a result belonging to someone else"


@pytest.mark.parametrize("extra", ["../escape.json", "data/reference/../../escape.json",
                                    "data/reference/tool.py", "data/reference/e/results/extra.json"])
def test_malicious_or_wrong_archive_is_rejected_without_writes(payload, tmp_path, extra):
    root, _, contents, manifest = payload
    archive = tmp_path / "untrusted.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, value in contents.items():
            bundle.writestr(name, value)
        bundle.writestr(extra, b"bad")
    with pytest.raises(ValueError):
        data.restore(root, manifest, archive=archive)
    assert all(not (root / name).exists() for name in contents)
    assert not (tmp_path / "escape.json").exists()


@pytest.mark.parametrize("path", ["C:/data/reference/a.json", "data/reference/../a.json",
                                   "data\\reference\\a.json", "data/reference/CON.json",
                                   "data/reference/a.json:stream", "data/reference/a.json ",
                                   "src/uav_path_planning/models.py", "data/reference/helper.py"])
def test_manifest_rejects_unsafe_or_executable_paths(payload, path):
    root, _, _, manifest = payload
    item = next(iter(manifest["files"].values()))
    manifest["files"] = {path: item}
    (root / data.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        data.load_manifest(root)


def test_manifest_rejects_windows_case_collision(payload):
    root, _, _, manifest = payload
    name, item = next(iter(manifest["files"].items()))
    manifest["files"][name.replace("a.json", "A.json")] = item
    (root / data.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="case-insensitive"):
        data.load_manifest(root)


def test_check_never_downloads_and_missing_url_is_explicit(payload, monkeypatch, capsys):
    root, _, _, _ = payload
    monkeypatch.setattr(data, "ROOT", root)
    monkeypatch.setattr(data, "download", lambda *_: pytest.fail("Unexpected network operation"))
    assert data.main(["--check"]) == 1
    assert "Missing 2" in capsys.readouterr().err
    assert data.main([]) == 1
    assert "No data download URL is configured" in capsys.readouterr().err


def test_download_rejects_non_https_before_opening(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "urlopen", lambda *_a, **_k: pytest.fail("Unexpected network operation"))
    with pytest.raises(ValueError, match="HTTPS"):
        data.download("file:///C:/secret", tmp_path / "download.zip")


def test_import_supports_filesystems_without_hardlinks(payload, monkeypatch):
    root, full, _, manifest = payload
    def unavailable(*_a, **_k):
        raise OSError("Hard links are unsupported on this volume")
    monkeypatch.setattr(data.os, "link", unavailable)
    assert data.restore(root, manifest, full=full) == 2
    data.check(root)
