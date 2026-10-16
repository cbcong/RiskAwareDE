import ctypes
import json
import os
from pathlib import Path
import threading
import time

import pytest

from reproducibility import atomic_io


def test_transient_permission_error_keeps_then_replaces_complete_json(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    atomic_io.write_json(path, {"completed": 3135})
    replace = os.replace
    attempts = []

    def busy(source, destination):
        attempts.append(1)
        assert json.loads(path.read_text()) == {"completed": 3135}
        if len(attempts) < 3:
            raise PermissionError(13, "file temporarily in use")
        replace(source, destination)

    monkeypatch.setattr(atomic_io.os, "replace", busy)
    atomic_io.write_json(path, {"completed": 3136})
    assert len(attempts) == 3
    assert json.loads(path.read_text()) == {"completed": 3136}
    assert not list(tmp_path.glob("*.tmp"))


def test_permanent_failure_preserves_old_state_and_pending_payload(tmp_path, monkeypatch):
    old = tmp_path / "state.json"
    new = tmp_path / "pending.tmp"
    old.write_text('{"completed":3135}')
    new.write_text('{"completed":3136}')

    def forbidden(*args):
        raise PermissionError(13, "persistent denial")

    monkeypatch.setattr(atomic_io.os, "replace", forbidden)
    with pytest.raises(PermissionError):
        atomic_io.replace_with_retry(new, old, timeout=.04)
    assert json.loads(old.read_text())["completed"] == 3135
    assert json.loads(new.read_text())["completed"] == 3136


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_real_windows_reader_temporarily_denies_replacement(tmp_path):
    path = tmp_path / "state.json"
    atomic_io.write_json(path, {"completed": 3135})
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    ready = threading.Event()
    errors = []

    def reader():
        handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            errors.append(ctypes.get_last_error())
            ready.set()
            return
        ready.set()
        time.sleep(.18)
        kernel.CloseHandle(handle)

    thread = threading.Thread(target=reader)
    thread.start()
    assert ready.wait(3)
    try:
        assert not errors
        atomic_io.write_json(path, {"completed": 3136})
    finally:
        thread.join(3)
    assert json.loads(path.read_text()) == {"completed": 3136}
