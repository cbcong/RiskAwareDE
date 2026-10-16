"""Atomic JSON checkpoints with bounded retries for Windows sharing violations."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time


def replace_with_retry(source, destination, timeout=8.0):
    deadline = time.monotonic() + timeout
    delay = 0.025
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (None, 5, 32, 33) or time.monotonic() >= deadline:
                raise
            time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
            delay = min(delay * 2, 0.4)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    # Serialize before creating a file. Each writer has a distinct temporary;
    # a persistent error preserves the original file and the pending payload.
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(name, path)
