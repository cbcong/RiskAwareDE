"""Small experiment lifecycle record; never silently overwrite a prior run."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys


@contextmanager
def recorded_run(output, protocol, *, allowed_existing=()):
    """Persist protocol and source hashes before work, then record completion.

    A killed process leaves ``running`` and must be treated as incomplete.
    Completion records execution only, not scientific validity or independence.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    unexpected = [p.name for p in output.iterdir() if p.name not in allowed_existing]
    if unexpected:
        raise FileExistsError(f"Refusing to overwrite prior artifacts in {output}: {unexpected[:5]}")
    root = Path(__file__).resolve().parents[2]
    files = [*Path(__file__).parent.glob('*.py')]
    entry = Path(sys.argv[0]).resolve()
    if entry.is_file(): files.append(entry)
    manifest = dict(protocol)
    manifest.update({
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'running', 'python': sys.version, 'platform': platform.platform(),
        'argv': sys.argv,
        'source_sha256': {str(p.relative_to(root) if p.is_relative_to(root) else p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    })
    target = output / 'execution_manifest.json'
    with target.open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, default=str)
    try:
        yield manifest
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=f'{type(exc).__name__}: {exc}')
        raise
    else:
        manifest['status'] = 'complete'
    finally:
        manifest['ended_at_utc'] = datetime.now(timezone.utc).isoformat()
        target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
