"""Internal native-runtime worker, driven by cohorts.py through JSON lines."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback

from .cohorts import checksum, execute, read


def send(value):
    print(json.dumps(value, ensure_ascii=True, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    import numpy
    environment = dict(executable=sys.executable, implementation=platform.python_implementation(),
                       python_version=list(sys.version_info[:3]), numpy_version=numpy.__version__)
    from .fixed_environment import identity
    environment_sha = identity()
    protocol = read(args.protocol)
    send(dict(ok=True, environment=environment))
    for line in sys.stdin:
        try:
            bundle = execute(json.loads(line), protocol)
            bundle["reproduction_runtime"] = environment
            bundle["environment_lock_sha256"] = environment_sha
            bundle["checkpoint_sha256"] = checksum(bundle)
            send(dict(ok=True, bundle=bundle))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            send(dict(ok=False, error=repr(exc)))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
