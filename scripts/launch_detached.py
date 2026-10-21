from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Sequence


def launch_detached(command: Sequence[str], log_path: Path) -> int:
    """Start a detached process with stdout and stderr captured in one log."""
    if not command:
        raise ValueError("command must not be empty")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    creationflags = (
        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    )
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
            creationflags=creationflags,
        )
    return int(process.pid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a reproducible experiment command without a terminal session."
    )
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    process_id = launch_detached(command, args.log)
    print(json.dumps({"pid": process_id, "log": str(args.log)}))


if __name__ == "__main__":
    main()
