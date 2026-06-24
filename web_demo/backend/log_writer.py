"""JSONL log writer for per-phrase request logs.

One file per session run; rotates on user request via `start_new_log()`.
Writes are append-only, line-delimited JSON, flushed on every record so
crashes don't lose data.
"""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Optional


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LOG_DIR = os.path.join(_REPO_ROOT, "logs")

_current_path: Optional[str] = None
_lock = threading.Lock()


def _new_path() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(_LOG_DIR, f"{ts}.jsonl")


def start_new_log() -> str:
    global _current_path
    with _lock:
        os.makedirs(_LOG_DIR, exist_ok=True)
        _current_path = _new_path()
        # Touch the file so listing tools / tail -f see it immediately.
        with open(_current_path, "a", encoding="utf-8"):
            pass
    return _current_path


def current_log_path() -> str:
    if _current_path is None:
        return start_new_log()
    return _current_path


def append(record: dict) -> None:
    path = current_log_path()
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
