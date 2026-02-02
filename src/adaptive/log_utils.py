"""
Logging utilities for adaptive DSP failure detection.
"""
import os
import json
from enum import Enum
from .types import FailureEvent

FAIL_LOG_PATH = os.path.join("results", "adaptive_logs", "failures.jsonl")

def append_jsonl(path: str, obj: dict) -> None:
    """Append a JSON object as a line in a .jsonl file, create folder if needed.
    Serializes Enum to string if needed."""
    def default(o):
        if hasattr(o, "__dict__"):
            return o.__dict__
        if isinstance(o, Enum):
            return o.name
        return str(o)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=default) + "\n")

def log_failure_event(ev: FailureEvent, path: str = FAIL_LOG_PATH) -> None:
    """Log a failure event to the .jsonl file."""
    # Explicit conversion of symptom Enum to string for serialization
    d = ev.__dict__.copy()
    if hasattr(ev, "symptom") and isinstance(ev.symptom, Enum):
        d["symptom"] = ev.symptom.name
    append_jsonl(path, d)

def log_failure(log_data: dict, log_path: str = FAIL_LOG_PATH) -> None:
    """Log a detailed dictionary (enriched test result) to the .jsonl file."""
    append_jsonl(log_path, log_data)
