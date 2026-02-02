"""
Load adaptive DSP configuration.
"""
import json
import os
from typing import Any
from .types import Thresholds

def safe_get(d: dict, key: str, default: Any) -> Any:
    """Get dictionary value with default fallback."""
    return d[key] if key in d else default

def load_thresholds(path: str | None = "data/adaptive/policy.json") -> Thresholds:
    """Load thresholds from JSON file, or return defaults."""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Thresholds(
            over_torque_k=safe_get(data, "over_torque_k", 1.8),
            over_torque_time_s=safe_get(data, "over_torque_time_s", 1.5),
            no_sep_force_N=safe_get(data, "no_sep_force_N", 10.0),
            timeout_s=safe_get(data, "timeout_s", 8.0),
            low_value_cutoff=safe_get(data, "low_value_cutoff", 0.2),
        )
    return Thresholds()
