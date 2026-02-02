"""
Types and enumerations for adaptive DSP failure detection.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Any

class Symptom(Enum):
    """
    Canonical failure symptoms mapped 1→1 to common local responses in adaptive DSP:
    - OVER_TORQUE_NO_ADVANCE → change tool/method for (un)screwing
    - ACCESS_BLOCKED → bypass if precedence allows; else replan
    - NO_SEPARATION → change process (thermal/solvent/mech) for adhesives
    - LOW_VALUE_BOTTLENECK → destroy if permitted by policy; else replan

    Adaptive decision layers in the literature often encode these as fuzzy rules via
    Fuzzy Reasoning/Attributed Petri Nets (FRPN/FAPN).
    """

    OVER_TORQUE_NO_ADVANCE = auto()
    ACCESS_BLOCKED = auto()
    NO_SEPARATION = auto()
    LOW_VALUE_BOTTLENECK = auto()
    TIMEOUT = auto()
    MANUAL_FLAG = auto()

@dataclass
class FailureEvent:
    comp_id: str
    op_type: str
    tool_id: Optional[str]
    symptom: Symptom
    signals: dict[str, float]
    context: dict[str, Any]
    history: dict[str, Any]
    timestamp: str  # UTC ISOformat

@dataclass
class Thresholds:
    over_torque_k: float = 1.8
    over_torque_time_s: float = 1.5
    no_sep_force_N: float = 10.0
    timeout_s: float = 8.0
    low_value_cutoff: float = 0.2
