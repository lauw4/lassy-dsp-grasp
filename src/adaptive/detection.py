"""
Failure detection module for adaptive DSP pipeline.

Detects 4 failure types based on physical signals:
- OVER_TORQUE_NO_ADVANCE: Torque exceeded, no rotation
- ACCESS_BLOCKED: Physical obstruction
- NO_SEPARATION: Joint won't open despite force
- LOW_VALUE_BOTTLENECK: Low-value component blocking high-value targets
"""
from typing import Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
import os
import json
from src.adaptive.types import FailureEvent, Thresholds, Symptom
from src.adaptive.log_utils import log_failure_event
from src.core.graph_api import node_value, no_destroy_flag, feasible_now_nodes, blocks_high_value_downstream

def build_failure_event(context: dict[str, Any], symptom: Symptom, signals: Optional[dict[str, float]] = None) -> FailureEvent:
    """
    Build a complete FailureEvent from context, symptom and signals.
    Args:
        context: Context dictionary (must contain 'comp_id' and 'op_type').
        symptom: Detected symptom.
        signals: Physical signals dictionary (optional).
    Returns:
        FailureEvent
    """
    comp_id = context.get("comp_id", "unknown")
    op_type = context.get("op_type", "unknown")
    tool_id = context.get("tool_id")
    signals = signals if signals is not None else {}
    history = context.get("history", {})
    timestamp = datetime.now(timezone.utc).isoformat()
    return FailureEvent(
        comp_id=comp_id,
        op_type=op_type,
        tool_id=tool_id,
        symptom=symptom,
        signals=signals,
        context=context,
        history=history,
        timestamp=timestamp
    )

def detect_failure(signals: dict[str, float], context: dict[str, Any], thresholds: Thresholds) -> Optional[FailureEvent]:
    """
    Detect failure according to 4 precise rules, returns FailureEvent or None.
    Args:
        signals: Physical signals dictionary.
        context: Operational context dictionary.
        thresholds: Detection thresholds.
    Returns:
        FailureEvent or None
    """
    comp_id = context.get("comp_id", "unknown")
    op_type = context.get("op_type", "unknown")
    # A) OVER_TORQUE_NO_ADVANCE
    torque_max = signals.get("torque_max")
    torque_nominal = signals.get("torque_nominal")
    duration_over = signals.get("duration_over")
    angle_delta = signals.get("angle_delta")
    if (
        torque_max is not None and torque_nominal is not None and duration_over is not None and angle_delta is not None and
        torque_max > thresholds.over_torque_k * torque_nominal and
        duration_over > thresholds.over_torque_time_s and
        abs(angle_delta) <= 1e-3
    ):
        ev = build_failure_event(context, Symptom.OVER_TORQUE_NO_ADVANCE, signals)
        log_failure_event(ev)
        return ev
    # B) ACCESS_BLOCKED
    access_blocked = signals.get("access_blocked")
    if access_blocked == 1 or access_blocked is True:
        # alt_paths
        alt_paths = context.get("alt_paths")
        if alt_paths is None and "state" in context:
            state = context["state"]
            exclude = {comp_id}
            alt_paths = len(feasible_now_nodes(state, exclude=exclude)) > 0
            context["alt_paths"] = alt_paths
        ev = build_failure_event(context, Symptom.ACCESS_BLOCKED, signals)
        log_failure_event(ev)
        return ev
    # C) NO_SEPARATION
    force_axial = signals.get("force_axial")
    joint_opened = signals.get("joint_opened")
    if (
        force_axial is not None and force_axial > thresholds.no_sep_force_N and
        (joint_opened is not None and (joint_opened == 0 or joint_opened is False))
    ):
        ev = build_failure_event(context, Symptom.NO_SEPARATION, signals)
        log_failure_event(ev)
        return ev
    # D) LOW_VALUE_BOTTLENECK
    value_piece = context.get("value_piece")
    if value_piece is None:
        try:
            value_piece = node_value(comp_id)
            context["value_piece"] = value_piece
        except Exception:
            value_piece = 0.0
    no_destroy = context.get("no_destroy")
    if no_destroy is None:
        try:
            no_destroy = no_destroy_flag(comp_id)
            context["no_destroy"] = no_destroy
        except Exception:
            no_destroy = False
    if (
        value_piece is not None and value_piece < thresholds.low_value_cutoff and
        blocks_high_value_downstream(comp_id) and
        no_destroy is False
    ):
        ev = build_failure_event(context, Symptom.LOW_VALUE_BOTTLENECK, signals)
        log_failure_event(ev)
        return ev
    # E) TIMEOUT
    timeout = signals.get("timeout")
    if timeout == 1 or timeout is True:
        ev = build_failure_event(context, Symptom.TIMEOUT, signals)
        log_failure_event(ev)
        return ev
    return None

def flag_failure_manual(comp_id: str, reason: str, context: dict[str, Any]) -> FailureEvent:
    """
    Build and log a MANUAL_FLAG FailureEvent.
    Args:
        comp_id: Component identifier.
        reason: Textual reason for manual flag.
        context: Operational context.
    Returns:
        FailureEvent
    """
    context = dict(context)  # defensive copy
    context["manual_reason"] = reason
    signals = {"reason_hash": hash(reason) % 1_000_000}
    context["comp_id"] = comp_id
    context.setdefault("op_type", "manual")
    ev = build_failure_event(context, Symptom.MANUAL_FLAG, signals)
    log_failure_event(ev)
    return ev
