"""
Unit tests for adaptive DSP failure detection (Part 2).

""
Adaptive failure detection for DSP execution.

This module detects four canonical, literature-backed failure symptoms in
disassembly execution and normalizes them into a FailureEvent:

A) OVER_TORQUE_NO_ADVANCE (stuck screw): torque > k * nominal and no angle progress
   — torque–angle based detection is standard in (un)screw operations; thresholds or
     patterns are widely reported in robotic unscrewing and fastening literature. 
     See Mironov (RecyBot) for FT thresholds/patterns and control insights, and
     broader treatment of torque–angle curves in screwdriving.

B) ACCESS_BLOCKED (tool access/collision): local access plan fails / unexpected contact,
   choose to postpone and continue elsewhere if precedence allows.
   — Adaptive DSP under uncertainty recommends local bypass when feasible, before
     re-planning globally. See recent DSP surveys covering dynamic/uncertain settings. 

C) NO_SEPARATION (adhesive not released): high axial force with no joint opening.
   — Debonding-on-demand / controlled debonding methods (thermal, solvent, mechanical)
     are standard alternatives; switching process/tool is a common response. 

D) LOW_VALUE_BOTTLENECK (low-value part blocks higher value downstream):
   — Multi-objective DSP commonly trades off recovery value, time/cost, and risk; if
     policy allows destruction and risk is low, “destroy” is a standard local option
     prior to global re-planning. See DSP surveys on value-driven planning & uncertainty. 

Design rationale:
- Standardize failure into a compact FailureEvent for downstream decision (rules/fuzzy)
- classic approach in dynamic/adaptive DSP using Fuzzy (Attributed) to encode local rules under uncertainty.
"""
import pytest
from src.adaptive.detection import detect_failure, flag_failure_manual
from src.adaptive.types import Thresholds, Symptom, FailureEvent

# 1) OVER_TORQUE_NO_ADVANCE
def test_over_torque_no_advance():
    signals = {"torque_max": 0.95, "torque_nominal": 0.5, "duration_over": 1.8, "angle_delta": 0.0}
    context = {"comp_id": "C03", "op_type": "unscrew", "tool_id": "T_hex", "value_piece": 0.4, "no_destroy": False}
    thresholds = Thresholds()
    ev = detect_failure(signals, context, thresholds)
    assert ev is not None
    assert ev.symptom == Symptom.OVER_TORQUE_NO_ADVANCE

# 2) ACCESS_BLOCKED (alt_paths provided)
def test_access_blocked_with_alt_paths(monkeypatch):
    signals = {"access_blocked": 1}
    context = {"comp_id": "C05", "op_type": "unscrew", "tool_id": "T_hex", "alt_paths": True, "value_piece": 0.5, "no_destroy": True}
    thresholds = Thresholds()
    ev = detect_failure(signals, context, thresholds)
    assert ev is not None
    assert ev.symptom == Symptom.ACCESS_BLOCKED

    # alt_paths not provided, but state present, feasible_now_nodes monkeypatched
    context2 = {"comp_id": "C05", "op_type": "unscrew", "tool_id": "T_hex", "state": {}}
    def fake_feasible_now_nodes(state, exclude=None):
        return ["C99"]
    monkeypatch.setattr("src.core.graph_api.feasible_now_nodes", fake_feasible_now_nodes)
    ev2 = detect_failure(signals, context2, thresholds)
    assert ev2 is not None
    assert ev2.symptom == Symptom.ACCESS_BLOCKED

# 3) NO_SEPARATION
def test_no_separation():
    signals = {"force_axial": 12.5, "joint_opened": 0}
    context = {"comp_id": "C06", "op_type": "debond", "tool_id": "hot_air", "value_piece": 0.7, "no_destroy": True}
    thresholds = Thresholds()
    ev = detect_failure(signals, context, thresholds)
    assert ev is not None
    assert ev.symptom == Symptom.NO_SEPARATION

# 4) LOW_VALUE_BOTTLENECK
def test_low_value_bottleneck(monkeypatch):
    signals = {}
    context = {"comp_id": "C08", "op_type": "pry", "tool_id": "pry_tool", "value_piece": 0.1, "no_destroy": False}
    thresholds = Thresholds()
    import src.adaptive.detection as detection_mod
    monkeypatch.setattr(detection_mod, "blocks_high_value_downstream", lambda comp_id, value_cutoff=0.5: True)
    monkeypatch.setattr(detection_mod, "no_destroy_flag", lambda comp_id: False)
    ev = detection_mod.detect_failure(signals, context, thresholds)
    assert ev is not None
    assert ev.symptom == Symptom.LOW_VALUE_BOTTLENECK

# 5) MANUAL_FLAG et log (tmp_path)
def test_manual_flag_logs(tmp_path, monkeypatch):
    logs = []
    import src.adaptive.detection as detection_mod
    def fake_log_failure_event(ev, path=None):
        logs.append(ev)
    monkeypatch.setattr(detection_mod, "log_failure_event", fake_log_failure_event)
    ev = detection_mod.flag_failure_manual("C10", "operator_marked_issue", {"op_type": "unscrew"})
    assert ev.symptom == Symptom.MANUAL_FLAG
    assert "manual_reason" in ev.context
    assert logs and logs[0].symptom == Symptom.MANUAL_FLAG
