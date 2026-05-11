import pytest
from src.adaptive.types import FailureEvent, Symptom, Thresholds
from src.adaptive.decision import choose_action

def make_fe(symptom: Symptom, context: dict | None = None) -> FailureEvent:
    return FailureEvent(
        comp_id="CX",
        op_type="unscrew",
        tool_id=None,
        symptom=symptom,
        signals={},
        context=context or {},
        history={},
        timestamp="2025-09-19T00:00:00Z"
    )

def test_decision_over_torque_change_tool():
    fe = make_fe(Symptom.OVER_TORQUE_NO_ADVANCE, {"alt_tools_available":["T_extractor"], "alt_paths":False})
    assert choose_action(fe) == "change_tool"

def test_decision_over_torque_no_tools_then_bypass():
    fe = make_fe(Symptom.OVER_TORQUE_NO_ADVANCE, {"alt_tools_available":[], "alt_paths":True})
    assert choose_action(fe) == "bypass"

def test_decision_no_separation_change_tool():
    fe = make_fe(Symptom.NO_SEPARATION, {"alt_tools_available":["hot_air"]})
    assert choose_action(fe) == "change_tool"

def test_decision_access_blocked_bypass():
    fe = make_fe(Symptom.ACCESS_BLOCKED, {"alt_paths":True})
    assert choose_action(fe) == "bypass"

def test_decision_access_blocked_replan_when_no_alt():
    fe = make_fe(Symptom.ACCESS_BLOCKED, {"alt_paths":False})
    assert choose_action(fe) == "replan"

def test_decision_low_value_bottleneck_destroy_allowed():
    T = Thresholds()
    fe = make_fe(Symptom.LOW_VALUE_BOTTLENECK, {"value_piece": T.low_value_cutoff - 1e-6, "no_destroy": False})
    assert choose_action(fe, thresholds=T) == "destroy"

def test_decision_low_value_bottleneck_replan_when_forbidden():
    T = Thresholds()
    fe = make_fe(Symptom.LOW_VALUE_BOTTLENECK, {"value_piece": T.low_value_cutoff - 1e-6, "no_destroy": True})
    assert choose_action(fe, thresholds=T) == "replan"

def test_decision_manual_flag_default_replan():
    fe = make_fe(Symptom.MANUAL_FLAG, {})
    assert choose_action(fe) == "replan"
