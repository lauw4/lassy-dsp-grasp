# python -m  pytest -q .\experiments_2\test_actions.py
"""
Tests unitaires pour les actions adaptatives (Partie 3).
"""
import pytest
from src.adaptive.overlay import init_state, is_destroyed
from src.adaptive.actions import try_bypass, try_change_tool, try_destroy, replan_grasp, dispatch, ActionResult
from src.adaptive.types import FailureEvent, Symptom

# 1) Bypass success
def test_try_bypass_success(monkeypatch):
    state = init_state()
    def fake_feasible_now_nodes(state_arg, exclude=None):
        return ["C10", "C11"] if exclude == {"C03"} else []
    monkeypatch.setattr("src.core.graph_api.feasible_now_nodes", fake_feasible_now_nodes)
    from src.adaptive.overlay import feasible_now_excluding
    fe = FailureEvent(comp_id="C03", op_type="unscrew", tool_id=None, symptom=Symptom.ACCESS_BLOCKED,
                      signals={}, context={}, history={}, timestamp="2025-09-16T00:00:00Z")
    res = try_bypass(state, fe)
    assert res.success is True
    assert "next_nodes" in res.updates

# 2) Bypass fail
def test_try_bypass_fail(monkeypatch):
    state = init_state()
    monkeypatch.setattr("src.core.graph_api.feasible_now_nodes", lambda state_arg, exclude=None: [])
    fe = FailureEvent(comp_id="C03", op_type="unscrew", tool_id=None, symptom=Symptom.ACCESS_BLOCKED,
                      signals={}, context={}, history={}, timestamp="2025-09-16T00:00:00Z")
    res = try_bypass(state, fe)
    assert res.success is False

# 3) Change tool success
def test_try_change_tool_success():
    state = init_state()
    fe = FailureEvent(comp_id="C05", op_type="unscrew", tool_id=None, symptom=Symptom.ACCESS_BLOCKED,
                      signals={}, context={"alt_tools_available": ["T_extractor_3mm"]}, history={}, timestamp="2025-09-16T00:00:00Z")
    res = try_change_tool(state, fe)
    assert res.success is True
    assert res.updates["tool_used"] == "T_extractor_3mm"

# 4) Change tool fail
def test_try_change_tool_fail():
    state = init_state()
    fe = FailureEvent(comp_id="C05", op_type="unscrew", tool_id=None, symptom=Symptom.ACCESS_BLOCKED,
                      signals={}, context={}, history={}, timestamp="2025-09-16T00:00:00Z")
    res = try_change_tool(state, fe)
    assert res.success is False

# 5) Destroy success
def test_try_destroy_success(monkeypatch):
    state = init_state()
    monkeypatch.setattr("src.core.graph_api.no_destroy_flag", lambda comp_id: False)
    fe = FailureEvent(comp_id="C08", op_type="pry", tool_id=None, symptom=Symptom.LOW_VALUE_BOTTLENECK,
                      signals={}, context={}, history={}, timestamp="2025-09-16T00:00:00Z")
    res = try_destroy(state, fe)
    assert res.success is True
    assert is_destroyed(state, "C08") is True

# 6) Destroy forbidden
def test_try_destroy_forbidden(monkeypatch):
    state = init_state()
    monkeypatch.setattr("src.adaptive.actions.no_destroy_flag", lambda comp_id: True)
    fe = FailureEvent(comp_id="C08", op_type="pry", tool_id=None, symptom=Symptom.LOW_VALUE_BOTTLENECK,
                      signals={}, context={}, history={}, timestamp="2025-09-16T00:00:00Z")
    res = try_destroy(state, fe)
    assert res.success is False

# 7) Replan success
def test_replan_grasp_success(monkeypatch):
    state = init_state()
    monkeypatch.setattr("src.adaptive.actions.replan_with_overlay", lambda state_arg: ["C01", "C02", "C03"])
    res = replan_grasp(state)
    assert res.success is True
    assert res.updates["new_plan"] == ["C01", "C02", "C03"]

# 8) Replan fail
def test_replan_grasp_fail(monkeypatch):
    state = init_state()
    monkeypatch.setattr("src.core.graph_api.replan_with_overlay", lambda state_arg: [])
    res = replan_grasp(state)
    assert res.success is False

# 9) Dispatch unknown action
def test_dispatch_unknown_action():
    state = init_state()
    fe = FailureEvent(comp_id="C01", op_type="unscrew", tool_id=None, symptom=Symptom.ACCESS_BLOCKED,
                      signals={}, context={}, history={}, timestamp="2025-09-16T00:00:00Z")
    with pytest.raises(ValueError):
        dispatch("unknown", fe, state)
