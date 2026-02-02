"""Quick test of fuzzy integration."""

from src.adaptive.decision import choose_action_fuzzy
from src.adaptive.types import FailureEvent, Symptom

print("=" * 60)
print("FUZZY TEST - Quick Integration")
print("=" * 60)

# Test 1: Gray zone (p_fail=40 is low/med boundary, fuzzy hesitates)
print("\n[TEST 1] Boundary zone (force=10, torque=8, p_fail=40)")
fe1 = FailureEvent(
    comp_id="N01",
    op_type="unscrew",
    tool_id="T_hex",
    symptom=Symptom.LOW_VALUE_BOTTLENECK,
    signals={'force': 10, 'torque': 8},
    context={},
    history={'failure_count': 0},
    timestamp="2025-11-06T10:00:00"
)

action1 = choose_action_fuzzy(fe1)
print(f"Action decided: {action1}")
# p_fail=40 is boundary, bypass OR change_tool are acceptable
print("OK" if action1 in ["change_tool", "bypass"] else f"Expected: change_tool/bypass, got: {action1}")

# Test 2: Critical case
print("\n[TEST 2] Very likely failure (force=40, torque=50, 2 previous failures)")
fe2 = FailureEvent(
    comp_id="N05",
    op_type="unscrew",
    tool_id="T_hex",
    symptom=Symptom.TIMEOUT,
    signals={'force': 40, 'torque': 50},
    context={},
    history={'failure_count': 2},
    timestamp="2025-11-06T10:05:00"
)

action2 = choose_action_fuzzy(fe2)
print(f"Action decided: {action2}")
print("OK" if action2 in ["replan", "destroy"] else f"Expected: replan/destroy, got: {action2}")

# Test 3: Missing sensors
print("\n[TEST 3] Missing sensors (defaults force=50, torque=50)")
fe3 = FailureEvent(
    comp_id="N07",
    op_type="unscrew",
    tool_id="T_hex",
    symptom=Symptom.ACCESS_BLOCKED,
    signals={},
    context={},
    history={'failure_count': 0},
    timestamp="2025-11-06T10:10:00"
)

action3 = choose_action_fuzzy(fe3)
print(f"Action decided: {action3}")
print("OK - System did not crash (defaults applied)")

print("\n" + "=" * 60)
print("Tests completed")
print("=" * 60)
