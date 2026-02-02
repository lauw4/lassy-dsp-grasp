"""
Adaptive Decision Module for DSP (Disassembly Sequence Planning)

This module contains the decision logic for selecting appropriate recovery actions
when failures occur during disassembly execution.

DECISION SYSTEMS AVAILABLE:
==========================

1. choose_action_fuzzy() [CURRENT - PRODUCTION USE]
   - Mamdani fuzzy inference system (Ye et al., 2022)
   - 8 fuzzy rules with 3 inputs (force, torque, failure_probability)
   - Gradual decision boundaries with uncertainty handling
   - DEFAULT approach for all adaptive scheduling

2. choose_action() [LEGACY - REFERENCE ONLY]
   - If/else hard-coded rule-based logic
   - Kept for comparison and validation purposes
   - Can be enabled via use_fuzzy=False in scheduler for testing

MIGRATION NOTES:
================
The fuzzy decision system replaced the if/else logic as the production approach.
Both implementations are kept in this file for reference and comparison.
For production use, always use choose_action_fuzzy().

References:
- Ye et al., 2022: Fuzzy logic for adaptive disassembly planning
- Tang et al., 2021: Failure probability estimation
- Pedrosa et al., 2023: Robust scheduling with adaptive decision making
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from dataclasses import dataclass
from src.adaptive.types import FailureEvent, Symptom, Thresholds
from src.adaptive.fuzzy_decision import FuzzyDecisionSystem
from src.core.graph_api import feasible_now_nodes
import logging

logger = logging.getLogger(__name__)

# Mapping Symptom -> Base failure probability (per literature)
# Reference: Ye et al. (2022) - Section 4.2 "Failure probability estimation"
SYMPTOM_PROBABILITIES = {
    Symptom.OVER_TORQUE_NO_ADVANCE: 65.0,
    Symptom.ACCESS_BLOCKED: 50.0,
    Symptom.NO_SEPARATION: 70.0,
    Symptom.LOW_VALUE_BOTTLENECK: 40.0,
    Symptom.TIMEOUT: 80.0,
    Symptom.MANUAL_FLAG: 90.0
}

@dataclass
class DecisionFeatures:
    alt_tools: bool
    alt_paths: bool
    value_piece: float
    no_destroy: bool
    value_downstream: float
    symptom: Symptom

def extract_features(fe: FailureEvent, defaults: Optional[Dict[str, Any]] = None) -> DecisionFeatures:
    """
    Convert FailureEvent -> minimal decision variables.
    `defaults` can provide values if missing (e.g., {"value_downstream": 0.0}).
    """
    d = defaults or {}
    ctx = fe.context or {}
    return DecisionFeatures(
        alt_tools = bool((ctx.get("alt_tools_available") or [])),
        alt_paths = bool(ctx.get("alt_paths", False)),
        value_piece = float(ctx.get("value_piece", d.get("value_piece", 0.0))),
        no_destroy = bool(ctx.get("no_destroy", d.get("no_destroy", False))),
        value_downstream = float(ctx.get("value_downstream", d.get("value_downstream", 0.0))),
        symptom = fe.symptom,
    )

# ==============================================================================
# LEGACY DECISION SYSTEM (IF/ELSE LOGIC)
# ==============================================================================
# NOTE: This function is kept for reference and comparison purposes only.
# The fuzzy decision system (choose_action_fuzzy) is now the default approach.
# See: Ye et al., 2022 for justification of fuzzy logic in adaptive disassembly.
# ==============================================================================

def choose_action(fe: FailureEvent, thresholds: Optional[Thresholds] = None) -> str:
    """
    LEGACY: Politique V1 (if/else), conforme aux cas probants.
    
    This is the original decision logic based on hard-coded if/else rules.
    Kept for comparison and validation purposes.
    
    For production use, prefer choose_action_fuzzy() which uses Mamdani fuzzy inference.
    """
    T = thresholds or Thresholds()
    f = extract_features(fe, defaults={"value_downstream": 0.0})
    # Order of actions to try (per literature)
    actions_order = ["bypass", "change_tool", "destroy", "replan"]
    # List of feasible actions based on context
    possible_actions = []
    # Bypass possible?
    if f.alt_paths:
        possible_actions.append("bypass")
    # Tool change possible?
    if f.alt_tools:
        possible_actions.append("change_tool")
    # Destruction possible?
    if (not f.no_destroy) and (
        (f.symptom == Symptom.LOW_VALUE_BOTTLENECK and f.value_piece < T.low_value_cutoff)
        or (f.symptom == Symptom.NO_SEPARATION)
    ):
        possible_actions.append("destroy")
    # Replan always possible
    possible_actions.append("replan")
    # Traverse in order
    for act in actions_order:
        if act in possible_actions:
            return act
    # Fail safe: no action possible
    print("[FAIL SAFE] No adaptive action possible, manual intervention required!")
    return "manual_intervention"

# ==============================================================================
# CURRENT DECISION SYSTEM (FUZZY LOGIC)
# ==============================================================================

def calculate_failure_probability(fe: FailureEvent) -> float:
    """
    Calculate failure probability based on symptom and history.
    
    Logic (literature - Tang et al. 2021):
    - Base: Probability according to symptom type
    - Adjustment: +10% per previous failure
    - Cap: 95%
    
    Args:
        fe: Failure event
    
    Returns:
        Failure probability [0-100]
    """
    base_p_fail = SYMPTOM_PROBABILITIES.get(fe.symptom, 50.0)
    
    failure_count = 0
    if fe.history:
        failure_count = fe.history.get("failure_count", 0)
    
    adjustment = failure_count * 10.0
    p_fail = min(95.0, base_p_fail + adjustment)
    
    return p_fail

def extract_fuzzy_inputs(fe: FailureEvent) -> tuple[float, float, float]:
    """
    Extract 3 fuzzy inputs from a FailureEvent.
    
    Applies sensor relevance principle (Tang et al. 2021):
    - For mechanical failures (NO_SEPARATION, OVER_TORQUE, ACCESS_BLOCKED): 
      use measured force/torque
    - For temporal failures (TIMEOUT): 
      use neutral values (50.0) as force/torque not relevant
    
    Args:
        fe: Failure event
    
    Returns:
        Tuple (force, torque, p_fail) normalized [0-100]
    """
    # Symptoms where force/torque are NOT relevant
    temporal_symptoms = {Symptom.TIMEOUT, Symptom.MANUAL_FLAG}
    
    if fe.symptom in temporal_symptoms:
        # Temporal failures: neutral force/torque (Tang et al. 2021)
        force = 50.0
        torque = 50.0
    else:
        # Mechanical failures: use actual measurements
        force = 50.0
        if fe.signals:
            force = fe.signals.get('force', 50.0)
        
        torque = 50.0
        if fe.signals:
            torque = fe.signals.get('torque', 50.0)
    
    p_fail = calculate_failure_probability(fe)
    
    return (force, torque, p_fail)

def choose_action_fuzzy(fe: FailureEvent, verbose: bool = True) -> str:
    """
    CURRENT: Adaptive decision via fuzzy logic system (Mamdani).
    
    This is the PRODUCTION decision system using Mamdani fuzzy inference.
    Based on Ye et al. (2022) - 8 Mamdani fuzzy rules for adaptive disassembly.
    
    Args:
        fe: Detected failure event
        verbose: If True, display detailed analysis. If False, silent.
    
    Returns:
        Recommended action: "change_tool" | "bypass" | "destroy" | "replan"
    
    Note:
        This replaces the legacy if/else decision logic (choose_action).
        The fuzzy approach provides better handling of uncertainty and gradual transitions
        between decision boundaries.
    """
    force, torque, p_fail = extract_fuzzy_inputs(fe)
    
    # CONTEXT_SCORE CALCULATION (4th fuzzy input)
    # Encodes alternative availability: tools AND paths
    # 0 = no alternatives, 100 = alternatives available
    
    alt_tools_available = fe.context.get("alt_tools_available", [])
    
    # IMPORTANT: Recalculate alt_paths with CURRENT state
    # The value in fe.context may be stale (computed before bypass/destroy/replan)
    alt_paths = False
    if "state" in fe.context:
        state = fe.context["state"]
        exclude = {fe.comp_id}
        feasible = feasible_now_nodes(state, exclude=exclude)
        alt_paths = len(feasible) > 0
    
    # Score based on TYPE of alternative (critical for choosing right action)
    # Note: alt_tools → change_tool (R0a), alt_paths → bypass (original rules)
    context_score = 0.0
    if alt_tools_available and len(alt_tools_available) > 0:
        context_score = 100.0  # Tools → strong context (R0a → change_tool)
    elif alt_paths:
        context_score = 0.0    # Paths only → NO context (let R4/R5 handle bypass)
    
    # context_score final: 0 (rien ou chemins), 100 (outils)
    # Logique: R0a ne s'active QUE pour alt_tools, pas alt_paths
    
    fuzzy_system = FuzzyDecisionSystem()
    result = fuzzy_system.evaluate(force, torque, p_fail, context_score)
    
    # Detailed display for analysis (only in verbose mode)
    if verbose:
        print("\n" + "="*70)
        print(f"[FUZZY ANALYSIS] Component: {fe.comp_id} | Symptom: {fe.symptom.name}")
        print("="*70)
        
        print(f"\n1. INPUTS:")
        print(f"   - Force:    {force:>6.1f} / 100")
        print(f"   - Torque:   {torque:>6.1f} / 100")
        print(f"   - p_fail:   {p_fail:>6.1f}% (base={SYMPTOM_PROBABILITIES.get(fe.symptom, 50.0):.0f}% + history={fe.history.get('failure_count', 0) * 10}%)")
        print(f"   - context:  {context_score:>6.1f} / 100 (alt_tools={len(alt_tools_available)}, alt_paths={alt_paths})")
        
        print(f"\n2. MEMBERSHIP DEGREES:")
        memberships = result['membership']
        for action in ['change_tool', 'bypass', 'destroy', 'replan']:
            degree = memberships.get(action, 0.0)
            bar_length = int(degree * 20)
            bar = "#" * bar_length  # Use # instead of █ for Windows compatibility
            print(f"   - {action:12s}: {degree:>5.1%} [{bar:<20s}]")
        
        print(f"\n3. ACTIVATED RULES:")
        if 'activated_rules' in result and result['activated_rules']:
            for rule_name in result['activated_rules']:
                print(f"   - {rule_name}")
        else:
            print("   - (no rules tracked)")
        
        print(f"\n4. DEFUZZIFICATION:")
        print(f"   - Raw score: {result['action_score']:.2f} / 100")
        print(f"   - Winner:    {result['label'].upper()}")
        print(f"   - Confidence: {memberships[result['label']]:.1%}")
        
        print("="*70 + "\n")
    
    # Log for file (always)
    logger.info(f"[FUZZY] Symptom: {fe.symptom.name}")
    logger.info(f"[FUZZY] Inputs: force={force:.1f}, torque={torque:.1f}, p_fail={p_fail:.1f}, context={context_score:.1f}")
    logger.info(f"[FUZZY] Score: {result['action_score']:.2f}")
    logger.info(f"[FUZZY] Decision: {result['label']} (confidence={result['membership'][result['label']]:.1%})")
    
    # FEASIBILITY CHECK: Mark BYPASS as infeasible if no alternative path
    # Fuzzy can recommend bypass, but we check REAL feasibility
    bypass_infeasible = False
    if result['label'] == 'bypass' and not alt_paths:
        bypass_infeasible = True
        if verbose:
            print(f"[FUZZY] ⚠️  BYPASS recommended but INFEASIBLE (alt_paths=False)")
            print(f"[FUZZY] → Automatic fallback will be triggered")
        logger.warning(f"[FUZZY] BYPASS infaisable pour {fe.comp_id}, alt_paths=False")
    
    # Return complete result (label + metadata for display)
    result['_force'] = force
    result['_torque'] = torque
    result['_p_fail'] = p_fail
    result['_context_score'] = context_score
    result['_alt_paths'] = alt_paths
    result['_bypass_infeasible'] = bypass_infeasible
    
    return result  # Returns complete dict instead of just label

__all__ = [
    "choose_action", 
    "choose_action_fuzzy", 
    "extract_features", 
    "DecisionFeatures",
    "calculate_failure_probability",
    "extract_fuzzy_inputs"
]
