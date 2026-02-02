"""
Adaptive Scheduler Module

Executes disassembly sequences with adaptive failure handling:
- Fuzzy decision system (Ye et al., 2022)
- Fallback hierarchy (bypass → change_tool → destroy → replan)
- Sequence modification support (replan, destroy)
- Loop protection (max attempts per component)

References:
- Ye et al., 2022 (Self-evolving adaptive disassembly)
- Pedrosa et al., 2023 (Robust scheduling with reactive procedures)
- Frizziero et al., 2020 (DSP sequence planning)
"""

import logging
from typing import Dict, List, Any, Tuple
from src.adaptive import actions
from src.adaptive.decision import choose_action_fuzzy
from src.adaptive.types import FailureEvent, Symptom

MAX_ATTEMPTS_PER_COMPONENT = 3  # Protection against infinite loops


def execute_adaptive_sequence(
    solution: List[str],
    state: Dict[str, Any],
    failures: List[Dict[str, Any]],
    verbose: bool = False
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Execute disassembly sequence with adaptive failure handling.
    
    Args:
        solution: Initial disassembly sequence
        state: Overlay state dict (must contain 'graph', 'targets')
        failures: List of failure scenarios (comp_id → failure dict)
        verbose: Show detailed fuzzy analysis
    
    Returns:
        (final_state, actions_log)
    """
    actions_log = []
    
    # Create failure map
    failure_map = {f['comp_id']: f for f in failures}
    
    # Protection against infinite loops
    attempt_counter = {}
    
    print(f"\nStarting disassembly sequence ({len(solution)} operations)...\n")
    
    # Execute sequence with while loop (allows sequence modification)
    plan_effectif = solution.copy()
    i = 0
    
    while i < len(plan_effectif):
        comp = plan_effectif[i]
        idx = i + 1
        
        # Skip locked/destroyed/skipped
        if comp in state['locked'] or comp in state['destroyed'] or comp in state.get('skipped', set()):
            i += 1
            continue
        
        # Protection against infinite loops
        attempt_counter[comp] = attempt_counter.get(comp, 0) + 1
        if attempt_counter[comp] > MAX_ATTEMPTS_PER_COMPONENT:
            print(f"\n[ERROR] Too many failures on {comp} ({attempt_counter[comp]} attempts). Skipping component.")
            state['locked'].add(comp)
            i += 1
            continue
        
        # Check if failure scenario exists for this component
        if comp in failure_map:
            # FAILURE PATH
            state, actions_log, i, plan_effectif = _handle_failure(
                comp, idx, failure_map[comp], state, actions_log, 
                i, plan_effectif, verbose
            )
        else:
            # NOMINAL PATH
            if not verbose:
                print(f"[{idx:2d}/{len(plan_effectif)}] Disassembling {comp}... [OK]")
            state['done'].add(comp)
            i += 1
    
    print()  # Empty line
    return state, actions_log


def _handle_failure(
    comp: str,
    idx: int,
    fail_dict: Dict[str, Any],
    state: Dict[str, Any],
    actions_log: List[Dict[str, Any]],
    i: int,
    plan_effectif: List[str],
    verbose: bool
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], int, List[str]]:
    """
    Handle a single failure scenario.
    
    Returns:
        (updated_state, updated_actions_log, new_index, updated_plan)
    """
    if not verbose:
        print(f"[{idx:2d}/{len(plan_effectif)}] Disassembling {comp}... ", end='', flush=True)
    
    # Build FailureEvent
    fe = FailureEvent(
        comp_id=fail_dict['comp_id'],
        op_type=fail_dict['op_type'],
        tool_id=fail_dict.get('tool_id'),
        symptom=Symptom[fail_dict['symptom']],
        signals=fail_dict.get('signals', {}),
        context=fail_dict.get('context', {}),
        history=fail_dict.get('history', {}),
        timestamp=fail_dict.get('timestamp', '')
    )
    
    if not verbose:
        print(f"FAILURE ({fe.symptom.name})")
        print(f"     > Analyzing failure... ", end='', flush=True)
    
    # FUZZY DECISION
    result = choose_action_fuzzy(fe, verbose=verbose)
    action = result['label']
    
    if not verbose:
        score = result['action_score']
        confidence = result['membership'][action]
        print(f"Action: {action.upper()} (fuzzy: {score:.1f}, conf={confidence:.0%})")
        print(f"     > Executing {action}... ", end='', flush=True)
    
    # Execute primary action
    res = actions.dispatch(action, fe, state)
    logging.info(f"[ACTION] {action}: success={res.success}, notes={res.notes}")
    
    if res.success:
        # PRIMARY ACTION SUCCEEDED
        if not verbose:
            print(f"[OK]")
        
        actions_log.append({
            "comp_id": fe.comp_id,
            "symptom": fe.symptom.name,
            "action": action,
            "success": True,
            "notes": res.notes
        })
        
        # Handle sequence modification
        i, plan_effectif = _handle_action_success(
            action, res, fe, state, i, plan_effectif, verbose
        )
    
    else:
        # PRIMARY ACTION FAILED → FALLBACK
        if not verbose:
            print(f"[FAILED] ({res.notes})")
            print(f"     > Trying fallback...")
        
        actions_log.append({
            "comp_id": fe.comp_id,
            "symptom": fe.symptom.name,
            "action": action,
            "success": False,
            "notes": res.notes
        })
        
        # Attempt fallback
        i, plan_effectif, fallback_ok = _attempt_fallback(
            action, fe, state, actions_log, i, plan_effectif, verbose
        )
        
        if not fallback_ok:
            if not verbose:
                print(f"     > All actions failed. Stopping.")
            state['locked'].add(comp)
            # If mission impossible detected (i >= len(plan)), signal stop
            if i >= len(plan_effectif):
                # Return a very large index to force stop
                return state, actions_log, 999999, plan_effectif
    
    return state, actions_log, i, plan_effectif


def _handle_action_success(
    action: str,
    res: Any,
    fe: FailureEvent,
    state: Dict[str, Any],
    i: int,
    plan_effectif: List[str],
    verbose: bool
) -> Tuple[int, List[str]]:
    """
    Handle successful action execution (sequence modification if needed).
    
    Returns:
        (new_index, updated_plan)
    """
    if action == "replan":
        plan_effectif = res.updates.get("new_plan", plan_effectif)
        if not verbose:
            print(f"     > Sequence replanned, restarting...")
        return 0, plan_effectif
    
    elif action == "destroy":
        if not verbose:
            print(f"     > Component destroyed, replanning...")
        res_replan = actions.dispatch("replan", fe, state)
        if res_replan.success:
            plan_effectif = res_replan.updates.get("new_plan", plan_effectif)
            if not verbose:
                print(f"     > Replan successful, restarting...")
            return 0, plan_effectif
        else:
            if not verbose:
                # Display specific error message if mission impossible
                if res_replan.updates.get("mission_failed", False):
                    print(f"     > {res_replan.notes}")
                else:
                    print(f"     > Replan failed. Stopping.")
            # Signal to stop (index beyond plan length)
            return len(plan_effectif), plan_effectif
    
    elif action == "bypass":
        if not verbose:
            print(f"     > Component bypassed, continuing...")
        return i + 1, plan_effectif
    
    elif action == "change_tool":
        state['done'].add(fe.comp_id)
        if not verbose:
            print(f"     > Tool changed, continuing...")
        return i + 1, plan_effectif
    
    else:
        state['done'].add(fe.comp_id)
        return i + 1, plan_effectif


def _attempt_fallback(
    primary_action: str,
    fe: FailureEvent,
    state: Dict[str, Any],
    actions_log: List[Dict[str, Any]],
    i: int,
    plan_effectif: List[str],
    verbose: bool
) -> Tuple[int, List[str], bool]:
    """
    Attempt fallback actions in hierarchy order.
    
    IMPORTANT: If primary action was 'destroy' and failed (e.g., no_destroy=true),
    we keep 'destroy' in fallbacks to retry with relaxed criteria (last resort 
    to save the mission).
    
    FALLBACK ORDER:
    1. bypass - Circumvent the component
    2. change_tool - Change tool
    3. destroy - Destroy component (before replan as it may unblock the situation)
    4. replan - Replan (last because if destroy fails, replan will also fail)
    
    Returns:
        (new_index, updated_plan, fallback_succeeded)
    """
    # Fallback hierarchy - destroy BEFORE replan
    fallback_hierarchy = ["bypass", "change_tool", "destroy", "replan"]
    # Do NOT remove destroy from fallback list, even if it was the primary action
    # to allow a 2nd attempt with relaxed criteria
    if primary_action in fallback_hierarchy and primary_action != "destroy":
        fallback_hierarchy.remove(primary_action)
    
    for fb_action in fallback_hierarchy:
        if not verbose:
            print(f"     > Trying {fb_action}... ", end='', flush=True)
        
        # Mark that this is a fallback attempt to relax criteria
        if fb_action == "destroy" and primary_action == "destroy":
            # This is a 2nd attempt at destroy in fallback
            fe.context['fallback_destroy'] = True
        
        fb_res = actions.dispatch(fb_action, fe, state)
        
        if fb_res.success:
            if not verbose:
                print(f"[OK]")
            
            actions_log.append({
                "comp_id": fe.comp_id,
                "symptom": fe.symptom.name,
                "action": f"{primary_action}→{fb_action}",
                "success": True,
                "notes": "Fallback succeeded"
            })
            
            # Handle fallback sequence modification
            if fb_action == "replan":
                plan_effectif = fb_res.updates.get("new_plan", plan_effectif)
                if not verbose:
                    print(f"     > Replanned, restarting...")
                return 0, plan_effectif, True
            
            elif fb_action == "destroy":
                res_replan = actions.dispatch("replan", fe, state)
                if res_replan.success:
                    plan_effectif = res_replan.updates.get("new_plan", plan_effectif)
                    if not verbose:
                        print(f"     > Replanned after destroy, restarting...")
                    return 0, plan_effectif, True
                else:
                    # Check if this is a mission failure (target blocked)
                    if res_replan.updates.get("mission_failed", False):
                        if not verbose:
                            print(f"     > {res_replan.notes}")
                        # Stop execution completely (index beyond sequence)
                        return len(plan_effectif), plan_effectif, False
            
            elif fb_action == "bypass":
                if not verbose:
                    print(f"     > Bypassed, continuing...")
                return i + 1, plan_effectif, True
            
            else:
                state['done'].add(fe.comp_id)
                return i + 1, plan_effectif, True
        
        else:
            # Fallback failed
            if not verbose:
                print(f"[FAILED]")
            
            # CRITICAL: If replan failed for mission impossible
            if fb_action == "replan" and fb_res.updates.get("mission_failed", False):
                if not verbose:
                    print(f"     > {fb_res.notes}")
                # Add all remaining nodes to locked to force stop
                remaining_nodes = set(plan_effectif) - state['done'] - state['destroyed'] - state.get('skipped', set())
                state['locked'].update(remaining_nodes)
                # Stop completely
                return len(plan_effectif), plan_effectif, False
    
    # All fallbacks failed
    return i, plan_effectif, False
