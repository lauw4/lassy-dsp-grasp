"""
Adaptive Scheduler V2 — Enhanced High-Stress Recovery

Key improvements over scheduler.py:
1. Multi-target fallback: when a target is blocked, try remaining targets
2. Global replanning: after all individual failures handled, replan on residual graph
3. Dynamic threshold relaxation: under high stress, relax destroy constraints
4. Graceful degradation: maximize recovered targets instead of all-or-nothing

References:
- Ye et al., 2022 (Self-evolving adaptive disassembly)
- Pedrosa et al., 2023 (Robust scheduling with reactive procedures)
- Frizziero et al., 2020 (DSP sequence planning)
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from src.adaptive import actions
from src.adaptive.decision import choose_action_fuzzy
from src.adaptive.types import FailureEvent, Symptom

MAX_ATTEMPTS_PER_COMPONENT = 3


def execute_adaptive_sequence_v2(
    solution: List[str],
    state: Dict[str, Any],
    failures: List[Dict[str, Any]],
    verbose: bool = False,
    enable_multitarget: bool = True,
    enable_phase2: bool = True,
    enable_phase3: bool = True,
    enable_relaxation: bool = True
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Execute disassembly sequence with enhanced adaptive failure handling.
    
    V2 improvements (each can be toggled for ablation study):
    - Multi-target: if one target is blocked, pursue remaining targets
    - Global replan (Phase 2): after individual failures, replan on residual graph
    - Aggressive recovery (Phase 3): force-destroy blocking ancestors
    - Dynamic threshold relaxation: under high stress, relax destroy constraints
    """
    actions_log = []
    failure_map = {f['comp_id']: f for f in failures}
    attempt_counter = {}
    
    # Detect stress level from number of failures
    n_failures = len(failures)
    n_components = len(state.get('graph', {}).nodes) if state.get('graph') else len(solution)
    stress_ratio = n_failures / max(n_components, 1)
    is_high_stress = n_failures >= 4 or stress_ratio > 0.05
    
    if is_high_stress and enable_relaxation:
        # Relax destroy thresholds under high stress
        _relax_destroy_thresholds(state, stress_ratio)
    
    all_targets = list(state.get('targets', []))
    remaining_targets = list(all_targets)
    recovered_targets = []
    
    if not verbose:
        print(f"\n[V2] Starting adaptive sequence ({len(solution)} ops, {n_failures} failures)")
        if is_high_stress:
            print(f"[V2] HIGH-STRESS detected (ratio={stress_ratio:.3f}), thresholds relaxed")
        print()
    
    # Phase 1: Execute sequence with individual failure handling
    plan_effectif = solution.copy()
    i = 0
    mission_failed_targets = set()
    
    while i < len(plan_effectif):
        comp = plan_effectif[i]
        idx = i + 1
        
        if comp in state['locked'] or comp in state['destroyed'] or comp in state.get('skipped', set()):
            i += 1
            continue
        
        attempt_counter[comp] = attempt_counter.get(comp, 0) + 1
        if attempt_counter[comp] > MAX_ATTEMPTS_PER_COMPONENT:
            state['locked'].add(comp)
            i += 1
            continue
        
        if comp in failure_map:
            state, actions_log, new_i, plan_effectif, mission_blocked = _handle_failure_v2(
                comp, idx, failure_map[comp], state, actions_log,
                i, plan_effectif, verbose, is_high_stress
            )
            
            if mission_blocked:
                if not enable_multitarget:
                    # V1 behavior: stop on first blocked target
                    break
                # V2: Don't stop! Record blocked target and try to continue
                current_target = _get_current_target(state, remaining_targets)
                if current_target:
                    mission_failed_targets.add(current_target)
                    remaining_targets = [t for t in remaining_targets if t != current_target]
                    
                    if not verbose:
                        print(f"[V2] Target {current_target} blocked, {len(remaining_targets)} targets remaining")
                    
                    if remaining_targets:
                        # Try replanning for next target
                        new_plan = _replan_for_target(state, remaining_targets[0], verbose)
                        if new_plan:
                            plan_effectif = new_plan
                            i = 0
                            continue
                
                # If no more targets or replan failed, try global recovery (Phase 2)
                break
            else:
                i = new_i
        else:
            if not verbose:
                print(f"[{idx:2d}/{len(plan_effectif)}] Disassembling {comp}... [OK]")
            state['done'].add(comp)
            i += 1
    
    # Check which targets were recovered in Phase 1
    for t in all_targets:
        if t in state.get('done', set()) and t not in state.get('destroyed', set()):
            if t not in recovered_targets:
                recovered_targets.append(t)
    
    # Phase 2: Global recovery for missed targets
    if enable_phase2:
        missed_targets = [t for t in all_targets 
                          if t not in recovered_targets 
                          and t not in mission_failed_targets]
        
        # Also retry previously failed targets with fresh global replan
        retry_targets = list(mission_failed_targets) + missed_targets
        
        if retry_targets and not verbose:
            print(f"\n[V2-PHASE2] Attempting global recovery for {len(retry_targets)} target(s)...")
        
        for target in retry_targets:
            if target in recovered_targets:
                continue
            if target in state.get('done', set()) and target not in state.get('destroyed', set()):
                recovered_targets.append(target)
                continue
                
            # Unlock previously locked components for fresh attempt
            # but keep destroyed ones (irreversible)
            saved_locked = set(state['locked'])
            
            # Only unlock components that were locked due to failed recovery attempts
            # Don't unlock components that have actual physical failures
            failure_comps = set(failure_map.keys())
            releasable = saved_locked - failure_comps
            for comp in releasable:
                state['locked'].discard(comp)
            
            new_plan = _replan_for_target(state, target, verbose)
            
            if new_plan:
                if not verbose:
                    print(f"[V2-PHASE2] Found path to {target} ({len(new_plan)} steps)")
                
                # Execute the recovery plan
                success = _execute_recovery_plan(
                    new_plan, state, failure_map, actions_log, 
                    attempt_counter, verbose, is_high_stress
                )
                
                if success and target in state.get('done', set()):
                    recovered_targets.append(target)
                    if not verbose:
                        print(f"[V2-PHASE2] Target {target} RECOVERED!")
                else:
                    if not verbose:
                        print(f"[V2-PHASE2] Target {target} still blocked")
                    # Restore locked state
                    state['locked'] = saved_locked
            else:
                # Restore locked state
                state['locked'] = saved_locked
                if not verbose:
                    print(f"[V2-PHASE2] No path to {target}")
    
    # Phase 3: Last resort — aggressive destroy + replan
    still_missed = [t for t in all_targets if t not in recovered_targets]
    
    if enable_phase3 and still_missed and is_high_stress:
        if not verbose:
            print(f"\n[V2-PHASE3] Aggressive recovery for {len(still_missed)} target(s)...")
        
        for target in still_missed:
            recovered = _aggressive_recovery(state, target, failure_map, actions_log, verbose)
            if recovered:
                recovered_targets.append(target)
                if not verbose:
                    print(f"[V2-PHASE3] Target {target} RECOVERED (aggressive)")
    
    if not verbose:
        print(f"\n[V2-SUMMARY] Recovered {len(recovered_targets)}/{len(all_targets)} targets")
    
    return state, actions_log


def _relax_destroy_thresholds(state: Dict[str, Any], stress_ratio: float):
    """
    Under high stress, relax destruction thresholds to allow more destructions.
    Higher stress = more relaxation.
    """
    import numpy as np
    G = state.get('graph')
    if G is None:
        return
    
    profits = [G.nodes[n].get('profit', 0.0) for n in G.nodes]
    if not profits:
        return
    
    # Base percentile is 20th. Under stress, increase up to 50th
    # stress_ratio 0.05 → percentile 25, ratio 0.15 → percentile 45
    relaxed_percentile = min(50, 20 + int(stress_ratio * 200))
    
    from src.adaptive.types import Thresholds
    cutoff = float(np.percentile(profits, relaxed_percentile))
    
    if state.get('thresholds') is None:
        state['thresholds'] = Thresholds(low_value_cutoff=cutoff)
    else:
        state['thresholds'].low_value_cutoff = cutoff


def _get_current_target(state: Dict[str, Any], remaining_targets: List[str]) -> Optional[str]:
    """Get the current active target."""
    if remaining_targets:
        return remaining_targets[0]
    targets = state.get('targets', [])
    return targets[0] if targets else None


def _replan_for_target(state: Dict[str, Any], target: str, verbose: bool) -> Optional[List[str]]:
    """
    Attempt to replan specifically for a given target.
    Returns new plan or None.
    """
    from src.core.graph_api import TargetBlockedException
    import networkx as nx
    from src.grasp.constructive import run_grasp
    
    G_orig = state.get('graph')
    if G_orig is None:
        return None
    
    G = G_orig.copy()
    
    destroyed = set(state.get('destroyed', set()))
    locked = set(state.get('locked', set()))
    done = set(state.get('done', set()))
    skipped = set(state.get('skipped', set()))
    
    # Set profit=0 for destroyed
    for n in destroyed:
        if n in G.nodes:
            G.nodes[n]['profit'] = 0.0
    
    # Remove done + skipped nodes
    G.remove_nodes_from(done | skipped)
    
    # Check target is still in graph and not blocked
    if target not in G.nodes:
        return None
    
    # Cascade blocking
    blocked = set(locked)
    remaining = set(G.nodes) - blocked
    changed = True
    while changed:
        changed = False
        newly_blocked = set()
        for node in list(remaining):
            preds = set(G.predecessors(node))
            preds -= done | skipped | destroyed
            if preds and preds.issubset(blocked):
                newly_blocked.add(node)
        if newly_blocked:
            blocked.update(newly_blocked)
            remaining -= newly_blocked
            changed = True
    
    if target in blocked:
        return None
    
    # Generate plan
    needed_nodes = [n for n in G.nodes if n not in blocked]
    
    try:
        seqs = run_grasp(G, algorithm='vnd', mode='selectif',
                         target_nodes=[target], runs=1, needed_nodes=needed_nodes)
        if isinstance(seqs, tuple) and len(seqs) == 2 and isinstance(seqs[0], list):
            seq = seqs[0]
        elif isinstance(seqs, list) and len(seqs) > 0:
            first = seqs[0]
            if isinstance(first, tuple):
                seq = first[0]
            elif isinstance(first, list):
                seq = first
            else:
                seq = seqs
        else:
            seq = []
        
        # Filter blocked/destroyed
        seq = [n for n in seq if n not in blocked and n not in destroyed]
        
        if target in seq:
            return seq
        return None
    except Exception:
        return None


def _execute_recovery_plan(
    plan: List[str],
    state: Dict[str, Any],
    failure_map: Dict[str, Dict],
    actions_log: List[Dict],
    attempt_counter: Dict[str, int],
    verbose: bool,
    is_high_stress: bool
) -> bool:
    """Execute a recovery sub-plan. Returns True if completed without mission failure."""
    i = 0
    while i < len(plan):
        comp = plan[i]
        
        if comp in state['locked'] or comp in state['destroyed'] or comp in state.get('skipped', set()):
            i += 1
            continue
        
        if comp in state.get('done', set()):
            i += 1
            continue
        
        attempt_counter[comp] = attempt_counter.get(comp, 0) + 1
        if attempt_counter[comp] > MAX_ATTEMPTS_PER_COMPONENT:
            state['locked'].add(comp)
            i += 1
            continue
        
        if comp in failure_map:
            state, actions_log, new_i, plan, mission_blocked = _handle_failure_v2(
                comp, i + 1, failure_map[comp], state, actions_log,
                i, plan, verbose, is_high_stress
            )
            if mission_blocked:
                return False
            i = new_i
        else:
            if not verbose:
                print(f"  [{i+1}/{len(plan)}] {comp}... [OK]")
            state['done'].add(comp)
            i += 1
    
    return True


def _aggressive_recovery(
    state: Dict[str, Any],
    target: str,
    failure_map: Dict[str, Dict],
    actions_log: List[Dict],
    verbose: bool
) -> bool:
    """
    Last resort: find path to target, force-destroy any blocking locked component.
    """
    import networkx as nx
    
    G = state.get('graph')
    if G is None:
        return False
    
    locked = set(state.get('locked', set()))
    done = set(state.get('done', set()))
    destroyed = set(state.get('destroyed', set()))
    skipped = set(state.get('skipped', set()))
    
    if target not in G.nodes or target in destroyed:
        return False
    
    # Find which locked components block the target
    # Try unlocking them by force-destroying the cheapest ones
    blocking_locked = set()
    
    # Find all ancestors of target
    try:
        ancestors = nx.ancestors(G, target)
    except Exception:
        return False
    
    blocking_locked = locked & ancestors
    
    if not blocking_locked:
        # Target not blocked by locked nodes — it's blocked by structure
        return False
    
    # Sort by profit (destroy cheapest first)
    blocking_sorted = sorted(
        blocking_locked,
        key=lambda n: G.nodes[n].get('profit', 0.0) if n in G.nodes else 0.0
    )
    
    # Force destroy up to 3 blocking components
    for comp in blocking_sorted[:3]:
        if not verbose:
            print(f"  [AGGRESSIVE] Force-destroying {comp} (profit={G.nodes.get(comp, {}).get('profit', 0):.2f})")
        
        state['locked'].discard(comp)
        state['destroyed'].add(comp)
        if comp in G.nodes:
            G.nodes[comp]['profit'] = 0.0
        
        actions_log.append({
            "comp_id": comp,
            "symptom": "AGGRESSIVE_RECOVERY",
            "action": "force_destroy",
            "success": True,
            "notes": f"Aggressive destruction to unblock target {target}"
        })
    
    # Now try replanning
    new_plan = _replan_for_target(state, target, verbose)
    if new_plan:
        attempt_counter = {}
        success = _execute_recovery_plan(
            new_plan, state, failure_map, actions_log,
            attempt_counter, verbose, True
        )
        return success and target in state.get('done', set())
    
    return False


def _handle_failure_v2(
    comp: str,
    idx: int,
    fail_dict: Dict[str, Any],
    state: Dict[str, Any],
    actions_log: List[Dict[str, Any]],
    i: int,
    plan_effectif: List[str],
    verbose: bool,
    is_high_stress: bool
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], int, List[str], bool]:
    """
    Handle a single failure. Returns (state, log, new_i, plan, mission_blocked).
    mission_blocked=True means the current target path is dead.
    """
    if not verbose:
        print(f"[{idx:2d}/{len(plan_effectif)}] Disassembling {comp}... ", end='', flush=True)
    
    ctx = dict(fail_dict.get('context', {}))
    ctx['state'] = state
    
    # V2: inject stress info for potential threshold relaxation
    if is_high_stress:
        ctx['high_stress_mode'] = True
    
    fe = FailureEvent(
        comp_id=fail_dict['comp_id'],
        op_type=fail_dict['op_type'],
        tool_id=fail_dict.get('tool_id'),
        symptom=Symptom[fail_dict['symptom']],
        signals=fail_dict.get('signals', {}),
        context=ctx,
        history=fail_dict.get('history', {}),
        timestamp=fail_dict.get('timestamp', '')
    )
    
    if not verbose:
        print(f"FAILURE ({fe.symptom.name})")
        print(f"     > Analyzing... ", end='', flush=True)
    
    result = choose_action_fuzzy(fe, verbose=verbose)
    action = result['label']
    
    if not verbose:
        score = result['action_score']
        confidence = result['membership'][action]
        print(f"Action: {action.upper()} (fuzzy: {score:.1f}, conf={confidence:.0%})")
        print(f"     > Executing {action}... ", end='', flush=True)
    
    res = actions.dispatch(action, fe, state)
    
    if res.success:
        if not verbose:
            print(f"[OK]")
        
        actions_log.append({
            "comp_id": fe.comp_id,
            "symptom": fe.symptom.name,
            "action": action,
            "success": True,
            "notes": res.notes
        })
        
        new_i, plan_effectif, mission_blocked = _handle_action_success_v2(
            action, res, fe, state, i, plan_effectif, verbose
        )
        return state, actions_log, new_i, plan_effectif, mission_blocked
    
    else:
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
        
        new_i, plan_effectif, fallback_ok = _attempt_fallback_v2(
            action, fe, state, actions_log, i, plan_effectif, verbose, is_high_stress
        )
        
        if not fallback_ok:
            if not verbose:
                print(f"     > All actions failed for {comp}")
            state['locked'].add(comp)
            mission_blocked = (new_i >= len(plan_effectif))
            return state, actions_log, new_i, plan_effectif, mission_blocked
        
        return state, actions_log, new_i, plan_effectif, False


def _handle_action_success_v2(
    action: str,
    res: Any,
    fe: FailureEvent,
    state: Dict[str, Any],
    i: int,
    plan_effectif: List[str],
    verbose: bool
) -> Tuple[int, List[str], bool]:
    """Returns (new_i, plan, mission_blocked)."""
    if action == "replan":
        plan_effectif = res.updates.get("new_plan", plan_effectif)
        if not verbose:
            print(f"     > Sequence replanned, restarting...")
        return 0, plan_effectif, False
    
    elif action == "destroy":
        if not verbose:
            print(f"     > Component destroyed, replanning...")
        res_replan = actions.dispatch("replan", fe, state)
        if res_replan.success:
            plan_effectif = res_replan.updates.get("new_plan", plan_effectif)
            if not verbose:
                print(f"     > Replan successful, restarting...")
            return 0, plan_effectif, False
        else:
            if res_replan.updates.get("mission_failed", False):
                if not verbose:
                    print(f"     > {res_replan.notes}")
                # V2: signal mission blocked but DON'T stop
                return len(plan_effectif), plan_effectif, True
            else:
                if not verbose:
                    print(f"     > Replan failed.")
                return len(plan_effectif), plan_effectif, True
    
    elif action == "bypass":
        if not verbose:
            print(f"     > Component bypassed, continuing...")
        return i + 1, plan_effectif, False
    
    elif action == "change_tool":
        state['done'].add(fe.comp_id)
        if not verbose:
            print(f"     > Tool changed, continuing...")
        return i + 1, plan_effectif, False
    
    else:
        state['done'].add(fe.comp_id)
        return i + 1, plan_effectif, False


def _attempt_fallback_v2(
    primary_action: str,
    fe: FailureEvent,
    state: Dict[str, Any],
    actions_log: List[Dict[str, Any]],
    i: int,
    plan_effectif: List[str],
    verbose: bool,
    is_high_stress: bool
) -> Tuple[int, List[str], bool]:
    """Fallback with V2 enhancements."""
    fallback_hierarchy = ["bypass", "change_tool", "destroy", "replan"]
    if primary_action in fallback_hierarchy and primary_action != "destroy":
        fallback_hierarchy.remove(primary_action)
    
    for fb_action in fallback_hierarchy:
        if not verbose:
            print(f"     > Trying {fb_action}... ", end='', flush=True)
        
        if fb_action == "destroy":
            fe.context['fallback_destroy'] = True
            # V2: under high stress, simulate human approval for last-resort destroy
            if is_high_stress:
                fe.context['human_approved'] = True
        
        fb_res = actions.dispatch(fb_action, fe, state)
        
        if fb_res.success:
            if not verbose:
                print(f"[OK]")
            
            actions_log.append({
                "comp_id": fe.comp_id,
                "symptom": fe.symptom.name,
                "action": f"{primary_action}->{fb_action}",
                "success": True,
                "notes": "Fallback succeeded"
            })
            
            if fb_action == "replan":
                plan_effectif = fb_res.updates.get("new_plan", plan_effectif)
                return 0, plan_effectif, True
            elif fb_action == "destroy":
                res_replan = actions.dispatch("replan", fe, state)
                if res_replan.success:
                    plan_effectif = res_replan.updates.get("new_plan", plan_effectif)
                    return 0, plan_effectif, True
                else:
                    if res_replan.updates.get("mission_failed", False):
                        return len(plan_effectif), plan_effectif, False
            elif fb_action == "bypass":
                return i + 1, plan_effectif, True
            else:
                state['done'].add(fe.comp_id)
                return i + 1, plan_effectif, True
        else:
            if not verbose:
                print(f"[FAILED]")
            
            if fb_action == "replan" and fb_res.updates.get("mission_failed", False):
                return len(plan_effectif), plan_effectif, False
    
    return i, plan_effectif, False
