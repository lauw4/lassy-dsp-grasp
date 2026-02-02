"""
Adaptive actions for DSP failure handling.

Four recovery actions based on fuzzy decision:
- bypass: Skip component via alternative path
- change_tool: Switch to alternative tool
- destroy: Destructive removal (low-value components)
- replan: Full sequence re-planning
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from src.adaptive.types import FailureEvent
from src.adaptive.overlay import (
    mark_done, lock_node, unlock_node, mark_destroyed, feasible_now_excluding
)
from src.core.graph_api import (
    node_value, no_destroy_flag, blocks_high_value_downstream, feasible_now_nodes, replan_with_overlay
)

@dataclass
class ActionResult:
    success: bool
    action_taken: str
    updates: Dict[str, Any]
    notes: str = ""

def try_bypass(state: Dict[str, Any], fe: FailureEvent) -> ActionResult:
    """
    Check if targets can be reached by removing the failed component from sequence.
    
    Logic (Ye et al. 2022):
    - Check if alt_paths exists in context
    - If yes, remove failed component from sequence and verify validity
    - Sequence without this component must remain valid and reach targets
    
    Returns:
        ActionResult with success=True if bypass possible
    """
    # Check if alternative paths exist (context)
    if not fe.context.get("alt_paths", False):
        state["notes"].append(f"Bypass impossible for {fe.comp_id}: no alternative path")
        return ActionResult(False, "bypass", {}, notes="No alternative path in graph.")
    
    # Mark component as "skipped" (label): we don't execute it,
    # but this is NOT a physical blockage (so should not go into locked).
    state.setdefault('skipped', set()).add(fe.comp_id)
    # Safety: if old logic put it in locked, remove it.
    if 'locked' in state:
        state['locked'].discard(fe.comp_id)
    
    print(f"[ACTIONS] try_bypass: {fe.comp_id} excluded from sequence")
    # Backward compatibility: also keep 'bypassed' key in updates.
    return ActionResult(True, "bypass", {"skipped": fe.comp_id, "bypassed": fe.comp_id}, notes=f"Component {fe.comp_id} bypassed (skipped).")

def try_change_tool(state: Dict[str, Any], fe: FailureEvent) -> ActionResult:
    """
    Try using an alternative tool if available.
    """
    alt_tools = fe.context.get("alt_tools_available")
    print(f"[ACTIONS] try_change_tool: alt_tools={alt_tools}")
    if alt_tools:
        mark_done(state, fe.comp_id)
        return ActionResult(True, "change_tool", {"tool_used": alt_tools[0], "effect": "improve_success"}, notes="Alternative tool suggested.")
    else:
        state["notes"].append(f"No alternative tool for {fe.comp_id}")
        return ActionResult(False, "change_tool", {}, notes="No alternative tool.")

def try_destroy(state: Dict[str, Any], fe: FailureEvent) -> ActionResult:
    """
    Try destructive removal if allowed (low-value component).
    """
    # Get profit from failure context or graph
    profit = None
    if hasattr(fe, 'context') and fe.context and 'value_piece' in fe.context:
        profit = fe.context['value_piece']
    
    if profit is None:
        from src.core.graph_api import node_value
        profit = node_value(fe.comp_id)
    
    if profit is None:
        profit = 0.0  # Default if profit undefined
    
    # Check no_destroy flag
    no_destroy = False
    is_fallback_destroy = False
    
    if hasattr(fe, 'context') and fe.context:
        is_fallback_destroy = fe.context.get('fallback_destroy', False)
        no_destroy = fe.context.get('no_destroy', False)
    
    if not no_destroy:
        G = state.get('graph', None)
        if G is not None and fe.comp_id in G.nodes:
            no_destroy = G.nodes[fe.comp_id].get('no_destroy', False)
    
    # Get threshold from state
    cutoff = None
    percentile = None
    thresholds = state.get('thresholds', None)
    if thresholds and hasattr(thresholds, 'low_value_cutoff'):
        cutoff = thresholds.low_value_cutoff
        percentile = getattr(thresholds, 'low_value_percentile', 20)
    else:
        G = state.get('graph', None)
        if G is not None:
            profits = [G.nodes[n].get('profit', 0.0) for n in G.nodes]
            import numpy as np
            percentile = getattr(thresholds, 'low_value_percentile', 20) if thresholds else 20
            cutoff = float(np.percentile(profits, percentile)) if profits else 0.0
    
    if cutoff is None:
        cutoff = 0.0  # Default value if threshold undefined
    if percentile is None:
        percentile = 20  # Default value
    
    # Determine if destruction is allowed
    reason = ""
    
    # Human approval (optional): default False if absent.
    # Backward compat: accept old field `human_approval_simulated`.
    human_approved = False
    if hasattr(fe, 'context') and fe.context:
        human_approved = bool(
            fe.context.get('human_approved', fe.context.get('human_approval_simulated', False))
        )
    
    # FALLBACK DESTROY with human approval: relaxed criteria as last resort
    if is_fallback_destroy and human_approved:
        # Simulated human approval: destruction allowed despite no_destroy
        reason = f"Destruction approved by operator (last resort): profit={profit:.2f} lost to save mission."
        nd_flag = False
        print(f"[ACTIONS] try_destroy [HUMAN APPROVED]: profit={profit:.2f} | {reason}")
    elif is_fallback_destroy and not human_approved:
        # Fallback without human approval: cannot destroy protected part
        reason = f"Destruction refused: protected part (no_destroy=true), human approval required."
        nd_flag = True
        print(f"[ACTIONS] try_destroy [FALLBACK DENIED]: profit={profit:.2f} | {reason}")
    elif no_destroy:
        reason = f"Flag 'no_destroy' active (context or graph)."
        nd_flag = True
    elif profit > cutoff:
        reason = f"Profit={profit:.2f} > threshold={cutoff:.2f} (percentile {percentile}th)."
        nd_flag = True
    else:
        reason = f"Destruction allowed (profit={profit:.2f} <= threshold={cutoff:.2f}, percentile {percentile}th)."
        nd_flag = False
    
    if not is_fallback_destroy:
        print(f"[ACTIONS] try_destroy: no_destroy_flag={nd_flag} | profit={profit:.2f} | threshold={cutoff:.2f} | percentile={percentile} | {reason}")
    
    if nd_flag:
        state["notes"].append(f"Destruction forbidden for {fe.comp_id}: {reason}")
        return ActionResult(False, "destroy", {}, notes=f"Destruction not allowed: {reason}")
    else:
        mark_destroyed(state, fe.comp_id)
        return ActionResult(True, "destroy", {"destroyed": fe.comp_id}, notes=f"Destruction allowed (cutoff percentile {percentile}th).")

def replan_grasp(state: Dict[str, Any], fe: FailureEvent = None) -> ActionResult:
    """
    Replan sequence considering overlay state.
    If fe is provided AND component is not destroyed, lock it before replan
    (avoids falling back on same component - Ye et al. 2022).
    """
    from src.core.graph_api import TargetBlockedException
    
    try:
        # Lock problematic component if provided AND not destroyed
        if fe is not None and fe.comp_id not in state.get('destroyed', set()):
            state['locked'].add(fe.comp_id)
            print(f"[ACTIONS] replan_grasp: locking {fe.comp_id} before replan")
            logging.info(f"[REPLAN] Locking {fe.comp_id} before replan")
        
        # Log states before replan
        done = state.get('done', set())
        destroyed = state.get('destroyed', set())
        locked = state.get('locked', set())
        skipped = state.get('skipped', set())
        
        logging.info(f"[REPLAN] States before replan:")
        logging.info(f"[REPLAN]   done={sorted(done)}")
        logging.info(f"[REPLAN]   destroyed={sorted(destroyed)}")
        logging.info(f"[REPLAN]   locked={sorted(locked)}")
        logging.info(f"[REPLAN]   skipped={sorted(skipped)}")
        
        print(f"[ACTIONS] replan_grasp: calling replan_with_overlay with state overlay (done={state.get('done')}, destroyed={state.get('destroyed')}, locked={state.get('locked')})")
        new_plan = replan_with_overlay(state)
        print(f"[ACTIONS] replan_grasp: new_plan={new_plan}")
        
        # Log new sequence
        logging.info(f"[REPLAN] New sequence: {new_plan}")
        
        if new_plan:
            return ActionResult(True, "replan", {"new_plan": new_plan}, notes="Replanning successful.")
        else:
            return ActionResult(False, "replan", {}, notes="No plan possible.")
    except TargetBlockedException as e:
        logging.info(f"[REPLAN] MISSION IMPOSSIBLE: {str(e)}")
        return ActionResult(False, "replan", {"mission_failed": True}, notes=f"MISSION IMPOSSIBLE: {str(e)}")
    except NotImplementedError:
        return ActionResult(False, "replan", {}, notes="Replan not implemented.")

def dispatch(action: str, fe: FailureEvent, state: Dict[str, Any]) -> ActionResult:
    """
    Dispatch the requested action.
    """
    ACTIONS = {
        "bypass": try_bypass,
        "change_tool": try_change_tool,
        "destroy": try_destroy,
        "replan": replan_grasp,  # Now passes fe correctly
    }
    if action == "manual_intervention":
        print("[FAIL SAFE] No adaptive action possible, human intervention required!")
        state["notes"].append("Manual intervention required: no adaptive action possible.")
        return ActionResult(False, "manual_intervention", {}, notes="Manual intervention required.")
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")
    return ACTIONS[action](state, fe)


