class TargetBlockedException(Exception):
    """Exception raised when target is physically blocked by a locked component."""
    pass


def replan_with_overlay(state: dict[str, object]) -> list[str]:
    """
    Replan sequence considering runtime overlay state.
    
    Handles:
        - state["destroyed"]: profit=0 (excluded from value)
        - state["locked"]: temporarily inaccessible
        - state["done"]: already processed (removed)
    
    Propagates blocking: descendants of locked nodes become inaccessible
    unless they have alternative paths (DAG convergence).
    
    Raises:
        TargetBlockedException: If target is physically unreachable
    """
    import networkx as nx
    from src.utils import graph_io
    from src.grasp import constructive

    G_orig = state.get("graph")
    if G_orig is None:
        raise ValueError("Graph must be provided in state['graph'].")
    G = G_orig.copy()

    destroyed = set(state.get("destroyed", set()))
    locked = set(state.get("locked", set()))
    done = set(state.get("done", set()))
    skipped = set(state.get("skipped", set()))

    # Set profit=0 for destroyed nodes
    for n in destroyed:
        if n in G.nodes:
            G.nodes[n]["profit"] = 0.0

    # Remove processed nodes + skipped (bypassed) nodes
    # Note: skipped is not a physical blockage (unlike locked)
    G.remove_nodes_from(done | skipped)

    # Cascade blocking: node blocked only if ALL remaining predecessors are blocked
    # Avoids blocking descendants with alternative access paths (DAG convergence)
    blocked_cascade = set(locked)
    remaining_nodes = set(G.nodes) - blocked_cascade
    changed = True
    while changed:
        changed = False
        newly_blocked = set()
        for node in list(remaining_nodes):
            preds = set(G.predecessors(node))
            # Removed predecessors (done/skipped/destroyed) don't count
            preds -= done
            preds -= skipped
            preds -= destroyed
            if preds and preds.issubset(blocked_cascade):
                newly_blocked.add(node)
        if newly_blocked:
            blocked_cascade.update(newly_blocked)
            remaining_nodes -= newly_blocked
            changed = True
            print(f"[REPLAN] Cascade blocking: {newly_blocked}")
    
    print(f"[REPLAN] Initial locked: {locked}")
    print(f"[REPLAN] Blocked after propagation: {blocked_cascade}")

    # Exclude locked nodes and their blocked descendants
    needed_nodes = [n for n in G.nodes if n not in blocked_cascade]

    # Debug info (commented for production)
    # print("[REPLAN] Remaining nodes:", list(G.nodes))
    # print("[REPLAN] Destroyed:", destroyed)
    # print("[REPLAN] Locked:", locked)
    # print("[REPLAN] Skipped:", skipped)
    # print("[replan_with_overlay][DEBUG] Done nodes:", done)
    # print("[replan_with_overlay][DEBUG] needed_nodes:", needed_nodes)
    # print("[replan_with_overlay][DEBUG] Edges:", list(G.edges))

    # Strict determination of user target (robust DSP)
    user_targets = state.get('targets', None)
    if user_targets is None or len(user_targets) == 0:
        # fallback: graph leaves
        user_targets = [n for n in G.nodes if G.out_degree(n) == 0]
    # Keep only user targets still present in graph
    user_targets_present = [t for t in user_targets if t in G.nodes]
    if len(user_targets_present) == 0:
        print("[replan_with_overlay][CRITICAL] User target unreachable after adaptation. Stopping (cf. Ye2022, Pedrosa2023, Frizziero2020)")
        return []
    
    # Check if target is physically blocked
    forced_target = user_targets_present[0]
    if forced_target in blocked_cascade:
        print(f"[replan_with_overlay][CRITICAL] Target {forced_target} is physically blocked by a non-removed component (locked). Mission impossible.")
        print(f"[replan_with_overlay][CRITICAL] Locked components: {locked}")
        print(f"[replan_with_overlay][CRITICAL] Blocking cascade: {blocked_cascade}")
        # Raise exception to signal mission is impossible
        raise TargetBlockedException(f"Target {forced_target} physically blocked by {locked}")
    
    # Force user target as sole goal, even if not a leaf
    # print(f"[replan_with_overlay][DEBUG] User target forced for selective replanning: {forced_target}")
    try:
        seqs = constructive.run_grasp(G, algorithm='vnd', mode='selectif', target_nodes=[forced_target], runs=1, needed_nodes=needed_nodes)
        # print(f"[replan_with_overlay][DEBUG] run_grasp result: {seqs}")
        # run_grasp returns (sequence, score) or [(sequence, score), ...]
        # We want to extract the sequence (list of ids)
        if isinstance(seqs, tuple) and len(seqs) == 2 and isinstance(seqs[0], list):
            seq = seqs[0]
        elif isinstance(seqs, list) and len(seqs) > 0:
            first = seqs[0]
            if isinstance(first, tuple) and len(first) == 2 and isinstance(first[0], list):
                seq = first[0]
            elif isinstance(first, list):
                seq = first
            else:
                seq = seqs
        else:
            seq = []
        # Explicit verification: target must be reached in sequence
        if forced_target not in seq:
            print(f"[replan_with_overlay][CRITICAL] User target {forced_target} unreachable in generated sequence after replan. Stop (cf. Ye2022, Pedrosa2023, Frizziero2020)")
            return []
        # Strict filtering: remove any blocked/destroyed part from sequence (DSP robustness)
        # CORRECTION: Use blocked_cascade instead of locked alone
        seq_filtered = [n for n in seq if n not in blocked_cascade and n not in destroyed]
        if len(seq_filtered) < len(seq):
            print(f"[replan_with_overlay][WARNING] GRASP sequence contained blocked/destroyed parts, filtered per literature (Ye2022, Pedrosa2023, Frizziero2020).")
        # Explicit verification: target must be reached in filtered sequence
        if forced_target not in seq_filtered:
            print(f"[replan_with_overlay][CRITICAL] User target {forced_target} unreachable in generated sequence after replan (after locked/destroyed filtering). Stop (cf. Ye2022, Pedrosa2023, Frizziero2020)")
            return []
        return seq_filtered
    except Exception as e:
        print(f"[replan_with_overlay] GRASP Error: {e}")
        return []
"""
Facades (stubs) to interface with existing code (GRASP/DAG/Dijkstra/MILP).
"""
from typing import Any

def node_value(comp_id: str) -> float:
    """Return node (component) value from current graph ('profit' attribute).
    Args:
        comp_id: Component identifier.
    Returns:
        float: Component value (profit), or 0.0 if absent.
    """
    import inspect
    # Try to find current graph in call stack (state['graph'])
    for frame_info in inspect.stack():
        frame = frame_info.frame
        if 'state' in frame.f_locals and 'graph' in frame.f_locals['state']:
            G = frame.f_locals['state']['graph']
            if comp_id in G.nodes:
                return float(G.nodes[comp_id].get('profit', 0.0))
    # Fallback: dummy value
    return 0.0

def no_destroy_flag(comp_id: str) -> bool:
    """
    Check if component should not be destroyed.
    - Priority to explicit 'no_destroy' attribute (present on node)
    - Otherwise, compare node profit to dynamic threshold (low_value_cutoff)
    
    References:
    - Ye et al., 2022, Self-evolving system for robotic DSP (low value cutoff)
    - Pedrosa et al., 2023, Reactive construction procedures (dynamic threshold)
    - Frizziero et al., 2020, DSP Applied to a Gear Box (flexible policy)
    """
    import inspect
    # Search for current graph and threshold in call stack
    G = None
    thresholds = None
    for frame_info in inspect.stack():
        frame = frame_info.frame
        if 'state' in frame.f_locals:
            state = frame.f_locals['state']
            if 'graph' in state:
                G = state['graph']
            # Look for explicit threshold in state or thresholds
            if 'thresholds' in state:
                thresholds = state['thresholds']
        if 'thresholds' in frame.f_locals:
            thresholds = frame.f_locals['thresholds']
    if G is None:
        return False  # permissive fallback
    # 1. Explicit attribute on node
    if comp_id in G.nodes and G.nodes[comp_id].get('no_destroy', False):
        return True
    # 2. Dynamic threshold (percentile on profits)
    import numpy as np
    # a) Threshold provided in thresholds
    if thresholds and hasattr(thresholds, 'low_value_cutoff'):
        cutoff = thresholds.low_value_cutoff
    else:
        profits = [G.nodes[n].get('profit', 0.0) for n in G.nodes]
        percentile = getattr(thresholds, 'low_value_percentile', 20) if thresholds else 20
        cutoff = float(np.percentile(profits, percentile)) if profits else 0.0
    profit = G.nodes[comp_id].get('profit', 0.0) if comp_id in G.nodes else 0.0
    return profit > cutoff


def feasible_now_nodes(state: dict[str,Any], exclude: set[str] | None = None) -> list[str]:
    """Return list of feasible nodes in current state (predecessors in done)."""
    G = state.get("graph")
    if G is None:
        return []
    done = set(state.get("done", set()))
    destroyed = set(state.get("destroyed", set()))
    locked = set(state.get("locked", set()))
    skipped = set(state.get("skipped", set()))
    if exclude is None:
        exclude = set()
    candidates = []
    for n in G.nodes:
        if n in done or n in destroyed or n in locked or n in skipped or n in exclude:
            continue
        preds = set(G.predecessors(n))
        satisfied = done | skipped | destroyed
        if preds.issubset(satisfied):
            candidates.append(n)
    return candidates


def blocks_high_value_downstream(comp_id: str, value_cutoff: float = 0.5) -> bool:
    """Check if this component blocks access to high-value descendants."""
    import networkx as nx
    import inspect
    G = None
    for frame_info in inspect.stack():
        frame = frame_info.frame
        if 'state' in frame.f_locals and 'graph' in frame.f_locals['state']:
            G = frame.f_locals['state']['graph']
            break
    if G is None or comp_id not in G.nodes:
        return False
    descendants = nx.descendants(G, comp_id)
    for d in descendants:
        profit = G.nodes[d].get('profit', 0.0)
        if profit > value_cutoff:
            return True
    return False
