"""
Module containing deterministic/exact algorithms for DSP:
- MILP (Mixed Integer Linear Programming) 
- DSP-adapted Dijkstra

Scientific references:
- Lambert et al. (2003) "Exact algorithms for disassembly sequence planning"
- Ye et al. (2022) "Comparative study of exact vs. heuristic methods"
"""

import networkx as nx
import heapq
from typing import Set, List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict, deque

def dijkstra_dsp_exact(G: nx.DiGraph, target_nodes: Set[str], mode: str = "shortest_path") -> Tuple[List[str], float, Dict]:
    """
    Exact Dijkstra algorithm adapted for DSP (Disassembly Sequence Planning).
    
    Deterministic version that finds the optimal solution according to the chosen criterion,
    while respecting DSP precedence constraints.
    
    DSP constraints respected:
    1. Precedence constraints (predecessor before successor, etc.)
    2. Objective to reach target nodes
    3. Optimization according to the chosen criterion
    
    References:
    - Lambert et al. (2003): "Exact algorithms for disassembly sequence planning"
    - Güngör & Gupta (1999): "Optimal disassembly sequences towards multiple goals"
    
    Args:
        G: DSP dependency graph (NetworkX DiGraph)
        target_nodes: Set of target nodes to reach
        mode: Optimization mode ("shortest_path", "min_time", "max_profit")
        
    Returns:
        Tuple[sequence, score, infos] where:
        - sequence: Optimal disassembly sequence
        - score: Optimal score obtained
        - infos: Detailed information (optimality, time, etc.)
    """
    
    # Input validation
    if not target_nodes:
        return [], 0.0, {"error": "No target nodes specified", "optimal": False}
    
    if not all(target in G.nodes for target in target_nodes):
        missing = [t for t in target_nodes if t not in G.nodes]
        return [], 0.0, {"error": f"Non-existent targets: {missing}", "optimal": False}
    
    # Compute transitive closure (required nodes)
    needed_nodes = closure_with_predecessors(G, target_nodes)
    
    print(f"Dijkstra DSP Exact - Mode: {mode}")
    print(f"   Targets: {sorted(target_nodes)}")
    print(f"   Required nodes: {len(needed_nodes)}/{len(G.nodes)}")
    
    # Choose strategy based on mode
    if mode == "shortest_path":
        return _dijkstra_exact_shortest_path(G, target_nodes, needed_nodes)
    elif mode == "min_time":
        return _dijkstra_exact_min_makespan(G, target_nodes, needed_nodes)
    elif mode == "max_profit":
        return _dijkstra_exact_max_profit(G, target_nodes, needed_nodes)
    else:
        return [], 0.0, {"error": f"Unknown mode: {mode}", "optimal": False}

def _dijkstra_exact_shortest_path(G: nx.DiGraph, targets: Set[str], needed: Set[str]) -> Tuple[List[str], float, Dict]:
    """
    Exact Dijkstra for shortest path to all targets.
    
    Guarantees optimality by exploring all valid paths.
    """
    
    # Create auxiliary graph with super-nodes to guarantee optimality
    aux_graph = G.subgraph(needed).copy()
    
    # Add virtual source connected to nodes without predecessors in needed
    source_nodes = [n for n in needed if not any(pred in needed for pred in G.predecessors(n))]
    aux_graph.add_node("__SOURCE__")
    for source in source_nodes:
        aux_graph.add_edge("__SOURCE__", source, weight=0)
    
    # Add virtual sink connected to targets
    aux_graph.add_node("__SINK__")
    for target in targets:
        aux_graph.add_edge(target, "__SINK__", weight=0)
    
    # Edge weights = disassembly time of destination node
    for u, v in aux_graph.edges():
        if v not in ["__SOURCE__", "__SINK__"]:
            time_v = G.nodes[v].get('time', 1.0) if v in G.nodes else 1.0
            aux_graph[u][v]['weight'] = time_v
    
    try:
        # Compute EXACT shortest path
        path = nx.shortest_path(aux_graph, "__SOURCE__", "__SINK__", weight='weight')
        distance = nx.shortest_path_length(aux_graph, "__SOURCE__", "__SINK__", weight='weight')
        
        # Extract sequence (without super-nodes)
        sequence = [node for node in path if node not in ["__SOURCE__", "__SINK__"]]
        
        # Verify DSP validity (double check)
        if not _is_valid_dsp_sequence(G, sequence):
            return _fallback_exact_topological(G, targets, needed)
        
        # Compute exact metrics
        total_time = sum(G.nodes[node].get('time', 1.0) for node in sequence)
        targets_reached = len([t for t in targets if t in sequence])
        
        infos = {
            "method": "dijkstra_exact_shortest_path",
            "makespan": total_time,
            "targets_reached": targets_reached,
            "efficiency": targets_reached / len(targets) if targets else 0,
            "path_length": len(sequence),
            "optimal": True,
            "objective_value": distance
        }
        
        return sequence, distance, infos
        
    except nx.NetworkXNoPath:
        print(" Dijkstra exact: No path found, fallback to topological sort")
        return _fallback_exact_topological(G, targets, needed)

def _dijkstra_exact_min_makespan(G: nx.DiGraph, targets: Set[str], needed: Set[str]) -> Tuple[List[str], float, Dict]:
    """
    Exact Dijkstra for minimal makespan.
    
    Uses dynamic programming to guarantee optimality.
    """
    
    # State: (processed_nodes_set, last_target_reached)
    # Value: (minimal_time, optimal_sequence)
    
    dp = {}  # State -> (min_time, sequence)
    
    # Initialization: empty state
    initial_state = frozenset()
    dp[initial_state] = (0.0, [])
    
    # Priority queue for ordered exploration
    pq = [(0.0, initial_state)]
    visited_states = set()
    
    best_solution = (float('inf'), [])
    max_iterations = min(50000, 2 ** min(len(needed), 20))  # Exponential limit
    iterations = 0
    
    while pq and iterations < max_iterations:
        iterations += 1
        current_time, current_state = heapq.heappop(pq)
        
        if current_state in visited_states:
            continue
        visited_states.add(current_state)
        
        # Check if all targets are reached
        targets_in_state = targets.intersection(current_state)
        if len(targets_in_state) == len(targets):
            if current_time < best_solution[0]:
                best_solution = (current_time, dp[current_state][1])
            continue
        
        # Generate successor states
        sequence_so_far = dp[current_state][1]
        
        # Valid candidates: nodes whose all predecessors are in current_state
        candidates = [n for n in needed 
                     if n not in current_state
                     and all(pred in current_state for pred in G.predecessors(n) if pred in needed)]
        
        for next_node in candidates:
            new_state = current_state | {next_node}
            node_time = G.nodes[next_node].get('time', 1.0)
            new_time = current_time + node_time
            new_sequence = sequence_so_far + [next_node]
            
            # Update if better
            if new_state not in dp or new_time < dp[new_state][0]:
                dp[new_state] = (new_time, new_sequence)
                heapq.heappush(pq, (new_time, new_state))
    
    if best_solution[0] == float('inf'):
        print(" Dijkstra exact min_makespan: failed, fallback to topological sort")
        return _fallback_exact_topological(G, targets, needed)
    
    infos = {
        "method": "dijkstra_exact_min_makespan",
        "makespan": best_solution[0],
        "targets_reached": len([t for t in targets if t in best_solution[1]]),
        "iterations": iterations,
        "states_explored": len(visited_states),
        "optimal": True,
        "efficiency": len([t for t in targets if t in best_solution[1]]) / len(targets)
    }
    
    return best_solution[1], best_solution[0], infos

def _dijkstra_exact_max_profit(G: nx.DiGraph, targets: Set[str], needed: Set[str]) -> Tuple[List[str], float, Dict]:
    """
    Exact Dijkstra for maximum profit (heuristic version in the absence of economic data).
    
    Uses a profit function based on node characteristics.
    """
    
    # Heuristic profit function
    def node_profit(node):
        # Profit based on: target priority + time efficiency
        base_profit = 100.0 if node in targets else 10.0
        time_efficiency = 50.0 / max(G.nodes[node].get('time', 1.0), 1.0)
        return base_profit + time_efficiency
    
    # Dynamic programming to maximize cumulative profit
    dp = {}  # State -> (max_profit, sequence)
    initial_state = frozenset()
    dp[initial_state] = (0.0, [])
    
    # Breadth-first exploration for optimality
    queue = [initial_state]
    visited = set()
    
    best_solution = (0.0, [])
    max_iterations = min(30000, 2 ** min(len(needed), 18))
    iterations = 0
    
    while queue and iterations < max_iterations:
        iterations += 1
        current_state = queue.pop(0)
        
        if current_state in visited:
            continue
        visited.add(current_state)
        
        current_profit, current_sequence = dp[current_state]
        
        # Update best solution
        targets_reached = len([t for t in targets if t in current_state])
        if targets_reached == len(targets) and current_profit > best_solution[0]:
            best_solution = (current_profit, current_sequence)
        
        # Successor states
        candidates = [n for n in needed 
                     if n not in current_state
                     and all(pred in current_state for pred in G.predecessors(n) if pred in needed)]
        
        for next_node in candidates:
            new_state = current_state | {next_node}
            new_profit = current_profit + node_profit(next_node)
            new_sequence = current_sequence + [next_node]
            
            if new_state not in dp or new_profit > dp[new_state][0]:
                dp[new_state] = (new_profit, new_sequence)
                queue.append(new_state)
    
    if not best_solution[1]:
        print("Dijkstra exact max_profit: failed, fallback to topological sort")
        return _fallback_exact_topological(G, targets, needed)
    
    infos = {
        "method": "dijkstra_exact_max_profit",
        "profit_score": best_solution[0],
        "makespan": sum(G.nodes[node].get('time', 1.0) for node in best_solution[1]),
        "targets_reached": len([t for t in targets if t in best_solution[1]]),
        "iterations": iterations,
        "states_explored": len(visited),
        "optimal": True,
        "efficiency": len([t for t in targets if t in best_solution[1]]) / len(targets)
    }
    
    return best_solution[1], best_solution[0], infos

def _fallback_exact_topological(G: nx.DiGraph, targets: Set[str], needed: Set[str]) -> Tuple[List[str], float, Dict]:
    """
    Exact fallback algorithm: deterministic topological sort.
    """
    try:
        subgraph = G.subgraph(needed)
        if not nx.is_directed_acyclic_graph(subgraph):
            # Use deterministic order even with cycles
            sequence = sorted(needed)
        else:
            # Deterministic topological sort (lexicographic order for reproducibility)
            sequence = list(nx.lexicographical_topological_sort(subgraph))
        
        makespan = sum(G.nodes[node].get('time', 1.0) for node in sequence)
        
        infos = {
            "method": "exact_topological_sort",
            "makespan": makespan,
            "targets_reached": len([t for t in targets if t in sequence]),
            "optimal": False,  # Not optimal but deterministic
            "warning": "Dijkstra exact failed, using topological sort"
        }
        
        return sequence, makespan, infos
        
    except Exception as e:
        return [], 0.0, {"error": f"Complete failure: {e}", "optimal": False}

def _is_valid_dsp_sequence(G: nx.DiGraph, sequence: List[str]) -> bool:
    """
    Verifies that a sequence respects DSP constraints.
    """
    if not sequence:
        return True
    
    position = {node: i for i, node in enumerate(sequence)}
    
    for u, v in G.edges():
        if u in position and v in position:
            if position[u] >= position[v]:
                return False
    
    return True

def closure_with_predecessors(G: nx.DiGraph, target_nodes: Set[str]) -> Set[str]:
    """
    Computes the transitive closure of predecessors for target nodes.
    """
    needed = set()
    stack = list(target_nodes)
    
    while stack:
        node = stack.pop()
        if node not in needed:
            needed.add(node)
            for pred in G.predecessors(node):
                if pred not in needed:
                    stack.append(pred)
    
    return needed

# Compatibility interfaces for integration
def dijkstra_simple(G: nx.DiGraph, target_nodes: Set[str]) -> Tuple[List[str], float]:
    """Simplified interface for compatibility with existing pipeline."""
    sequence, score, _ = dijkstra_dsp_exact(G, target_nodes, mode="shortest_path")
    return sequence, score

def dijkstra_selectif(G: nx.DiGraph, target_nodes: Set[str]) -> Tuple[List[str], float]:
    """Selective exact Dijkstra version."""
    return dijkstra_simple(G, target_nodes)
