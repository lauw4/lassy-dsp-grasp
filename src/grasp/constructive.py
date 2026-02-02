"""GRASP & variants for Disassembly Sequence Planning (DSP)

Core algorithms: greedy randomized construction + local search (VND, Tabu, MNS).
Convention: lower score() = better sequence.

References:
- Feo & Resende (1995/2020), GRASP - J. Global Optimization
- Martins et al. (2024), Hybrid GRASP for DLBP
- Santos et al. (2022), Hybrid GRASP with Tabu Search for DSP
- Ye et al. (2022), Self-evolving system for robotic DSP
"""
import random
import numpy as np
import time
import networkx as nx
from src.utils.metrics import score_profit_net
from src.grasp.local_search import (
    swap_2,
    relocate,
    two_opt,
    mns_local_search,
    tabu_search,
    multi_neighborhood_tabu_search,
    insert_target_valid,
    is_valid,
    swap_selectif,
)
from .config import PARAMS, classify, ALPHA_POOL

def _check_order_rules(seq, order_rules):
    """Check custom order rules compliance. Returns violation count."""
    if not seq or not order_rules:
        return 0
    violations = 0
    for rule in order_rules:
        cond = rule.get('condition', '').lower()
        rule_if = rule.get('if', [])
        if len(rule_if) == 2:
            n1, n2 = rule_if[0], rule_if[1]
            if n1 in seq and n2 in seq:
                idx_n1 = seq.index(n1)
                idx_n2 = seq.index(n2)
                if 'avant' in cond and idx_n1 > idx_n2:
                    violations += 1
                if 'apres' in cond and idx_n1 < idx_n2:
                    violations += 1
    return violations

def greedy_randomized_construction(G, alpha, target_nodes=None, needed_nodes=None):
    """
    Greedy randomized construction (Feo & Resende, 2020).
    
    Args:
        alpha: Greediness (1=greedy, 0=random)
    Returns:
        Valid disassembly sequence.
    """
    remaining = set(needed_nodes) if needed_nodes else set(G.nodes())
    sequence = []
    target_left = set(target_nodes) if target_nodes else set()
    while remaining:
        candidates = [n for n in remaining if all(pred in sequence for pred in G.predecessors(n))]
        if not candidates:
            return sequence
        scores = [1.0 / (G.nodes[n].get("time", 1.0)) for n in candidates]
        max_s, min_s = max(scores), min(scores)
        threshold = min_s + alpha * (max_s - min_s)
        rcl = [n for n, s in zip(candidates, scores) if s >= threshold]
        chosen = random.choice(rcl)
        sequence.append(chosen)
        remaining.remove(chosen)
    return sequence


def closure_with_predecessors(G, target_nodes):
    """Return nodes needed to reach all targets (targets + all predecessors)."""
    needed = set()
    stack = list(target_nodes)
    while stack:
        node = stack.pop()
        if node not in needed:
            needed.add(node)
            stack.extend(G.predecessors(node))
    return needed


def randomized_topological_sort(G):
    """
    Stochastic topological sort.
    
    Reference: Santos et al. (2022), "Hybrid GRASP with Tabu Search for DSP."
    """
    G_copy = G.copy()
    sequence = []
    
    while G_copy.nodes():
        available = [n for n in G_copy.nodes() if G_copy.in_degree(n) == 0]
        if not available:
            available = [min(G_copy.nodes(), key=lambda x: G_copy.in_degree(x))]
        
        chosen = random.choice(available)
        sequence.append(chosen)
        G_copy.remove_node(chosen)
    
    return sequence


def reactive_grasp(G, alpha_pool=(0.1, 0.3, 0.5, 0.7, 0.9), max_iterations=100, update_window=8, local_search_fn=None, time_budget=None, early_stop=True, 
                   target_nodes=None, needed_nodes=None):
    """
    Reactive GRASP with adaptive alpha pool weights.
    
    Reference: Pedrosa et al. (2023), "Reactive construction procedures."
    """
    start_time = time.perf_counter() if time_budget else None
    p = np.ones(len(alpha_pool)) / len(alpha_pool)
    history = {a: [] for a in alpha_pool}
    best, best_score = None, float('inf')
    no_improvement_count = 0
    
    for it in range(max_iterations):
        if time_budget and start_time:
            elapsed = time.perf_counter() - start_time
            if elapsed > time_budget:
                break
        
        alpha = random.choices(alpha_pool, weights=p)[0]
        solution = greedy_randomized_construction(G, alpha, target_nodes=target_nodes, needed_nodes=needed_nodes)
        
        if local_search_fn:
            solution = local_search_fn(solution, G)
        
        s = score_profit_net(solution, G)
        history[alpha].append(s)
        
        if best_score == float('inf') or s > best_score:
            best, best_score = solution, s
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if early_stop and no_improvement_count >= 20:
            break
        
        if (it + 1) % update_window == 0:
            avg_scores = np.array([np.mean(history[a]) if history[a] else 0 for a in alpha_pool])
            if avg_scores.max() > 0:
                p = 0.1 + 0.9 * (avg_scores / avg_scores.max())
            else:
                p = np.ones(len(alpha_pool))
            p /= p.sum()
    return best


def grasp_with_vnd(G, alpha_pool=(0.1, 0.3, 0.5, 0.7, 0.9), max_iterations=100, update_window=8, 
                   neighborhoods=None, time_budget=30, target_nodes=None, needed_nodes=None, early_stop=True):
    """
    Reactive GRASP with VND (Variable Neighborhood Descent).
    
    Reference: Ye et al. (2022), "Self-evolving system for robotic DSP."
    """
    def enhanced_vnd_search(solution, graph):
        """VND multi-pass with intelligent restart."""
        if not neighborhoods:
            return solution
        
        current = solution[:]
        graph_size = len(graph.nodes)
        max_rounds = 2 if graph_size <= 50 else 3
        restart_threshold = 3 if graph_size <= 50 else 5
        
        for round_num in range(max_rounds):
            improved_this_round = True
            stagnation = 0  
            local_iterations = 0 
            
            while improved_this_round and stagnation < restart_threshold and local_iterations < 20:
                improved_this_round = False
                for neighborhood_fn in neighborhoods:
                    new_solution = neighborhood_fn(current, graph, target_nodes=target_nodes) 
                    if score_profit_net(new_solution, graph) > score_profit_net(current, graph):
                        current = new_solution
                        improved_this_round = True
                        stagnation = 0
                        break
                if not improved_this_round:
                    stagnation += 1
                local_iterations += 1
            if round_num > 0 and stagnation >= restart_threshold:
                break
        
        return current
    
    best_sequence = reactive_grasp(G, alpha_pool, max_iterations, update_window, enhanced_vnd_search,
                                   time_budget, early_stop=early_stop, target_nodes=target_nodes, needed_nodes=needed_nodes)
    if target_nodes:
        subgraph = G.subgraph(target_nodes)
        if best_sequence is None or not all(t in best_sequence for t in target_nodes):
            best_sequence = randomized_topological_sort(subgraph)
    return best_sequence


def grasp_with_tabu(G, alpha_pool=(0.1, 0.3, 0.5, 0.7, 0.9), max_iterations=100, update_window=10, neighborhoods=None,
                    max_iter=20, target_nodes=None, needed_nodes=None, mode="complet", time_budget=None, early_stop=True):
    """
    GRASP+Tabu Search for DSP.
    
    Reference: Santos et al. (2022), "Hybrid GRASP with Tabu Search for DSP."
    """
    if neighborhoods is None:
        neighborhoods = [swap_2, relocate, two_opt]
    if mode == "selectif" and target_nodes:
        from src.grasp.local_search import insert_target_valid
        neighborhoods.append(lambda seq, G, target_nodes=None: insert_target_valid(seq, G, target_nodes[0]))
    
    graph_size = len(G.nodes)
    size_key = classify(graph_size)
    params = PARAMS[size_key]
    adapted_iterations = min(max_iterations, params.max_iterations)
    adapted_max_iter = min(max_iter, params.tabu_iter)

    def local_search(solution, graph):
        return multi_neighborhood_tabu_search(solution, graph, neighborhoods, max_iter=adapted_max_iter, target_nodes=target_nodes)

    best_sequence = reactive_grasp(G, alpha_pool, adapted_iterations, update_window, local_search, time_budget=time_budget, early_stop=early_stop, target_nodes=target_nodes,
                                   needed_nodes=needed_nodes)
    if target_nodes:
        subgraph = G.subgraph(target_nodes)
    if best_sequence is None or not all(t in best_sequence for t in target_nodes):
        best_sequence = randomized_topological_sort(subgraph)
    return best_sequence


def enhanced_mns_solver(G, max_time_budget=20, max_iterations=100, target_nodes=None, needed_nodes=None, mode="complet", early_stop=True):
    """
    Enhanced MNS (Multi-Neighborhood Search).
    
    Reference: Santos et al. (2022), "Multi-neighborhood search for DSP."
    """
    graph_size = len(G.nodes)
    if mode == "selectif" and target_nodes:
        neighborhoods = [swap_selectif, relocate, two_opt]
    else:
        neighborhoods = [swap_2, relocate, two_opt]
    
    if graph_size <= 50:
        iters = max_iterations
    elif graph_size <= 100:
        iters = int(max_iterations * 0.8)
    else:
        iters = int(max_iterations * 0.6)
    
    def enhanced_mns_search(solution, graph):
        return mns_local_search(solution, graph, neighborhoods, target_nodes=target_nodes)

    best_sequence = reactive_grasp(
        G,
        alpha_pool=(0.1, 0.3, 0.5, 0.7, 0.9),
        max_iterations=iters,
        update_window=8,
        local_search_fn=enhanced_mns_search,
        time_budget=max_time_budget,
        early_stop=early_stop,
        target_nodes=target_nodes,
        needed_nodes=needed_nodes
    )
    if target_nodes:
        needed = closure_with_predecessors(G, target_nodes)
        subgraph = G.subgraph(needed)
        if best_sequence is None or not is_valid(best_sequence, subgraph) or not all(t in best_sequence for t in target_nodes):
            best_sequence = randomized_topological_sort(subgraph)
    return best_sequence


# ===============================
# SELECTIVE DSP FUNCTIONS
# ===============================

def grasp_simple_selectif(G, target_nodes=None, run_idx=0, **kwargs):
    """GRASP simple for selective DSP (targets only)."""
    needed_nodes = closure_with_predecessors(G, target_nodes)
    
    order_rules = kwargs.get('order_rules', [])
    max_attempts = 10
    for attempt in range(max_attempts):
        seq = greedy_randomized_construction(G, alpha=0.5, target_nodes=target_nodes, needed_nodes=needed_nodes)
        # Quick local search
        for _ in range(10):
            seq = swap_selectif(seq, G, target_nodes=target_nodes)
            if isinstance(seq, list) and any(isinstance(x, list) for x in seq):
                seq = [item for sublist in seq for item in (sublist if isinstance(sublist, list) else [sublist])]
            if len(seq) > len(needed_nodes) * 2:
                seq = seq[:len(needed_nodes)]
        seq = [x for x in seq if isinstance(x, str)]
        subgraph = G.subgraph(needed_nodes)
        if not is_valid(seq, subgraph) or not all(t in seq for t in target_nodes):
            seq = randomized_topological_sort(subgraph)
        # Check: target should be last in sequence
        if _check_order_rules(seq, order_rules) == 0:
            if target_nodes and seq and seq[-1] != target_nodes[-1]:
                print(f"[grasp_simple_selectif][WARNING] User target {target_nodes[-1]} is not last in generated sequence: {seq}")
            return seq
    # If no valid sequence found, force topological sort
    seq = randomized_topological_sort(subgraph)
    if _check_order_rules(seq, order_rules) == 0:
        return seq
    # If still not valid, return best found
    return seq

def grasp_with_vnd_selectif(G, target_nodes=None, run_idx=0, **kwargs):
    """GRASP+VND for selective DSP."""
    neighborhoods = [swap_selectif, relocate, two_opt]
    needed_nodes = closure_with_predecessors(G, target_nodes)
    adaptive_iterations = 80 if len(needed_nodes) > 100 else 150
    user_iterations = kwargs.get('max_iterations') or kwargs.get('grasp_iterations')
    max_iterations = user_iterations if user_iterations else adaptive_iterations
    time_budget = kwargs.get('time_budget') or kwargs.get('vnd_time_budget') or kwargs.get('grasp_time_budget')
    early_stop = kwargs.get('early_stop', True)
    order_rules = kwargs.get('order_rules', [])
    max_attempts = 10
    for attempt in range(max_attempts):
        seq = grasp_with_vnd(
            G,
            neighborhoods=neighborhoods,
            target_nodes=target_nodes,
            needed_nodes=needed_nodes,
            max_iterations=max_iterations,
            time_budget=time_budget if time_budget else 30,
            early_stop=early_stop
        )
        subgraph = G.subgraph(needed_nodes)
        if not is_valid(seq, subgraph) or not all(t in seq for t in target_nodes):
            seq = randomized_topological_sort(subgraph)
        if _check_order_rules(seq, order_rules) == 0:
            if target_nodes and seq and seq[-1] != target_nodes[-1]:
                print(f"[grasp_with_vnd_selectif][WARNING] User target {target_nodes[-1]} is not last in generated sequence: {seq}")
            return seq
    seq = randomized_topological_sort(subgraph)
    if _check_order_rules(seq, order_rules) == 0:
        return seq
    return seq


def mns_selectif(G, target_nodes=None, run_idx=0, **kwargs):
    """MNS for selective DSP."""
    needed_nodes = closure_with_predecessors(G, target_nodes)
    adaptive_iterations = 80 if len(needed_nodes) > 100 else 150
    user_iterations = kwargs.get('max_iterations') or kwargs.get('grasp_iterations')
    max_iterations = user_iterations if user_iterations else adaptive_iterations
    time_budget = kwargs.get('time_budget') or kwargs.get('mns_time_budget') or kwargs.get('grasp_time_budget') or 30
    early_stop = kwargs.get('early_stop', True)
    order_rules = kwargs.get('order_rules', [])
    max_attempts = 10
    for attempt in range(max_attempts):
        seq = enhanced_mns_solver(
            G,
            max_time_budget=time_budget,
            max_iterations=max_iterations,
            target_nodes=target_nodes,
            needed_nodes=needed_nodes,
            mode="selectif",
            early_stop=early_stop
        )
        subgraph = G.subgraph(needed_nodes)
        if not is_valid(seq, subgraph) or not all(t in seq for t in target_nodes):
            seq = randomized_topological_sort(subgraph)
        if _check_order_rules(seq, order_rules) == 0:
            if target_nodes and seq and seq[-1] != target_nodes[-1]:
                print(f"[mns_selectif][WARNING] User target {target_nodes[-1]} is not last in generated sequence: {seq}")
            return seq
    seq = randomized_topological_sort(subgraph)
    if _check_order_rules(seq, order_rules) == 0:
        return seq
    return seq


def grasp_with_tabu_selectif(G, target_nodes=None, run_idx=0, **kwargs):
    """GRASP+Tabu for selective DSP."""
    neighborhoods = [swap_selectif, relocate, two_opt]
    needed_nodes = closure_with_predecessors(G, target_nodes)
    adaptive_iterations = 80 if len(needed_nodes) > 100 else 150
    user_iterations = kwargs.get('max_iterations') or kwargs.get('grasp_iterations')
    max_iterations = user_iterations if user_iterations else adaptive_iterations
    tabu_iter = kwargs.get('max_iter') or kwargs.get('tabu_iter') or 20
    time_budget = kwargs.get('time_budget') or kwargs.get('tabu_time_budget') or kwargs.get('grasp_time_budget')
    early_stop = kwargs.get('early_stop', True)
    order_rules = kwargs.get('order_rules', [])
    max_attempts = 10
    for attempt in range(max_attempts):
        seq = grasp_with_tabu(
            G,
            neighborhoods=neighborhoods,
            target_nodes=target_nodes,
            needed_nodes=needed_nodes,
            max_iterations=max_iterations,
            max_iter=tabu_iter,
            mode="selectif",
            time_budget=time_budget,
            early_stop=early_stop
        )
        subgraph = G.subgraph(needed_nodes)
        if not is_valid(seq, subgraph) or not all(t in seq for t in target_nodes):
            seq = randomized_topological_sort(subgraph)
        if _check_order_rules(seq, order_rules) == 0:
            if target_nodes and seq and seq[-1] != target_nodes[-1]:
                print(f"[grasp_with_tabu_selectif][WARNING] User target {target_nodes[-1]} is not last in generated sequence: {seq}")
            return seq
    seq = randomized_topological_sort(subgraph)
    if _check_order_rules(seq, order_rules) == 0:
        return seq
    return seq


# ============================================================
# Unified interface
# ============================================================

_ALGO_MAP_COMPLET = {
    'vnd': grasp_with_vnd,
    'tabu': grasp_with_tabu,
    'mns': enhanced_mns_solver,
}

_ALGO_MAP_SELECTIF = {
    'simple': grasp_simple_selectif,
    'vnd': grasp_with_vnd_selectif,
    'tabu': grasp_with_tabu_selectif,
    'mns': mns_selectif,
}


def run_grasp(G, algorithm='vnd', mode='selectif', target_nodes=None, runs=1, **kwargs):
    """
    Unified entry point for all GRASP variants.

    Args:
        G: NetworkX DiGraph
        algorithm: 'vnd' | 'tabu' | 'mns' | 'simple' (simple only for selective mode)
        mode: 'complet' or 'selectif'
        target_nodes: target list (required for selective mode)
        runs: independent runs (keeps best)
        **kwargs: additional parameters

    Returns:
        (best_sequence, best_score)
    """
    if mode not in ('complet', 'selectif'):
        raise ValueError("mode must be 'complet' or 'selectif'")
    if mode == 'selectif' and not target_nodes:
        raise ValueError("target_nodes required for selectif mode")

    algo_map = _ALGO_MAP_SELECTIF if mode == 'selectif' else _ALGO_MAP_COMPLET
    if algorithm not in algo_map:
        raise ValueError(f"Unknown algorithm: {algorithm} (available: {list(algo_map.keys())})")

    best_seq, best_s = None, float('-inf')
    for _ in range(runs):
        seq = algo_map[algorithm](G, target_nodes=target_nodes, **kwargs)
        s = score_profit_net(seq, G)
        if s > best_s:
            best_seq, best_s = seq, s
    return best_seq, best_s


__all__ = [
    'greedy_randomized_construction', 'reactive_grasp', 'grasp_with_vnd', 'grasp_with_tabu',
    'enhanced_mns_solver', 'grasp_simple_selectif', 'grasp_with_vnd_selectif', 'grasp_with_tabu_selectif',
    'mns_selectif', 'run_grasp'
]

