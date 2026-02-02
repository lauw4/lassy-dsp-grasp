
"""
Local search operators for DSP (Disassembly Sequence Planning)

Operators: swap_2, relocate, two_opt, insert_target_valid
Meta-heuristics: VND, Tabu Search, MNS (Multi-Neighborhood Search)
"""
import random
from collections import deque

def is_valid(sequence, G):
    """Validate sequence against precedence constraints.

    Convention (DSP & scheduling literature):
        u -> v (directed edge) means "u must be removed BEFORE v" (u precedes v),
        hence constraint pos[u] < pos[v].

    References:
        - Ye et al. (2022), "A self-evolving system for robotic DSP"
        - Santos et al. (2022), "Hybrid GRASP with Tabu Search for DSP"

    Args:
        sequence (list): proposed disassembly order
        G (nx.DiGraph): precedence graph

    Returns:
        bool: True if all edges (u,v) satisfy pos[u] < pos[v]
    """
    pos = {node: i for i, node in enumerate(sequence)}
    sequence_nodes = set(sequence)
    for u, v in G.edges:
        if u in sequence_nodes and v in sequence_nodes:
            if pos[u] >= pos[v]:
                return False
    return True

def _targets_preserved(seq, target_nodes):
    """Return True if all targets are present (or no targets specified)."""
    if not target_nodes:
        return True
    target_set = set(target_nodes)
    return target_set.issubset(seq)

def swap_2(sequence, G, target_nodes=None):
    from src.utils.metrics import score
    """
    Swap neighborhood operator: exchange two components.
    
    Reference: Ye et al. (2022), Section 3.3 "Neighborhood structures for DSP"
    
    Returns:
        Modified sequence or original if no valid swap found.
    """
    seq = sequence[:]
    n = len(seq)
    for _ in range(10):
        i, j = sorted(random.sample(range(n), 2))
        seq[i], seq[j] = seq[j], seq[i]
        if is_valid(seq, G) and _targets_preserved(seq, target_nodes):
            return seq
        seq[i], seq[j] = seq[j], seq[i]
    return sequence

def relocate(sequence, G, target_nodes=None):
    from src.utils.metrics import score
    """
    Relocate operator: move one component to another position.
    
    Reference: Martins et al. (2024), "A hybrid GRASP algorithm for disassembly line balancing."
    
    Returns:
        Modified sequence or original if no valid move found.
    """
    seq = sequence[:]
    n = len(seq)
    for _ in range(10):
        i, j = random.sample(range(n), 2)
        node = seq.pop(i)
        seq.insert(j, node)
        if is_valid(seq, G) and _targets_preserved(seq, target_nodes):
            return seq
        seq.pop(j)
        seq.insert(i, node)
    return sequence

def two_opt(sequence, G, target_nodes=None):
    from src.utils.metrics import score
    """
    Two-opt operator: reverse a segment of the sequence.
    
    Reference: Kalaycilar et al. (2022), "A reinforcement learning approach for the DLBP."
    
    Returns:
        Modified sequence or original if no valid reversal found.
    """
    seq = sequence[:]
    n = len(seq)
    for _ in range(10):
        i, j = sorted(random.sample(range(n), 2))
        seq[i:j] = reversed(seq[i:j])
        if is_valid(seq, G) and _targets_preserved(seq, target_nodes):
            return seq
        seq[i:j] = reversed(seq[i:j])
    return sequence

def insert(sequence, G, return_all=False):
    from src.utils.metrics import score
    """
    Insert operator: insert an element at a new position.
    
    Reference: Santos et al. (2022), "Hybrid GRASP with Tabu Search for DSP."
    
    Args:
        return_all: If True, returns all valid neighbors.
    Returns:
        New sequence or list of neighbors if return_all=True.
    """
    if return_all:
        neighbors = []
        n = len(sequence)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                seq = sequence[:]
                element = seq.pop(i)
                seq.insert(j, element)
                if is_valid(seq, G):
                    neighbors.append(seq)
        return neighbors
    
    seq = sequence[:]
    n = len(seq)
    best_seq = seq[:]
    best_score = score(seq, G)
    for _ in range(min(15, n)):
        i = random.randint(0, n-1)
        element = seq.pop(i)
        j = random.randint(0, len(seq))
        seq.insert(j, element)
        if is_valid(seq, G):
            current_score = score(seq, G)
            if current_score < best_score:
                return seq
        seq = sequence[:]
    return best_seq

def insert_target_valid(sequence, G, target):
    from src.utils.metrics import score
    """Insert target at first valid position in sequence."""
    for i in range(len(sequence)+1):
        seq = sequence[:i] + [target] + sequence[i:]
        if is_valid(seq, G):
            return seq
    return sequence

def mns_local_search(solution, G, neighborhoods, target_nodes=None):
    from src.utils.metrics import score
    """
    Multi-Neighborhood Search (MNS): first-improvement local search.
    
    Reference: Santos et al. (2022), "Multi-neighborhood search for DSP."
    
    Applies multiple neighborhood operators in sequence. When any improvement
    is found, restarts from the beginning of the neighborhood list.
    """
    current = solution[:]
    improved = True
    while improved:
        improved = False
        best_score = score(current, G)
        for fn in neighborhoods:
            candidate = fn(current, G, target_nodes)
            s = score(candidate, G)
            if s < best_score:
                current = candidate
                improved = True
                break
    return current

def tabu_search(solution, G, neighborhood_fn, max_iter=20, tabu_size=5, max_neighbors=10, target_nodes=None):
    from src.utils.metrics import score
    """
    Tabu Search for DSP.
    
    Reference: Santos et al. (2022), "Hybrid GRASP with Tabu Search for DSP."
    Parameters tuned per Paschko et al. (2023).
    """
    current = solution[:]
    best = current[:]
    best_score = score(best, G)
    tabu_list = deque(maxlen=tabu_size)

    for iter_num in range(max_iter):
        neighbors = neighborhood_fn(current, G, return_all=True)
        random.shuffle(neighbors)
        neighbors = neighbors[:max_neighbors]

        candidates = [n for n in neighbors if n not in tabu_list]

        if not candidates:
            break

        next_sol = min(candidates, key=lambda s: score(s, G))

        tabu_list.append(next_sol)
        current = next_sol

        s = score(current, G)
        if s < best_score:
            best, best_score = current, s
    return best


def multi_neighborhood_tabu_search(solution, G, neighborhoods, max_iter=20, tabu_size=7, max_neighbors=10,target_nodes=None):
    from src.utils.metrics import score
    """
    Multi-Neighborhood Tabu Search for DSP.
    
    Reference: Santos et al. (2022), "Hybrid GRASP with Tabu Search for DSP."
    Alternates neighborhood operators at each iteration.
    """
    if not neighborhoods:
        return solution
    current = solution[:]
    best = current[:]
    best_score_value = score(best, G)
    tabu_list = []
    
    for iteration in range(max_iter):
        current_neighborhood = neighborhoods[iteration % len(neighborhoods)]
        neighbors = []
        for _ in range(max_neighbors):
            neighbor = current_neighborhood(current[:], G, target_nodes=target_nodes)
            neighbors.append(neighbor)
        best_neighbor = None
        best_neighbor_score = float('inf')
        
        for neighbor in neighbors:
            neighbor_hash = tuple(neighbor)
            neighbor_score = score(neighbor, G)
            # Aspiration criterion: accept tabu if better than global best
            if neighbor_score < best_score_value:
                best_neighbor = neighbor
                best_neighbor_score = neighbor_score
                break
            if neighbor_hash not in tabu_list and neighbor_score < best_neighbor_score:
                best_neighbor = neighbor
                best_neighbor_score = neighbor_score
        if best_neighbor is None:
            continue
            
        current = best_neighbor[:]
        current_score = best_neighbor_score
        if current_score < best_score_value:
            best = current[:]
            best_score_value = current_score
            
        tabu_list.append(tuple(current))
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
    return best


def swap_selectif(seq, G, target_nodes=None):
    from src.utils.metrics import score
    """Swap operator for selective mode. Returns improved sequence."""
    if not seq:
        return seq
    
    n = len(seq)
    best_seq = seq[:]
    best_score = score(seq, G)
    
    for _ in range(10):
        i, j = random.sample(range(n), 2)
        new_seq = seq[:]
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        
        if is_valid(new_seq, G):
            new_score = score(new_seq, G)
            if new_score < best_score:
                return new_seq
    
    return best_seq