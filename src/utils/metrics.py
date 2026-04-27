"""Evaluation metrics for DSP (Disassembly Sequence Planning).

Conventions:
    - score / selective_score / adaptive_score: minimize
    - profit (MILP): maximize
    - makespan: minimize
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Dict, Any, List, Tuple, Optional
from src.grasp.local_search import is_valid


def score(sequence, G):
    from src.grasp.local_search import is_valid
    """
    Compute disassembly sequence score (lower is better).
    
    Reference: Frizziero et al. (2020), "DSP applied to a Gear Box."
    
    score = Σ(time_i × position_i)
    """
    if not is_valid(sequence, G):
        return float('inf')
    total = 0.0
    for i, node in enumerate(sequence):
        t = G.nodes[node].get("time", 1.0)
        total += t * (i + 1)
    return total


def score_profit_net(sequence, G):
    """Managerial net profit for DSP (to MAXIMIZE) — ORDER-SENSITIVE.

    Formulation (Section 5, IWSPE 2026):
        Z = Σ(r_i − c_i − h·p_i)         [Term 1: Operating margin]
          − α·Σ 1[r_i < θ]·r_i           [Term 2: Destruction risk]
          − ρ·Σ v_i·C_i(π)               [Term 3: Value exposure (ORDER)]

    Budget feasibility:  Σ(c_i + h·p_i) ≤ B_max   (penalty if violated)
    Precedence:  hard constraint checked via is_valid()

    Term 3 makes the objective sequence-dependent → NP-hard
    (equivalent to 1|prec|Σw_jC_j, Lawler 1978).
    """
    from src.grasp.config import (
        OVERHEAD_RATE, DESTRUCTION_THRESHOLD, RISK_FACTOR,
        VALUE_EXPOSURE_RATE,
    )

    h     = G.graph.get('overhead_rate', OVERHEAD_RATE)
    theta = G.graph.get('destruction_threshold', DESTRUCTION_THRESHOLD)
    alpha = G.graph.get('risk_factor', RISK_FACTOR)
    rho   = G.graph.get('value_exposure_rate', VALUE_EXPOSURE_RATE)
    B_max = G.graph.get('budget_max', None)

    prec_penalty   = -1000.0     # hard constraint violation
    budget_penalty = -10.0       # per EUR over B_max

    # ── Precedence check ──
    if not is_valid(sequence, G):
        return prec_penalty * len(sequence)

    if not sequence:
        return 0.0

    profit = 0.0
    cumul_time = 0.0
    budget_used = 0.0

    for node in sequence:
        r_i = G.nodes[node].get('profit', 0.0)
        c_i = G.nodes[node].get('cost', 0.0)
        p_i = G.nodes[node].get('time', G.nodes[node].get('duration', 1.0))

        # Term 1 — operating margin
        profit += (r_i - c_i - h * p_i)

        # Term 2 — destruction risk
        if r_i < theta:
            profit -= alpha * r_i

        # Term 3 — value exposure (order-dependent)
        cumul_time += p_i
        v_i = max(r_i - c_i, 0.0)
        profit -= rho * v_i * cumul_time

        budget_used += (c_i + h * p_i)

    # ── Budget penalty ──
    if B_max is not None and budget_used > B_max:
        profit += budget_penalty * (budget_used - B_max)

    return profit


def score_obj1(sequence, G, lam: float = None) -> float:
    """Objective 1 — Discounted Recovery Profit (to MAXIMIZE, order-sensitive).

    Formulation (IWSPE 2026, aligned with 1|prec|ΣC_j):
        Z = Σ_{i ∈ S} (v_i - λ·C_i)
          where v_i = r_i - c_i  (net recovery margin)
                C_i = Σ_{j: σ(j)≤σ(i)} p_j  (completion time at position i)
                λ   = lambda_discount (EUR/s, default 0.05)

    NP-hard: equivalent to 1|prec|ΣC_j (Lenstra & Rinnooy Kan 1978).
    Sequence-sensitive: later positions incur larger λ·C_i penalty.
    """
    from src.grasp.config import LAMBDA_DISCOUNT
    if not is_valid(sequence, G):
        return -1000.0 * len(sequence)
    if not sequence:
        return 0.0
    if lam is None:
        if 'lambda_discount' in G.graph:
            lam = G.graph['lambda_discount']
        else:
            # Auto-calibrate: λ = 5 % of mean net-value rate (scales with instance)
            from src.utils.graph_io import auto_lambda
            lam = auto_lambda(G)
            G.graph['lambda_discount'] = lam   # cache
    total = 0.0
    cumul_time = 0.0
    for node in sequence:
        r_i = G.nodes[node].get('profit', 0.0)
        c_i = G.nodes[node].get('cost', 0.0)
        p_i = G.nodes[node].get('time', G.nodes[node].get('duration', 1.0))
        cumul_time += p_i          # C_i = completion time (includes p_i itself)
        v_i = r_i - c_i
        total += v_i - lam * cumul_time
    return total


def adaptive_score(sequence, G):
    """Score with optional tau attribute."""
    total = 0.0
    for i, node in enumerate(sequence):
        t = G.nodes[node].get("time", 1.0)
        tau = G.nodes[node].get("tau", 1.0)
        total += t * tau * (i + 1)
    return total


def selective_score(sequence, G, target_nodes):
    """Score of partial sequence up to last target (for selective DSP)."""
    try:
        positions = []
        for t in target_nodes:
            if t not in sequence:
                return float('inf')
            positions.append(sequence.index(t))
        last_target_idx = max(positions)
        partial_seq = sequence[:last_target_idx + 1]
        G.graph["target_nodes"] = target_nodes
        return -score_profit_net(partial_seq, G)
    except Exception:
        return float('inf')


# --- Normalization ---------------------------------------------------------

@dataclass
class MetricSpec:
    """Metric specification for normalization to [0,1] range."""
    name: str
    direction: str  # 'min' or 'max'
    best_ref: float
    worst_ref: float

    def normalize(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if self.best_ref is None or self.worst_ref is None:
            return None
        if self.worst_ref == self.best_ref:
            return 0.0
        if self.direction == 'min':
            return (value - self.best_ref) / (self.worst_ref - self.best_ref)
        else:
            return (self.best_ref - value) / (self.best_ref - self.worst_ref)


def build_specs_from_rows(rows: List[Dict[str, Any]], fields: Dict[str, str]) -> Dict[str, MetricSpec]:
    """Build MetricSpec from result rows."""
    specs: Dict[str, MetricSpec] = {}
    for metric, direction in fields.items():
        values = [r.get(metric) for r in rows if r.get(metric) is not None]
        numeric = [v for v in values if isinstance(v, (int, float)) and v not in (float('inf'), float('-inf'))]
        if not numeric:
            continue
        if direction == 'min':
            best = min(numeric)
            worst = max(numeric)
        else:
            best = max(numeric)
            worst = min(numeric)
        specs[metric] = MetricSpec(metric, direction, best, worst)
    return specs


def apply_normalization(rows: List[Dict[str, Any]], specs: Dict[str, MetricSpec], suffix: str = '_norm') -> None:
    """Add normalized values in-place (key: metric+suffix)."""
    for r in rows:
        for name, spec in specs.items():
            val = r.get(name)
            r[name + suffix] = spec.normalize(val)


def compute_economic_metrics(
    G,
    nominal_sequence: List[str],
    done: set,
    destroyed: set,
    skipped: set = None
) -> Dict[str, Any]:
    """
    Compute economic metrics: nominal vs adaptive profit.
    
    Returns dict with: nominal_profit, adaptive_profit, profit_loss, 
    profit_loss_pct, destroyed_value, skipped_value, details.
    """
    if skipped is None:
        skipped = set()
    
    nominal_profit = 0.0
    nominal_cost = 0.0
    details = {}
    
    for node in nominal_sequence:
        profit = G.nodes[node].get('profit', 0.0)
        cost = G.nodes[node].get('cost', 0.0)
        nominal_profit += profit - cost
        details[node] = {'profit': profit, 'cost': cost, 'net': profit - cost}
    
    adaptive_profit = 0.0
    destroyed_value = 0.0
    skipped_value = 0.0
    
    for node in nominal_sequence:
        profit = G.nodes[node].get('profit', 0.0)
        cost = G.nodes[node].get('cost', 0.0)
        
        if node in done and node not in destroyed:
            adaptive_profit += profit - cost
            details[node]['status'] = 'recovered'
        elif node in destroyed:
            adaptive_profit -= cost
            destroyed_value += profit
            details[node]['status'] = 'destroyed'
        elif node in skipped:
            skipped_value += profit
            details[node]['status'] = 'skipped'
        else:
            details[node]['status'] = 'blocked'
    
    profit_loss = nominal_profit - adaptive_profit
    profit_loss_pct = (profit_loss / nominal_profit * 100) if nominal_profit > 0 else 0.0
    
    return {
        'nominal_profit': nominal_profit,
        'adaptive_profit': adaptive_profit,
        'profit_loss': profit_loss,
        'profit_loss_pct': profit_loss_pct,
        'destroyed_value': destroyed_value,
        'skipped_value': skipped_value,
        'details': details
    }


def compute_time_metrics(
    G,
    nominal_sequence: List[str],
    done: set,
    destroyed: set,
    skipped: set = None
) -> Dict[str, Any]:
    """
    Compute time metrics: nominal vs adaptive time.
    
    Returns dict with: nominal_time, adaptive_time, time_diff, 
    time_diff_pct, destroyed_time, skipped_time, details.
    """
    if skipped is None:
        skipped = set()
    
    details = {}
    
    nominal_time = 0.0
    for node in nominal_sequence:
        duration = G.nodes[node].get('time', G.nodes[node].get('duration', 1.0))
        nominal_time += duration
        details[node] = {'duration': duration}
    
    adaptive_time = 0.0
    destroyed_time = 0.0
    skipped_time = 0.0
    
    for node in nominal_sequence:
        duration = G.nodes[node].get('time', G.nodes[node].get('duration', 1.0))
        
        if node in done and node not in destroyed:
            adaptive_time += duration
            details[node]['status'] = 'done'
        elif node in destroyed:
            adaptive_time += duration
            destroyed_time += duration
            details[node]['status'] = 'destroyed'
        elif node in skipped:
            skipped_time += duration
            details[node]['status'] = 'skipped'
        else:
            details[node]['status'] = 'not_processed'
    
    time_diff = nominal_time - adaptive_time
    time_diff_pct = (time_diff / nominal_time * 100) if nominal_time > 0 else 0.0
    
    return {
        'nominal_time': nominal_time,
        'adaptive_time': adaptive_time,
        'time_diff': time_diff,
        'time_diff_pct': time_diff_pct,
        'destroyed_time': destroyed_time,
        'skipped_time': skipped_time,
        'details': details
    }


def score_weighted(
    sequence: List[str],
    G,
    w1: float = 1/3,
    w2: float = 1/3,
    w3: float = 1/3
) -> float:
    """
    Weighted global score: Score = w₁∑Pᵢ - w₂∑Cᵢ - w₃∑Tᵢ
    
    References: Frizziero et al. (2020), Ye et al. (2022)
    
    Returns: Higher score = better.
    """
    if not sequence:
        return 0.0
    
    total_profit = 0.0
    total_cost = 0.0
    total_time = 0.0
    
    for node in sequence:
        total_profit += G.nodes[node].get('profit', 0.0)
        total_cost += G.nodes[node].get('cost', 0.0)
        total_time += G.nodes[node].get('time', G.nodes[node].get('duration', 1.0))
    
    time_normalized = total_time / 100.0
    
    score = w1 * total_profit - w2 * total_cost - w3 * time_normalized
    return score


def prune_sequence(sequence, G, protected_nodes=None):
    """Remove unprofitable nodes from sequence while preserving precedence.

    A node n is removed if its absence improves score_obj1 AND no node
    placed after it in the sequence has n as a predecessor.

    Args:
        sequence:        list of node ids
        G:               NetworkX DiGraph
        protected_nodes: set of nodes that must stay (e.g. targets)

    Returns:
        Shorter (or equal) sequence with improved or equal score_obj1.
    """
    if protected_nodes is None:
        protected_nodes = set()
    current = list(sequence)
    improved = True
    while improved:
        improved = False
        current_score = score_obj1(current, G)
        for i in range(len(current) - 1, -1, -1):
            n = current[i]
            if n in protected_nodes:
                continue
            # Don't remove if a successor is still in the sequence after it
            has_req_successor = any(
                v in current[i + 1:] for v in G.successors(n)
            )
            if has_req_successor:
                continue
            candidate = current[:i] + current[i + 1:]
            if score_obj1(candidate, G) >= current_score:
                current = candidate
                improved = True
                current_score = score_obj1(current, G)
                break   # restart
    return current


__all__ = [
    'score', 'score_obj1', 'prune_sequence',
    'adaptive_score', 'selective_score', 'score_profit_net',
    'compute_economic_metrics', 'compute_time_metrics', 'score_weighted',
    'MetricSpec', 'build_specs_from_rows', 'apply_normalization'
]