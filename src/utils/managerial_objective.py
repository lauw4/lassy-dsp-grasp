"""DSP Objective — Managerial formulation (Order-Sensitive, NP-hard).

Aligned with Section 5 (Managerial Implications) of IWSPE 2026 manuscript.

FORMULATION:
    max  Σ(r_i - c_i - h·p_i)·x_i           [Term 1: Net profit after overhead]
       - α·Σ 1[r_i < θ]·r_i·x_i             [Term 2: Destruction risk]
       - ρ·Σ v_i·C_i(π)                     [Term 3: Protect value early (ORDER)]

    s.t. Σ(c_i + h·p_i)·x_i ≤ B_max         [Budget constraint in EUR]
         precedence constraints on π

PARAMETERS (all interpretable by managers):
    h     : overhead rate (EUR/s) — cost of cell occupation
    θ     : destruction threshold (EUR) = h·Δt + κ
    α     : risk factor [0,1] — probability low-value component blocks
    ρ     : value exposure rate (EUR/EUR·s) — "risk of waiting"
    B_max : total budget (EUR) — financial limit for the batch
    v_i   : value to protect = max(r_i - c_i, 0) or r_i

WHY NP-HARD:
    Term 3 (ρ·Σv_i·C_i) makes the objective ORDER-DEPENDENT.
    This is equivalent to 1|prec|Σw_jC_j (Lawler 1978, Lenstra & Rinnooy Kan 1978).

MANAGERIAL INTERPRETATION:
    "Each euro of recoverable value, each second it waits, is exposed to 
    downstream failures. Recover high-value components early to protect them."
"""
from __future__ import annotations


def score_profit_managerial(sequence, G):
    """Managerial profit (to MAXIMIZE) — ORDER-SENSITIVE.
    
    Parameters read from G.graph:
        - overhead_rate (h)           : EUR/s, default 0.05 (~180 EUR/h)
        - destruction_threshold (θ)   : EUR, default 10
        - risk_factor (α)             : [0,1], default 0.2
        - value_exposure_rate (ρ)     : EUR/(EUR·s), default 0.0001
        - budget_max (B_max)          : EUR, default None (no limit)
        - budget_penalty (β)          : penalty multiplier, default 10.0
        - value_mode                  : 'margin' or 'revenue', default 'margin'
    
    Returns:
        profit (float) : Z value (higher is better)
    """
    from src.grasp.local_search import is_valid
    
    # ── Economic parameters ──
    h = G.graph.get('overhead_rate', 0.05)
    theta = G.graph.get('destruction_threshold', 10.0)
    alpha = G.graph.get('risk_factor', 0.2)
    rho = G.graph.get('value_exposure_rate', 0.0001)  # ORDER TERM
    B_max = G.graph.get('budget_max', None)
    beta = G.graph.get('budget_penalty', 10.0)
    value_mode = G.graph.get('value_mode', 'margin')
    
    order_penalty = -1000.0  # Hard constraint violation
    
    # ── Precedence check ──
    if not is_valid(sequence, G):
        return order_penalty * len(sequence)
    
    n = len(sequence)
    if n == 0:
        return 0.0
    
    profit = 0.0
    total_budget_used = 0.0
    cumul_time = 0.0  # C_i(π) = completion time
    
    for pos, node in enumerate(sequence):
        r_i = G.nodes[node].get('profit', 0.0)
        c_i = G.nodes[node].get('cost', 0.0)
        p_i = G.nodes[node].get('time', G.nodes[node].get('duration', 1.0))
        
        # ── Term 1: Net profit after overhead ──
        # (r_i - c_i - h·p_i)
        net_i = r_i - c_i - h * p_i
        profit += net_i
        
        # ── Term 2: Destruction risk ──
        # α·1[r_i < θ]·r_i
        if r_i < theta:
            profit -= alpha * r_i
        
        # ── Term 3: Protect value early (ORDER-DEPENDENT) ──
        # ρ·v_i·C_i  where C_i = cumulative time up to and including i
        cumul_time += p_i
        if value_mode == 'margin':
            v_i = max(r_i - c_i, 0.0)  # Positive margin only
        else:
            v_i = r_i  # Raw revenue
        
        # Penalty: value × completion time × exposure rate
        # Interpretation: "this value waited C_i seconds, exposed to risk"
        profit -= rho * v_i * cumul_time
        
        # Budget tracking
        total_budget_used += (c_i + h * p_i)
    
    # ── Budget constraint penalty ──
    if B_max is not None and total_budget_used > B_max:
        profit -= beta * (total_budget_used - B_max)
    
    return profit


def compute_managerial_metrics(sequence, G):
    """Detailed P&L breakdown for managers.
    
    Returns dict with line-by-line profit decomposition.
    """
    from src.grasp.local_search import is_valid
    
    h = G.graph.get('overhead_rate', 0.05)
    theta = G.graph.get('destruction_threshold', 10.0)
    alpha = G.graph.get('risk_factor', 0.2)
    rho = G.graph.get('value_exposure_rate', 0.0001)
    B_max = G.graph.get('budget_max', None)
    value_mode = G.graph.get('value_mode', 'margin')
    
    gross_revenue = 0.0
    direct_cost = 0.0
    overhead_cost = 0.0
    destruction_risk = 0.0
    value_exposure_cost = 0.0
    cumul_time = 0.0
    total_time = 0.0
    
    for node in sequence:
        r_i = G.nodes[node].get('profit', 0.0)
        c_i = G.nodes[node].get('cost', 0.0)
        p_i = G.nodes[node].get('time', G.nodes[node].get('duration', 1.0))
        
        gross_revenue += r_i
        direct_cost += c_i
        overhead_cost += h * p_i
        total_time += p_i
        cumul_time += p_i
        
        if r_i < theta:
            destruction_risk += alpha * r_i
        
        v_i = max(r_i - c_i, 0.0) if value_mode == 'margin' else r_i
        value_exposure_cost += rho * v_i * cumul_time
    
    net_operating = gross_revenue - direct_cost - overhead_cost
    budget_used = direct_cost + overhead_cost
    budget_overrun = max(0, budget_used - B_max) if B_max else 0.0
    
    final_profit = net_operating - destruction_risk - value_exposure_cost
    if B_max and budget_used > B_max:
        final_profit -= G.graph.get('budget_penalty', 10.0) * budget_overrun
    
    return {
        # Revenue & costs
        'gross_revenue': gross_revenue,
        'direct_cost': direct_cost,
        'overhead_cost': overhead_cost,
        'net_operating_profit': net_operating,
        # Risk terms
        'destruction_risk': destruction_risk,
        'value_exposure_cost': value_exposure_cost,
        # Budget
        'total_time_s': total_time,
        'budget_used_eur': budget_used,
        'budget_max_eur': B_max,
        'budget_overrun': budget_overrun,
        # Final
        'final_profit': final_profit,
        # Ratios
        'margin_pct': (net_operating / gross_revenue * 100) if gross_revenue > 0 else 0,
        'risk_pct': ((destruction_risk + value_exposure_cost) / gross_revenue * 100) if gross_revenue > 0 else 0,
    }


def calibrate_rho(G, target_impact_pct=5.0):
    """Calibrate rho so that Term 3 represents ~target_impact_pct of Term 1.
    
    Rule of thumb: rho = (target% × avg_margin) / (avg_time × n² / 2)
    
    Returns recommended rho value.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return 0.0001
    
    total_margin = 0.0
    total_time = 0.0
    for node in nodes:
        r_i = G.nodes[node].get('profit', 0.0)
        c_i = G.nodes[node].get('cost', 0.0)
        p_i = G.nodes[node].get('time', G.nodes[node].get('duration', 1.0))
        total_margin += max(r_i - c_i, 0)
        total_time += p_i
    
    avg_margin = total_margin / n
    avg_time = total_time / n
    
    # For a random sequence, E[C_i] ≈ avg_time × (n+1)/2
    # E[Σv_i·C_i] ≈ n × avg_margin × avg_time × (n+1)/2
    expected_sum_vC = n * avg_margin * avg_time * (n + 1) / 2
    
    # We want: rho × expected_sum_vC = target% × total_margin
    if expected_sum_vC > 0:
        rho = (target_impact_pct / 100) * total_margin / expected_sum_vC
    else:
        rho = 0.0001
    
    return rho


__all__ = ['score_profit_managerial', 'compute_managerial_metrics', 'calibrate_rho']
