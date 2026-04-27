"""Compare academic vs. managerial objective formulations (ORDER-SENSITIVE)."""
import random
random.seed(42)

from src.utils.graph_io import load_adaptive_graph
from src.utils.metrics import score_profit_net
from src.utils.managerial_objective import (
    score_profit_managerial, 
    compute_managerial_metrics,
    calibrate_rho
)
from src.grasp.constructive import closure_with_predecessors, run_grasp
import networkx as nx

# Load instance
G, _, _ = load_adaptive_graph('data/instances_base/gearpump.json')

# Calibrate rho for ~5% impact of Term 3
recommended_rho = calibrate_rho(G, target_impact_pct=5.0)

# Set managerial parameters
G.graph['overhead_rate'] = 0.05           # 50 cents/second = 180 EUR/h
G.graph['destruction_threshold'] = 8.0    # Components < 8 EUR are expendable
G.graph['risk_factor'] = 0.25             # 25% chance low-value blocks
G.graph['value_exposure_rate'] = recommended_rho  # Calibrated rho
G.graph['budget_max'] = 120.0             # 120 EUR budget for this batch
G.graph['budget_penalty'] = 5.0
G.graph['value_mode'] = 'margin'          # Protect positive margins

print("="*70)
print("MANAGERIAL OBJECTIVE with ORDER-DEPENDENT TERM")
print("="*70)

# Generate test sequences
target = ['G20']
needed = closure_with_predecessors(G, target)
subG = G.subgraph(needed)

topos = []
for t in nx.all_topological_sorts(subG):
    topos.append(list(t))
    if len(topos) >= 5:
        break

print(f"\nInstance: Gearpump, Target: G20, Closure: {len(needed)} nodes\n")

# Economic parameters summary
print("Economic parameters:")
print(f"  h (overhead rate)      : {G.graph['overhead_rate']:.3f} EUR/s  (~{G.graph['overhead_rate']*3600:.0f} EUR/h)")
print(f"  theta (destruction)    : {G.graph['destruction_threshold']:.1f} EUR")
print(f"  alpha (risk factor)    : {G.graph['risk_factor']:.1%}")
print(f"  rho (value exposure)   : {G.graph['value_exposure_rate']:.6f} EUR/(EUR*s)")
print(f"  B_max (budget)         : {G.graph['budget_max']:.1f} EUR")
print(f"  value_mode             : {G.graph['value_mode']}")
print()

# Compare objectives on topological sorts
print("-"*70)
print("ORDER-SENSITIVITY TEST: Same nodes, different orders")
print("-"*70)

academic_scores = []
managerial_scores = []

for i, seq in enumerate(topos):
    academic = score_profit_net(seq, G)
    managerial = score_profit_managerial(seq, G)
    academic_scores.append(academic)
    managerial_scores.append(managerial)
    
    print(f"\nSequence {i+1}:")
    print(f"  Academic Z   : {academic:>10.4f}")
    print(f"  Managerial Z : {managerial:>10.4f}")

# Check order-sensitivity
academic_all_same = max(academic_scores) - min(academic_scores) < 0.01
managerial_all_same = max(managerial_scores) - min(managerial_scores) < 0.01

print("\n" + "="*70)
print("ORDER-SENSITIVITY VERDICT")
print("="*70)
print(f"  Academic scores range   : {min(academic_scores):.4f} to {max(academic_scores):.4f}")
print(f"  Managerial scores range : {min(managerial_scores):.4f} to {max(managerial_scores):.4f}")
print()
print(f"  Academic ORDER-dependent?   : {'YES - NP-hard' if not academic_all_same else 'NO - polynomial'}")
print(f"  Managerial ORDER-dependent? : {'YES - NP-hard' if not managerial_all_same else 'NO - polynomial'}")

# Detailed breakdown for best sequence
best_idx = managerial_scores.index(max(managerial_scores))
print("\n" + "="*70)
print(f"DETAILED P&L BREAKDOWN (Best sequence #{best_idx+1})")
print("="*70)

metrics = compute_managerial_metrics(topos[best_idx], G)

print(f"\n  REVENUE")
print(f"    Gross revenue              : +{metrics['gross_revenue']:>8.2f} EUR")
print(f"\n  COSTS")
print(f"    Direct operating costs     : -{metrics['direct_cost']:>8.2f} EUR")
print(f"    Overhead (h*T)             : -{metrics['overhead_cost']:>8.2f} EUR  (T={metrics['total_time_s']:.0f}s)")
print(f"    ───────────────────────────────────────")
print(f"    Net Operating Profit       :  {metrics['net_operating_profit']:>8.2f} EUR")
print(f"\n  RISK ADJUSTMENTS")
print(f"    Destruction risk (alpha)   : -{metrics['destruction_risk']:>8.2f} EUR")
print(f"    Value exposure (rho*v*C)   : -{metrics['value_exposure_cost']:>8.2f} EUR  <-- ORDER TERM")
print(f"\n  BUDGET")
print(f"    Budget used (c+h*p)        :  {metrics['budget_used_eur']:>8.2f} EUR / {metrics['budget_max_eur']:.1f} EUR")
print(f"    Budget overrun penalty     : -{metrics['budget_overrun']:>8.2f} EUR")
print(f"\n  ═══════════════════════════════════════════")
print(f"    FINAL PROFIT (Z)           :  {metrics['final_profit']:>8.2f} EUR")
print(f"  ═══════════════════════════════════════════")
print(f"\n    Margin %                   :  {metrics['margin_pct']:>7.1f}%")
print(f"    Risk impact %              :  {metrics['risk_pct']:>7.1f}%")

# Show the formulation
print("\n" + "="*70)
print("MATHEMATICAL FORMULATION")
print("="*70)
print("""
max  Z = Σ(r_i - c_i - h·p_i)·x_i           [Net profit after overhead]
       - α·Σ 1[r_i < θ]·r_i·x_i             [Destruction risk]
       - ρ·Σ v_i·C_i(π)                     [Value exposure = ORDER TERM]

s.t. Σ(c_i + h·p_i)·x_i ≤ B_max             [Budget constraint]
     precedence constraints

WHY NP-HARD:
  Term 3 ρ·Σv_i·C_i makes objective ORDER-DEPENDENT.
  ≡ 1|prec|Σw_jC_j (Lawler 1978) = NP-hard.

MANAGERIAL INTERPRETATION:
  "Each euro of margin waiting in the sequence is exposed to downstream
   failures. Recover high-value components EARLY to protect them."
""")
