"""Centralized GRASP parameters.

Size-based adaptation (number of nodes):
  - small (<=50)
  - medium (51-100)
  - large (101-200)
  - xlarge (>200)
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SizeParams:
    max_iterations: int
    tabu_iter: int
    mns_iterations: int
    vnd_rounds: int


def classify(n: int) -> str:
    if n <= 50: return 'small'
    if n <= 100: return 'medium'
    if n <= 200: return 'large'
    return 'xlarge'


PARAMS = {
    'small':  SizeParams(150, 20, 100, 2),
    'medium': SizeParams(120, 15,  80, 3),
    'large':  SizeParams(100, 12,  70, 3),
    'xlarge': SizeParams( 60,  10, 50, 3),
}

ALPHA_POOL = (0.1, 0.3, 0.5, 0.7, 0.9)

# ── Managerial parameters (Section 5, IWSPE 2026) ─────────────────────
# max Σ(r_i - c_i - h·p_i)·x_i  - α·Σ1[r_i<θ]·r_i·x_i  - ρ·Σv_i·C_i(π)
# s.t. Σ(c_i + h·p_i)·x_i ≤ B_max,  precedence on π

OVERHEAD_RATE: float = 0.05          # h  : EUR/s  (~180 EUR/h cell occupation)
DESTRUCTION_THRESHOLD: float = 8.0   # θ  : EUR    component at-risk if r_i < θ
RISK_FACTOR: float = 0.20            # α  : [0,1]  probability low-value blocks line
VALUE_EXPOSURE_RATE: float = 0.0001  # ρ  : EUR/(EUR·s)  "risk of waiting"
BUDGET_RATIO: float = 0.90           # B_max = ratio × Σ(c_i + h·p_i)

# ── Objective 1 — Discounted Recovery Profit (IWSPE 2026) ─────────────────
# max Σ(v_i - λ·C_i)   where v_i = r_i - c_i,  C_i = completion time
# λ = h = 0.05 EUR/s: "each second a part waits costs the same as running
#                       the disassembly cell" (capital + energy per second).
# NP-hard via 1|prec|ΣC_j (Lenstra & Rinnooy Kan 1978).
LAMBDA_DISCOUNT: float = 0.05        # λ  : EUR/s  discount on completion time

__all__ = [
    'PARAMS', 'classify', 'ALPHA_POOL', 'SizeParams',
    'OVERHEAD_RATE', 'DESTRUCTION_THRESHOLD', 'RISK_FACTOR',
    'VALUE_EXPOSURE_RATE', 'BUDGET_RATIO', 'LAMBDA_DISCOUNT',
]
