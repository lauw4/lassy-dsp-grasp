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

__all__ = ['PARAMS', 'classify', 'ALPHA_POOL', 'SizeParams']
