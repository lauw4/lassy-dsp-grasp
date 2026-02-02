"""Top-level package exports for lassy-dsp-grasp.

Convenience re-exports so users can:

	from src import run_grasp, closure_with_predecessors

Versioning kept simple (manual bump).
"""

__all__ = [
	'run_grasp', 'closure_with_predecessors', 'VERSION'
]

from .grasp.constructive import run_grasp, closure_with_predecessors

VERSION = '0.1.0'
