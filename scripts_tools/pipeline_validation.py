"""Pipeline validation tracker for DSP comparison experiments.

Tracks key steps (load instance, Dijkstra, GRASP, MILP, aggregate) to ensure
the experimental protocol follows the defined sequence and is reproducible.

Usage:
    from scripts_tools.pipeline_validation import PipelineTracker
    
    tracker = PipelineTracker(run_id, out_dir)
    tracker.log_step('load_instance', instance=name, status='ok')
    tracker.log_step('run_dijkstra', sequence_len=len(seq), status='ok')
    tracker.finalize_report()
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import os
import hashlib

_ORDER = [
	'load_instance',
	'run_dijkstra',
	'run_grasp',
	'run_milp',
	'aggregate'
]

@dataclass
class StepRecord:
	name: str
	timestamp: str
	status: str
	details: Dict[str, Any]

class PipelineTracker:
	def __init__(self, run_id: str, output_dir: str):
		self.run_id = run_id
		self.output_dir = output_dir
		self.steps: List[StepRecord] = []
		self.meta: Dict[str, Any] = {
			'run_id': run_id,
			'start_time': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
			'expected_order': _ORDER,
		}
		os.makedirs(output_dir, exist_ok=True)

	def log_step(self, name: str, status: str = 'ok', **details: Any) -> None:
		rec = StepRecord(
			name=name,
			timestamp=datetime.utcnow().isoformat(timespec='seconds') + 'Z',
			status=status,
			details=details
		)
		self.steps.append(rec)

	def _compute_order_warnings(self) -> List[str]:
		warnings: List[str] = []
		# simple order check: each expected step appears at most once and in order
		last_index = -1
		for s in self.steps:
			if s.name in _ORDER:
				idx = _ORDER.index(s.name)
				if idx < last_index:
					warnings.append(f"Step '{s.name}' out of order (index {idx} < {last_index})")
				last_index = max(last_index, idx)
		return warnings

	def finalize_report(self) -> str:
		warnings = self._compute_order_warnings()
		report = {
			'meta': self.meta,
			'steps': [asdict(s) for s in self.steps],
			'order_warnings': warnings,
			'hash': None,
		}
		# hash for integrity
		digest_src = json.dumps(report['steps'], sort_keys=True).encode('utf-8')
		report['hash'] = hashlib.sha256(digest_src).hexdigest()[:16]

		out_path = os.path.join(self.output_dir, f"validation_{self.run_id}.json")
		with open(out_path, 'w', encoding='utf-8') as f:
			json.dump(report, f, indent=2, ensure_ascii=False)
		return out_path

__all__ = ["PipelineTracker"]
