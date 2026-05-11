# LAsSy-DSP

**Reactive Disassembly Sequence Planning under Uncertain Component Conditions**

A two-block framework for end-of-life disassembly:

1. **Offline planner** — GRASP + Variable Neighborhood Descent (VND) builds a precedence-feasible sequence that maximises net recovery profit on directed acyclic product graphs, with both complete and selective disassembly modes. CPLEX (MILP) and CP-SAT (OR-Tools) serve as exact baselines.
2. **Online adaptive layer** — a Mamdani fuzzy decision system monitors process signals (force, torque, p_fail, context) and triggers one of four local actions (bypass, tool change, controlled destruction, local replanning) when execution deviates from the plan.

## Installation

```bash
pip install -r requirements.txt
```

Optional: IBM CPLEX (for the MILP baseline) and Google OR-Tools (`ortools`, for CP-SAT).

## Usage

Single instance with optional failure scenario:

```bash
python run_pipeline.py --instance gearpump
python run_pipeline.py --instance dagtest --failures demo_mixed --verbose
```

Full GRASP vs CPLEX vs CP-SAT benchmark on all instances:

```bash
PYTHONPATH=. python experiments/run_benchmark_scholl_salbp.py
```

Fuzzy adaptation benchmark (87 failure scenarios across simple / intermediate / complex difficulty). The launcher reads `data/adaptive/scenarios_list.csv` and writes per-scenario logs plus the aggregated `RESUME_FUZZY_DETAILED.csv`:

```bash
python fuzzy_test_launcher.py
```

Regenerate the per-scenario CSV from existing logs (without re-running the benchmark):

```bash
python fuzzy_test_launcher.py --resume
```

## Project layout

```
src/
├── grasp/                 GRASP constructor and VND local search
├── adaptive/              Fuzzy Mamdani decision system, scheduler, actions
├── exact_milp_dijkstra/   MILP formulation (CPLEX, time-aware objective)
├── exact_cpsat/           CP-SAT formulation (OR-Tools, IntervalVar + NoOverlap)
├── core/                  Shared data structures
└── utils/                 Graph I/O, metrics, scoring

experiments/
├── run_benchmark_scholl_salbp.py    3-way benchmark (GRASP / CPLEX / CP-SAT)
├── convert_salbp_to_dsp.py          SALBP → DSP instance conversion
├── convert_scholl_to_dsp.py         Scholl SALBP → DSP instance conversion
├── generate_stress_scenarios.py     Failure scenario generation (3 difficulty levels)
├── generate_stress_for_originals.py Failure scenarios for legacy instances
└── json_to_txt.py                   JSON → MILP .txt format conversion

tests/                               pytest suite for the fuzzy layer

data/
├── instances_milp/                  DSP instances in MILP .txt format
├── instances_base/                  DSP instances in JSON format
└── adaptive/
    ├── scenarios_list.csv           List of the 87 failure scenarios to run
    └── generated_failures/          Per-instance failure JSON files referenced by the list

results/benchmark_2026/
├── grasp_vs_milp/
│   ├── RESUME_GRASP_MILP.csv                  Aggregated 3-way benchmark results
│   ├── benchmark_grasp_vs_exact_results.xlsx  Multi-sheet Excel of the same data
│   └── <instance>/                            Per-instance details (results.csv, terminal.log, validation.json)
└── fuzzy/
    ├── RESUME_FUZZY_DETAILED.csv                  Per-scenario fuzzy metrics (18 columns)
    ├── benchmark_fuzzy_detailed_results.xlsx      Multi-sheet Excel of the same data
    └── {simple,intermediate,complex}/scenario_<k>/  Per-scenario logs and reports
```

## Adaptive actions

| Action | Description |
|---|---|
| `bypass` | Skip a non-critical component via an alternative path |
| `change_tool` | Switch to an alternative tool and retry |
| `destroy` | Controlled destructive removal when economic conditions allow |
| `replan` | Local replanning on the residual graph |

## Failure scenarios

| Difficulty | Failure location | What is tested |
|---|---|---|
| Simple | Non-critical node, alternative path available | Basic bypass |
| Intermediate | Moderately important node | Adaptation under constraints |
| Complex | Critical node on every path to a target | Maximum adaptation: fallback, local replan, controlled destruction |

## Citing

If you use this code, please cite:

- W. El Morabit, M.-A. Abdous, F. Lucas, G. Bluvstein, F.-A. Brunnenkant, L. Streibel. *Reactive Disassembly Sequence Planning under Uncertain Component Conditions.* IFAC World Congress, 2026.

## Acknowledgements

LAsSy project (*Learning Assistance Systems for Efficient Disassembly*), funded by the German-French Academy for the Industry of the Future.