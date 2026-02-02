# LAsSy-DSP: Adaptive Disassembly Sequence Planning

GRASP-based metaheuristics and adaptive fuzzy decision system for the Disassembly Sequence Planning (DSP) problem.

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python 3.8+, NetworkX, NumPy, scikit-fuzzy, PuLP, ...  
Optional: IBM CPLEX (for MILP baseline)

## Usage

### GRASP + Adaptive Pipeline 

```bash
python run_pipeline.py --instance dagtest
python run_pipeline.py --instance dagtest --failures demo_mixed --verbose
```

### Benchmark GRASP vs Milp

```bash
python run_all.py
python run_all.py --mode selectif --n_runs 10
python experiments/compare_grasp_milp_dijkstra.py --instance scholl_lutz1_n=32.json
```

### API

```python
from src.utils.graph_io import load_adaptive_graph
from src.grasp.constructive import run_grasp

G, _, _ = load_adaptive_graph("data/instances_base/scholl_lutz1_n=32.json")
sequence, score = run_grasp(G, algorithm='vnd', mode='selectif', target_nodes=['C001'])
```

## Project Structure

```
src/
├── grasp/                   # GRASP metaheuristics (constructive.py, local_search.py)
├── adaptive/                # Fuzzy decision system: fuzzy_decision.py (Mamdani), scheduler.py, actions.py
├── exact_milp_dijkstra/     # Baselines: partial_disassembly_model.py (MILP), dijkstra.py
└── utils/                   # graph_io.py (load JSON), metrics.py (scoring functions)

data/
├── instances_base/          # DSP instances in JSON format (nodes, edges, targets)
├── instances_milp/          # Converted instances for MILP solver (.txt format)
└── adaptive/
    └── generated_failures/  # Failure scenarios for adaptive pipeline testing

experiments/
├── compare_grasp_milp_dijkstra.py  # Compare GRASP vs MILP vs Dijkstra on instances
├── run_complete_benchmark.py       # Full benchmark on all instances
├── convert_scholl_to_dsp.py        # Convert Scholl SALBP instances to DSP format
└── json_to_txt.py                  # Convert JSON instances to MILP text format

results/
├── benchmark_otto/          # Benchmark results on Otto and Scholl instances
├── benchmarks_2/            # Benchmark results on Otto and Scholl instances
├── compare/                 # GRASP vs MILP comparison outputs (CSV, logs)
├── grasp_selectif/          # GRASP selective mode results
├── grasp_complet/           # GRASP complete mode results
├── milp_save_selectif/      # MILP solver logs and solutions
├── batch/                   # Batch run outputs
└── save/                    # Adaptive pipeline outputs (dagtest, test_failures, test_nominal)
```

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| GRASP | Greedy Randomized Adaptive Search Procedure |
| GRASP+VND | Variable Neighborhood Descent |
| GRASP+MNS | Multi-Neighborhood Search |
| GRASP+TABU | Tabu Search |
| MILP | Mixed-Integer Linear Programming (baseline) |
| Dijkstra | Shortest path on precedence closure (baseline) |

## Adaptive Actions

| Action | Description |
|--------|-------------|
| change_tool | Switch to alternative tool |
| bypass | Skip via alternative path |
| destroy | Destructive removal |
| replan | Full sequence re-optimization |

## Failure Scenarios

The adaptive pipeline is tested with failure scenarios at three difficulty levels:

| Level | Description | Characteristics |
|-------|-------------|-----------------|
| **Simple** | Failure on non-critical component | - Leaf nodes or secondary branches<br>- Alternative paths available<br>- Easy bypass/workaround<br>- Tests basic robustness |
| **Intermediate** | Failure on moderately important component | - Not a bottleneck but requires adaptation<br>- May force less optimal paths<br>- Tests adaptive capabilities under constraints |
| **Complex** | Failure on critical component | - Cut nodes / mandatory passage points<br>- No alternative paths<br>- May block target access<br>- Tests maximum adaptation capacity (fallback, local replan) |

Failure scenarios are generated automatically based on graph topology to ensure each difficulty level represents a meaningful robustness challenge. See `data/adaptive/generated_failures/` for examples.


IMT NE


