# Disassembly Pipeline

This repository provides a pipeline for Disassembly Sequence Planning (DSP) using GRASP-based metaheuristics, with two main modes: **Full** and **Selective**.

## Project Objective
Optimize the disassembly sequence of complex products (automotive, electronics, etc.) to maximize recovered value, with:
- A full mode (complete disassembly)
- A selective mode (targeted disassembly with adaptive fallback)

## Installation & Dependencies

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Repository Structure

- `src/` : all source code (GRASP, local search, utils, etc.)
- `notebooks/` : analysis and visualization notebooks
- `data/` : datasets (benchmarks)
- `results/` : benchmark results
- `saved_runs/` : last 3 useful runs
- `docs/` : documentation, diagrams (optional)
- `experiments/` : scripts for generation and testing
- `old/` : legacy/unused archives (can be ignored)

## Available Modes

- **Full Mode**:
  - Complete disassembly
  - GRASP, VND, Tabu, MNS metaheuristics
  - Visualization, scoring, timing
- **Selective Mode**:
  - Partial disassembly (target nodes)
  - Adaptive fallback if blocked
  - Visualization of actual path, partial scoring

## Work in Progress

- The **Full Mode** (complete) is stable and fully reproducible.
- The **Selective Mode** (targeted) is still under active development and may not provide optimal or fully robust results yet. Use it for experimentation and feedback, but expect possible improvements and changes in future updates.

## How to Run the Pipeline

### 1. With script (recommended)
```bash
python run_all.py --mode complet   # or --mode selectif
```

### 2. With notebook
- Open a notebook (just for complete mode) in `notebooks/`
- Adjust data paths if needed

### 3. Custom benchmarks
- Use scripts in `experiments/`

## Benchmarks and Selective Targets

The list of benchmarks (graph datasets) and selective targets (nodes to be disassembled in selective mode) are coded in the script `experiments/run_complete_benchmark.py`.

- To add or modify a benchmark, edit the `benchmarks` dictionary in that script.
- To change the selective targets, edit the `selective_targets` dictionary in the same script.

This design ensures reproducibility of the experiments. 
If you want to run the pipeline on new data or with different targets, you must update these variables directly in the code.

**Example:**
```python
benchmarks = {
    "small_40": "data/small_40.json",
    "electronics_89": "data/electronics_89.json",
    # ...
}
selective_targets = {
    "small_40": ["C001", "C032"],
    # ...
}
```

## Pipeline Overview

1. Load graph (JSON data)
2. Initial construction (GRASP)
3. Local search (VND, Tabu, MNS...)
4. Visualization & result saving
5. Analysis via notebooks

## About `old/` folders
All folders named `old/` (in `data/`, `experiments/`, `results/`...) contain legacy files, scripts, or results not used in the current version. You can ignore them for normal usage.

## Going Further
- Modify benchmarks: see `experiments/`
- Add new datasets in `data/`
- Adapt heuristics in `src/grasp/`

## Reproducibility
- All dependencies are listed in `requirements.txt`
- Notebooks allow full result reproduction
- Provided datasets are directly usable


