#!/usr/bin/env python3
"""
DSP Metaheuristics Benchmark (IWSPE 2026)

Compares GRASP, GRASP+VND, GRASP+MNS, and GRASP+TABU on DSP instances.

Usage:
    python run_all.py
    python run_all.py --mode selectif --n_runs 10
"""

import sys
import os
from datetime import datetime
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(description="DSP Metaheuristics Benchmark")
    parser.add_argument("--mode", choices=["complet", "selectif"], default="complet",
                        help="Disassembly mode: complet (all nodes) or selectif (target nodes)")
    parser.add_argument("--n_runs", type=int, default=15,
                        help="Number of runs per algorithm-instance pair")
    args = parser.parse_args()

    from experiments.run_complete_benchmark import run_complete_benchmark

    # Update mode in imported module
    import experiments.run_complete_benchmark as rcb
    rcb.mode = args.mode

    # Run benchmark
    try:
        # Print benchmark information
        
        print("=" * 60)
        print("DSP METAHEURISTICS BENCHMARK")
        print("=" * 60)
        print("Algorithms: GRASP, GRASP+VND, GRASP+MNS, GRASP+TABU")
        print(f"Mode: {args.mode}")
        print(f"Runs per configuration: {args.n_runs}")
        print()
        
        start_time = datetime.now()
        print(f"Benchmark started at: {start_time}")
        print()
        
        # Run complete benchmark
        results_file = run_complete_benchmark(n_runs=args.n_runs)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print()
        print("=" * 60)
        print("BENCHMARK COMPLETED")
        print("=" * 60)
        print(f"Duration: {duration}")
        print(f"Results saved to: {results_file}")
        print("=" * 60)
        
    except ImportError as e:
        print(f"Error: Could not import benchmark module: {e}")
        print("Make sure the experiments module is properly configured.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during benchmark execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
