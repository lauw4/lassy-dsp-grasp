#!/usr/bin/env python3
"""
Main execution script for DSP benchmark comparison
==================================================

Launches the complete benchmark comparison of 4 metaheuristics
on 6 industrial DSP instances.

Usage:
    python run_all.py

This script will:
1. Run complete benchmark comparison of GRASP, GRASP+VND, GRASP+MNS, and GRASP+TABU
2. Generate timestamped results in results/ directory
3. Create analysis-ready CSV file for notebook processing
"""

import sys
import os
from datetime import datetime
import argparse

# src vers path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["complet", "selectif"], default="complet")
    parser.add_argument("--n_runs", type=int, default=15)
    args = parser.parse_args()

    from experiments.run_complete_benchmark import run_complete_benchmark, mode as global_mode

    # Met à jour la variable globale mode dans le module importé
    import experiments.run_complete_benchmark as rcb
    rcb.mode = args.mode

    """Execute complete DSP benchmark"""
    try:
        # Print benchmark information
        
        print("=" * 60)
        print("DSP METAHEURISTICS BENCHMARK")
        print("=" * 60)
        print("Algorithms: GRASP, GRASP+VND, GRASP+MNS, GRASP+TABU")
        print("Benchmarks: 6 industrial/fictifs instances (40-200 nodes)")
        print("Protocol: 15 runs per algorithm-benchmark pair")
        print("Total experiments: 360")
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
        print(f"Analysis ready: Open notebooks/DSP_Analysis.ipynb")
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
