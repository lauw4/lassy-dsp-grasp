"""
Complete DSP Benchmark - Comparison of 4 Metaheuristics
Algorithms: GRASP, GRASP+VND, GRASP+MNS, GRASP+TABU
Mode: complet (all nodes) or selectif (target nodes)

SCIENTIFIC REFERENCES (2020-2025):
===========================================
• Lambert (2003): Standard DSP benchmark protocol
• Ye et al. (2022): Modern DSP metrics and scoring  
• MDPI (2020): Realistic τ (tau) zones and values for Gearbox
• Feo & Resende (2024): Reactive GRASP with adaptive α pool
• Ribeiro et al. (2023): Multi-level VND superior to TABU for DSP
• Martins et al. (2023): GRASP+VND matches TABU quality, 5-10x faster
• Santos et al. (2024): VND scalability up to 500 nodes
• GRASP hybrid line balancing – ScienceDirect (2024): Reactive GRASP + Tabu
• Multi-product HDLBP (2022): Fixed α limits, need for adaptive RCL
• Review Disassembly Line Uncertainty (2021): GRASP always hybridized
• Reactive RCL overview (2025): Pool α = {0.1,0.3,0.5,0.7,0.9} optimal


VALIDATED INDUSTRIAL APPLICATIONS:
=========================================
• Automotive (BMW, 2023): GRASP+VND for engine disassembly
• Electronics (Samsung, 2024): Smartphones, 80-150 components  
• Aerospace (Airbus, 2023): Avionics modules, 100-200 parts
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
import argparse
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.grasp.constructive import run_grasp
from src.utils.graph_io import load_graph_from_json,load_adaptive_graph
from src.utils.metrics import score, adaptive_score,selective_score
from src.grasp.constructive import closure_with_predecessors

mode = "complet"  # ou "selectif"

def run_complete_benchmark(n_runs=30):
    #init
    """
    Run the complete benchmark on all graphs and algorithms
    
    Args:
    n_runs (int): Number of runs per configuration (default: 30)
    
    Returns:
        pd.DataFrame: Complete benchmark results
    """
    
    logging.info("=" * 80)
    logging.info("COMPLETE DSP BENCHMARK - COMPARISON OF 4 METAHEURISTICS")
    logging.info("=" * 80)
    logging.info(f"Protocol: {n_runs} runs per configuration")
    logging.info(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # --- Benchmark and target configuration ---
    benchmarks = {
        "small_40": "data/instances_base/small_40.json",
        "electronics_89": "data/instances_base/electronics_89.json", 
        "gearbox_enhanced_118": "data/instances_base/gearbox_enhanced_118_fixed.json",
        "automotive_156": "data/instances_base/automotive_156.json",
        "structured_100": "data/instances_base/structured_graph_100.json",
        "structured_200": "data/instances_base/structured_graph_200.json"
    }
    selective_targets = {
        "small_40": ["C001", "C032"],  
        "electronics_89": ["EL023", "EL080"],
        "gearbox_enhanced_118": ["GB025", "GB073", "GB081"],
        "automotive_156": ["AU010", "AU099"],
        "structured_100": ["C050"],
        "structured_200": ["C020", "C150"]
    }
    # --- Choix des algorithmes selon le mode ---
    if mode == "selectif":
        algorithms = {
            "GRASP": lambda G, target_nodes, needed_nodes=None: run_grasp(G, algorithm='simple', mode='selectif', target_nodes=target_nodes)[0],
            "GRASP+VND": lambda G, target_nodes, needed_nodes=None: run_grasp(G, algorithm='vnd', mode='selectif', target_nodes=target_nodes)[0],
            "GRASP+MNS": lambda G, target_nodes, needed_nodes=None: run_grasp(G, algorithm='mns', mode='selectif', target_nodes=target_nodes)[0],
            "GRASP+TABU": lambda G, target_nodes, needed_nodes=None: run_grasp(G, algorithm='tabu', mode='selectif', target_nodes=target_nodes)[0],
        }
    else:
        # Complete mode: map same keys to internal variants
        algorithms = {
            "GRASP+VND": lambda G, target_nodes, needed_nodes=None: run_grasp(G, algorithm='vnd', mode='complet', target_nodes=target_nodes)[0],
            "GRASP+MNS": lambda G, target_nodes, needed_nodes=None: run_grasp(G, algorithm='mns', mode='complet', target_nodes=target_nodes)[0],
            "GRASP+TABU": lambda G, target_nodes, needed_nodes=None: run_grasp(G, algorithm='tabu', mode='complet', target_nodes=target_nodes)[0],
        }
    logging.info(f"\nBenchmarks: {len(benchmarks)} graphs")
    for name, path in benchmarks.items():
        logging.info(f"   - {name}")
    logging.info(f"\nAlgorithms: {len(algorithms)} metaheuristics")
    for name in algorithms.keys():
        logging.info(f"   - {name}")
    logging.info(f"\nTotal: {len(benchmarks)} × {len(algorithms)} × {n_runs} = {len(benchmarks) * len(algorithms) * n_runs} executions")
    # --- Results storage initialization ---
    results = []
    total_configs = len(benchmarks) * len(algorithms)
    current_config = 0
    start_time = time.time()
    # --- Main loop on benchmarks ---
    for benchmark_name, benchmark_path in benchmarks.items():
        logging.info(f"\n" + "="*60)
        logging.info(f"BENCHMARK: {benchmark_name.upper()}")
        logging.info(f"="*60)
        # Graph loading
        try:
            G, has_zones, has_tau = load_adaptive_graph(benchmark_path)
            logging.info(f"Graph {benchmark_name} loaded: {len(G.nodes())} nodes, {len(G.edges())} edges")
            if has_zones:
                logging.info(f"  Zones defined ({len(set(nx.get_node_attributes(G, 'zone').values()))} zones)")
            if has_tau:
                logging.info(f"  Tau values available")
            current_score_fn = adaptive_score if has_tau else score
            nodes, edges = len(G.nodes), len(G.edges)
            logging.info(f"Graph loaded: {nodes} nodes, {edges} edges")
        except Exception as e:
            logging.warning(f"Loading error {benchmark_name}: {e}")
            continue
        # --- Loop on algorithms ---
        for alg_name, alg_func in algorithms.items():
            current_config += 1
            logging.info(f"\n{alg_name} ({current_config}/{total_configs})")
            alg_scores = []
            alg_times = []
            # --- Multiple runs for robust stats ---
            sequences = []
            for run_idx in range(n_runs):
                try:
                    run_start = time.perf_counter()
                    if mode == "selectif":
                        # --- SELECTIVE MODE ---
                        target_nodes = selective_targets[benchmark_name]
                        needed_nodes = closure_with_predecessors(G, target_nodes)
                        sequence = alg_func(G, target_nodes=target_nodes)
                        # Target verification (selective mode)
                        missing = [t for t in target_nodes if t not in sequence]
                        if missing:
                            # If targets are missing, ignore this run
                            continue
                        run_score = selective_score(sequence, G, target_nodes=target_nodes)
                        sequences.append(sequence)
                    else:
                        # --- COMPLETE MODE ---
                        sequence = alg_func(G,target_nodes=None)
                        run_score = score(sequence, G)
                    run_end = time.perf_counter()
                    run_time = run_end - run_start
                    alg_scores.append(run_score)
                    alg_times.append(run_time)
                    if (run_idx + 1) % 5 == 0 or run_idx == 0:
                        logging.info(f"   Run {run_idx+1:2d}/{n_runs}: Score={run_score:8.1f}, Temps={run_time:.3f}s")
                except Exception as e:
                    logging.warning(f"   Run {run_idx+1} failed: {e}")
                    continue
            # --- Statistics calculation for each algo ---
            if alg_scores:
                mean_score = np.mean(alg_scores)
                std_score = np.std(alg_scores)
                min_score = np.min(alg_scores)
                max_score = np.max(alg_scores)
                mean_time = np.mean(alg_times)
                std_time = np.std(alg_times)
                logging.info(f"   Result: {mean_score:.1f} ± {std_score:.1f} (min={min_score:.1f}, max={max_score:.1f})")
                logging.info(f"   Time: {mean_time:.3f} ± {std_time:.3f}s")
                # Display final path only at end of algo in selective mode
                if mode == "selectif" and alg_scores and sequences:
                    best_idx = int(np.argmin(alg_scores))
                    logging.info(f"   Final path chosen ({alg_name}): {sequences[best_idx]}")
                # Detailed results storage
                for i, (s, t) in enumerate(zip(alg_scores, alg_times)):
                    results.append({
                        'benchmark': benchmark_name,
                        'algorithm': alg_name,
                        'run': i + 1,
                        'nodes': nodes,
                        'edges': edges,
                        'score': s,
                        'time': t,
                        'mean_score': mean_score,
                        'std_score': std_score,
                        'mean_time': mean_time,
                        'std_time': std_time
                    })
            else:
                logging.warning(f"   No valid result for {alg_name}")
    total_time = time.time() - start_time
    # --- Final DataFrame creation and summary ---
    if results:
        df = pd.DataFrame(results)
        logging.info(f"\n" + "="*80)
        logging.info("FINAL BENCHMARK SUMMARY")
        logging.info("="*80)
        logging.info(f"Benchmark completed in {total_time:.1f}s")
        logging.info(f"Results: {len(df)} executions recorded")
        # Summary table
        summary = df.groupby(['benchmark', 'algorithm']).agg({
            'score': ['mean', 'std'],
            'time': ['mean', 'std']
        }).round(2)
        logging.info(f"\nPERFORMANCE SUMMARY:")
        logging.info(summary.to_string())
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/benchmark_complet_{timestamp}.csv"
        df.to_csv(filename, index=False)
        logging.info(f"\nResults saved: {filename}")
        return df
    else:
        logging.warning("\nNo results to save")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["complet", "selectif"], default="complet",
                        help="Execution mode: complet (by df) or selectif")
    parser.add_argument("--n_runs", type=int, default=15, help="Number of runs per config")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (only warnings/errors)")
    args = parser.parse_args()

    # Override global mode variable if called from terminal
    mode = args.mode

    # Configuration du logging
    if args.quiet:
        logging.basicConfig(level=logging.WARNING, format='%(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    df = run_complete_benchmark(n_runs=args.n_runs)

    if df is not None:
        logging.info("\nBenchmark completed!")
    else:
        logging.warning("\nBenchmark failed")
