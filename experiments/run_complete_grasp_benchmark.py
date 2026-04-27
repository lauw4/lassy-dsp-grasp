#!/usr/bin/env python3
"""run_complete_grasp_benchmark.py
Benchmark GRASP+VND sur toutes les instances TXT selectif disponibles.
On utilise les fichiers data/instances_milp/*_selectif.txt qui sont compatibles
avec load_graph_from_txt().

1 RUN PAR INSTANCE pour comparaison équitable avec CPLEX.

Total: ~315 instances (Scholl + SALBP/Otto), de n=7 à n=1000.
Output: results/grasp_complete_benchmark.csv
"""
import sys, os, time, glob, math, json
from datetime import datetime, timedelta
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.grasp.constructive import run_grasp
from src.utils.graph_io import load_graph_from_txt
from src.utils.metrics import score_obj1, prune_sequence

def _bar(done, total, width=20):
    filled = int(width * done / max(total, 1))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {done}/{total}"

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
GRASP_RUNS = 1          # 1 run per instance (for fair comparison with CPLEX)
GRASP_ITERS = 150       # max iterations per restart
GRASP_TIME = 30         # time budget per restart (s)
RESULTS_DIR = "results"
OUTPUT_CSV = os.path.join(RESULTS_DIR, "grasp_complete_benchmark.csv")
MILP_DIR = "data/instances_milp"


# ─────────────────────────────────────────────
#  Collect all TXT selectif instances
# ─────────────────────────────────────────────
def collect_instances():
    """Return list of (label, txt_path) for all TXT selectif instances."""
    all_txt = sorted(glob.glob(os.path.join(MILP_DIR, "*_selectif.txt")))
    instances = []
    for p in all_txt:
        label = os.path.basename(p).replace("_selectif.txt", "")
        instances.append((label, p))

    print(f"\n[INFO] Collected {len(instances)} instances from {MILP_DIR}/*_selectif.txt")
    
    # Count by type
    scholl_count = len([l for l, _ in instances if l.startswith('scholl_')])
    salbp_20 = len([l for l, _ in instances if 'n=20' in l])
    salbp_50 = len([l for l, _ in instances if 'n=50' in l])
    salbp_100 = len([l for l, _ in instances if 'n=100' in l])
    salbp_1000 = len([l for l, _ in instances if 'n=1000' in l])
    
    print(f"  - Scholl: {scholl_count}")
    print(f"  - SALBP n=20: {salbp_20}")
    print(f"  - SALBP n=50: {salbp_50}")
    print(f"  - SALBP n=100: {salbp_100}")
    print(f"  - SALBP n=1000: {salbp_1000}")

    return instances


# ─────────────────────────────────────────────
#  GRASP runner
# ─────────────────────────────────────────────
def run_grasp_on_txt(txt_path, n_runs=GRASP_RUNS):
    """Run GRASP n_runs times on a TXT instance, return (best, avg, std, lambda, n, elapsed)."""
    G = load_graph_from_txt(txt_path)
    targets = G.graph.get('targets', [])
    lam = G.graph.get('lambda_discount')
    n = G.number_of_nodes()

    scores = []
    t0 = time.perf_counter()
    for _ in range(n_runs):
        seq, _ = run_grasp(
            G,
            algorithm='vnd',
            mode='complet',
            runs=1,
            max_iterations=GRASP_ITERS,
            time_budget=GRASP_TIME,
            early_stop=True,
        )
        seq_pruned = prune_sequence(seq, G, protected_nodes=set(targets))
        s = score_obj1(seq_pruned, G)
        scores.append(s)
    elapsed = time.perf_counter() - t0

    best_score = max(scores)
    avg_score = sum(scores) / len(scores)
    std_score = math.sqrt(sum((s - avg_score)**2 for s in scores) / len(scores))
    return best_score, avg_score, std_score, lam, n, elapsed


# ─────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────
def main():
    instances = collect_instances()
    total = len(instances)
    t_start_all = time.perf_counter()

    print(f"\n{'='*80}")
    print(f"  COMPLETE GRASP BENCHMARK  |  {total} instances  |  1 run/instance")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    rows = []
    for idx, (label, txt_path) in enumerate(instances, 1):
        t_inst = time.perf_counter()

        # ETA computation
        if idx > 1:
            elapsed_total = t_inst - t_start_all
            avg_per_inst = elapsed_total / (idx - 1)
            eta_s = avg_per_inst * (total - idx + 1)
            eta_str = str(timedelta(seconds=int(eta_s)))
        else:
            eta_str = "?"

        print(f"\n{_bar(idx, total)}  ETA ~{eta_str}")
        print(f"  ► [{idx:3d}/{total}] {label}", flush=True)

        # ── GRASP ──
        print(f"    GRASP (1 run)...", end="", flush=True)
        try:
            best_g, avg_g, std_g, lam, n, grasp_time = run_grasp_on_txt(txt_path)
            print(f"  ✓  best={best_g:.2f}  avg={avg_g:.2f}±{std_g:.2f}  n={n}  λ={lam:.6f}  t={grasp_time:.1f}s", flush=True)

            rows.append({
                "instance": label,
                "n": n,
                "lambda": round(lam, 6) if lam else None,
                "grasp_best": round(best_g, 4) if best_g is not None else None,
                "grasp_avg": round(avg_g, 4) if avg_g is not None else None,
                "grasp_std": round(std_g, 4) if std_g is not None else None,
                "grasp_time_s": round(grasp_time, 2) if grasp_time is not None else None,
            })
        except Exception as e:
            print(f"  ✗  ERROR: {e}", flush=True)
            rows.append({
                "instance": label,
                "n": None,
                "lambda": None,
                "grasp_best": None,
                "grasp_avg": None,
                "grasp_std": None,
                "grasp_time_s": None,
            })

        inst_time = time.perf_counter() - t_inst
        print(f"    Instance done in {inst_time:.1f}s", flush=True)

    # Save results
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    total_time = time.perf_counter() - t_start_all

    print(f"\n{'='*80}")
    print(f"  ✓ Saved → {OUTPUT_CSV}  ({len(df)} rows)")
    print(f"  Total runtime: {str(timedelta(seconds=int(total_time)))}")
    print(f"{'='*80}")

    # Summary statistics
    print(f"\n── SUMMARY ──────────────────────────────────────────────────────")
    print(f"  Total instances: {len(df)}")
    print(f"  Avg best score: {df['grasp_best'].mean():.2f}")
    print(f"  Avg time per instance: {df['grasp_time_s'].mean():.2f}s")
    print(f"  Max time: {df['grasp_time_s'].max():.2f}s")
    print()

    # Group by size
    print(f"── BY SIZE ──────────────────────────────────────────────────────")
    for n_val in sorted(df['n'].dropna().unique()):
        subset = df[df['n'] == n_val]
        print(f"  n={int(n_val):4d}: {len(subset):3d} instances, avg_time={subset['grasp_time_s'].mean():.2f}s")


if __name__ == '__main__':
    main()
