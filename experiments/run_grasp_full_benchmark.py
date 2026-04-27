"""run_grasp_full_benchmark.py
Run GRASP+VND on all 40 Scholl + SALBP instances.
Calculate gaps against CPLEX optimal solutions (when available).

Output: results/grasp_full_benchmark.csv + display results here.
"""
import sys, os, time, glob, math
from datetime import datetime, timedelta
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.utils.graph_io import load_graph_from_txt
from src.grasp.constructive import run_grasp
from src.utils.metrics import score_obj1, prune_sequence

def _bar(done, total, width=20):
    filled = int(width * done / max(total, 1))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {done}/{total}"

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
MILP_DIR   = "data/instances_milp"
RESULTS_DIR = "results"
OUTPUT_CSV  = os.path.join(RESULTS_DIR, "grasp_full_benchmark.csv")
MILESTONE_CSV = os.path.join(RESULTS_DIR, "benchmark_scholl_salbp.csv")  # existing CPLEX results

GRASP_RUNS  = 20        # independent GRASP restarts per instance (more thorough)
GRASP_ITERS = 150       # max iterations per run
GRASP_TIME  = 30        # VND time budget per run (seconds)

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  Instance selection
# ─────────────────────────────────────────────
def collect_instances():
    """Return list of (label, txt_path) for Scholl + SALBP instances."""
    instances = []
    seen = set()

    # All Scholl instances (25 instances, n=7 to n=297)
    for p in sorted(glob.glob(os.path.join(MILP_DIR, "scholl_*.txt"))):
        key = os.path.basename(p)
        if key not in seen:
            seen.add(key)
            label = os.path.basename(p).replace("_selectif.txt", "")
            instances.append((label, p))

    # SALBP n=20: first 10 instances
    salbp20 = sorted(glob.glob(os.path.join(MILP_DIR, "salbp_instance_n=20_*.txt")))[:10]
    for p in salbp20:
        key = os.path.basename(p)
        if key not in seen:
            seen.add(key)
            label = os.path.basename(p).replace("_selectif.txt", "")
            instances.append((label, p))

    # SALBP n=50: first 5 instances
    salbp50 = sorted(glob.glob(os.path.join(MILP_DIR, "salbp_instance_n=50_*.txt")))[:5]
    for p in salbp50:
        key = os.path.basename(p)
        if key not in seen:
            seen.add(key)
            label = os.path.basename(p).replace("_selectif.txt", "")
            instances.append((label, p))

    return instances


# ─────────────────────────────────────────────
#  GRASP runner
# ─────────────────────────────────────────────
def run_grasp_on_txt(txt_path, n_runs=GRASP_RUNS):
    G = load_graph_from_txt(txt_path)              # same data as MILP (TXT)
    targets = G.graph.get('targets', [])
    lam     = G.graph.get('lambda_discount')
    n       = G.number_of_nodes()

    scores = []
    t0 = time.perf_counter()
    for _ in range(n_runs):
        seq, _s = run_grasp(
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
    avg_score  = sum(scores) / len(scores)
    std_score  = math.sqrt(sum((s - avg_score)**2 for s in scores) / len(scores))
    return best_score, avg_score, std_score, lam, n, elapsed


# ─────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────
def main():
    instances = collect_instances()
    total = len(instances)
    t_start_all = time.perf_counter()

    # Load existing CPLEX results for comparison
    cplex_map = {}
    if os.path.exists(MILESTONE_CSV):
        df_cplex = pd.read_csv(MILESTONE_CSV)
        for _, row in df_cplex.iterrows():
            if pd.notna(row.get('milp_obj')):
                cplex_map[row['instance']] = float(row['milp_obj'])

    print(f"\n{'='*80}")
    print(f"  GRASP BENCHMARK  |  {total} instances  |  {GRASP_RUNS} runs/instance")
    print(f"  Started          :  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CPLEX ref        :  {len(cplex_map)} instances loaded")
    print(f"{'='*80}\n")
    sys.stdout.flush()

    rows = []
    for idx, (label, txt_path) in enumerate(instances, 1):
        t_inst = time.perf_counter()

        # ETA computation
        if idx > 1:
            elapsed_total = t_inst - t_start_all
            avg_per_inst  = elapsed_total / (idx - 1)
            eta_s         = avg_per_inst * (total - idx + 1)
            eta_str       = str(timedelta(seconds=int(eta_s)))
        else:
            eta_str = "?"

        print(f"\n{_bar(idx, total)}  ETA ~{eta_str}")
        print(f"  ► [{idx:3d}/{total}] {label:<45}", end="", flush=True)

        # ── GRASP ──
        best_g, avg_g, std_g, lam, n, grasp_time = run_grasp_on_txt(txt_path)
        print(f"  ✓  best={best_g:7.2f}  avg={avg_g:7.2f}±{std_g:5.2f}  n={n:3d}  λ={lam:.4f}  t={grasp_time:5.1f}s", end="", flush=True)

        # ── Compare to CPLEX if available ──
        gap_pct = None
        cplex_obj = None
        if label in cplex_map:
            cplex_obj = cplex_map[label]
            if cplex_obj != 0:
                gap_pct = (cplex_obj - best_g) / abs(cplex_obj) * 100.0
            flag = "⚠️" if gap_pct is not None and gap_pct > 5 else "✓"
            print(f"  {flag}  gap={gap_pct:6.2f}%", flush=True)
        else:
            print(f"    (no CPLEX ref)", flush=True)

        rows.append({
            "instance":     label,
            "n":            n,
            "lambda":       round(lam, 5) if lam else None,
            "grasp_best":   round(best_g, 4) if best_g is not None else None,
            "grasp_avg":    round(avg_g, 4) if avg_g is not None else None,
            "grasp_std":    round(std_g, 4) if std_g is not None else None,
            "grasp_time_s": round(grasp_time, 2) if grasp_time is not None else None,
            "cplex_obj":    round(cplex_obj, 4) if cplex_obj is not None else None,
            "gap_pct":      round(gap_pct, 2) if gap_pct is not None else None,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    total_time = time.perf_counter() - t_start_all
    print(f"\n{'='*80}")
    print(f"  ✓ Saved → {OUTPUT_CSV}  ({len(df)} rows)")
    print(f"  Total runtime: {str(timedelta(seconds=int(total_time)))}")
    print(f"{'='*80}")
    sys.stdout.flush()

    # ── Summary ──
    comp = df.dropna(subset=["cplex_obj", "grasp_best"])
    if not comp.empty:
        print(f"\n{'─'*80}")
        print(f"  GRASP vs CPLEX ({len(comp)} instances with both results)")
        print(f"{'─'*80}")
        print(comp[["instance", "n", "lambda", "grasp_best", "cplex_obj", "gap_pct", "grasp_time_s"]].to_string(index=False))
        
        avg_gap = comp["gap_pct"].mean()
        max_gap = comp["gap_pct"].max()
        min_gap = comp["gap_pct"].min()
        print(f"\n  Average gap : {avg_gap:7.2f} %")
        print(f"  Max gap     : {max_gap:7.2f} %  ({comp.loc[comp['gap_pct'].idxmax(), 'instance']})")
        print(f"  Min gap     : {min_gap:7.2f} %  ({comp.loc[comp['gap_pct'].idxmin(), 'instance']})")
        
        # Count gaps by range
        gaps_perfect = len(comp[comp["gap_pct"] <= 0.5])
        gaps_excellent = len(comp[(comp["gap_pct"] > 0.5) & (comp["gap_pct"] <= 1)])
        gaps_good = len(comp[(comp["gap_pct"] > 1) & (comp["gap_pct"] <= 3)])
        gaps_acceptable = len(comp[(comp["gap_pct"] > 3) & (comp["gap_pct"] <= 5)])
        gaps_poor = len(comp[comp["gap_pct"] > 5])
        print(f"\n  Gap distribution:")
        print(f"    ≤ 0.5%  : {gaps_perfect} instances")
        print(f"    0.5-1%  : {gaps_excellent} instances")
        print(f"    1-3%    : {gaps_good} instances")
        print(f"    3-5%    : {gaps_acceptable} instances")
        print(f"    > 5%    : {gaps_poor} instances")

    print(f"\n  GRASP-only (no CPLEX ref):")
    grasp_only = df[df["cplex_obj"].isna()]
    if not grasp_only.empty:
        print(f"    {len(grasp_only)} instances: {', '.join(grasp_only['instance'].head(5).tolist())}...")
        print(f"    Avg best score: {grasp_only['grasp_best'].mean():.2f}")
    
    print(f"\n{'='*80}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
