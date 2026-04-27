"""run_benchmark_scholl_salbp.py
Benchmark GRASP+VND vs MILP-CPLEX on Scholl and SALBP instances.

KEY FIX: Both GRASP and MILP load from the SAME TXT file via
load_graph_from_txt(), so profits/costs/times are identical.
λ is auto-calibrated per instance (5% of mean net-value rate).
CPLEX is used as solver (fallback to CBC if not found).

Output: results/benchmark_scholl_salbp.csv
"""
import sys, os, time, glob, math
from datetime import datetime, timedelta
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pulp
from src.utils.graph_io import load_graph_from_txt
from src.grasp.constructive import run_grasp
from src.utils.metrics import score_obj1, prune_sequence
from src.exact_milp_dijkstra.partial_disassembly_model import (
    load_data, build_model, solve_model,
    _parse_last_status_from_cplex_log, _parse_last_gap_from_cplex_log
)

def _bar(done, total, width=20):
    filled = int(width * done / max(total, 1))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {done}/{total}"

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
MILP_DIR    = "data/instances_milp"
RESULTS_DIR = "results"
OUTPUT_CSV  = os.path.join(RESULTS_DIR, "benchmark_scholl_salbp_n100.csv")

GRASP_RUNS  = 1         # independent GRASP restarts per instance
GRASP_ITERS = 150       # max reactive-GRASP iterations per run
GRASP_TIME  = 30        # VND time budget per run (seconds)

MILP_TIMELIM = 120      # CBC time limit (seconds)
MILP_GAP     = 0.01     # 1 % relative gap — tight for fair comparison
MAX_N_MILP   = 100      # only run MILP for instances with |V| ≤ this

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  Instance selection
# ─────────────────────────────────────────────
def collect_instances():
    """Return list of (label, txt_path) for all instances with n <= MAX_N_MILP."""
    instances = []
    seen = set()

    for p in sorted(glob.glob(os.path.join(MILP_DIR, "*.txt"))):
        label = os.path.basename(p)
        label = label.replace("_selectif.txt", "").replace(".txt", "")
        if label in seen:
            continue

        try:
            data = load_data(p)
            n = len(data.get("V", []))
        except Exception:
            continue

        if n <= MAX_N_MILP:
            seen.add(label)
            instances.append((label, p))

    return instances


# ─────────────────────────────────────────────
#  GRASP runner (uses TXT → NetworkX via load_graph_from_txt)
# ─────────────────────────────────────────────
def run_grasp_on_txt(txt_path, n_runs=GRASP_RUNS):
    G = load_graph_from_txt(txt_path)              # same data as MILP (TXT)
    targets = G.graph.get('targets', [])           # mandatory targets
    lam     = G.graph.get('lambda_discount')
    n       = G.number_of_nodes()

    scores = []
    t0 = time.perf_counter()
    for _ in range(n_runs):
        # Use complet mode so GRASP can consider all nodes (same scope as MILP)
        seq, _s = run_grasp(
            G,
            algorithm='vnd',
            mode='complet',
            runs=1,
            max_iterations=GRASP_ITERS,
            time_budget=GRASP_TIME,
            early_stop=True,
        )
        # prune_sequence removes nodes that hurt the objective (mirrors MILP freedom)
        # but targets stay protected (mirrors MILP x_i=1 for i in T constraint)
        seq_pruned = prune_sequence(seq, G, protected_nodes=set(targets))
        s = score_obj1(seq_pruned, G)
        scores.append(s)
    elapsed = time.perf_counter() - t0

    best_score = max(scores)
    avg_score  = sum(scores) / len(scores)
    std_score  = math.sqrt(sum((s - avg_score)**2 for s in scores) / len(scores))
    return best_score, avg_score, std_score, lam, n, elapsed


# ─────────────────────────────────────────────
#  MILP runner
# ─────────────────────────────────────────────
def run_milp_on_txt(txt_path, label=""):
    data = load_data(txt_path)
    n    = len(data['V'])
    if n > MAX_N_MILP:
        return None, None, None, n, f"n={n} > MAX_N_MILP={MAX_N_MILP}", None
    try:
        model, variables = build_model(data)
        lam_milp = data.get('lambda_discount', 0.05)
        t0 = time.perf_counter()
        optimal, log_file = solve_model(
            model, time_limit=MILP_TIMELIM, use_cplex=True, gap_limit=MILP_GAP,
            instance_name=label
        )
        elapsed = time.perf_counter() - t0
        # Use real CPLEX log status (PuLP wrongly reports 'Optimal' on TimeLimit)
        status = _parse_last_status_from_cplex_log(log_file)
        if status == 'Unknown':
            status = pulp.LpStatus[model.status]  # fallback
        obj    = pulp.value(model.objective)
        cplex_gap = _parse_last_gap_from_cplex_log(log_file)
        return obj, status, elapsed, n, lam_milp, cplex_gap
    except Exception as e:
        return None, f"ERROR: {e}", None, n, None, None


# ─────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────
def main():
    instances = collect_instances()
    total = len(instances)
    t_start_all = time.perf_counter()
    print(f"\n{'='*70}")
    print(f"  BENCHMARK  |  {total} instances  |  GRASP runs={GRASP_RUNS}  |  MILP n≤{MAX_N_MILP}")
    print(f"  Started    :  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
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
        print(f"  ► [{idx:3d}/{total}] {label}", flush=True)

        # ── GRASP ──
        print(f"    GRASP ({GRASP_RUNS} runs)...", end="", flush=True)
        best_g, avg_g, std_g, lam, n, grasp_time = run_grasp_on_txt(txt_path)
        print(f"  ✓  best={best_g:.2f}  avg={avg_g:.2f}±{std_g:.2f}  n={n}  λ={lam:.4f}  t={grasp_time:.1f}s", flush=True)

        # ── MILP (only for small instances) ──
        if n <= MAX_N_MILP:
            print(f"    MILP  (CPLEX, lim={MILP_TIMELIM}s)...", end="", flush=True)
        milp_obj, milp_status, milp_time, _, _, cplex_gap = run_milp_on_txt(txt_path, label=label)

        if milp_obj is not None and best_g is not None and milp_obj != 0:
            gap_pct = (milp_obj - best_g) / abs(milp_obj) * 100.0
        else:
            gap_pct = None

        if milp_obj is not None:
            flag = "⚠️" if gap_pct is not None and gap_pct > 5 else "✓"
            gap_info = f"cplex_gap={cplex_gap:.1f}%" if cplex_gap is not None else ""
            print(f"  {flag}  obj={milp_obj:.2f}  ({milp_status}, {milp_time:.1f}s, {gap_info})  gap_vs_grasp={gap_pct:.1f}%", flush=True)
        elif n > MAX_N_MILP:
            print(f"    MILP  → skipped (n={n} > {MAX_N_MILP})", flush=True)
        else:
            print(f"  ✗  ({milp_status})", flush=True)

        inst_time = time.perf_counter() - t_inst
        print(f"    Instance done in {inst_time:.1f}s", flush=True)

        rows.append({
            "instance":     label,
            "n":            n,
            "lambda":       round(lam, 5) if lam else None,
            "grasp_best":   round(best_g, 4) if best_g is not None else None,
            "grasp_avg":    round(avg_g, 4) if avg_g is not None else None,
            "grasp_std":    round(std_g, 4) if std_g is not None else None,
            "grasp_time_s": round(grasp_time, 2) if grasp_time is not None else None,
            "milp_obj":     round(milp_obj, 4) if milp_obj is not None else None,
            "milp_status":  milp_status,
            "milp_time_s":  round(milp_time, 2) if milp_time is not None else None,
            "cplex_gap_pct": round(cplex_gap, 2) if cplex_gap is not None else None,
            "gap_pct":      round(gap_pct, 2) if gap_pct is not None else None,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    total_time = time.perf_counter() - t_start_all
    print(f"\n{'='*70}")
    print(f"  ✓ Saved → {OUTPUT_CSV}  ({len(df)} rows)")
    print(f"  Total runtime: {str(timedelta(seconds=int(total_time)))}")
    print(f"{'='*70}")
    sys.stdout.flush()

    # ── Summary: instances with MILP comparison ──
    comp = df.dropna(subset=["milp_obj", "grasp_best"])
    if not comp.empty:
        print(f"\n── GRASP vs MILP (n ≤ {MAX_N_MILP}) ─────────────────────────────────────")
        print(comp[["instance","n","lambda","grasp_best","milp_obj","gap_pct","milp_status","grasp_time_s","milp_time_s"]].to_string(index=False))
        avg_gap = comp["gap_pct"].mean()
        max_gap = comp["gap_pct"].max()
        print(f"\n  Average gap : {avg_gap:.2f} %")
        print(f"  Max gap     : {max_gap:.2f} %")

    print(f"\n── GRASP only (n > {MAX_N_MILP}) ─────────────────────────────────────────")
    large = df[df["n"] > MAX_N_MILP]
    if not large.empty:
        print(large[["instance","n","lambda","grasp_best","grasp_time_s"]].to_string(index=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
