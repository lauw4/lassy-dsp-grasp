"""
Benchmark — Objective 1: Discounted Recovery Profit
=====================================================
Objective: max Σ(v_i − λ·C_i)   v_i = r_i − c_i,  λ = 0.05 EUR/s
NP-hard via 1|prec|ΣC_j  (Lenstra & Rinnooy Kan, 1978)

Protocol
--------
* GRASP+VND  : all instances, N_RUNS independent starts, best+avg+std reported
* MILP (CBC) : small instances only (n ≤ 20), 120 s time limit

Results saved to:  results/updated_benchmark_obj1.csv
Run from project root:
    python experiments/run_benchmark_obj1.py
"""

import sys, os, time, json, math
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.utils.graph_io  import load_adaptive_graph
from src.utils.metrics   import score_obj1
from src.grasp.config    import LAMBDA_DISCOUNT
from src.grasp.constructive import run_grasp, closure_with_predecessors

# ── MILP imports ──────────────────────────────────────────────────────────────
from src.exact_milp_dijkstra.partial_disassembly_model import (
    load_data, build_model, solve_model, extract_sequence
)
import pulp

# =============================================================================
# Configuration
# =============================================================================
N_RUNS       = 10      # GRASP independent starts per instance
MILP_TIMELIM = 120     # seconds, CBC only (no CPLEX required)
MILP_GAP     = 0.05    # 5 % optimality gap tolerance
MAX_N_MILP   = 20      # only run MILP when |V| ≤ this value

RESULTS_DIR  = os.path.join(ROOT, "results")
OUTPUT_CSV   = os.path.join(RESULTS_DIR, "updated_benchmark_obj1.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# Instance catalogue
# =============================================================================

# -- Industrial instances (JSON) ----------------------------------------------
INDUSTRIAL = {
    "mini_test"           : "data/instances_base/mini_test.json",
    "dagtest"             : "data/instances_base/dagtest.json",
    "gearbox_118"         : "data/instances_base/gearbox_118.json",
    "electronics_89"      : "data/instances_base/electronics_89.json",
    "automotive_156"      : "data/instances_base/automotive_156.json",
}

# -- Scholl benchmark instances (JSON, all sizes) ------------------------------
SCHOLL = {
    f"scholl_{k}": f"data/instances_base/{k}.json"
    for k in [
        "scholl_mertens_n=7",
        "scholl_bowman8_n=8",
        "scholl_jaeschke_n=9",
        "scholl_jackson_n=11",
        "scholl_mansoor_n=11",
        "scholl_roszieg_n=25",
        "scholl_heskia_n=28",
        "scholl_buxey_n=29",
        "scholl_lutz1_n=32",
        "scholl_gunther_n=35",
        "scholl_kilbrid_n=45",
        "scholl_hahn_n=53",
        "scholl_warnecke_n=58",
        "scholl_tonge70_n=70",
        "scholl_wee-mag_n=75",
        "scholl_lutz2_n=89",
        "scholl_lutz3_n=89",
        "scholl_arc83_n=83",
        "scholl_mukherje_n=94",
        "scholl_arc111_n=111",
    ]
}

# -- SALBP n=100 subset (10 instances) ----------------------------------------
SALBP100 = {
    f"salbp100_{i}": f"data/instances_base/salbp_instance_n=100_{i}.json"
    for i in [1, 10, 20, 21, 22, 23, 24, 25, 26, 27]
}

# -- MILP small instances (txt format, n ≤ 20) --------------------------------
MILP_SMALL = {
    "milp_mertens_n7"  : "data/instances_milp/scholl_mertens_n=7_selectif.txt",
    "milp_bowman8_n8"  : "data/instances_milp/scholl_bowman8_n=8_selectif.txt",
    "milp_jaeschke_n9" : "data/instances_milp/scholl_jaeschke_n=9_selectif.txt",
    "milp_jackson_n11" : "data/instances_milp/scholl_jackson_n=11_selectif.txt",
    "milp_mansoor_n11" : "data/instances_milp/scholl_mansoor_n=11_selectif.txt",
    "milp_mini_test"   : "data/instances_milp/mini_test_selectif.txt",
    "milp_dagtest"     : "data/instances_milp/dagtest_selectif.txt",
}

ALL_JSON = {}
ALL_JSON.update(INDUSTRIAL)
ALL_JSON.update(SCHOLL)
ALL_JSON.update(SALBP100)

# =============================================================================
# GRASP runner
# =============================================================================

def run_grasp_instance(name, json_path):
    """Run GRASP+VND on one JSON instance; return metrics dict."""
    row = {"instance": name, "solver": "GRASP+VND", "lambda": LAMBDA_DISCOUNT}
    if not os.path.exists(json_path):
        row["error"] = "file not found"
        return row

    try:
        G, has_zones, has_tau = load_adaptive_graph(json_path)
    except Exception as e:
        row["error"] = str(e)
        return row

    n = len(G.nodes())
    row["n"] = n

    # Determine targets and mode
    raw = json.load(open(json_path, encoding="utf-8"))
    targets = [str(t) for t in raw.get("targets", [])] if raw.get("targets") else None
    # Verify targets are in graph
    if targets:
        targets = [t for t in targets if t in G.nodes()]
    mode = "selectif" if targets else "complet"
    
    if targets:
        needed = closure_with_predecessors(G, targets)
        row["n_selected"] = len(needed)
        row["n_targets"]  = len(targets)
    else:
        row["n_selected"] = n
        row["n_targets"]  = n

    scores = []
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        try:
            seq, _ = run_grasp(G, algorithm="vnd", mode=mode, target_nodes=targets)
            if seq:
                scores.append(score_obj1(seq, G))
        except Exception:
            pass
    elapsed = time.perf_counter() - t0

    if not scores:
        row["error"] = "no valid solution"
        return row

    row["best_obj1"]   = round(max(scores), 4)
    row["avg_obj1"]    = round(float(np.mean(scores)), 4)
    row["std_obj1"]    = round(float(np.std(scores)), 4)
    row["runtime_s"]   = round(elapsed, 2)
    row["n_runs"]      = len(scores)
    row["status"]      = "OK"
    return row


# =============================================================================
# MILP runner
# =============================================================================

def run_milp_instance(name, txt_path):
    """Run MILP (CBC) on one txt instance; return metrics dict."""
    row = {"instance": name, "solver": "MILP-CBC", "lambda": LAMBDA_DISCOUNT}
    if not os.path.exists(txt_path):
        row["error"] = "file not found"
        return row

    try:
        data = load_data(txt_path)
    except Exception as e:
        row["error"] = f"load: {e}"
        return row

    n = len(data["V"])
    row["n"] = n

    if n > MAX_N_MILP:
        row["skipped"] = f"n={n} > MAX_N_MILP={MAX_N_MILP}"
        return row

    row["n_targets"]  = len(data.get("T", []))
    row["n_selected"] = n   # MILP selects a subset; full set is upper bound

    try:
        model, variables = build_model(data)
    except Exception as e:
        row["error"] = f"build: {e}"
        return row

    t0 = time.perf_counter()
    try:
        # CBC, no CPLEX required
        optimal, _ = solve_model(
            model,
            time_limit=MILP_TIMELIM,
            use_cplex=False,
            gap_limit=MILP_GAP
        )
    except Exception as e:
        row["error"] = f"solve: {e}"
        return row
    elapsed = time.perf_counter() - t0

    status   = pulp.LpStatus[model.status]
    obj_val  = pulp.value(model.objective)
    row["milp_status"]  = status
    row["milp_obj1"]    = round(obj_val, 4) if obj_val is not None else None
    row["runtime_s"]    = round(elapsed, 2)

    # Also extract sequence and verify score_obj1 is consistent
    # (MILP uses completion-time linearisation so values should match)
    row["status"] = "Optimal" if optimal else "Feasible/TimeLimit"
    return row


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  DSP BENCHMARK — Objective 1: Discounted Recovery Profit")
    print(f"  λ = {LAMBDA_DISCOUNT} EUR/s  |  GRASP runs = {N_RUNS}  |  MILP limit = {MILP_TIMELIM}s")
    print("=" * 70)

    results = []

    # ── GRASP on all JSON instances ──────────────────────────────────────────
    total_json = len(ALL_JSON)
    print(f"\n[GRASP+VND] {total_json} instances …")
    for idx, (name, path) in enumerate(ALL_JSON.items(), 1):
        print(f"  [{idx:3d}/{total_json}] {name:<40}", end="", flush=True)
        row = run_grasp_instance(name, path)
        status = row.get("status", row.get("error", row.get("skipped", "?")))
        best   = row.get("best_obj1", "—")
        rt     = row.get("runtime_s", "—")
        print(f"  Z*={best}  t={rt}s  [{status}]")
        results.append(row)

    # ── MILP on small txt instances ──────────────────────────────────────────
    total_milp = len(MILP_SMALL)
    print(f"\n[MILP-CBC]  {total_milp} small instances (n ≤ {MAX_N_MILP}) …")
    for idx, (name, path) in enumerate(MILP_SMALL.items(), 1):
        print(f"  [{idx:3d}/{total_milp}] {name:<40}", end="", flush=True)
        row = run_milp_instance(name, path)
        status = row.get("status", row.get("error", row.get("skipped", "?")))
        obj    = row.get("milp_obj1", "—")
        rt     = row.get("runtime_s", "—")
        print(f"  Z*={obj}  t={rt}s  [{status}]")
        results.append(row)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    # Reorder columns for readability
    col_order = [
        "instance", "solver", "n", "n_selected", "n_targets",
        "lambda",
        "best_obj1", "milp_obj1",
        "avg_obj1", "std_obj1",
        "milp_status", "runtime_s", "n_runs", "status", "error", "skipped"
    ]
    cols = [c for c in col_order if c in df.columns] + \
           [c for c in df.columns if c not in col_order]
    df = df[cols]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Results saved to: {OUTPUT_CSV}")
    print(f"  Rows: {len(df)}  |  GRASP: {(df.solver == 'GRASP+VND').sum()}  |  MILP: {(df.solver == 'MILP-CBC').sum()}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Summary (GRASP+VND best Z*) ────────────────────────────────────")
    grasp_df = df[df.solver == "GRASP+VND"][["instance", "n", "best_obj1", "runtime_s", "status"]].copy()
    grasp_df = grasp_df.sort_values("n")
    print(grasp_df.to_string(index=False))

    milp_df = df[df.solver == "MILP-CBC"]
    if not milp_df.empty:
        print("\n── Summary (MILP-CBC) ─────────────────────────────────────────────")
        print(milp_df[["instance", "n", "milp_obj1", "milp_status", "runtime_s"]].to_string(index=False))

    return OUTPUT_CSV


if __name__ == "__main__":
    out = main()
    print(f"\nDone. CSV: {out}")
