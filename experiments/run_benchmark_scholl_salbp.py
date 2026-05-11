"""run_benchmark_scholl_salbp.py
3-way benchmark: GRASP+VND vs CPLEX (MILP) vs CP-SAT (OR-Tools)
on Scholl and SALBP instances, all on the slide-9 time-aware objective:
    max Σ v_i · x_i  −  λ · Σ C_i

All three solvers load from the same TXT file via load_graph_from_txt(),
so profits/costs/times are identical. λ is auto-calibrated per instance
(5% of mean net-value rate). The MILP and CP-SAT share a uniform 180s
budget per instance.

Per-instance output (results/benchmark_2026/grasp_vs_milp/<label>/):
  - results.csv      : 1-line summary with all GRASP/CPLEX/CP-SAT columns
  - terminal.log     : human-readable, 3 sections + cross-gaps
  - compare.json     : structured (grasp/cplex/cpsat/comparison)
  - validation.json  : full row + parameters
  - solver_cplex.log : CPLEX solver log (if any)
  - solver_cpsat.log : CP-SAT solver log (mirror of cpsat_logs/<label>.log)

Global summary: RESUME_GRASP_MILP.csv (kept under this name for back-compat).
"""
import sys, os, time, glob, math, json, argparse
from datetime import datetime, timedelta
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pulp
from src.utils.graph_io import load_graph_from_txt
from src.grasp.constructive import run_grasp
from src.utils.metrics import score_obj1, prune_sequence
from src.exact_milp_dijkstra.partial_disassembly_model import (
    load_data, build_model, solve_model, extract_sequence,
    _parse_last_status_from_cplex_log, _parse_last_gap_from_cplex_log
)
from src.exact_cpsat.partial_disassembly_cpsat import (
    build_model_cpsat, solve_cpsat,
)

def _bar(done, total, width=20):
    filled = int(width * done / max(total, 1))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {done}/{total}"

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
MILP_DIR    = "data/instances_milp"
RESULTS_DIR = "results"

GRASP_RUNS  = 1         # independent GRASP restarts per instance
GRASP_ITERS = 150       # max reactive-GRASP iterations per run
GRASP_TIME  = 900       # VND time budget per run (seconds) — 15 min

# Uniform exact-solver time limit, applied to both CPLEX and CP-SAT.
# Decision (Amine, 2026-05-06): same budget on every instance is cleaner
# scientifically than an adaptive split — the comparison is "what each
# solver finds in 180s, regardless of instance size". The empirical
# cutoff (n ≈ 25-35 closes, larger doesn't) emerges from the data, not
# from how we sized the budget.
EXACT_TIMELIM = 180
EXACT_TIMELIM_SMALL = EXACT_TIMELIM
EXACT_TIMELIM_LARGE = EXACT_TIMELIM
EXACT_SIZE_THRESHOLD = 100

def exact_timeout(n: int) -> int:
    return EXACT_TIMELIM_SMALL if n <= EXACT_SIZE_THRESHOLD else EXACT_TIMELIM_LARGE

# Backwards-compatible aliases (older scripts may import MILP_TIMELIM_*).
MILP_TIMELIM_SMALL  = EXACT_TIMELIM_SMALL
MILP_TIMELIM_LARGE  = EXACT_TIMELIM_LARGE
MILP_SIZE_THRESHOLD = EXACT_SIZE_THRESHOLD
def milp_timeout(n: int) -> int:
    return exact_timeout(n)

MILP_GAP   = 0.01     # 1% relative gap (stop when gap ≤ 1%)
MAX_N_MILP = 1000     # allow MILP on all instances; it will timeout/gap naturally

# ─────────────────────────────────────────────
#  Instance selection
# ─────────────────────────────────────────────
def collect_instances():
    """Return list of (label, txt_path, n) sorted by size (ascending)."""
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
            instances.append((label, p, n))

    # Sort by size ascending (smallest first)
    instances.sort(key=lambda x: x[2])
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
    best_seq_pruned = None
    best_score_val = -float('inf')

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

        # Keep best sequence
        if s > best_score_val:
            best_score_val = s
            best_seq_pruned = seq_pruned

    elapsed = time.perf_counter() - t0

    best_score = max(scores)
    avg_score  = sum(scores) / len(scores)
    std_score  = math.sqrt(sum((s - avg_score)**2 for s in scores) / len(scores))

    # Return sequence and targets as well
    return best_score, avg_score, std_score, lam, n, elapsed, best_seq_pruned, targets


# ─────────────────────────────────────────────
#  MILP runner
# ─────────────────────────────────────────────
def run_milp_on_txt(txt_path, label=""):
    """CPLEX runner on the time-aware objective (slide 9)."""
    data = load_data(txt_path)
    n    = len(data['V'])
    if n > MAX_N_MILP:
        return None, None, None, n, f"n={n} > MAX_N_MILP={MAX_N_MILP}", None, "", []
    try:
        model, variables = build_model(data, use_time_aware=True)
        lam_milp = data.get('lambda_discount', 0.05)
        tl = exact_timeout(n)
        t0 = time.perf_counter()
        optimal, log_file = solve_model(
            model, time_limit=tl, use_cplex=True, gap_limit=MILP_GAP,
            instance_name=label
        )
        elapsed = time.perf_counter() - t0
        # Use real CPLEX log status (PuLP wrongly reports 'Optimal' on TimeLimit)
        status = _parse_last_status_from_cplex_log(log_file)
        if status == 'Unknown':
            status = pulp.LpStatus[model.status]  # fallback
        obj    = pulp.value(model.objective)
        cplex_gap = _parse_last_gap_from_cplex_log(log_file)

        # Extract MILP sequence
        milp_seq = []
        try:
            milp_seq, _ = extract_sequence(model, data, variables)
        except:
            milp_seq = []

        # Read log file content
        log_content = ""
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    log_content = f.read()
            except:
                log_content = ""

        return obj, status, elapsed, n, lam_milp, cplex_gap, log_content, milp_seq
    except Exception as e:
        return None, f"ERROR: {e}", None, n, None, None, "", []


# ─────────────────────────────────────────────
#  CP-SAT runner (time-aware objective via OR-Tools)
# ─────────────────────────────────────────────
def run_cpsat_on_txt(txt_path, label="", outdir=None):
    """CP-SAT runner on the time-aware objective (slide 9), via OR-Tools."""
    data = load_data(txt_path)
    n    = len(data['V'])
    try:
        log_path = None
        if outdir is not None:
            log_dir = os.path.join(outdir, "cpsat_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"cpsat_{label}.log")
        model, vars_ = build_model_cpsat(data)
        result = solve_cpsat(
            model, vars_,
            time_limit=exact_timeout(n),
            log_path=log_path,
            instance_name=label,
        )
        return {
            'obj':        result['obj'],
            'bound':      result['bound'],
            'status':     result['status'],
            'gap_pct':    result['gap_pct'],
            'time_s':     result['solve_time'],
            'sequence':   result['sequence'],
            'lam':        vars_['lam'],
            'log_path':   result['log_path'],
            'n':          n,
        }
    except Exception as e:
        return {
            'obj': None, 'bound': None, 'status': f"ERROR: {e}", 'gap_pct': None,
            'time_s': None, 'sequence': [], 'lam': None, 'log_path': None, 'n': n,
        }


# ─────────────────────────────────────────────
#  Per-instance result saving
# ─────────────────────────────────────────────
def save_instance_results(instance_dir, row_data, terminal_output="", milp_log_content="",
                          grasp_seq=None, targets=None, label="", milp_seq=None,
                          cpsat_seq=None, cpsat_log_path=None):
    """Save results for single instance to subdirectory."""
    os.makedirs(instance_dir, exist_ok=True)

    # Save results.csv
    results_csv = os.path.join(instance_dir, "results.csv")
    df = pd.DataFrame([row_data])
    df.to_csv(results_csv, index=False)

    # Save terminal.log
    terminal_log = os.path.join(instance_dir, "terminal.log")
    with open(terminal_log, 'w') as f:
        f.write(terminal_output)

    # Save validation.json (full details for debugging)
    validation_json = os.path.join(instance_dir, "validation.json")
    validation_data = {
        "timestamp": datetime.now().isoformat(),
        "instance": row_data.get("instance", ""),
        "parameters": {
            "GRASP_RUNS": GRASP_RUNS,
            "GRASP_ITERS": GRASP_ITERS,
            "GRASP_TIME": GRASP_TIME,
            "EXACT_TIMELIM": EXACT_TIMELIM,
            "MILP_GAP": MILP_GAP,
            "MAX_N_MILP": MAX_N_MILP,
        },
        "results": row_data
    }
    with open(validation_json, 'w') as f:
        json.dump(validation_data, f, indent=2)

    # Save compare.json (3-way comparison with sequences)
    compare_json = os.path.join(instance_dir, "compare.json")
    compare_data = {
        "instance": label,
        "targets": targets if targets else [],
        "grasp": {
            "best_score": row_data.get("grasp_best"),
            "avg_score":  row_data.get("grasp_avg"),
            "std_score":  row_data.get("grasp_std"),
            "time_s":     row_data.get("grasp_time_s"),
            "sequence":   grasp_seq if grasp_seq else []
        },
        "cplex": {
            "objective":  row_data.get("cplex_obj"),
            "status":     row_data.get("cplex_status"),
            "time_s":     row_data.get("cplex_time_s"),
            "intgap_pct": row_data.get("cplex_intgap_pct"),
            "sequence":   milp_seq if milp_seq else []
        },
        "cpsat": {
            "objective":  row_data.get("cpsat_obj"),
            "bound":      row_data.get("cpsat_bound"),
            "status":     row_data.get("cpsat_status"),
            "time_s":     row_data.get("cpsat_time_s"),
            "intgap_pct": row_data.get("cpsat_intgap_pct"),
            "sequence":   cpsat_seq if cpsat_seq else []
        },
        "comparison": {
            "gap_grasp_vs_cplex_pct": row_data.get("gap_grasp_vs_cplex"),
            "gap_grasp_vs_cpsat_pct": row_data.get("gap_grasp_vs_cpsat"),
        }
    }
    with open(compare_json, 'w') as f:
        json.dump(compare_data, f, indent=2)

    # Save CPLEX solver log if available
    if milp_log_content:
        with open(os.path.join(instance_dir, "solver_cplex.log"), 'w') as f:
            f.write(milp_log_content)

    # Mirror CP-SAT log inside the per-instance dir (it's also kept under cpsat_logs/)
    if cpsat_log_path and os.path.exists(cpsat_log_path):
        try:
            with open(cpsat_log_path, 'r', encoding='utf-8', errors='replace') as fin, \
                 open(os.path.join(instance_dir, "solver_cpsat.log"), 'w', encoding='utf-8') as fout:
                fout.write(fin.read())
        except Exception:
            pass


def compile_resume(outdir):
    """Compile all per-instance results.csv into RESUME_GRASP_MILP.csv."""
    resume_path = os.path.join(outdir, "RESUME_GRASP_MILP.csv")

    all_rows = []
    for instance_subdir in sorted(os.listdir(outdir)):
        instance_path = os.path.join(outdir, instance_subdir)
        if not os.path.isdir(instance_path):
            continue

        results_csv = os.path.join(instance_path, "results.csv")
        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
            all_rows.append(df)

    if all_rows:
        resume_df = pd.concat(all_rows, ignore_index=True)
        resume_df.to_csv(resume_path, index=False)
        return resume_df
    return pd.DataFrame()


# ─────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────
def main(outdir=None):
    if outdir is None:
        outdir = os.path.join(RESULTS_DIR, "benchmark_2026", "grasp_vs_milp")

    os.makedirs(outdir, exist_ok=True)

    instances = collect_instances()
    total = len(instances)
    t_start_all = time.perf_counter()
    print(f"\n{'='*70}")
    print(f"  BENCHMARK  |  {total} instances (sorted by size)")
    print(f"  Solvers    : GRASP+VND  ·  CPLEX (MILP)  ·  CP-SAT (OR-Tools)")
    print(f"  Objective  : max Σ v_i·x_i − λ·Σ C_i  (slide-9 time-aware)")
    print(f"  Exact tlim : {EXACT_TIMELIM}s per instance (CPLEX & CP-SAT), gap {MILP_GAP*100:.1f}%")
    print(f"  Output dir : {outdir}")
    print(f"  Started    :  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    sys.stdout.flush()

    all_rows = []
    for idx, (label, txt_path, n) in enumerate(instances, 1):
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
        print(f"  ► [{idx:3d}/{total}] {label} (n={n})", flush=True)

        # ── GRASP ──
        print(f"    GRASP ({GRASP_RUNS} runs)...", end="", flush=True)
        best_g, avg_g, std_g, lam, _, grasp_time, grasp_seq, targets = run_grasp_on_txt(txt_path)
        print(f"  [OK]  best={best_g:.2f}  avg={avg_g:.2f}±{std_g:.2f}  lam={lam:.4f}  t={grasp_time:.1f}s", flush=True)

        # ── CPLEX (time-aware MILP, slide-9 objective) ──
        if n <= MAX_N_MILP:
            print(f"    CPLEX  ({exact_timeout(n)}s, gap={MILP_GAP*100:.1f}%)...", end="", flush=True)

        milp_obj, milp_status, milp_time, _, _, cplex_gap, milp_log_content, milp_seq = run_milp_on_txt(txt_path, label=label)

        if milp_obj is not None:
            cplex_vs_grasp = ((milp_obj - best_g) / abs(milp_obj) * 100.0
                              if best_g is not None and milp_obj != 0 else None)
            cv_str = f"{cplex_vs_grasp:.1f}%" if cplex_vs_grasp is not None else "N/A"
            cplex_g_str = f"cplex_gap={cplex_gap:.1f}%" if cplex_gap is not None else ""
            print(f"  obj={milp_obj:.2f}  ({milp_status}, {milp_time:.1f}s, {cplex_g_str})  gap_vs_grasp={cv_str}", flush=True)
        elif n > MAX_N_MILP:
            print(f"  [SKIP] CPLEX skipped (n={n} > MAX_N_MILP={MAX_N_MILP})", flush=True)
            cplex_vs_grasp = None
        else:
            print(f"  [ERROR] ({milp_status})", flush=True)
            cplex_vs_grasp = None

        # ── CP-SAT (time-aware, OR-Tools constraint programming) ──
        print(f"    CP-SAT ({exact_timeout(n)}s)...", end="", flush=True)
        cpsat = run_cpsat_on_txt(txt_path, label=label, outdir=outdir)
        cpsat_obj    = cpsat['obj']
        cpsat_bound  = cpsat['bound']
        cpsat_status = cpsat['status']
        cpsat_time   = cpsat['time_s']
        cpsat_intgap = cpsat['gap_pct']
        cpsat_seq    = cpsat['sequence']

        if cpsat_obj is not None:
            cpsat_vs_grasp = ((cpsat_obj - best_g) / abs(cpsat_obj) * 100.0
                              if best_g is not None and cpsat_obj != 0 else None)
            cv2 = f"{cpsat_vs_grasp:.1f}%" if cpsat_vs_grasp is not None else "N/A"
            cig = f"cpsat_gap={cpsat_intgap:.1f}%" if cpsat_intgap is not None else ""
            print(f"  obj={cpsat_obj:.2f}  ({cpsat_status}, {cpsat_time:.1f}s, {cig})  gap_vs_grasp={cv2}", flush=True)
        else:
            print(f"  [ERROR] ({cpsat_status})", flush=True)
            cpsat_vs_grasp = None

        # Cross-solver agreement on small instances (both Optimal → must match)
        if (milp_obj is not None and cpsat_obj is not None
                and milp_status == 'Optimal' and cpsat_status == 'OPTIMAL'):
            mismatch = abs(milp_obj - cpsat_obj) / max(abs(milp_obj), 1e-6) * 100.0
            if mismatch > 1.0:
                print(f"    [WARN] CPLEX/CP-SAT optimum mismatch: {mismatch:.2f}% (rounding from CP-SAT integer scale)", flush=True)

        inst_time = time.perf_counter() - t_inst
        print(f"    Instance done in {inst_time:.1f}s", flush=True)

        # Build row data
        row_data = {
            "instance":         label,
            "n":                n,
            "lambda":           round(lam, 5) if lam else None,
            # GRASP
            "grasp_best":       round(best_g, 4) if best_g is not None else None,
            "grasp_avg":        round(avg_g, 4) if avg_g is not None else None,
            "grasp_std":        round(std_g, 4) if std_g is not None else None,
            "grasp_time_s":     round(grasp_time, 2) if grasp_time is not None else None,
            # CPLEX (MILP, time-aware)
            "cplex_obj":        round(milp_obj, 4) if milp_obj is not None else None,
            "cplex_status":     milp_status,
            "cplex_time_s":     round(milp_time, 2) if milp_time is not None else None,
            "cplex_intgap_pct": round(cplex_gap, 2) if cplex_gap is not None else None,
            "gap_grasp_vs_cplex": round(cplex_vs_grasp, 2) if cplex_vs_grasp is not None else None,
            # CP-SAT (constraint programming, time-aware)
            "cpsat_obj":        round(cpsat_obj, 4) if cpsat_obj is not None else None,
            "cpsat_bound":      round(cpsat_bound, 4) if cpsat_bound is not None else None,
            "cpsat_status":     cpsat_status,
            "cpsat_time_s":     round(cpsat_time, 2) if cpsat_time is not None else None,
            "cpsat_intgap_pct": round(cpsat_intgap, 2) if cpsat_intgap is not None else None,
            "gap_grasp_vs_cpsat": round(cpsat_vs_grasp, 2) if cpsat_vs_grasp is not None else None,
        }
        # Backwards-compat aliases for older readers
        row_data["milp_obj"]    = row_data["cplex_obj"]
        row_data["milp_status"] = row_data["cplex_status"]
        row_data["milp_time_s"] = row_data["cplex_time_s"]
        row_data["gap_pct"]     = row_data["gap_grasp_vs_cplex"]

        # Save per-instance results
        instance_dir = os.path.join(outdir, label)
        milp_time_str  = f"{milp_time:.2f}s"  if milp_time  is not None else "N/A"
        cpsat_time_str = f"{cpsat_time:.2f}s" if cpsat_time is not None else "N/A"

        # Format sequences for display
        grasp_seq_str = " -> ".join(map(str, grasp_seq)) if grasp_seq else "N/A"
        milp_seq_str  = " -> ".join(map(str, milp_seq))  if milp_seq  else "N/A"
        cpsat_seq_str = " -> ".join(map(str, cpsat_seq)) if cpsat_seq else "N/A"
        targets_str   = ", ".join(map(str, targets))     if targets   else "none"
        cplex_gap_str = f"{cplex_gap:.2f}%"   if cplex_gap   is not None else "N/A"
        cpsat_gap_str = f"{cpsat_intgap:.2f}%" if cpsat_intgap is not None else "N/A"
        gvc_str       = f"{cplex_vs_grasp:+.2f}%" if cplex_vs_grasp is not None else "N/A"
        gvs_str       = f"{cpsat_vs_grasp:+.2f}%" if cpsat_vs_grasp is not None else "N/A"

        terminal_log = f"""Instance: {label}
Nodes: {n}
Targets (mandatory): {targets_str}
Lambda (time discount): {lam}

GRASP+VND (score={best_g:.4f}, time={grasp_time:.2f}s):
  {grasp_seq_str}

CPLEX (obj={milp_obj}, status={milp_status}, time={milp_time_str}, intgap={cplex_gap_str}):
  {milp_seq_str}

CP-SAT (obj={cpsat_obj}, bound={cpsat_bound}, status={cpsat_status}, time={cpsat_time_str}, intgap={cpsat_gap_str}):
  {cpsat_seq_str}

GRASP gap vs CPLEX:  {gvc_str}
GRASP gap vs CP-SAT: {gvs_str}

Total instance time: {inst_time:.1f}s
"""
        save_instance_results(instance_dir, row_data, terminal_log, milp_log_content,
                              grasp_seq, targets, label, milp_seq,
                              cpsat_seq=cpsat_seq, cpsat_log_path=cpsat.get('log_path'))
        all_rows.append(row_data)

    # Compile resume
    resume_df = compile_resume(outdir)
    resume_path = os.path.join(outdir, "RESUME_GRASP_MILP.csv")

    total_time = time.perf_counter() - t_start_all
    print(f"\n{'='*70}")
    print(f"  ✓ Saved → {outdir}")
    print(f"  ✓ Resume → {resume_path}  ({len(resume_df)} instances)")
    print(f"  Total runtime: {str(timedelta(seconds=int(total_time)))}")
    print(f"{'='*70}")
    sys.stdout.flush()

    # ── Summary: instances with both exact solvers ──
    comp = resume_df.dropna(subset=["cplex_obj", "grasp_best"])
    if not comp.empty:
        cols = ["instance","n","lambda","grasp_best",
                "cplex_obj","cplex_status","cplex_time_s","gap_grasp_vs_cplex",
                "cpsat_obj","cpsat_status","cpsat_time_s","gap_grasp_vs_cpsat"]
        cols = [c for c in cols if c in comp.columns]
        print(f"\n── GRASP+VND vs CPLEX vs CP-SAT ─────────────────────────────────────")
        print(comp[cols].to_string(index=False))
        if "gap_grasp_vs_cplex" in comp.columns:
            print(f"\n  GRASP vs CPLEX  — avg gap: {comp['gap_grasp_vs_cplex'].mean():.2f} %  |  max: {comp['gap_grasp_vs_cplex'].max():.2f} %")
        if "gap_grasp_vs_cpsat" in comp.columns:
            print(f"  GRASP vs CP-SAT — avg gap: {comp['gap_grasp_vs_cpsat'].mean():.2f} %  |  max: {comp['gap_grasp_vs_cpsat'].max():.2f} %")

    print(f"\n── GRASP only (n > {MAX_N_MILP}) ─────────────────────────────────────────")
    large = resume_df[resume_df["n"] > MAX_N_MILP]
    if not large.empty:
        print(large[["instance","n","lambda","grasp_best","grasp_time_s"]].to_string(index=False))
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GRASP+VND vs MILP")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory (default: results/benchmark_2026/grasp_vs_milp)")
    args = parser.parse_args()
    main(outdir=args.outdir)
