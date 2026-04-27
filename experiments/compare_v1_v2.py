#!/usr/bin/env python3
"""
Compare V1 (original) vs V2 (enhanced) adaptive fuzzy pipeline.
Runs all scenarios and produces a side-by-side comparison CSV.
"""
import os, sys, json, glob, time, csv, traceback
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.grasp.constructive import run_grasp
from src.adaptive import overlay
from src.adaptive.scheduler import execute_adaptive_sequence        # V1
from src.adaptive.scheduler_v2 import execute_adaptive_sequence_v2  # V2
from src.utils.graph_io import load_adaptive_graph
from src.utils.metrics import compute_economic_metrics, compute_time_metrics, score_weighted

FAILURES_DIR = "data/adaptive/generated_failures"
RESULTS_CSV  = "results/fuzzy_v1_vs_v2.csv"


def classify_scenario(filename):
    base = os.path.basename(filename).replace(".json", "")
    for lvl in ("simple", "intermediate", "complex"):
        if lvl in base:
            return lvl
    if "extreme" in base:
        return "extreme"
    for stress in ("stress_10", "stress_8", "stress_6", "stress_4", "stress", "cascade"):
        if stress in base:
            return stress
    return base


def run_single(instance_name, scenario_path, use_v2=False):
    """Run one scenario with V1 or V2 scheduler."""
    instance_path = f"data/instances_base/{instance_name}.json"
    if not os.path.exists(instance_path):
        return {"error": f"Instance not found: {instance_path}"}

    G, *_ = load_adaptive_graph(instance_path)
    with open(instance_path, 'r') as f:
        inst_data = json.load(f)
    targets = inst_data.get("targets", list(G.nodes()))

    # Solve GRASP+VND
    t0 = time.perf_counter()
    solution, score = run_grasp(G, algorithm='vnd', mode='selectif', target_nodes=targets)
    grasp_time = time.perf_counter() - t0

    # Load failures
    with open(scenario_path, 'r') as f:
        failures = json.load(f)

    # Init state (fresh copy of graph for each run)
    G2, *_ = load_adaptive_graph(instance_path)
    state = overlay.init_state()
    state['graph'] = G2
    state['G'] = G2
    state['targets'] = targets

    # Execute
    t0 = time.perf_counter()
    if use_v2:
        state, actions_log = execute_adaptive_sequence_v2(
            solution=solution, state=state, failures=failures, verbose=False
        )
    else:
        state, actions_log = execute_adaptive_sequence(
            solution=solution, state=state, failures=failures, verbose=False
        )
    adapt_time = time.perf_counter() - t0

    # Metrics
    eco = compute_economic_metrics(G2, solution,
                                   state.get('done', set()),
                                   state.get('destroyed', set()),
                                   state.get('skipped', set()))

    targets_achieved = [t for t in targets
                        if t in state.get('done', set()) and t not in state.get('destroyed', set())]
    success_rate = len(targets_achieved) / len(targets) * 100 if targets else 0

    action_counts = Counter()
    for a in actions_log:
        act = a.get("action", "unknown")
        action_counts[act] += 1

    return {
        "instance": instance_name,
        "n": len(G2.nodes()),
        "scenario_file": os.path.basename(scenario_path),
        "scenario_type": classify_scenario(scenario_path),
        "n_failures": len(failures),
        "n_actions": len(actions_log),
        "target_total": len(targets),
        "target_achieved": len(targets_achieved),
        "target_recovery_pct": round(success_rate, 1),
        "nominal_profit": round(eco['nominal_profit'], 2),
        "adaptive_profit": round(eco['adaptive_profit'], 2),
        "profit_loss_pct": round(eco['profit_loss_pct'], 1),
        "n_bypass": action_counts.get("bypass", 0),
        "n_change_tool": action_counts.get("change_tool", 0),
        "n_destroy": action_counts.get("destroy", 0),
        "n_replan": action_counts.get("replan", 0),
        "n_force_destroy": action_counts.get("force_destroy", 0),
        "n_fallback": sum(1 for a in actions_log if "->" in a.get("action", "") or "→" in a.get("action", "")),
        "n_destroyed_comps": len(state.get('destroyed', set())),
        "grasp_time": round(grasp_time, 3),
        "adapt_time": round(adapt_time, 3),
        "error": None
    }


def severity(stype):
    """Map scenario type to severity level."""
    if stype == 'simple':
        return 'simple'
    elif stype == 'intermediate':
        return 'intermediate'
    elif stype == 'complex':
        return 'complex'
    else:
        return 'high_stress'


def main():
    # Collect all (instance, scenario) pairs
    pairs = []
    for instance_dir in sorted(glob.glob(os.path.join(FAILURES_DIR, "*"))):
        if not os.path.isdir(instance_dir):
            continue
        instance_name = os.path.basename(instance_dir)
        for scenario_file in sorted(glob.glob(os.path.join(instance_dir, "*.json"))):
            pairs.append((instance_name, scenario_file))

    print(f"╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  FUZZY PIPELINE COMPARISON: V1 (original) vs V2 (enhanced)     ║")
    print(f"║  {len(pairs)} scenario(s) across {len(set(p[0] for p in pairs))} instance(s)                              ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝\n")

    results = []

    for idx, (inst, scen) in enumerate(pairs, 1):
        stype = classify_scenario(scen)
        print(f"[{idx:3d}/{len(pairs)}] {inst:40s} | {stype:15s}")

        # Run V1
        try:
            print(f"        V1: ", end="", flush=True)
            r1 = run_single(inst, scen, use_v2=False)
            if r1.get("error"):
                print(f"ERROR: {r1['error']}")
            else:
                print(f"rec={r1['target_recovery_pct']:5.1f}%  loss={r1['profit_loss_pct']:5.1f}%  t={r1['adapt_time']:.2f}s")
        except Exception as e:
            print(f"EXCEPTION: {e}")
            r1 = {"error": str(e)}

        # Run V2
        try:
            print(f"        V2: ", end="", flush=True)
            r2 = run_single(inst, scen, use_v2=True)
            if r2.get("error"):
                print(f"ERROR: {r2['error']}")
            else:
                print(f"rec={r2['target_recovery_pct']:5.1f}%  loss={r2['profit_loss_pct']:5.1f}%  t={r2['adapt_time']:.2f}s")
        except Exception as e:
            print(f"EXCEPTION: {e}")
            r2 = {"error": str(e)}

        # Delta
        if not r1.get("error") and not r2.get("error"):
            delta_rec = r2['target_recovery_pct'] - r1['target_recovery_pct']
            delta_loss = r2['profit_loss_pct'] - r1['profit_loss_pct']
            marker = "IMPROVED" if delta_rec > 0 else ("SAME" if delta_rec == 0 else "WORSE")
            print(f"        --> Δrec={delta_rec:+.1f}%  Δloss={delta_loss:+.1f}%  [{marker}]")

        results.append({
            "instance": inst,
            "scenario_file": os.path.basename(scen),
            "scenario_type": stype,
            "severity": severity(stype),
            "n_failures": r1.get("n_failures", r2.get("n_failures", 0)),
            # V1
            "v1_recovery": r1.get("target_recovery_pct", None),
            "v1_profit_loss": r1.get("profit_loss_pct", None),
            "v1_actions": r1.get("n_actions", None),
            "v1_destroyed": r1.get("n_destroyed_comps", None),
            "v1_time": r1.get("adapt_time", None),
            # V2
            "v2_recovery": r2.get("target_recovery_pct", None),
            "v2_profit_loss": r2.get("profit_loss_pct", None),
            "v2_actions": r2.get("n_actions", None),
            "v2_destroyed": r2.get("n_destroyed_comps", None),
            "v2_force_destroy": r2.get("n_force_destroy", None),
            "v2_time": r2.get("adapt_time", None),
            # Errors
            "v1_error": r1.get("error"),
            "v2_error": r2.get("error"),
        })
        print()

    # Write CSV
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    fieldnames = list(results[0].keys()) if results else []
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\n✅ Results saved to {RESULTS_CSV}")

    # Summary
    valid = [r for r in results if r.get('v1_recovery') is not None and r.get('v2_recovery') is not None]
    if not valid:
        print("No valid results!")
        return

    print(f"\n{'='*75}")
    print(f"  SUMMARY — {len(valid)} scenarios")
    print(f"{'='*75}")

    # By severity
    by_sev = defaultdict(list)
    for r in valid:
        by_sev[r['severity']].append(r)

    print(f"\n{'Severity':15s} | {'Count':5s} | {'V1 rec%':8s} | {'V2 rec%':8s} | {'Δrec':8s} | {'V1 loss%':9s} | {'V2 loss%':9s}")
    print("-" * 80)

    for sev in ["simple", "intermediate", "complex", "high_stress"]:
        rows = by_sev.get(sev, [])
        if not rows:
            continue
        v1_rec = sum(r['v1_recovery'] for r in rows) / len(rows)
        v2_rec = sum(r['v2_recovery'] for r in rows) / len(rows)
        v1_loss = sum(r['v1_profit_loss'] for r in rows) / len(rows)
        v2_loss = sum(r['v2_profit_loss'] for r in rows) / len(rows)
        delta = v2_rec - v1_rec
        print(f"{sev:15s} | {len(rows):5d} | {v1_rec:7.1f}% | {v2_rec:7.1f}% | {delta:+7.1f}% | {v1_loss:8.1f}% | {v2_loss:8.1f}%")

    # Overall
    v1_rec_all = sum(r['v1_recovery'] for r in valid) / len(valid)
    v2_rec_all = sum(r['v2_recovery'] for r in valid) / len(valid)
    v1_loss_all = sum(r['v1_profit_loss'] for r in valid) / len(valid)
    v2_loss_all = sum(r['v2_profit_loss'] for r in valid) / len(valid)
    delta_all = v2_rec_all - v1_rec_all
    print("-" * 80)
    print(f"{'OVERALL':15s} | {len(valid):5d} | {v1_rec_all:7.1f}% | {v2_rec_all:7.1f}% | {delta_all:+7.1f}% | {v1_loss_all:8.1f}% | {v2_loss_all:8.1f}%")

    # Full recovery comparison
    v1_full = sum(1 for r in valid if r['v1_recovery'] == 100.0)
    v2_full = sum(1 for r in valid if r['v2_recovery'] == 100.0)
    print(f"\n  Full recovery (100%): V1={v1_full}/{len(valid)} ({v1_full/len(valid)*100:.1f}%)  V2={v2_full}/{len(valid)} ({v2_full/len(valid)*100:.1f}%)")

    # HS-specific
    hs = by_sev.get('high_stress', [])
    if hs:
        v1_hs_full = sum(1 for r in hs if r['v1_recovery'] == 100.0)
        v2_hs_full = sum(1 for r in hs if r['v2_recovery'] == 100.0)
        v2_force = sum(r.get('v2_force_destroy', 0) or 0 for r in hs)
        print(f"  HS full recovery: V1={v1_hs_full}/{len(hs)} ({v1_hs_full/len(hs)*100:.1f}%)  V2={v2_hs_full}/{len(hs)} ({v2_hs_full/len(hs)*100:.1f}%)")
        print(f"  V2 force-destroys in HS: {v2_force}")

    # Improved / Same / Worse
    improved = sum(1 for r in valid if r['v2_recovery'] > r['v1_recovery'])
    same = sum(1 for r in valid if r['v2_recovery'] == r['v1_recovery'])
    worse = sum(1 for r in valid if r['v2_recovery'] < r['v1_recovery'])
    print(f"\n  V2 vs V1: {improved} improved, {same} same, {worse} worse")

    print(f"\n{'='*75}")
    print(f"  Done.")


if __name__ == "__main__":
    main()
