#!/usr/bin/env python3
"""
Run ALL adaptive fuzzy scenarios across all instances.
Collects: instance, scenario_type, n_failures, n_actions, target_recovery,
          profit_loss_pct, time_diff_pct, score_loss_pct, action_breakdown.
"""
import os, sys, json, glob, time, csv, traceback
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.grasp.constructive import run_grasp
from src.adaptive import overlay
from src.adaptive.scheduler import execute_adaptive_sequence
from src.utils.graph_io import load_adaptive_graph
from src.utils.metrics import compute_economic_metrics, compute_time_metrics, score_weighted

FAILURES_DIR = "data/adaptive/generated_failures"
RESULTS_CSV  = "results/fuzzy_all_scenarios.csv"

# ── helpers ──────────────────────────────────────────────────────────

def classify_scenario(filename):
    """Return scenario type."""
    base = os.path.basename(filename).replace(".json", "")
    for lvl in ("simple", "intermediate", "complex"):
        if lvl in base:
            return lvl
    # Must check "extreme" before "stress" to avoid mis-classifying "extreme" as generic
    if "extreme" in base:
        return "extreme"
    for stress in ("stress_10", "stress_8", "stress_6", "stress_4", "stress", "cascade"):
        if stress in base:
            return stress
    return base  # demo, failures_1, etc.

def count_failures_in_scenario(failures):
    return len(failures)

# ── core runner ──────────────────────────────────────────────────────

def run_single(instance_name, scenario_path, verbose=False):
    """
    Run one instance + one failure scenario, return summary dict.
    """
    # Locate instance JSON
    instance_path = f"data/instances_base/{instance_name}.json"
    if not os.path.exists(instance_path):
        return {"error": f"Instance not found: {instance_path}"}

    # Load graph + targets
    G, *_ = load_adaptive_graph(instance_path)
    with open(instance_path, 'r') as f:
        inst_data = json.load(f)
    targets = inst_data.get("targets", list(G.nodes()))

    # Solve GRASP+VND (selective)
    t0 = time.perf_counter()
    solution, score = run_grasp(G, algorithm='vnd', mode='selectif', target_nodes=targets)
    grasp_time = time.perf_counter() - t0

    # Load failures
    with open(scenario_path, 'r') as f:
        failures = json.load(f)

    # Init state
    state = overlay.init_state()
    state['graph'] = G
    state['G'] = G
    state['targets'] = targets

    # Execute adaptive scheduler
    t0 = time.perf_counter()
    state, actions_log = execute_adaptive_sequence(
        solution=solution, state=state, failures=failures, verbose=verbose
    )
    adapt_time = time.perf_counter() - t0

    # Metrics
    eco = compute_economic_metrics(G, solution,
                                   state.get('done', set()),
                                   state.get('destroyed', set()),
                                   state.get('skipped', set()))

    tm = compute_time_metrics(G, solution,
                              state.get('done', set()),
                              state.get('destroyed', set()),
                              state.get('skipped', set()))

    nominal_score = score_weighted(solution, G)
    adaptive_seq = [c for c in solution
                    if c in state.get('done', set()) and c not in state.get('destroyed', set())]
    adaptive_score_val = score_weighted(adaptive_seq, G)

    targets_achieved = [t for t in targets
                        if t in state.get('done', set()) and t not in state.get('destroyed', set())]
    success_rate = len(targets_achieved) / len(targets) * 100 if targets else 0

    # Action breakdown
    action_counts = Counter()
    for a in actions_log:
        act = a.get("action", "unknown")
        action_counts[act] += 1

    score_loss_pct = ((nominal_score - adaptive_score_val) / nominal_score * 100
                      if nominal_score > 0 else 0.0)

    return {
        "instance": instance_name,
        "n": len(G.nodes()),
        "scenario_file": os.path.basename(scenario_path),
        "scenario_type": classify_scenario(scenario_path),
        "n_failures": count_failures_in_scenario(failures),
        "n_actions": len(actions_log),
        "target_total": len(targets),
        "target_achieved": len(targets_achieved),
        "target_recovery_pct": round(success_rate, 1),
        "nominal_profit": round(eco['nominal_profit'], 2),
        "adaptive_profit": round(eco['adaptive_profit'], 2),
        "profit_loss_pct": round(eco['profit_loss_pct'], 1),
        "nominal_time": round(tm['nominal_time'], 1),
        "adaptive_time": round(tm['adaptive_time'], 1),
        "time_diff_pct": round(tm['time_diff_pct'], 1),
        "nominal_score": round(nominal_score, 2),
        "adaptive_score": round(adaptive_score_val, 2),
        "score_loss_pct": round(score_loss_pct, 1),
        "n_bypass": action_counts.get("bypass", 0),
        "n_change_tool": action_counts.get("change_tool", 0),
        "n_destroy": action_counts.get("destroy", 0),
        "n_replan": action_counts.get("replan", 0),
        "n_fallback": sum(1 for a in actions_log if "→" in a.get("action", "") or "->" in a.get("action", "")),
        "n_destroyed_comps": len(state.get('destroyed', set())),
        "n_skipped_comps": len(state.get('skipped', set())),
        "grasp_time": round(grasp_time, 3),
        "adapt_time": round(adapt_time, 3),
        "error": None
    }


# ── main ─────────────────────────────────────────────────────────────

def main():
    # Collect all (instance, scenario) pairs
    pairs = []
    for instance_dir in sorted(glob.glob(os.path.join(FAILURES_DIR, "*"))):
        if not os.path.isdir(instance_dir):
            continue
        instance_name = os.path.basename(instance_dir)
        for scenario_file in sorted(glob.glob(os.path.join(instance_dir, "*.json"))):
            pairs.append((instance_name, scenario_file))

    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║  FUZZY ADAPTIVE BENCHMARK — ALL SCENARIOS               ║")
    print(f"║  {len(pairs)} scenario(s) across {len(set(p[0] for p in pairs))} instance(s)  ║")
    print(f"╚══════════════════════════════════════════════════════════╝\n")

    results = []
    for idx, (inst, scen) in enumerate(pairs, 1):
        stype = classify_scenario(scen)
        print(f"[{idx:3d}/{len(pairs)}] {inst:40s} | {stype:15s} ... ", end="", flush=True)
        try:
            row = run_single(inst, scen, verbose=False)
            results.append(row)
            if row.get("error"):
                print(f"ERROR: {row['error']}")
            else:
                rec = row['target_recovery_pct']
                loss = row['profit_loss_pct']
                print(f"recovery={rec:5.1f}%  profit_loss={loss:5.1f}%  actions={row['n_actions']}")
        except Exception as e:
            print(f"EXCEPTION: {e}")
            traceback.print_exc()
            results.append({
                "instance": inst,
                "scenario_file": os.path.basename(scen),
                "scenario_type": stype,
                "error": str(e)
            })

    # ── Write CSV ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    fieldnames = [
        "instance", "n", "scenario_file", "scenario_type",
        "n_failures", "n_actions",
        "target_total", "target_achieved", "target_recovery_pct",
        "nominal_profit", "adaptive_profit", "profit_loss_pct",
        "nominal_time", "adaptive_time", "time_diff_pct",
        "nominal_score", "adaptive_score", "score_loss_pct",
        "n_bypass", "n_change_tool", "n_destroy", "n_replan", "n_fallback",
        "n_destroyed_comps", "n_skipped_comps",
        "grasp_time", "adapt_time", "error"
    ]

    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n✅ Results saved to {RESULTS_CSV}")

    # ── Summary statistics ───────────────────────────────────────────
    valid = [r for r in results if not r.get("error")]
    if not valid:
        print("No valid results!")
        return

    print(f"\n{'='*70}")
    print(f"  SUMMARY — {len(valid)} scenarios completed ({len(results)-len(valid)} errors)")
    print(f"{'='*70}")

    # By scenario type
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in valid:
        by_type[r['scenario_type']].append(r)

    print(f"\n{'Type':15s} | {'Count':5s} | {'Recovery%':10s} | {'ProfitLoss%':12s} | {'ScoreLoss%':11s}")
    print("-" * 70)

    for stype in ["simple", "intermediate", "complex", "stress", "extreme", "stress_4", "stress_6", "stress_8", "stress_10", "cascade"]:
        rows = by_type.get(stype, [])
        if not rows:
            continue
        avg_rec = sum(r['target_recovery_pct'] for r in rows) / len(rows)
        avg_pl  = sum(r['profit_loss_pct'] for r in rows) / len(rows)
        avg_sl  = sum(r['score_loss_pct'] for r in rows) / len(rows)
        print(f"{stype:15s} | {len(rows):5d} | {avg_rec:9.1f}% | {avg_pl:11.1f}% | {avg_sl:10.1f}%")

    # Overall
    avg_rec = sum(r['target_recovery_pct'] for r in valid) / len(valid)
    avg_pl  = sum(r['profit_loss_pct'] for r in valid) / len(valid)
    avg_sl  = sum(r['score_loss_pct'] for r in valid) / len(valid)
    print("-" * 70)
    print(f"{'OVERALL':15s} | {len(valid):5d} | {avg_rec:9.1f}% | {avg_pl:11.1f}% | {avg_sl:10.1f}%")

    # Action distribution
    total_actions = sum(r['n_actions'] for r in valid)
    total_bypass = sum(r['n_bypass'] for r in valid)
    total_ctool  = sum(r['n_change_tool'] for r in valid)
    total_destroy= sum(r['n_destroy'] for r in valid)
    total_replan = sum(r['n_replan'] for r in valid)
    total_fb     = sum(r['n_fallback'] for r in valid)

    print(f"\n  Action distribution ({total_actions} total actions):")
    if total_actions > 0:
        print(f"    bypass:      {total_bypass:4d} ({total_bypass/total_actions*100:5.1f}%)")
        print(f"    change_tool: {total_ctool:4d} ({total_ctool/total_actions*100:5.1f}%)")
        print(f"    destroy:     {total_destroy:4d} ({total_destroy/total_actions*100:5.1f}%)")
        print(f"    replan:      {total_replan:4d} ({total_replan/total_actions*100:5.1f}%)")
        print(f"    fallback:    {total_fb:4d} ({total_fb/total_actions*100:5.1f}%)")

    # Full recovery rate
    full_recovery = sum(1 for r in valid if r['target_recovery_pct'] == 100.0)
    print(f"\n  Full target recovery: {full_recovery}/{len(valid)} ({full_recovery/len(valid)*100:.1f}%)")

    # Worst-case profit loss
    worst_pl = max(r['profit_loss_pct'] for r in valid)
    worst_inst = next(r for r in valid if r['profit_loss_pct'] == worst_pl)
    print(f"  Worst profit loss: {worst_pl:.1f}% on {worst_inst['instance']} ({worst_inst['scenario_type']})")

    print(f"\n{'='*70}")
    print(f"  Done.")

if __name__ == "__main__":
    main()
