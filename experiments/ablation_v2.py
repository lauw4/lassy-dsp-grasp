#!/usr/bin/env python3
"""
Ablation study: test each V2 strategy independently by toggling flags.
Uses the enable_* flags added to execute_adaptive_sequence_v2.

Variants tested (on high-stress scenarios, 4+ failures):
  - V1:           original baseline (scheduler.py)
  - relax_only:   V1-like + dynamic threshold relaxation only
  - P1_only:      multi-target continuation only (no Phase 2/3)
  - P1+P2:        multi-target + global replan (no Phase 3)
  - P1+P3:        multi-target + aggressive recovery (no Phase 2)
  - V2_full:      all strategies combined
"""
import os, sys, json, glob, time, csv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.grasp.constructive import run_grasp
from src.adaptive import overlay
from src.adaptive.scheduler import execute_adaptive_sequence
from src.adaptive.scheduler_v2 import execute_adaptive_sequence_v2
from src.utils.graph_io import load_adaptive_graph
from src.utils.metrics import compute_economic_metrics

FAILURES_DIR = "data/adaptive/generated_failures"
RESULTS_CSV  = "results/ablation_v2_strategies.csv"

# (name, use_v2, v2_kwargs)
VARIANTS = [
    ("V1",         False, {}),
    ("relax_only", True,  dict(enable_multitarget=False, enable_phase2=False, enable_phase3=False, enable_relaxation=True)),
    ("P1_only",    True,  dict(enable_multitarget=True,  enable_phase2=False, enable_phase3=False, enable_relaxation=True)),
    ("P1+P2",      True,  dict(enable_multitarget=True,  enable_phase2=True,  enable_phase3=False, enable_relaxation=True)),
    ("P1+P3",      True,  dict(enable_multitarget=True,  enable_phase2=False, enable_phase3=True,  enable_relaxation=True)),
    ("V2_full",    True,  dict(enable_multitarget=True,  enable_phase2=True,  enable_phase3=True,  enable_relaxation=True)),
]


def run_variant(instance_name, scenario_path, use_v2, v2_kwargs):
    """Run one scenario with a specific variant."""
    instance_path = f"data/instances_base/{instance_name}.json"
    if not os.path.exists(instance_path):
        return None

    G, *_ = load_adaptive_graph(instance_path)
    with open(instance_path, 'r') as f:
        inst_data = json.load(f)
    targets = inst_data.get("targets", list(G.nodes()))

    solution, score = run_grasp(G, algorithm='vnd', mode='selectif', target_nodes=targets)

    with open(scenario_path, 'r') as f:
        failures = json.load(f)

    G2, *_ = load_adaptive_graph(instance_path)
    state = overlay.init_state()
    state['graph'] = G2
    state['G'] = G2
    state['targets'] = targets

    t0 = time.perf_counter()
    if use_v2:
        state, actions_log = execute_adaptive_sequence_v2(
            solution=solution, state=state, failures=failures,
            verbose=True, **v2_kwargs
        )
    else:
        state, actions_log = execute_adaptive_sequence(
            solution=solution, state=state, failures=failures, verbose=True
        )
    elapsed = time.perf_counter() - t0

    eco = compute_economic_metrics(G2, solution,
                                   state.get('done', set()),
                                   state.get('destroyed', set()),
                                   state.get('skipped', set()))

    targets_achieved = [t for t in targets
                        if t in state.get('done', set()) and t not in state.get('destroyed', set())]
    success_rate = len(targets_achieved) / len(targets) * 100 if targets else 0

    return {
        "recovery_pct": round(success_rate, 1),
        "profit_loss_pct": round(eco['profit_loss_pct'], 1),
        "n_destroyed": len(state.get('destroyed', set())),
        "time": round(elapsed, 3),
    }


def main():
    # Collect all high-stress scenarios (4+ failures)
    pairs = []
    for instance_dir in sorted(glob.glob(os.path.join(FAILURES_DIR, "*"))):
        if not os.path.isdir(instance_dir):
            continue
        instance_name = os.path.basename(instance_dir)
        for scenario_file in sorted(glob.glob(os.path.join(instance_dir, "*.json"))):
            with open(scenario_file) as f:
                nf = len(json.load(f))
            if nf >= 4:
                pairs.append((instance_name, scenario_file, nf))

    vnames = [v[0] for v in VARIANTS]
    print("=" * 80)
    print(f"  ABLATION STUDY -- {len(pairs)} high-stress scenarios (4+ failures)")
    print(f"  Variants: {vnames}")
    print("=" * 80)
    print()

    rows = []
    for idx, (inst, scen, nf) in enumerate(pairs, 1):
        print(f"[{idx:3d}/{len(pairs)}] {inst:30s} nf={nf}", end="", flush=True)
        row = {"instance": inst, "scenario": os.path.basename(scen), "n_failures": nf}

        for vname, use_v2, kwargs in VARIANTS:
            try:
                r = run_variant(inst, scen, use_v2, kwargs)
                if r:
                    row[f"{vname}_rec"] = r["recovery_pct"]
                    row[f"{vname}_loss"] = r["profit_loss_pct"]
                    row[f"{vname}_destr"] = r["n_destroyed"]
                else:
                    row[f"{vname}_rec"] = None
            except Exception as e:
                row[f"{vname}_rec"] = None
                print(f" ERR({vname}:{e})", end="")

        recs = " | ".join(f"{v}={row.get(f'{v}_rec','?')}" for v in vnames)
        print(f"  => {recs}")
        rows.append(row)

    # Summary table
    print()
    print("=" * 80)
    print(f"  RESULTS -- {len(rows)} high-stress scenarios")
    print("=" * 80)
    print()
    hdr = f"{'Variant':15s} | {'Full rec%':>9s} | {'Avg rec%':>9s} | {'Avg loss%':>9s} | {'Avg destr':>9s}"
    print(hdr)
    print("-" * len(hdr))

    v1_full_rate = 0.0
    for vname, _, _ in VARIANTS:
        recs = [r[f"{vname}_rec"] for r in rows if r.get(f"{vname}_rec") is not None]
        losses = [r.get(f"{vname}_loss", 0) for r in rows if r.get(f"{vname}_rec") is not None]
        destrs = [r.get(f"{vname}_destr", 0) for r in rows if r.get(f"{vname}_rec") is not None]
        if recs:
            full = sum(1 for x in recs if x == 100.0) / len(recs) * 100
            avg = sum(recs) / len(recs)
            avg_l = sum(losses) / len(losses) if losses else 0
            avg_d = sum(destrs) / len(destrs) if destrs else 0
            print(f"{vname:15s} | {full:8.1f}% | {avg:8.1f}% | {avg_l:8.1f}% | {avg_d:8.1f}")
            if vname == "V1":
                v1_full_rate = full

    # Marginal contribution
    print()
    print("=" * 80)
    print("  MARGINAL CONTRIBUTION (delta full recovery rate vs V1)")
    print("=" * 80)
    print()
    for vname, _, _ in VARIANTS:
        if vname == "V1":
            continue
        recs = [r[f"{vname}_rec"] for r in rows if r.get(f"{vname}_rec") is not None]
        if recs:
            full = sum(1 for x in recs if x == 100.0) / len(recs) * 100
            delta = full - v1_full_rate
            sign = "+" if delta >= 0 else ""
            print(f"  {vname:15s}: {full:5.1f}%  (delta = {sign}{delta:.1f}%)")

    # Save CSV
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    if rows:
        with open(RESULTS_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
