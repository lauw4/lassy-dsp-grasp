#!/usr/bin/env python3
"""
Fuzzy Test Launcher - Runs all failure scenarios listed in
data/adaptive/scenarios_list.csv (29 instances x 3 difficulty levels = 87 runs)
and saves results in:
  results/benchmark_2026/fuzzy/[simple|intermediate|complex]/scenario_X/

At the end, builds the aggregated per-scenario CSV
  results/benchmark_2026/fuzzy/RESUME_FUZZY_DETAILED.csv
by parsing each scenario's terminal_python_log.txt.
"""
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

SCENARIOS_CSV = "data/adaptive/scenarios_list.csv"
OUTPUT_BASE = Path("results/benchmark_2026/fuzzy")
RESUME_CSV = OUTPUT_BASE / "RESUME_FUZZY_DETAILED.csv"
DIFFICULTY_ORDER = ["simple", "intermediate", "complex"]


# ---------- Per-scenario log parsing (used to build RESUME_FUZZY_DETAILED.csv) ----------

def parse_time_to_seconds(text):
    """Convert 'Xmin Ys' / 'Xs' / '-0s' to total seconds (int)."""
    text = text.strip()
    m = re.match(r"(-?\d+)\s*min\s*(-?\d+)\s*s", text)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.match(r"(-?\d+)\s*s", text)
    if m:
        return int(m.group(1))
    return None


def parse_scenario_log(log_path: Path, results_json: dict) -> dict:
    """Parse one terminal_python_log.txt and return a flat dict of metrics."""
    text = log_path.read_text(encoding="utf-8", errors="replace")

    def grab(pattern, group=1, cast=str, default=None):
        m = re.search(pattern, text)
        if not m:
            return default
        try:
            return cast(m.group(group))
        except (ValueError, IndexError):
            return default

    nodes = grab(r"Nodes:\s*(\d+),", cast=int)
    num_failures = grab(r"Loaded failures:\s*(\d+)", cast=int, default=0)
    status_raw = grab(r"Status:\s*(SUCCESS|PARTIAL SUCCESS|FAILURE)", default="UNKNOWN")

    tgt_done = grab(r"Targets achieved:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)", group=1, cast=int)
    tgt_total = grab(r"Targets achieved:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)", group=2, cast=int)
    tgt_pct = grab(r"Targets achieved:\s*\d+/\d+\s*\(([\d.]+)%\)", cast=float)
    target_achieved = (tgt_done is not None and tgt_total is not None and tgt_done == tgt_total)

    actions_taken = grab(r"Actions taken:\s*(\d+)", cast=int, default=0)
    cd_num = grab(r"Components done:\s*(\d+)/(\d+)", group=1, cast=int)
    cd_total = grab(r"Components done:\s*(\d+)/(\d+)", group=2, cast=int)
    algorithm_runtime_s = grab(r"Algorithm runtime:\s*([\d.]+)\s*s", cast=float)

    nominal_profit = grab(r"Nominal profit \(planned\):\s*([\d.\-]+)\s*units", cast=float)
    adaptive_profit = grab(r"Adaptive profit \(recovered\):\s*([\d.\-]+)\s*units", cast=float)
    profit_loss_pct = grab(r"Profit loss:\s*[\d.\-]+\s*units\s*\(([\-\d.]+)%\)", cast=float)

    nominal_time_str = grab(r"Nominal time \(planned\):\s*(.+)")
    adaptive_time_str = grab(r"Adaptive time \(actual\):\s*(.+)")
    nominal_time_s = parse_time_to_seconds(nominal_time_str) if nominal_time_str else None
    adaptive_time_s = parse_time_to_seconds(adaptive_time_str) if adaptive_time_str else None

    if nominal_time_s is not None and adaptive_time_s is not None and nominal_time_s > 0:
        time_overhead_pct = (adaptive_time_s - nominal_time_s) / nominal_time_s * 100
    else:
        time_overhead_pct = None

    replans = len(re.findall(r"\[REPLAN\] New sequence", text))

    return {
        "instance": results_json.get("instance"),
        "difficulty": results_json.get("difficulty"),
        "scenario_id": results_json.get("scenario_id"),
        "n": nodes,
        "num_failures": num_failures,
        "status": status_raw,
        "target_achieved": target_achieved,
        "targets_pct": tgt_pct,
        "components_done": f"{cd_num}/{cd_total}" if cd_num is not None else None,
        "nominal_profit": nominal_profit,
        "adaptive_profit": adaptive_profit,
        "profit_loss_pct": profit_loss_pct,
        "nominal_time_s": nominal_time_s,
        "adaptive_time_s": adaptive_time_s,
        "time_overhead_pct": round(time_overhead_pct, 2) if time_overhead_pct is not None else None,
        "actions_taken": actions_taken,
        "replans": replans,
        "algorithm_runtime_s": algorithm_runtime_s,
    }


def build_resume_csv():
    """Walk per-scenario subfolders and build RESUME_FUZZY_DETAILED.csv. Prints stats."""
    rows = []
    for difficulty in DIFFICULTY_ORDER:
        d_dir = OUTPUT_BASE / difficulty
        if not d_dir.is_dir():
            continue
        for sc_dir in sorted(d_dir.iterdir()):
            if not sc_dir.is_dir():
                continue
            rj_path = sc_dir / "results.json"
            log_path = sc_dir / "terminal_python_log.txt"
            if not rj_path.exists() or not log_path.exists():
                continue
            with rj_path.open() as f:
                rj = json.load(f)
            rows.append(parse_scenario_log(log_path, rj))

    if not rows:
        print("[WARN] No scenarios found, RESUME CSV not written.")
        return

    diff_rank = {d: i for i, d in enumerate(DIFFICULTY_ORDER)}
    rows.sort(key=lambda r: (diff_rank.get(r["difficulty"], 99), r["instance"] or ""))

    RESUME_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESUME_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] {len(rows)} scenarios -> {RESUME_CSV}")

    print("\n=== STATS BY DIFFICULTY ===")
    for diff in DIFFICULTY_ORDER:
        sub = [r for r in rows if r["difficulty"] == diff]
        if not sub:
            continue
        n_success = sum(1 for r in sub if r["target_achieved"])
        pl = [r["profit_loss_pct"] for r in sub if r["profit_loss_pct"] is not None]
        to = [r["time_overhead_pct"] for r in sub if r["time_overhead_pct"] is not None]
        avg_pl = sum(pl) / len(pl) if pl else 0.0
        avg_to = sum(to) / len(to) if to else 0.0
        status_count = Counter(r["status"] for r in sub)
        print(f"\n[{diff.upper()}]  ({len(sub)} scenarios)")
        print(f"  Target achieved: {n_success}/{len(sub)} ({100*n_success/len(sub):.1f}%)")
        print(f"  Status: {dict(status_count)}")
        print(f"  Avg profit loss: {avg_pl:.2f}%   Avg time overhead: {avg_to:+.2f}%")


# ---------- Benchmark runner ----------

def run_fuzzy_tests():
    """Run all fuzzy scenarios from SCENARIOS_CSV."""
    if not os.path.exists(SCENARIOS_CSV):
        print(f"Error: {SCENARIOS_CSV} not found")
        return False

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SCENARIOS_CSV)
    print(f"\nFuzzy Test Launcher: {len(df)} scenarios to run\n")

    scenarios_by_difficulty = {
        diff: df[df["scenario_type"] == diff] for diff in DIFFICULTY_ORDER
    }
    for diff in DIFFICULTY_ORDER:
        print(f"  {diff}: {len(scenarios_by_difficulty[diff])} scenarios")

    total_count = completed = failed = 0

    for difficulty in DIFFICULTY_ORDER:
        difficulty_dir = OUTPUT_BASE / difficulty
        difficulty_dir.mkdir(parents=True, exist_ok=True)
        scenarios = scenarios_by_difficulty[difficulty]

        for idx, (_, row) in enumerate(scenarios.iterrows(), 1):
            scenario_dir = difficulty_dir / f"scenario_{idx}"
            scenario_dir.mkdir(parents=True, exist_ok=True)

            instance = row["instance"]
            scenario_file = row["scenario_file"]

            print(f"\n[{difficulty}/{idx}/{len(scenarios)}] {instance} - {scenario_file}")

            instance_path = f"data/instances_base/{instance}.json"
            failures_path = f"data/adaptive/generated_failures/{instance}/{scenario_file}"

            if not os.path.exists(instance_path):
                print(f"  ERROR: Instance file not found: {instance_path}")
                failed += 1
                continue
            if not os.path.exists(failures_path):
                print(f"  ERROR: Scenario file not found: {failures_path}")
                failed += 1
                continue

            try:
                cmd = [sys.executable, "run_pipeline.py",
                       "--instance", instance, "--failures", failures_path]
                print(f"  Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, timeout=3600)

                if result.returncode == 0:
                    print(f"  OK - completed")
                    completed += 1
                else:
                    print(f"  ERROR - return code {result.returncode}")
                    if result.stderr:
                        print(f"  {result.stderr.decode()[:200]}")
                    failed += 1

                with (scenario_dir / "results.json").open("w") as f:
                    json.dump({
                        "timestamp": datetime.now().isoformat(),
                        "instance": instance,
                        "difficulty": difficulty,
                        "scenario_id": f"scenario_{idx}",
                        "scenario_file": scenario_file,
                        "status": "completed" if result.returncode == 0 else "failed",
                        "return_code": result.returncode,
                    }, f, indent=2)

                old_result_dir = Path("results/save/test_failures") / instance / difficulty
                if old_result_dir.exists():
                    for fname in ("execution_report.txt", "terminal_python_log.txt", "pipeline_log.txt"):
                        src = old_result_dir / fname
                        if src.exists():
                            (scenario_dir / fname).write_text(
                                src.read_text(encoding="utf-8", errors="replace"),
                                encoding="utf-8",
                            )

                total_count += 1

            except subprocess.TimeoutExpired:
                print(f"  ERROR - timeout (>1h)")
                failed += 1
            except Exception as e:
                print(f"  ERROR - {str(e)}")
                failed += 1

    # Build the aggregated RESUME_FUZZY_DETAILED.csv from the per-scenario logs
    print("\n\nBuilding RESUME_FUZZY_DETAILED.csv from scenario logs...")
    build_resume_csv()

    print(f"\nTotal scenarios: {total_count}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    return failed == 0


if __name__ == "__main__":
    # If called as `python fuzzy_test_launcher.py --resume`, only rebuild the CSV
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        build_resume_csv()
        sys.exit(0)
    success = run_fuzzy_tests()
    sys.exit(0 if success else 1)
