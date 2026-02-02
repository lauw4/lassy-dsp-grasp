"""
Adaptive DSP Pipeline with Fuzzy Decision System (IFAC 2026)

Usage:
    python run_pipeline.py --instance dagtest
    python run_pipeline.py --instance dagtest --failures demo_bypass
    python run_pipeline.py --instance dagtest --failures demo_bypass --verbose

Pipeline:
    1. Load DSP instance from data/instances_base/
    2. Solve initial sequence with GRASP+VND (selective mode)
    3. Execute with adaptive scheduler (fuzzy decision on failures)
    4. Generate execution report in results/save/

References:
    - Ye et al., 2022 (Adaptive Disassembly Planning)
    - Pedrosa et al., 2023 (Robust Scheduling)
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

# Fix Windows encoding issues (force UTF-8)
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Global verbose flag
VERBOSE_MODE = False

from src.grasp.constructive import run_grasp  # GRASP+VND solver
from src.adaptive import config, overlay
from src.adaptive.scheduler import execute_adaptive_sequence
from src.utils.graph_io import load_adaptive_graph
from src.utils.metrics import compute_economic_metrics, compute_time_metrics, score_weighted

# --- UTILITY FUNCTIONS ---
def format_time(seconds):
    """Format time in min:sec or just seconds if < 60"""
    if seconds >= 60:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}min {secs}s"
    else:
        return f"{seconds:.0f}s"

# --- CONFIGURATION ---
INSTANCE_PATH = "data/instances_base/dagtest.json"  # Change to your instance file
FAILURE_SCENARIO_PATH = "data/adaptive/generated_failures/dagtest/failures_1_on_dagtest.json"  

# --- LOGGING SETUP ---
def get_log_dir(instance_path, failures_path=None):
    instance_name = os.path.splitext(os.path.basename(instance_path))[0]
    if failures_path is None:
        prefix = "test_nominal"
        log_dir = os.path.join("results", "save", prefix, instance_name)
    else:
        prefix = "test_failures"
        # Special detection for scenario_article_pedagogique_copy
        if isinstance(failures_path, str) and "scenario_article_pedagogique_copy" in failures_path:
            log_dir = os.path.join("results", "save", "dagtest_article")
        else:
            # Detect scenario type (simple/intermediate/complex) from failure file name
            scenario_type = None
            if isinstance(failures_path, str):
                for t in ["simple", "intermediate", "complex"]:
                    if f"scenario_{t}_" in failures_path or f"_{t}_" in failures_path:
                        scenario_type = t
                        break
            if scenario_type:
                log_dir = os.path.join("results", "save", prefix, instance_name, scenario_type)
            else:
                log_dir = os.path.join("results", "save", prefix, instance_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

# --- LOAD DSP INSTANCE ---
def load_instance(path):
    """Load DSP instance from JSON file. Format: see data/instances_base/*.json
    Returns: (instance_dict, full_path)"""
    # 
    if not path.endswith('.json'):
        path = f"data/instances_base/{path}.json"
    with open(path, 'r', encoding='utf-8') as f:
        instance = json.load(f)
    return instance, path

# --- LOAD FAILURE SCENARIO ---
def load_failures(path):
    """Load failure scenario from JSON file. Format: List[Dict], e.g. [{"node": "A12", "type": "block"}]"""
    if path is None:
        return []
    # Auto-complete path if just scenario name given
    if not path.endswith('.json'):
        # Try multiple patterns to find the scenario file
        import glob
        
        # Pattern 1: demo files (demo_bypass.json, demo_mixed.json, etc.)
        pattern1 = f"data/adaptive/generated_failures/dagtest/{path}.json"
        matches = glob.glob(pattern1)
        
        if not matches:
            # Pattern 2: scenario files with _on_dagtest suffix
            pattern2 = f"data/adaptive/generated_failures/dagtest/{path}*_on_dagtest.json"
            matches = glob.glob(pattern2)
        
        if not matches:
            # Pattern 3: gearpump scenario files
            pattern3 = f"data/adaptive/generated_failures/gearpump/{path}.json"
            matches = glob.glob(pattern3)
        
        if matches:
            path = matches[0]  # Use first match
        else:
            # Fallback: add .json extension and hope it's in current dir
            path = f"{path}.json"
    
    with open(path, 'r', encoding='utf-8') as f:
        failures = json.load(f)
    return failures

# --- SOLVE DSP (SELECTIVE GRASP+VND) ---
def solve_dsp(instance_path):
    """Solve DSP using selective GRASP+VND. Returns (solution, G, targets)."""
    # Reference: Frizziero2020, Pedrosa2023
    # Load graph and extract targets
    G, *_ = load_adaptive_graph(instance_path)
    with open(instance_path, 'r', encoding='utf-8') as f:
        instance_data = json.load(f)
    targets = instance_data.get("targets", [])
    if not targets:
        print("[WARNING] No targets found in instance. Using all nodes.")
        targets = list(G.nodes())
    # Run GRASP+VND
    solution, score = run_grasp(G, algorithm='vnd', mode='selectif', target_nodes=targets)
    print(f"[INFO] GRASP+VND solution (score={score}): {solution}")
    return solution, G, targets

# --- ADAPTIVE SCHEDULER ---
def run_adaptive_scheduler(solution, G, targets, failures):
    """
    Run adaptive scheduler - orchestrates execution.
    Actual logic is in src/adaptive/scheduler.py
    """
    # Initialize state
    state = overlay.init_state()
    state['graph'] = G
    state['G'] = G
    state['targets'] = targets
    
    # Execute adaptive sequence
    state, actions_log = execute_adaptive_sequence(
        solution=solution,
        state=state,
        failures=failures,
        verbose=VERBOSE_MODE
    )
    
    return state, actions_log

# --- LOGGING RESULTS ---
def generate_narrative_report(solution, state, actions_log, failures, targets, log_dir, G=None, eco_metrics=None, time_metrics=None):
    """
    Generate narrative execution report, extracting fuzzy data and replans from terminal log.
    """
    narrative_path = os.path.join(log_dir, "execution_report.txt")

    BLOCK_SEP = "-" * 6
    SUB_SEP = "-" * 40

    def _write_block_title(file_obj, title: str) -> None:
        file_obj.write(f"{BLOCK_SEP}\n{title}\n{BLOCK_SEP}\n\n")

    def _write_underlined_title(file_obj, title: str) -> None:
        file_obj.write(f"{title}\n{SUB_SEP}\n")

    def _write_replan_banner(file_obj) -> None:
        file_obj.write(f"\n{SUB_SEP}\n")
        file_obj.write("SELECTIVE REPLANNING TRIGGERED\n")
        file_obj.write(f"{SUB_SEP}\n")

    def _normalize_action_label(action: str) -> str:
        return (action or "").replace("→", "->")

    def _normalize_notes_to_english(notes: str) -> str:
        if not notes:
            return "Action executed"
        replacements = {
            "Outil alternatif proposé.": "Alternative tool suggested.",
            "Destruction autorisée (cutoff percentile 20e).": "Destruction allowed (20th percentile cutoff).",
            "Destruction autorisée (valeur piece faible)(cutoff percentile 20e).": "Destruction allowed (low-value component) (20th percentile cutoff).",
            "Destruction autorisée (valeur piece faible)": "Destruction allowed (low-value component).",
        }
        return replacements.get(notes.strip(), notes)
    
    # Parse terminal_python_log.txt for fuzzy data and replans
    fuzzy_map = {}
    replans = []
    replan_states = []  # States associated with each replan
    terminal_log = os.path.join(log_dir, "terminal_python_log.txt")
    if os.path.exists(terminal_log):
        with open(terminal_log, 'r', encoding='utf-8', errors='replace') as tf:
            lines = tf.readlines()
            i = 0
            current_state = {'done': [], 'destroyed': [], 'locked': [], 'skipped': []}
            while i < len(lines):
                # Extract fuzzy decisions
                if '[FUZZY] Symptom:' in lines[i]:
                    symptom = lines[i].split('Symptom: ')[1].strip()
                    force = torque = p_fail = score = confidence = action = None
                    
                    if i+1 < len(lines) and '[FUZZY] Inputs:' in lines[i+1]:
                        inputs = lines[i+1]
                        if 'force=' in inputs:
                            force = inputs.split('force=')[1].split(',')[0]
                        if 'torque=' in inputs:
                            torque = inputs.split('torque=')[1].split(',')[0]
                        if 'p_fail=' in inputs:
                            p_fail = inputs.split('p_fail=')[1].split(',')[0]
                    
                    if i+2 < len(lines) and '[FUZZY] Score:' in lines[i+2]:
                        score = lines[i+2].split('Score: ')[1].strip()
                    
                    if i+3 < len(lines) and '[FUZZY] Decision:' in lines[i+3]:
                        dec = lines[i+3].split('Decision: ')[1]
                        action = dec.split(' (')[0]
                        if 'confidence=' in dec:
                            confidence = dec.split('confidence=')[1].split(')')[0].replace('%', '')
                    
                    for act in actions_log:
                        if act['symptom'] == symptom and act['comp_id'] not in fuzzy_map:
                            fuzzy_map[act['comp_id']] = {
                                'force': force,
                                'torque': torque,
                                'p_fail': p_fail,
                                'score': score,
                                'confidence': confidence,
                                'action': action
                            }
                            break
                
                # Extract replan states
                if '[REPLAN]   done=' in lines[i]:
                    current_state['done'] = lines[i].split('done=')[1].strip()
                if '[REPLAN]   destroyed=' in lines[i]:
                    current_state['destroyed'] = lines[i].split('destroyed=')[1].strip()
                if '[REPLAN]   locked=' in lines[i]:
                    current_state['locked'] = lines[i].split('locked=')[1].strip()
                if '[REPLAN]   skipped=' in lines[i]:
                    current_state['skipped'] = lines[i].split('skipped=')[1].strip()
                
                # Extract replans (new sequence)
                if '[REPLAN] Nouvelle séquence:' in lines[i] or '[REPLAN] New sequence:' in lines[i]:
                    if 'Nouvelle séquence:' in lines[i]:
                        plan_str = lines[i].split('Nouvelle séquence: ')[1].strip()
                    else:
                        plan_str = lines[i].split('New sequence: ')[1].strip()
                    replans.append(plan_str)
                    replan_states.append(dict(current_state))
                    current_state = {'done': [], 'destroyed': [], 'locked': [], 'skipped': []}
                
                i += 1
    
    failures_map = {f['comp_id']: f for f in failures}
    
    with open(narrative_path, 'w', encoding='utf-8') as f:
        _write_block_title(f, "ADAPTIVE DSP EXECUTION REPORT")

        _write_underlined_title(f, "INITIAL PLANNED SEQUENCE:")
        f.write(f"{', '.join(solution)}\n")
        f.write(f"Target = {', '.join(targets)}\n\n")

        _write_block_title(f, "EXECUTION WITH ADAPTATION:")
        
        actions_by_comp = {act['comp_id']: act for act in actions_log}
        done_set = state.get('done', set())
        replan_count = 0
        
        for comp in solution:
            if comp in actions_by_comp:
                act = actions_by_comp[comp]
                fuzzy = fuzzy_map.get(comp, {})

                _write_block_title(f, f"FAILURE ON {comp}")
                f.write(f"Symptom: {act['symptom']}\n")
                
                if fuzzy:
                    f.write(f"\nSensor values:\n")
                    f.write(f"  - Force:  {fuzzy.get('force', 'N/A')}\n")
                    f.write(f"  - Torque: {fuzzy.get('torque', 'N/A')}\n")
                    f.write(f"  - P_fail: {fuzzy.get('p_fail', 'N/A')}%\n")
                    f.write(f"\nFuzzy decision:\n")
                    f.write(f"  - Action: {_normalize_action_label(act.get('action', ''))}\n")
                    f.write(f"  - Score:  {fuzzy.get('score', 'N/A')}\n")
                    f.write(f"  - Confidence: {fuzzy.get('confidence', 'N/A')}%\n")

                f.write(f"\nResult: {_normalize_notes_to_english(act.get('notes', 'Action executed'))}\n")
                
                # Show replan if this action triggered one
                action_norm = _normalize_action_label(act.get('action', '')).lower()
                if 'replan' in action_norm or action_norm in ['destroy', 'bypass->destroy']:
                    if replan_count < len(replans):
                        _write_replan_banner(f)
                        
                        # Show state at replan
                        if replan_count < len(replan_states):
                            rs = replan_states[replan_count]
                            f.write(f"\nState at replan:\n")
                            f.write(f"  - Done:      {rs.get('done', '[]')}\n")
                            f.write(f"  - Destroyed: {rs.get('destroyed', '[]')}\n")
                            f.write(f"  - Locked:    {rs.get('locked', '[]')}\n")
                            f.write(f"  - Skipped:   {rs.get('skipped', '[]')}\n")
                        
                        f.write(f"\nNew sequence after replan:\n")
                        f.write(f"  {replans[replan_count]}\n")
                        replan_count += 1
                
            elif comp in done_set:
                f.write(f"{comp} - OK\n")

        # Execution metrics (compact)
        done_now = state.get('done', set())
        destroyed_now = state.get('destroyed', set())
        skipped_now = state.get('skipped', set())
        targets_achieved = [t for t in targets if t in done_now and t not in destroyed_now]
        success_rate = (len(targets_achieved) / len(targets) * 100) if targets else 0

        _write_block_title(f, "EXECUTION METRICS:")
        f.write(f"Targets achieved: {len(targets_achieved)}/{len(targets)} ({success_rate:.1f}%)\n")
        f.write(f"Failures handled: {len(failures)}\n")
        f.write(f"Actions taken:    {len(actions_log)}\n")
        f.write(f"Done:             {len(done_now)}/{len(solution)}\n")
        f.write(f"Destroyed:        {len(destroyed_now)}\n")
        f.write(f"Skipped:          {len(skipped_now)}\n")
        
        # Final state
        _write_block_title(f, "FINAL STATE")
        f.write(f"Done:      {sorted(state.get('done', set()))}\n")
        f.write(f"Destroyed: {sorted(state.get('destroyed', set()))}\n")
        f.write(f"Locked:    {sorted(state.get('locked', set()))}\n")
        f.write(f"Skipped:   {sorted(state.get('skipped', set()))}\n")
        
        # Economic analysis (if available)
        if eco_metrics is not None:
            _write_block_title(f, "ECONOMIC ANALYSIS")
            f.write(f"Nominal profit (planned):    {eco_metrics['nominal_profit']:.2f} units\n")
            f.write(f"Adaptive profit (recovered): {eco_metrics['adaptive_profit']:.2f} units\n")
            f.write(f"Profit loss:                 {eco_metrics['profit_loss']:.2f} units ({eco_metrics['profit_loss_pct']:.1f}%)\n")
            if eco_metrics['destroyed_value'] > 0:
                f.write(f"Value lost (destroyed):      {eco_metrics['destroyed_value']:.2f} units\n")
            if eco_metrics['skipped_value'] > 0:
                f.write(f"Value skipped (bypassed):    {eco_metrics['skipped_value']:.2f} units\n")
            
            # Details per component
            f.write(f"\nComponent breakdown:\n")
            for comp, info in eco_metrics['details'].items():
                status = info.get('status', 'unknown')
                net = info.get('net', 0)
                if status == 'destroyed':
                    f.write(f"  {comp}: profit={info['profit']}, cost={info['cost']} -> DESTROYED (lost {info['profit']} + paid {info['cost']})\n")
                elif status == 'skipped':
                    f.write(f"  {comp}: profit={info['profit']}, cost={info['cost']} -> SKIPPED (not recovered)\n")
                elif status == 'recovered':
                    f.write(f"  {comp}: profit={info['profit']}, cost={info['cost']} -> net={net:.2f}\n")
        
        # Time analysis (if available)
        if time_metrics is not None:
            _write_block_title(f, "TIME ANALYSIS")
            f.write(f"Nominal time (planned):      {format_time(time_metrics['nominal_time'])}\n")
            f.write(f"Adaptive time (actual):      {format_time(time_metrics['adaptive_time'])}\n")
            if time_metrics['time_diff'] > 0:
                f.write(f"Time saved:                  {format_time(time_metrics['time_diff'])} ({time_metrics['time_diff_pct']:.1f}%)\n")
            else:
                f.write(f"Time overhead:               {format_time(-time_metrics['time_diff'])} ({-time_metrics['time_diff_pct']:.1f}%)\n")
            if time_metrics['destroyed_time'] > 0:
                f.write(f"Time on destroyed (lost):    {format_time(time_metrics['destroyed_time'])}\n")
            if time_metrics['skipped_time'] > 0:
                f.write(f"Time saved (skipped):        {format_time(time_metrics['skipped_time'])}\n")
            
            # Details per component
            f.write(f"\nComponent time breakdown:\n")
            for comp, info in time_metrics['details'].items():
                duration = info.get('duration', 0)
                status = info.get('status', 'unknown')
                if status == 'destroyed':
                    f.write(f"  {comp}: {duration:.0f}s -> DESTROYED\n")
                elif status == 'skipped':
                    f.write(f"  {comp}: {duration:.0f}s -> SKIPPED (saved)\n")
                elif status == 'done':
                    f.write(f"  {comp}: {duration:.0f}s\n")
        
        # Global score (weighted)
        if G is not None:
            nominal_score = score_weighted(solution, G)
            adaptive_seq = [c for c in solution if c in state.get('done', set()) and c not in state.get('destroyed', set())]
            adaptive_score_val = score_weighted(adaptive_seq, G)
            score_diff = nominal_score - adaptive_score_val
            score_diff_pct = (score_diff / nominal_score * 100) if nominal_score > 0 else 0.0
            
            _write_block_title(f, "GLOBAL SCORE")
            #f.write(f"Formula: Score = w1*∑P - w2*∑C - w3*∑T/100 (w1=w2=w3=1/3)\n")
            f.write(f"Nominal score:               {nominal_score:.2f}\n")
            f.write(f"Adaptive score:              {adaptive_score_val:.2f}\n")
            f.write(f"Score loss:                  {score_diff:.2f} ({score_diff_pct:.1f}%)\n")
        
        _write_block_title(f, "GLOSSARY")
        f.write("p_fail:\n")
        f.write("  Estimated failure risk. Higher = more likely to fail.\n\n")
        f.write("Fuzzy decision:\n")
        f.write("  AI-based analysis choosing best recovery action.\n")
        f.write("  Score (0-100) and confidence (%) indicate decision quality.\n\n")
        f.write("Actions:\n")
        f.write("  - change_tool: Switch to alternative tool\n")
        f.write("  - bypass: Skip inaccessible component\n")
        f.write("  - destroy: Remove component by destruction\n")
        f.write("  - replan: Recalculate sequence after changes\n\n")
        f.write("States:\n")
        f.write("  - done: Successfully removed components\n")
        f.write("  - destroyed: Components removed by destruction\n")
        f.write("  - locked: Components physically blocked (not removable)\n")
        f.write("  - skipped: Components bypassed (not on current path)\n")


def log_results(instance, solution, state, actions_log, failures, log_dir):
    """Save pipeline log, failures log, and actions taken in log_dir."""
    # Main pipeline log
    log_path = os.path.join(log_dir, "pipeline_log.txt")
    with open(log_path, 'w', encoding='utf-8') as logf:
        logf.write(f"--- Pipeline run {datetime.now()} ---\n")
        logf.write(f"Instance: {instance.get('name', 'unknown')}\n")
        logf.write(f"Initial solution: {solution}\n")
        logf.write(f"Final state:\n")
        logf.write(f"  - Locked: {state['locked']}\n")
        logf.write(f"  - Destroyed: {state['destroyed']}\n")
        logf.write(f"  - Done: {state['done']}\n")
        if 'skipped' in state:
            logf.write(f"  - Skipped: {state['skipped']}\n")
        logf.write(f"  - Notes: {state['notes']}\n")
        logf.write(f"Actions taken: {actions_log}\n")
    # Save failures log if any
    if failures:
        failures_path = os.path.join(log_dir, "failures_log.json")
        with open(failures_path, 'w', encoding='utf-8') as ff:
            json.dump(failures, ff, indent=2)
    # Save actions log
    if actions_log:
        actions_path = os.path.join(log_dir, "actions_log.json")
        with open(actions_path, 'w', encoding='utf-8') as fa:
            json.dump(actions_log, fa, indent=2)

# --- MAIN PIPELINE ---
def main():
    parser = argparse.ArgumentParser(description="DSP Adaptive Pipeline with Fuzzy Decision System")
    parser.add_argument('--instance', type=str, default="dagtest", 
                        help='Instance name (e.g., dagtest) or full path to JSON file')
    parser.add_argument('--failures', type=str, default=None, 
                        help='Path to failure scenario JSON file (optional)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed fuzzy analysis (default: clean output)')
    args = parser.parse_args()
    
    # Set global verbose flag
    global VERBOSE_MODE
    VERBOSE_MODE = args.verbose
    
    import time
    start_time = time.time()
    

    # Prepare log directory
    log_dir = get_log_dir(args.instance, args.failures)
    logging.basicConfig(filename=os.path.join(log_dir, "terminal_python_log.txt"), filemode="w", level=logging.INFO, format='%(asctime)s %(message)s')

    instance_name = os.path.splitext(os.path.basename(args.instance))[0]
    print(f"\nDSP ADAPTIVE PIPELINE - {instance_name.upper()}\n")
    logging.info(f"DSP ADAPTIVE PIPELINE - {instance_name.upper()}")

    instance, instance_path = load_instance(args.instance)
    logging.info(f"Loaded instance: {instance_name}")

    failures = load_failures(args.failures)
    logging.info(f"Loaded failures: {len(failures)}")


    print(f"\n[1/4] Loading instance and solving DSP...")
    logging.info("[1/4] Loading instance and solving DSP...")
    solution, G, targets = solve_dsp(instance_path)
    logging.info(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}, Targets: {len(targets)}, Solution: {len(solution)} operations")

    # Log targets at the start
    print(f"\nTarget(s) to achieve: {targets}")
    logging.info(f"Target(s) to achieve: {targets}")

    # Log the chosen disassembly sequence
    print(f"\nStarting disassembly sequence ({len(solution)} operations)...")
    logging.info(f"Starting disassembly sequence ({len(solution)} operations)...")
    for idx, comp in enumerate(solution, 1):
        msg = f"[{idx:2d}/{len(solution)}] Disassembling {comp}... [OK]"
        print(msg)
        logging.info(msg)

    if failures:
        print(f"\n[2/4] Failure scenario loaded: {len(failures)} failure(s)")
        logging.info(f"[2/4] Failure scenario loaded: {len(failures)} failure(s)")
    else:
        print(f"\n[2/4] No failures (nominal execution)")
        logging.info("[2/4] No failures (nominal execution)")

    print(f"\n[3/4] Executing adaptive scheduler...")
    logging.info("[3/4] Executing adaptive scheduler...")
    state, actions_log = run_adaptive_scheduler(solution, G, targets, failures)
    logging.info(f"Adaptive scheduler finished. Actions taken: {len(actions_log)}")

    elapsed = time.time() - start_time
    targets_achieved = [t for t in targets if t in state.get('done', set()) and t not in state.get('destroyed', set())]
    if not failures:
        targets_achieved = targets
    success_rate = len(targets_achieved) / len(targets) * 100 if targets else 0

    print("\nEXECUTION SUMMARY")
    logging.info("EXECUTION SUMMARY")
    print(f"Status: {'SUCCESS' if success_rate == 100 else 'PARTIAL SUCCESS'}")
    logging.info(f"Status: {'SUCCESS' if success_rate == 100 else 'PARTIAL SUCCESS'}")

    print(f"\nTarget components:")
    logging.info("Target components:")
    for t in targets:
        if t in targets_achieved:
            print(f"  [OK] {t} - Successfully disassembled")
            logging.info(f"[OK] {t} - Successfully disassembled")
        elif t in state.get('destroyed', set()):
            print(f"  [DESTROYED] {t}")
            logging.info(f"[DESTROYED] {t}")
        else:
            print(f"  [FAILED] {t} - NOT ACHIEVED")
            logging.info(f"[FAILED] {t} - NOT ACHIEVED")

    print(f"\nExecution metrics:")
    logging.info("Execution metrics:")
    print(f"  Targets achieved: {len(targets_achieved)}/{len(targets)} ({success_rate:.1f}%)")
    logging.info(f"Targets achieved: {len(targets_achieved)}/{len(targets)} ({success_rate:.1f}%)")
    print(f"  Failures handled: {len(failures)}")
    logging.info(f"Failures handled: {len(failures)}")
    print(f"  Actions taken:    {len(actions_log)}")
    logging.info(f"Actions taken: {len(actions_log)}")
    print(f"  Components done:  {len(state.get('done', set()))}/{len(solution)}")
    logging.info(f"Components done: {len(state.get('done', set()))}/{len(solution)}")
    if state.get('destroyed'):
        print(f"  Destroyed:        {len(state.get('destroyed', set()))} component(s)")
        logging.info(f"Destroyed: {len(state.get('destroyed', set()))} component(s)")
    if state.get('locked'):
        print(f"  Locked:           {len(state.get('locked', set()))} component(s)")
        logging.info(f"Locked: {len(state.get('locked', set()))} component(s)")
    print(f"  Algorithm runtime:   {elapsed:.2f}s")
    logging.info(f"Algorithm runtime: {elapsed:.2f}s")

    # === ECONOMIC ANALYSIS ===
    eco_metrics = compute_economic_metrics(
        G=G,
        nominal_sequence=solution,
        done=state.get('done', set()),
        destroyed=state.get('destroyed', set()),
        skipped=state.get('skipped', set())
    )
    
    print(f"\nECONOMIC ANALYSIS:")
    logging.info("ECONOMIC ANALYSIS:")
    print(f"  Nominal profit (planned):    {eco_metrics['nominal_profit']:.2f} units")
    logging.info(f"Nominal profit (planned): {eco_metrics['nominal_profit']:.2f} units")
    print(f"  Adaptive profit (recovered): {eco_metrics['adaptive_profit']:.2f} units")
    logging.info(f"Adaptive profit (recovered): {eco_metrics['adaptive_profit']:.2f} units")
    print(f"  Profit loss:                 {eco_metrics['profit_loss']:.2f} units ({eco_metrics['profit_loss_pct']:.1f}%)")
    logging.info(f"Profit loss: {eco_metrics['profit_loss']:.2f} units ({eco_metrics['profit_loss_pct']:.1f}%)")
    if eco_metrics['destroyed_value'] > 0:
        print(f"  Value lost (destroyed):      {eco_metrics['destroyed_value']:.2f} units")
        logging.info(f"Value lost (destroyed): {eco_metrics['destroyed_value']:.2f} units")
    if eco_metrics['skipped_value'] > 0:
        print(f"  Value skipped (bypassed):    {eco_metrics['skipped_value']:.2f} units")
        logging.info(f"Value skipped (bypassed): {eco_metrics['skipped_value']:.2f} units")

    # === TIME ANALYSIS ===
    time_metrics = compute_time_metrics(
        G=G,
        nominal_sequence=solution,
        done=state.get('done', set()),
        destroyed=state.get('destroyed', set()),
        skipped=state.get('skipped', set())
    )
    
    print(f"\nTIME ANALYSIS:")
    logging.info("TIME ANALYSIS:")
    print(f"  Nominal time (planned):      {format_time(time_metrics['nominal_time'])}")
    logging.info(f"Nominal time (planned): {format_time(time_metrics['nominal_time'])}")
    print(f"  Adaptive time (actual):      {format_time(time_metrics['adaptive_time'])}")
    logging.info(f"Adaptive time (actual): {format_time(time_metrics['adaptive_time'])}")
    if time_metrics['time_diff'] > 0:
        print(f"  Time saved:                  {format_time(time_metrics['time_diff'])} ({time_metrics['time_diff_pct']:.1f}%)")
        logging.info(f"Time saved: {format_time(time_metrics['time_diff'])} ({time_metrics['time_diff_pct']:.1f}%)")
    else:
        print(f"  Time overhead:               {format_time(-time_metrics['time_diff'])} ({-time_metrics['time_diff_pct']:.1f}%)")
        logging.info(f"Time overhead: {format_time(-time_metrics['time_diff'])} ({-time_metrics['time_diff_pct']:.1f}%)")
    if time_metrics['destroyed_time'] > 0:
        print(f"  Time on destroyed (lost):    {format_time(time_metrics['destroyed_time'])}")
        logging.info(f"Time on destroyed (lost): {format_time(time_metrics['destroyed_time'])}")
    if time_metrics['skipped_time'] > 0:
        print(f"  Time saved (skipped):        {format_time(time_metrics['skipped_time'])}")
        logging.info(f"Time saved (skipped): {format_time(time_metrics['skipped_time'])}")

    # === GLOBAL SCORE (weighted) ===
    nominal_score = score_weighted(solution, G)
    # Build adaptive sequence (done components in order)
    adaptive_seq = [c for c in solution if c in state.get('done', set()) and c not in state.get('destroyed', set())]
    adaptive_score_val = score_weighted(adaptive_seq, G)
    
    print(f"\nGLOBAL SCORE :")
    logging.info("GLOBAL SCORE :")
    print(f"  Nominal score:               {nominal_score:.2f}")
    logging.info(f"Nominal score: {nominal_score:.2f}")
    print(f"  Adaptive score:              {adaptive_score_val:.2f}")
    logging.info(f"Adaptive score: {adaptive_score_val:.2f}")
    score_diff = nominal_score - adaptive_score_val
    score_diff_pct = (score_diff / nominal_score * 100) if nominal_score > 0 else 0.0
    print(f"  Score loss:                  {score_diff:.2f} ({score_diff_pct:.1f}%)")
    logging.info(f"Score loss: {score_diff:.2f} ({score_diff_pct:.1f}%)")

    if actions_log:
        print(f"\nActions breakdown:")
        logging.info("Actions breakdown:")
        for act in actions_log:
            status = "[OK]" if act['success'] else "[FAILED]"
            print(f"  {status} {act['comp_id']:8s} {act['symptom']:20s} -> {act['action']:12s}")
            logging.info(f"{status} {act['comp_id']:8s} {act['symptom']:20s} -> {act['action']:12s}")

    print(f"\nDetailed logs: {log_dir}\n")
    logging.info(f"Detailed logs: {log_dir}")

    # Generate narrative report (with economic, time analysis and global score)
    generate_narrative_report(solution, state, actions_log, failures, targets, log_dir, G=G, eco_metrics=eco_metrics, time_metrics=time_metrics)
    
    log_results(instance, solution, state, actions_log, failures, log_dir)

    if success_rate < 100:
        print(f"\n[WARNING] Some targets were not achieved.")
        logging.warning("Some targets were not achieved.")
    print(f"\n[SUCCESS] Pipeline finished successfully")
    logging.info("Pipeline finished successfully")

if __name__ == "__main__":
    main()


