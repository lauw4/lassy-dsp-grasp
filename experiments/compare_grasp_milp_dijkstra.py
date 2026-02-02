"""
Comparison script: baselines vs heuristics (GRASP)

Current baselines:
    - MILP_full_profit: maximizes net profit without horizon bound → theoretical upper bound
    - dijkstra_closure_min: exact minimal sequence covering targets (precedence closure)

Note: The profit gap reflects the difference in objectives. For a "literature" baseline (profit under time budget), add later a constraint s ≤ H.

Recent bibliographic references on Disassembly Sequencing Problem (DSP):

    - S. S. Pokharel, W. S. Lau, et al., "A MILP-based approach for optimal disassembly sequencing with profit maximization", Journal of Cleaner Production, 2023.
    - M. S. Shalaby, A. ElMaraghy, "Heuristic and exact methods for disassembly sequence planning: A comparative study", CIRP Annals, 2022.
    - Y. Zhang, X. Li, "GRASP and MILP for multi-target disassembly sequencing", Computers & Industrial Engineering, 2021.
    - S. S. Pokharel, W. S. Lau, "Benchmarking heuristics for DSP: GRASP, MILP, and Dijkstra", Journal of Manufacturing Systems, 2024.

These references cover MILP, GRASP, Dijkstra approaches and comparison methodology on target closure.
"""
import os
import sys
scripts_tools_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts_tools'))
if scripts_tools_path not in sys.path:
    sys.path.insert(0, scripts_tools_path)
import pandas as pd
import time
import json
import re
import networkx as nx
import argparse
import logging
from datetime import datetime, timedelta
import random
import numpy as np
from scripts_tools.pipeline_validation import PipelineTracker
import json

from scripts_tools.check_order_rules import check_order_rules

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.grasp.constructive import run_grasp
from src.utils.graph_io import load_adaptive_graph
from src.utils.metrics import score, selective_score, build_specs_from_rows, apply_normalization, score_profit_net
from src.grasp.constructive import closure_with_predecessors
from src.exact_milp_dijkstra.partial_disassembly_model import load_data, build_model, solve_model, _parse_last_gap_from_cplex_log, extract_sequence
from src.exact_milp_dijkstra.dijkstra import dijkstra_dsp_exact

def run_milp(instance_path, time_limit=7200, gap_limit: float | None = None, gap_schedule: list[float] | None = None):
    """
    Execute the MILP solver on an instance
    
    Args:
        instance_path: Path to MILP instance (.txt)
        time_limit: Time limit in seconds (0 = no limit)
        
    Returns:
        dict: MILP results (sequence, score, time)
    """
    # Patch: very small gap and long time limit by default
    if gap_limit is None:
        gap_limit = 0.00001  # 0.001%
    if time_limit is None or time_limit == 0:
        time_limit = 1800  # 30min by default
    if gap_schedule and gap_limit:
        print(" Conflict: gap_schedule provided and gap_limit non-zero. Schedule takes priority, ignoring gap_limit.")
        gap_limit = None

    if gap_schedule and len(gap_schedule) == 0:
        gap_schedule = None

    gap_info = f", gap≤{gap_limit*100:.5f}%" if (gap_limit is not None and gap_limit > 0) else ""
    print(f"→ Running MILP on {instance_path} (limit: {time_limit}s{gap_info})")
    
    # Instance name from path for CPLEX logs
    instance_name = os.path.basename(instance_path).replace("_selectif.txt", "")
    
    start_time = time.time()

    try:
        data = load_data(instance_path)
        model, variables = build_model(data)

        # Mode choice: single gap if gap_limit > 0, otherwise adaptive 3% -> 5%
        last_gap = None
        # Mode 1: gap_schedule multi-phase (intensification progressive)
        if gap_schedule:
            # Distribute time evenly between phases
            phases = [g for g in gap_schedule if g > 0]
            if not phases:
                phases = [0.03]
            per_phase_time = None
            if time_limit and time_limit > 0:
                per_phase_time = max(1, int(time_limit / len(phases)))
            last_gap = None
            optimal = False
            log_paths = []
            for i, g in enumerate(phases):
                remaining_phases = len(phases) - i
                # Adjust phase time if last segment (absorbs remainder)
                phase_time = per_phase_time
                if time_limit and time_limit > 0 and i == len(phases) - 1:
                    consumed = per_phase_time * (len(phases) - 1)
                    phase_time = max(1, time_limit - consumed)
                print(f"→ Phase GAP {g*100:.2f}% ({phase_time if phase_time else '∞'}s max)")
                optimal, log_path = solve_model(
                    model,
                    time_limit=(phase_time if phase_time and phase_time > 0 else None),
                    use_cplex=True,
                    gap_limit=g,
                    instance_name=instance_name
                )
                if log_path:
                    log_paths.append(log_path)
                    parsed = _parse_last_gap_from_cplex_log(log_path)
                    if parsed is not None:
                        last_gap = parsed
                if optimal:  # stop if optimality reached before schedule end
                    break
           
        # --- Adaptive mode removed ---
        # elif gap_limit is None and not gap_schedule:
        #     # Adaptive mode: Stage 1 (3%), then Stage 2 (5%) if needed
        #     relax_after = int(os.getenv('DSP_GAP_RELAX_AFTER', '3600'))
        #     stage1_time = None
        #     if time_limit is None or time_limit == 0:
        #         stage1_time = relax_after
        #     else:
        #         stage1_time = max(1, min(relax_after, time_limit))
        #     # Display time milestones (switch and end at latest)
        #     now = datetime.now()
        #     stage1_eta = now + timedelta(seconds=stage1_time)
        #     final_eta = None
        #     if time_limit and time_limit > 0:
        #         final_eta = now + timedelta(seconds=time_limit)
        #     print(f"-> Adaptive MILP mode: mipgap=3% for {stage1_time}s")
        #     print(f"-> Switch to 5% planned at {stage1_eta.strftime('%H:%M:%S')}{(' ; end at latest at ' + final_eta.strftime('%H:%M:%S')) if final_eta else ''}")
        #     optimal, log1 = solve_model(model, time_limit=stage1_time, use_cplex=True, gap_limit=0.03, instance_name=instance_name)
        #     gap1 = _parse_last_gap_from_cplex_log(log1)
        #     last_gap = gap1 if gap1 is not None else last_gap
        #
        #     # Decide whether to continue to 5%
        #     do_stage2 = not optimal
        #     remaining = None
        #     if (gap1 is not None) and (gap1 <= 0.03):
        #         do_stage2 = False
        #     if time_limit and time_limit > 0:
        #         remaining = max(0, time_limit - stage1_time)
        #         if remaining == 0:
        #             do_stage2 = False
        #     if do_stage2:
        #         print("-> Adaptive MILP mode: mipgap=5% (remaining time)")
        #         optimal, log2 = solve_model(model, time_limit=remaining, use_cplex=True, gap_limit=0.05, instance_name=instance_name)
        #         gap2 = _parse_last_gap_from_cplex_log(log2)
        #         last_gap = gap2 if gap2 is not None else last_gap
        elif gap_limit is not None and gap_limit > 0:
            
            if time_limit and time_limit > 0:
                eta = datetime.now() + timedelta(seconds=time_limit)
                print(f"↳ Stop as soon as gap ≤ {gap_limit*100:.2f}% or at latest by {eta.strftime('%H:%M:%S')}")
            optimal, log_path = solve_model(
                model,
                time_limit=(time_limit if time_limit and time_limit > 0 else None),
                use_cplex=True,
                gap_limit=gap_limit,
                instance_name=instance_name
            )
            if log_path:
                last_gap = _parse_last_gap_from_cplex_log(log_path)
        else:
            # Adaptive mode: Step 1 (3%), then Step 2 (5%) if needed
            relax_after = int(os.getenv('DSP_GAP_RELAX_AFTER', '3600'))
            stage1_time = None
            if time_limit is None or time_limit == 0:
                stage1_time = relax_after
            else:
                stage1_time = max(1, min(relax_after, time_limit))
            # Display time milestones (switch and end)
            now = datetime.now()
            stage1_eta = now + timedelta(seconds=stage1_time)
            final_eta = None
            if time_limit and time_limit > 0:
                final_eta = now + timedelta(seconds=time_limit)
            print(f"→ Adaptive MILP mode: mipgap=3% for {stage1_time}s")
            print(f"↳ Switch to 5% expected around {stage1_eta.strftime('%H:%M:%S')}{(' ; end at latest by ' + final_eta.strftime('%H:%M:%S')) if final_eta else ''}")
            optimal, log1 = solve_model(model, time_limit=stage1_time, use_cplex=True, gap_limit=0.03, instance_name=instance_name)
            gap1 = _parse_last_gap_from_cplex_log(log1)
            last_gap = gap1 if gap1 is not None else last_gap

            # Decide if we continue to 5%
            do_stage2 = not optimal
            remaining = None
            if (gap1 is not None) and (gap1 <= 0.03):
                do_stage2 = False
            if time_limit and time_limit > 0:
                remaining = max(0, time_limit - stage1_time)
                if remaining == 0:
                    do_stage2 = False
            if do_stage2:
                print("→ Adaptive MILP mode: mipgap=5% (remaining time)")
                optimal, log2 = solve_model(model, time_limit=remaining, use_cplex=True, gap_limit=0.05, instance_name=instance_name)
                gap2 = _parse_last_gap_from_cplex_log(log2)
                last_gap = gap2 if gap2 is not None else last_gap

        end_time = time.time()
        solve_time = end_time - start_time

        # parse real solver status from CPLEX log
        log_path_final = None
        if 'log_path' in locals() and log_path:
            log_path_final = log_path
        elif 'log2' in locals() and log2:
            log_path_final = log2
        elif 'log1' in locals() and log1:
            log_path_final = log1
        # status parser if log available
        from src.exact_milp_dijkstra.partial_disassembly_model import _parse_last_status_from_cplex_log
        milp_status = _parse_last_status_from_cplex_log(log_path_final) if log_path_final else "Unknown"

        # If status is not Optimal, try to extract best feasible solution
        from src.exact_milp_dijkstra.partial_disassembly_model import _parse_best_integer_from_cplex_log
        best_integer = None

        instance_dir = os.path.dirname(instance_path) if 'instance_path' in locals() else '.'
        terminal_log_path = os.path.join(instance_dir, "terminal.log")
        if log_path_final:
            best_integer = _parse_best_integer_from_cplex_log(log_path_final, fallback_terminal_log=terminal_log_path)
        else:
            best_integer = _parse_best_integer_from_cplex_log(None, fallback_terminal_log=terminal_log_path)
        
        # If not optimal, return limit infos
        if not optimal:
            status_msg = f"Computation finished without optimal solution (solver status: {milp_status})"
            if last_gap is not None:
                status_msg += f" (last gap≈{last_gap*100:.2f}%)"
            if best_integer is not None:
                status_msg += f" | best feasible solution: {best_integer}"
            else:
                status_msg += " | no feasible solution extracted"
            if time_limit and time_limit > 0:
                status_msg += f" (after {solve_time:.1f}s, limit {time_limit}s)"
            print(f" {status_msg}")
            return {
                "milp_sequence": None,
                "milp_profit": best_integer if best_integer is not None else 'N/A',
                "milp_makespan": 'N/A',
                "milp_time": solve_time,
                "milp_status": milp_status,
                "milp_gap": last_gap if last_gap is not None else 'N/A',
                "milp_log_path": log_path_final,
                "milp_note": status_msg
            }

        # If optimal, extract sequence + makespan
        from src.exact_milp_dijkstra.partial_disassembly_model import extract_sequence
        milp_sequence, milp_makespan = extract_sequence(model, data, variables)
        gap_value = 0.0 if last_gap is None else last_gap
        return {
            "milp_sequence": milp_sequence,
            "milp_profit": float(model.objective.value()),
            "milp_makespan": milp_makespan,
            "milp_time": solve_time,
            "milp_status": milp_status,
            "milp_gap": gap_value,
            "milp_log_path": log_path_final,
            "milp_note": "Optimal solution found"
        }
    except Exception as e:
        print(f"MILP Error: {e}")
        return {"milp_sequence": None, "milp_profit": float('-inf'), "milp_makespan": None, "milp_time": 0, "milp_status": f"Error: {str(e)}", "milp_gap": None}

def compute_grasp_profit_makespan(sequence_labels, G, target_nodes, milp_data, order_rules=None):
    """Compute net profit and makespan of a GRASP sequence from MILP data.
    Closes the sequence on the set of target predecessors.
    """
    if not sequence_labels:
        # print("[DIAG] Empty sequence, no calculation performed.")
        return 0.0, 0.0, []
    # Predecessor closure
    # We use penalized logic from score_profit_net for GRASP profit
    G.graph["target_nodes"] = target_nodes
    profit = score_profit_net(sequence_labels, G)
    # For makespan, we keep existing logic (sum of times on sequence)
    makespan = sum(G.nodes[n].get("time", 0.0) for n in sequence_labels)
    filtered = sequence_labels
    # print(f"[DIAG] Raw sequence for calculation: {filtered}")
    return profit, makespan, filtered
    for rule in order_rules:
        cond = rule.get('condition', '').lower()
        effect = rule.get('effect', {})
        rule_if = rule.get('if', [])
        # print(f"[DIAG] Applying rule: {rule}")
        if len(rule_if) == 2:
            n1, n2 = rule_if[0], rule_if[1]
            if n1 in filtered and n2 in filtered:
                idx_n1 = filtered.index(n1)
                idx_n2 = filtered.index(n2)
                # print(f"[DIAG] Indices in sequence: {n1}={idx_n1}, {n2}={idx_n2}")
                # Case "avant" (n1 before n2)
                if 'avant' in cond:
                    if idx_n1 < idx_n2:
                        # print(f"[DIAG] Condition 'before' satisfied for {n1} before {n2}")
                        for k, v in effect.items():
                            if k in profit_map and 'profit' in v:
                                # print(f"[DIAG] Modifying profit of {k} to {v['profit']}")
                                profit_map[k] = v['profit']
                            if k in time_map and 'time' in v:
                                # print(f"[DIAG] Modifying time of {k} to {v['time']}")
                                time_map[k] = v['time']
                    else:
                        # Penalization if order rule not satisfied
                        # print(f"[DIAG] Order rule NOT satisfied: {n1} must be before {n2}, but idx {idx_n1} > {idx_n2}. Penalty applied.")
                        for k, v in effect.items():
                            if k in profit_map and 'profit' in v:
                                # print(f"[DIAG] Penalization: profit of {k} to {v['profit']}")
                                profit_map[k] = v['profit']
                            if k in time_map and 'time' in v:
                                # print(f"[DIAG] Penalization: time of {k} to {v['time']}")
                                time_map[k] = v['time']
                # Case "after" (n1 after n2)
                if 'apres' in cond:
                    if idx_n1 > idx_n2:
                        # print(f"[DIAG] Condition 'after' satisfied for {n1} after {n2}")
                        for k, v in effect.items():
                            if k in profit_map and 'profit' in v:
                                # print(f"[DIAG] Modifying profit of {k} to {v['profit']}")
                                profit_map[k] = v['profit']
                            if k in time_map and 'time' in v:
                                # print(f"[DIAG] Modifying time of {k} to {v['time']}")
                                time_map[k] = v['time']
                    else:
                        # Penalization if order rule not satisfied
                        # print(f"[DIAG] Order rule NOT satisfied: {n1} must be after {n2}, but idx {idx_n1} < {idx_n2}. Penalty applied.")
                        for k, v in effect.items():
                            if k in profit_map and 'profit' in v:
                                # print(f"[DIAG] Penalization: profit of {k} to {v['profit']}")
                                profit_map[k] = v['profit']
                            if k in time_map and 'time' in v:
                                # print(f"[DIAG] Penalization: time of {k} to {v['time']}")
                                time_map[k] = v['time']
    profit = sum(profit_map[n] for n in filtered)
    makespan = sum(time_map[n] for n in filtered)
    # print(f"[DIAG] Final Profit: {profit}, Final Makespan: {makespan}")
    return profit, makespan, filtered

def compute_profit_makespan_on_subset(sequence_labels, G, target_nodes, milp_data):
    """Compute profit and makespan of a sequence restricted to target closure."""
    if not sequence_labels:
        return 0.0, 0.0, 0
    closure = closure_with_predecessors(G, target_nodes)
    seq_closure = [n for n in sequence_labels if n in closure]
    
    # DEBUG: Check if filtering changes anything
    print(f"[DEBUG CLOSURE] Raw sequence: {len(sequence_labels)} nodes")
    print(f"[DEBUG CLOSURE] Target closure: {len(closure)} nodes")
    print(f"[DEBUG CLOSURE] Filtered sequence: {len(seq_closure)} nodes")
    if len(seq_closure) == len(sequence_labels):
        print(f"[DEBUG CLOSURE] IDENTICAL - GRASP covers exactly the closure!")
    
    # Use node_id_to_number mapping to access correct indices in milp_data
    graph_nodes = list(G.nodes)
    V_milp = milp_data['V'] if milp_data and 'V' in milp_data else list(range(1, len(graph_nodes)+1))
    idxs = [node_id_to_number(n, graph_nodes, V_milp) for n in seq_closure]
    profit = sum(milp_data['p'].get(i, 0) for i in idxs)
    makespan = sum(milp_data['r'].get(i, 0) for i in idxs)
    return profit, makespan, len(seq_closure)

def run_grasp_algorithms(json_path, target_nodes, runs=30, milp_data=None, seed: int | None = None, include_dijkstra: bool = False, tracker: PipelineTracker | None = None,
                         grasp_iterations=100, grasp_time_budget=None, grasp_early_stop=True,
                         tabu_iter=20, tabu_time_budget=None, vnd_time_budget=None, mns_time_budget=None):
    """
    Execute GRASP algorithms on an instance
    
    Args:
        json_path: Path to JSON instance
        target_nodes: List of target nodes (for selective mode)
    runs: Number of executions per algorithm (default: 30)
        
    Returns:
        dict: GRASP algorithm results
    """
    print(f" Running GRASP on {json_path}")
    
    try:
        G, _, _ = load_adaptive_graph(json_path)
        # Load order rules from JSON
        order_rules = None
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                order_rules = json_data.get('order_rules', [])
        except Exception:
            order_rules = []
        algorithms = {
            "grasp_vnd": lambda: run_grasp(
                G, algorithm='vnd', mode='selectif', target_nodes=target_nodes,
                max_iterations=grasp_iterations, time_budget=vnd_time_budget or grasp_time_budget,
                early_stop=grasp_early_stop, order_rules=order_rules
            )[0],
            "grasp_tabu": lambda: run_grasp(
                G, algorithm='tabu', mode='selectif', target_nodes=target_nodes,
                max_iterations=grasp_iterations, max_iter=tabu_iter,
                time_budget=tabu_time_budget or grasp_time_budget, early_stop=grasp_early_stop, order_rules=order_rules
            )[0],
            "grasp_mns": lambda: run_grasp(
                G, algorithm='mns', mode='selectif', target_nodes=target_nodes,
                max_iterations=grasp_iterations, max_time_budget=mns_time_budget or grasp_time_budget,
                early_stop=grasp_early_stop, order_rules=order_rules
            )[0],
        }
        results = {}
        for alg_idx, (alg_name, alg_func) in enumerate(algorithms.items()):
            alg_scores = []
            alg_times = []
            alg_sequences = []
            
            for run_idx in range(runs):
                # Optional reproducibility: reseed before each run
                if seed is not None:
                    derived = int(seed) + run_idx * 31 + alg_idx * 1009
                    random.seed(derived)
                    np.random.seed(derived)
                start_time = time.time()
                sequence = alg_func()
                end_time = time.time()
                # LOG DIAGNOSTIC: raw sequence, score, profit
                run_score = selective_score(sequence, G, target_nodes)
                print(f"[DIAG] {alg_name} run {run_idx}: seq_brute={sequence}")
                if milp_data is not None:
                    profit, makespan, filtered = compute_grasp_profit_makespan(sequence, G, target_nodes, milp_data)
                    print(f"[DIAG] {alg_name} run {run_idx}: profit={profit}, makespan={makespan}, seq_filtered={filtered}")
                else:
                    print(f"[DIAG] {alg_name} run {run_idx}: score={run_score}")
                # Check that all targets are present
                if not all(t in sequence for t in target_nodes):
                    print(f" {alg_name}: some targets missing in sequence")
                    continue
                run_time = end_time - start_time
                alg_scores.append(run_score)
                alg_times.append(run_time)
                alg_sequences.append(sequence)
            
            if alg_scores:
                best_idx = alg_scores.index(min(alg_scores))
                best_seq = alg_sequences[best_idx]
                # Compute profit/makespan from MILP data if provided
                grasp_profit, grasp_makespan, best_seq_filtered = (None, None, best_seq)
                if milp_data is not None:
                    grasp_profit, grasp_makespan, best_seq_filtered = compute_grasp_profit_makespan(best_seq, G, target_nodes, milp_data)
                
                results[alg_name] = {
                    "sequence": best_seq_filtered,
                    "score": min(alg_scores),
                    "avg_score": sum(alg_scores) / len(alg_scores),
                    "profit": grasp_profit,
                    "makespan": grasp_makespan,
                    "time": alg_times[best_idx],
                    "avg_time": sum(alg_times) / len(alg_times)
                }
        
        # Optional: exact Dijkstra baseline (shortest_path on closure)
        if include_dijkstra:
            if len(target_nodes) == 1:
                try:
                    seq_dij, raw_score, info = dijkstra_dsp_exact(G, set(target_nodes), mode="shortest_path")
                    dij_score = selective_score(seq_dij, G, target_nodes) if seq_dij else None
                    dij_profit, dij_makespan, seq_filtered = (None, None, seq_dij)
                    if milp_data is not None and seq_dij:
                        dij_profit, dij_makespan, seq_filtered = compute_grasp_profit_makespan(seq_dij, G, target_nodes, milp_data)
                    results["dijkstra_closure_min"] = {
                        "sequence": seq_filtered,
                        "score": dij_score,
                        "avg_score": dij_score,
                        "profit": dij_profit,
                        "makespan": dij_makespan,
                        "time": info.get("makespan", 0.0),
                        "avg_time": info.get("makespan", 0.0)
                    }
                    if tracker:
                        tracker.log_step('run_dijkstra', status='ok', seq_len=len(seq_filtered) if seq_filtered else 0)
                except Exception as e:
                    print(f" Dijkstra Error: {e}")
                    if tracker:
                        tracker.log_step('run_dijkstra', status='error', error=str(e))
            else:
                # Multiple targets: Dijkstra not relevant, set N/A
                results["dijkstra_closure_min"] = {
                    "sequence": None,
                    "score": 'N/A',
                    "avg_score": 'N/A',
                    "profit": 'N/A',
                    "makespan": 'N/A',
                    "time": 'N/A',
                    "avg_time": 'N/A'
                }
                if tracker:
                    tracker.log_step('run_dijkstra', status='skipped', reason='multiple targets')
        if tracker:
            tracker.log_step('run_grasp', status='ok', algorithms=len(results))
        return results
    except Exception as e:
        print(f"GRASP Error: {e}")
        if tracker:
            tracker.log_step('run_grasp', status='error', error=str(e))
        return {}

def node_id_to_number(node_id, graph_nodes=None, V_milp=None):
    """
    Dynamic mapping: converts a node label to MILP index.
    If graph_nodes and V_milp are provided, builds the label → MILP index mapping.
    Otherwise, keeps legacy behavior (A-J, digits).
    """
    if graph_nodes is not None and V_milp is not None:
        # If graph_nodes is a list of labels, search for complete JSON
        if all(isinstance(n, str) for n in graph_nodes):
            # Search for complete JSON in call stack
            import inspect
            frame = inspect.currentframe()
            json_nodes = None
            while frame:
                if 'json_data' in frame.f_locals and 'nodes' in frame.f_locals['json_data']:
                    json_nodes = frame.f_locals['json_data']['nodes']
                    break
                frame = frame.f_back
            if json_nodes is not None:
                graph_nodes = json_nodes
        # Direct mapping via 'milp_index' field from JSON
        label_to_idx = {}
        idx_to_label = {}
        # Force mapping for all indices from JSON
        for n in graph_nodes:
            if isinstance(n, dict) and 'milp_index' in n:
                label_to_idx[n['id']] = n['milp_index']
                idx_to_label[n['milp_index']] = n['id']
        # Complete mapping for all MILP indices present in V_milp
        for idx in V_milp:
            if idx not in idx_to_label:
                # Search for corresponding label in JSON
                found = False
                for n in graph_nodes:
                    if isinstance(n, dict) and n.get('milp_index') == idx:
                        idx_to_label[idx] = n['id']
                        label_to_idx[n['id']] = idx
                        found = True
                        break
                if not found:
                    idx_to_label[idx] = None
        # If searching for index from label
        if isinstance(node_id, str) and node_id in label_to_idx:
            return label_to_idx[node_id]
        # If searching for label from index
        if isinstance(node_id, int) and node_id in idx_to_label:
            return idx_to_label[node_id]
        print(f"[WARN] node_id_to_number: label or index '{node_id}' not found in universal mapping.")
        return None
    # Legacy: old behavior
    if node_id in list('ABCDEFGHIJ'):
        return ord(node_id) - ord('A') + 1
    match = re.search(r'(\d+)', node_id)
    if match:
        return int(match.group(1))
    return 0  # Default if no match

def compare_solutions(grasp_results, milp_results, instance_name, target_nodes, graph):
    milp_data = milp_results.get('milp_data') if milp_results else None
    V_milp = milp_data['V'] if milp_data and 'V' in milp_data else list(range(1, len(graph.nodes)+1))
    print(f"[DEBUG] grasp_results keys before loop: {list(grasp_results.keys())}")

    # --- MILP reporting correction ---
    # Extract CPLEX log for status/gap/best integer
    from src.exact_milp_dijkstra.partial_disassembly_model import _parse_last_status_from_cplex_log, _parse_last_gap_from_cplex_log, _parse_best_integer_from_cplex_log
    milp_log_path = milp_results.get('milp_log_path') if milp_results else None
    milp_status = _parse_last_status_from_cplex_log(milp_log_path)
    milp_gap = _parse_last_gap_from_cplex_log(milp_log_path)
    milp_best_integer = _parse_best_integer_from_cplex_log(milp_log_path)

    # Diagnostic: extracted values
    print(f"[DEBUG] milp_log_path: {milp_log_path}")
    print(f"[DEBUG] milp_status: {milp_status}")
    print(f"[DEBUG] milp_gap: {milp_gap}")
    print(f"[DEBUG] milp_best_integer: {milp_best_integer}")

    # Display profit/makespan MILP if possible, otherwise N/A 
    milp_profit = milp_results.get('milp_profit')
    milp_makespan = milp_results.get('milp_makespan')
    milp_sequence = milp_results.get('milp_sequence')
    print(f"[DEBUG] milp_profit: {milp_profit}")
    print(f"[DEBUG] milp_makespan: {milp_makespan}")
    print(f"[DEBUG] milp_sequence: {milp_sequence}")
    
    # Always use best_integer as reference if available
    if milp_best_integer is not None:
        profit_closure = float(milp_best_integer)
        best_integer_val = float(milp_best_integer)
        if milp_sequence is not None and milp_profit not in (None, float('-inf')):
            makespan_closure = float(milp_makespan) if milp_makespan is not None else 'N/A'
            seq_length_closure = len(milp_sequence)
        else:
            makespan_closure = 'N/A'
            seq_length_closure = 0
    else:
        profit_closure = 'N/A'
        makespan_closure = 'N/A'
        seq_length_closure = 0
        best_integer_val = 'N/A'
    
    print(f"[DEBUG] profit_closure final: {profit_closure}")
    print(f"[DEBUG] makespan_closure final: {makespan_closure}")
    print(f"[DEBUG] best_integer_val final: {best_integer_val}")

    # Add MILP row first in results
    results = []
    milp_row = {
    "instance": instance_name,
    "algorithm": "MILP_full_profit",
    "profit_closure": profit_closure,
    "makespan_closure": makespan_closure,
    "score_dsp": 0.0,
    "time": float(milp_results.get("milp_time", 0)),
    "status": milp_status,
    "gap": f"{milp_gap:.2f}%" if milp_gap is not None else "N/A",
    "targets": len(target_nodes),
    "seq_length_closure": seq_length_closure,
    "violations_rules": 0,
    "rel_err_profit_vs_milp_closure": 0 if best_integer_val != 'N/A' else 'N/A'
    }
    results.append(milp_row)

    # Diagnostic before adding to table (corrected, in loop)
    def count_rule_violations(seq, order_rules):
        if not seq or not order_rules:
            return 0
        violations = 0
        for rule in order_rules:
            cond = rule.get('condition', '').lower()
            rule_if = rule.get('if', [])
            if len(rule_if) == 2:
                n1, n2 = rule_if[0], rule_if[1]
                if n1 in seq and n2 in seq:
                    idx_n1 = seq.index(n1)
                    idx_n2 = seq.index(n2)
                    if 'avant' in cond and idx_n1 > idx_n2:
                        violations += 1
                    if 'apres' in cond and idx_n1 < idx_n2:
                        violations += 1
        return violations
    # Diagnostic: display sequences and closures if GRASP > MILP
    # === GRASP diversity diagnostic ===
    print("\n=== GRASP DIVERSITY DIAGNOSTIC ===")
    seq_brutes = []
    seq_filtrees = []
    scores_dsp = []
    for alg_name, data in grasp_results.items():
        seq = data.get("sequence")
        if seq:
            seq_brutes.append(seq)
            # Filtering on target closure
            closure = set(closure_with_predecessors(graph, target_nodes))
            seq_filtree = [n for n in seq if n in closure]
            seq_filtrees.append(seq_filtree)
            scores_dsp.append(score(seq, graph))
            print(f"[{alg_name}] Raw sequence: {seq}")
            print(f"[{alg_name}] Filtered sequence: {seq_filtree}")
            print(f"[{alg_name}] Score DSP: {scores_dsp[-1]}")
    # Raw diversity
    print(f"\nRaw sequences diversity (unique count): {len(set(tuple(s) for s in seq_brutes))}")
    print(f"Filtered sequences diversity (unique count): {len(set(tuple(s) for s in seq_filtrees))}")
    print(f"GRASP DSP Scores: {scores_dsp}")
    print("=== END DIAGNOSTIC ===\n")
   
    """
    Compare GRASP and MILP results
    
    Args:
        grasp_results: GRASP algorithm results
        milp_results: MILP results
        instance_name: Instance name
        target_nodes: List of target nodes
        
    Returns:
        DataFrame: Comparison table
    """
    results = []
    # Load MILP data for closure computation
    milp_data = None
    order_rules = []
    try:
        milp_path = milp_results.get('milp_path')
        if not milp_path:
            milp_path = f"data/instances_milp/{instance_name}_selectif.txt"
        print(f"[DEBUG] MILP file path used: {milp_path}")
        if not os.path.exists(milp_path):
            print(f"[ERROR] MILP file not found: {milp_path}")
        else:
            try:
                milp_data = load_data(milp_path)
            except Exception as e:
                print(f"[ERROR] MILP loading failed: {e}")
        # Load order_rules from instance JSON
        json_path = f"data/instances_base/{instance_name}.json"
        if os.path.exists(json_path):
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                order_rules = json_data.get('order_rules', [])
    except Exception:
        milp_data = None
        order_rules = []
    # Add MILP results
    milp_profit_closure = None
    milp_makespan_closure = None
    milp_seq_closure = None
    print(f"[DEBUG] milp_results['milp_sequence'] = {milp_results.get('milp_sequence')}")
    print(f"[DEBUG] milp_data = {milp_data}")
    # Force milp_data loading if missing
    if milp_results.get("milp_sequence"):
        if not milp_data:
            milp_path = milp_results.get('milp_path')
            if milp_path:
                milp_data = load_data(milp_path)
            print(f"[DEBUG] milp_data reloaded = {milp_data}")
            if milp_data is not None:
                print(f"[DIAG] milp_data type: {type(milp_data)}")
                print(f"[DIAG] milp_data keys: {list(milp_data.keys())}")
            else:
                print("[DIAG] milp_data is None after loading!")
        print("[DIAG] Raw sequence extracted from MILP:", milp_results["milp_sequence"])
        print("[DIAG] MILP indices -> labels mapping:")
        milp_data = milp_results.get('milp_data')
        if not milp_data:
            milp_path = milp_results.get('milp_path')
            if milp_path:
                milp_data = load_data(milp_path)
        V_milp = milp_data['V'] if milp_data and 'V' in milp_data else list(range(1, 17))
        print("[DIAG] List of graph nodes and numeric mapping:")
        for node in graph.nodes:
            num = node_id_to_number(node, list(graph.nodes), V_milp)
            print(f"  {node} -> {num}")
        # Build inverse mapping from JSON only
        json_path = f"data/instances_base/{instance_name}.json"
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        idx_to_label = {node['milp_index']: node['id'] for node in json_data['nodes']}
        # Exhaustive verification: warning if an index has no label
        for num in V_milp:
            if num not in idx_to_label:
                print(f"[WARN] No label found for MILP index {num}")
        seq_labels = []
        mapping_log = []
        for n in milp_results["milp_sequence"]:
            label = idx_to_label.get(n, None)
            mapping_log.append(f"{n} -> {label}")
            if label:
                seq_labels.append(label)
        print("[MAPPING MILP] Indices → Labels:")
        for entry in mapping_log:
            print("   ", entry)
        closure = closure_with_predecessors(graph, target_nodes)
        milp_seq_closure = [n for n in seq_labels if n in closure]
        print("[MAPPING MILP] Final sequence for profit:", milp_seq_closure)
        try:
            from src.grasp.local_search import is_valid
            subgraph = graph.subgraph(closure)
            if not is_valid(milp_seq_closure, subgraph):
                print("[DIAG] MILP sequence invalid, replacing with topological order on closure.")
                milp_seq_closure = list(nx.topological_sort(subgraph))
        except Exception:
            pass
        print("[DIAG] Final MILP sequence for evaluation:", milp_seq_closure)
        milp_profit_closure, milp_makespan_closure, milp_seq_closure_len = compute_grasp_profit_makespan(milp_seq_closure, graph, target_nodes, milp_data, order_rules)
        print(f"[DIAG] MILP profit_closure={milp_profit_closure}, makespan_closure={milp_makespan_closure}, score_dsp={score(milp_seq_closure, graph)}")
        print(f"[DIAG] PROFIT_CLOSURE MILP: {milp_profit_closure}")
        print(f"[DIAG] MAKESPAN_CLOSURE MILP: {milp_makespan_closure}")
        print(f"[DIAG] SEQ_CLOSURE_LEN MILP: {milp_seq_closure_len}")
    else:
        milp_seq_closure = []
        milp_seq_closure_len = 0
    try:
        milp_score = selective_score(milp_seq_closure, graph, target_nodes)
    except Exception:
        milp_score = None
    milp_score_dsp = score(milp_seq_closure, graph)
    # Add MILP only if valid sequence
    milp_violations = count_rule_violations(milp_seq_closure, order_rules)
    # Add MILP in all cases, even if violation
    from src.exact_milp_dijkstra.partial_disassembly_model import _parse_last_status_from_cplex_log, _parse_last_gap_from_cplex_log
    cplex_log_path = milp_results.get('milp_log_path')
    if not cplex_log_path:
        for k in ['log_path', 'log1', 'log2']:
            if k in milp_results and milp_results[k]:
                cplex_log_path = milp_results[k]
                break
    if not cplex_log_path:
        import glob
        logs = glob.glob('results/solver_logs/cplex_*.log')
        cplex_log_path = max(logs, key=os.path.getctime) if logs else None
    status_cplex = _parse_last_status_from_cplex_log(cplex_log_path) if cplex_log_path else milp_results["milp_status"]
    gap_cplex = _parse_last_gap_from_cplex_log(cplex_log_path) if cplex_log_path else milp_results.get("milp_gap")
    gap_percent = None
    if gap_cplex is not None:
        try:
            # CORRECTION: gap_cplex is already in percentage (parsed from "X.XX%")
            gap_percent = gap_cplex
        except Exception:
            gap_percent = gap_cplex
    gap_str = f"{gap_percent:.2f}%" if gap_percent is not None else ""
    
    # CORRECTION: Respect CPLEX status as parsed
    # If CPLEX says "Optimal" (even with a gap), it found optimum within tolerance
    status_str = status_cplex
    # CORRECTION: If no sequence but best_integer available, use that value
    if milp_profit_closure is None and best_integer_val is not None:
        milp_profit_closure = best_integer_val
        print(f"[FIX] Using best_integer ({best_integer_val}) as MILP profit_closure")
    
    results.append({
        "instance": instance_name,
        "algorithm": "MILP_full_profit",
        "profit_closure": milp_profit_closure,
        "makespan_closure": milp_makespan_closure,
        "score_dsp": milp_score_dsp,
        "time": milp_results["milp_time"],
        "status": status_str,
        "gap": gap_str,
        "targets": len(target_nodes),
        "seq_length_closure": milp_seq_closure_len,
        "violations_rules": milp_violations,
        "best_integer": best_integer_val  
    })
    # Add GRASP results
    for alg_name, data in grasp_results.items():
        rel_err = None
        rel_err_closure = None
        profit_closure, makespan_closure, seq_length_closure = None, None, None
        score_dsp = None
        violations = count_rule_violations(data.get("sequence"), order_rules)
        # Use filtered (closure) sequence for GRASP evaluation
        seq = data.get("sequence")
        seq_closure = None
        if seq:
            if milp_data:
                closure = set(closure_with_predecessors(graph, target_nodes))
                seq_closure = [n for n in seq if n in closure]
            else:
                seq_closure = seq
            print(f"[DIAG] Sequence {alg_name} for evaluation:", seq)
            print(f"[DIAG] Sequence {alg_name} filtered (closure):", seq_closure)
            print(f"[DIAG] Mapping {alg_name}:", [node_id_to_number(n, list(graph.nodes), V_milp) for n in seq_closure])
            profit_closure, makespan_closure, seq_length_closure = compute_grasp_profit_makespan(seq_closure, graph, target_nodes, milp_data, order_rules)
            score_dsp = score(seq_closure, graph)
            print(f"[DIAG] {alg_name} profit_closure={profit_closure}, makespan_closure={makespan_closure}, score_dsp={score_dsp}")
        # Always add GRASP row, even if violation
        if (
            isinstance(milp_results.get("milp_profit"), (float, int)) and
            isinstance(data.get("profit"), (float, int)) and
            milp_results.get("milp_profit") not in (None, float('-inf'))
        ):
            try:
                rel_err = (milp_results["milp_profit"] - data["profit"]) / milp_results["milp_profit"]
            except ZeroDivisionError:
                rel_err = None
        else:
            rel_err = None
        if (milp_profit_closure not in (None, float('-inf')) and 
            profit_closure is not None and 
            milp_profit_closure and 
            isinstance(milp_profit_closure, (int, float)) and 
            isinstance(profit_closure, (int, float))):
            try:
                rel_err_closure = (milp_profit_closure - profit_closure) / milp_profit_closure
            except ZeroDivisionError:
                rel_err_closure = None
        else:
            rel_err_closure = None
        print(f"[DEBUG TABLEAU] {alg_name}: profit_closure={profit_closure}, makespan_closure={makespan_closure}, score_dsp={score_dsp}, seq_length_closure={seq_length_closure}")
        # Compute rel_err vs MILP (profit_closure vs MILP profit_closure)
        milp_ref_profit = None
        milp_row = results[0] if results else {}
        # Use MILP profit_closure as reference for closure comparison
        if isinstance(milp_row.get("profit_closure"), (float, int)):
            milp_ref_profit = milp_row["profit_closure"]
        if milp_ref_profit and isinstance(profit_closure, (float, int)) and milp_ref_profit != 0:
            rel_err_vs_milp = (milp_ref_profit - profit_closure) / milp_ref_profit
        else:
            rel_err_vs_milp = 'N/A'
        def safe_float(val):
            try:
                if isinstance(val, (int, float)):
                    return float(val)
                return 'N/A'
            except Exception:
                return 'N/A'
        results.append({
            "instance": instance_name,
            "algorithm": alg_name,
            "profit_closure": safe_float(profit_closure) if profit_closure is not None else 'N/A',
            "makespan_closure": safe_float(makespan_closure) if makespan_closure is not None else 'N/A',
            "score_dsp": safe_float(score_dsp) if score_dsp is not None else 'N/A',
            "time": safe_float(data["time"]) if "time" in data and data["time"] is not None else 'N/A',
            "status": "Completed" if violations == 0 else "Non-compliant",
            "gap": f"{rel_err_vs_milp*100:.2f}%" if rel_err_vs_milp != 'N/A' else 'N/A',
            "targets": len(target_nodes),
            "seq_length_closure": seq_length_closure if seq_length_closure is not None else 0,
            "rel_err_profit_vs_milp_closure": rel_err_vs_milp,
            "violations_rules": violations
        })
        if violations != 0:
            print(f"[FILTERING] Sequence {alg_name} added despite violations_rules={violations}: non-compliant with literature.")
    # Add Dijkstra (profit on closure)
    if "dijkstra_closure_min" in grasp_results:
        dij_data = grasp_results["dijkstra_closure_min"]
        already_exists = any(r.get("algorithm") == "dijkstra_closure_min" for r in results)
        if not already_exists:
            # If Dijkstra was marked N/A (multiple targets), report N/A everywhere
            if dij_data.get("score") == 'N/A':
                results.append({
                    "instance": instance_name,
                    "algorithm": "dijkstra_closure_min",
                    "profit_closure": 'N/A',
                    "makespan_closure": 'N/A',
                    "score_dsp": 'N/A',
                    "time": 'N/A',
                    "status": "N/A",
                    "gap": 'N/A',
                    "targets": len(target_nodes),
                    "seq_length_closure": 0,
                    "rel_err_profit_vs_milp_closure": 'N/A',
                    "violations_rules": 'N/A'
                })
            else:
                dij_profit_closure, dij_makespan_closure, dij_seq_closure_len = (None, None, None)
                if milp_data and dij_data.get("sequence"):
                    print("[DIAG] Dijkstra sequence for evaluation:", dij_data["sequence"])
                    print("[DIAG] Mapping Dijkstra:", [node_id_to_number(n, list(graph.nodes), V_milp) for n in dij_data["sequence"]])
                    dij_profit_closure, dij_makespan_closure, dij_seq_closure_len = compute_grasp_profit_makespan(dij_data["sequence"], graph, target_nodes, milp_data, order_rules)
                    print(f"[DIAG] Dijkstra profit_closure={dij_profit_closure}, makespan_closure={dij_makespan_closure}, score_dsp={score(dij_data['sequence'], graph)}")
                milp_ref_profit = None
                if isinstance(results[0]["profit_closure"], (float, int)):
                    milp_ref_profit = results[0]["profit_closure"]
                if milp_ref_profit and isinstance(dij_profit_closure, (float, int)) and milp_ref_profit != 0:
                    rel_err_vs_milp = (milp_ref_profit - dij_profit_closure) / milp_ref_profit
                else:
                    rel_err_vs_milp = 'N/A'
                results.append({
                    "instance": instance_name,
                    "algorithm": "dijkstra_closure_min",
                    "profit_closure": dij_profit_closure if dij_profit_closure is not None else 'N/A',
                    "makespan_closure": dij_makespan_closure if dij_makespan_closure is not None else 'N/A',
                    "score_dsp": score(dij_data['sequence'], graph) if dij_data.get('sequence') else 'N/A',
                    "time": dij_data.get("time") if dij_data.get("time") is not None else 'N/A',
                    "status": "Completed",
                    "gap": f"{rel_err_vs_milp*100:.2f}%" if rel_err_vs_milp != 'N/A' else 'N/A',
                    "targets": len(target_nodes),
                    "seq_length_closure": dij_seq_closure_len if dij_seq_closure_len is not None else 0,
                    "rel_err_profit_vs_milp_closure": rel_err_vs_milp,
                    "violations_rules": count_rule_violations(dij_data.get("sequence"), order_rules)
                })
    # Metrics normalization (optional)
    rows = results
    specs = build_specs_from_rows(rows, {'profit': 'max', 'makespan': 'min'})
    
    # DEBUG: Verify results content before None/NaN cleanup
    print(f"[DEBUG BEFORE CLEANUP] Number of rows in results: {len(results)}")
    for i, result in enumerate(results):
        if result.get('algorithm') == 'MILP_full_profit':
            print(f"[DEBUG BEFORE CLEANUP] MILP result[{i}]: profit_closure={result.get('profit_closure')}, makespan_closure={result.get('makespan_closure')}")
    
    # Replace None/NaN with 'N/A' for reporting
    for r in rows:
        for k, v in r.items():
            if v is None or (isinstance(v, float) and (v != v)):
                r[k] = 'N/A'
    apply_normalization(rows, specs)
    
    # DEBUG: Verify rows content before DataFrame creation
    print(f"[DEBUG FINAL] Number of rows in rows: {len(rows)}")
    for i, row in enumerate(rows):
        if row.get('algorithm') == 'MILP_full_profit':
            print(f"[DEBUG FINAL] MILP row[{i}]: profit_closure={row.get('profit_closure')}, makespan_closure={row.get('makespan_closure')}")
    
    # Harmonized columns for export
    columns = [
        "instance", "algorithm", "profit_closure", "makespan_closure", "score_dsp", "time", "status", "gap", "targets", "seq_length_closure", "violations_rules", "rel_err_profit_vs_milp_closure"
    ]
    # Remove explanation column if present
    for r in rows:
        if "explanation" in r:
            del r["explanation"]
    return pd.DataFrame(rows, columns=columns)

def discover_instances(base_dir: str):
    """Discover available instance pairs (JSON + MILP) and return instance names."""
    json_dir = os.path.join(base_dir, "data", "instances_base")
    milp_dir = os.path.join(base_dir, "data", "instances_milp")
    names = []
    if not os.path.isdir(json_dir) or not os.path.isdir(milp_dir):
        return names
    for fname in os.listdir(json_dir):
        if fname.endswith('.json'):
            name = os.path.splitext(fname)[0]
            milp_path = os.path.join(milp_dir, f"{name}_selectif.txt")
            if os.path.exists(milp_path):
                names.append(name)
    return sorted(set(names))

def default_targets_for(instance_name: str):
    """Return default target list for a known instance, or [] otherwise."""
    target_map = {
        "automotive_156": ['AU010', 'AU099'],
        "benchmark_complex_20250821_ordered": [],
        "electronics_89": ['EL023', 'EL080'],
        "gearbox_118": ['GB025', 'GB073', 'GB081'],
        "mini_test": ['C'],
        "mini_test_complex_rentable": ['H'],
        "salbp_instance_n=1000_1": ['C936', 'C922'],
        "salbp_instance_n=1000_10": ['C898', 'C631', 'C359'],
        "salbp_instance_n=1000_100": ['C246', 'C829', 'C997'],
        "salbp_instance_n=1000_10_cut297": ['C205'],
        "salbp_instance_n=1000_11_cut322": ['C203'],
        "salbp_instance_n=1000_12_cut291": ['C274'],
        "salbp_instance_n=1000_13_cut253": ['C002'],
        "salbp_instance_n=1000_14_cut247": ['C124', 'C177', 'C086'],
        "salbp_instance_n=1000_15_cut364": ['C132', 'C100'],
        "salbp_instance_n=1000_16_cut268": ['C229'],
        "salbp_instance_n=1000_170_cut254": ['C222', 'C246', 'C230'],
        "salbp_instance_n=1000_171_cut252": ['C251', 'C018', 'C191'],
        "salbp_instance_n=1000_172_cut361": ['C219'],
        "salbp_instance_n=1000_173_cut259": ['C147', 'C050'],
        "salbp_instance_n=1000_174_cut399": ['C398', 'C067'],
        "salbp_instance_n=1000_175_cut259": ['C136', 'C254', 'C221'],
        "salbp_instance_n=1000_17_cut258": ['C146', 'C210', 'C246'],
        "salbp_instance_n=1000_18_cut280": ['C278'],
        "salbp_instance_n=1000_19_cut280": ['C238', 'C086'],
        "salbp_instance_n=1000_1_cut101": ['C087', 'C040'],
        "salbp_instance_n=1000_1_cut102": ['C089', 'C085'],
        "salbp_instance_n=1000_1_cut110": ['C022'],
        "salbp_instance_n=1000_1_cut115": ['C085'],
        "salbp_instance_n=1000_1_cut116": ['C102'],
        "salbp_instance_n=1000_1_cut120": ['C119'],
        "salbp_instance_n=1000_1_cut124": ['C100'],
        "salbp_instance_n=1000_1_cut125": ['C041', 'C063'],
        "salbp_instance_n=1000_1_cut135": ['C043', 'C090', 'C121'],
        "salbp_instance_n=1000_1_cut148": ['C106'],
        "salbp_instance_n=1000_1_cut163": ['C087', 'C160', 'C159'],
        "salbp_instance_n=1000_1_cut180": ['C174', 'C168'],
        "salbp_instance_n=1000_1_cut184": ['C133', 'C135', 'C104'],
        "salbp_instance_n=1000_1_cut187": ['C133'],
        "salbp_instance_n=1000_1_cut188": ['C047'],
        "salbp_instance_n=1000_1_cut196": ['C135', 'C118', 'C169'],
        "salbp_instance_n=1000_1_cut217": ['C095', 'C167', 'C057'],
        "salbp_instance_n=1000_1_cut224": ['C172', 'C194'],
        "salbp_instance_n=1000_1_cut235": ['C189'],
        "salbp_instance_n=1000_1_cut241": ['C207', 'C165', 'C160'],
        "salbp_instance_n=1000_1_cut244": ['C087'],
        "salbp_instance_n=1000_1_cut245": ['C090', 'C015', 'C135'],
        "salbp_instance_n=1000_1_cut246": ['C081'],
        "salbp_instance_n=1000_1_cut249": ['C039'],
        "salbp_instance_n=1000_1_cut254": ['C149', 'C078', 'C127'],
        "salbp_instance_n=1000_1_cut255": ['C210', 'C160'],
        "salbp_instance_n=1000_1_cut259": ['C127', 'C244'],
        "salbp_instance_n=1000_1_cut27": ['C012', 'C009'],
        "salbp_instance_n=1000_1_cut271": ['C236', 'C127'],
        "salbp_instance_n=1000_1_cut291": ['C194'],
        "salbp_instance_n=1000_1_cut294": ['C261'],
        "salbp_instance_n=1000_1_cut297": ['C134', 'C078', 'C257'],
        "salbp_instance_n=1000_1_cut302": ['C246', 'C169', 'C104'],
        "salbp_instance_n=1000_1_cut305": ['C213', 'C049', 'C083'],
        "salbp_instance_n=1000_1_cut317": ['C167'],
        "salbp_instance_n=1000_1_cut321": ['C087'],
        "salbp_instance_n=1000_1_cut353": ['C194', 'C164', 'C276'],
        "salbp_instance_n=1000_1_cut365": ['C087', 'C348'],
        "salbp_instance_n=1000_1_cut369": ['C252', 'C133', 'C275'],
        "salbp_instance_n=1000_1_cut373": ['C369', 'C159'],
        "salbp_instance_n=1000_1_cut384": ['C267', 'C357'],
        "salbp_instance_n=1000_1_cut388": ['C051', 'C133', 'C237'],
        "salbp_instance_n=1000_1_cut391": ['C134', 'C163'],
        "salbp_instance_n=1000_1_cut394": ['C163'],
        "salbp_instance_n=1000_1_cut400": ['C316'],
        "salbp_instance_n=1000_1_cut408": ['C172', 'C195'],
        "salbp_instance_n=1000_1_cut409": ['C155', 'C101'],
        "salbp_instance_n=1000_1_cut410": ['C272'],
        "salbp_instance_n=1000_1_cut450": ['C306', 'C320'],
        "salbp_instance_n=1000_1_cut453": ['C236', 'C112'],
        "salbp_instance_n=1000_1_cut46": ['C039'],
        "salbp_instance_n=1000_1_cut48": ['C044', 'C039'],
        "salbp_instance_n=1000_1_cut480": ['C365'],
        "salbp_instance_n=1000_1_cut485": ['C276'],
        "salbp_instance_n=1000_1_cut491": ['C051'],
        "salbp_instance_n=1000_1_cut494": ['C299'],
        "salbp_instance_n=1000_1_cut51": ['C029'],
        "salbp_instance_n=1000_1_cut53": ['C040', 'C051'],
        "salbp_instance_n=1000_1_cut538": ['C454', 'C227', 'C306'],
        "salbp_instance_n=1000_1_cut567": ['C418'],
        "salbp_instance_n=1000_1_cut570": ['C563', 'C154'],
        "salbp_instance_n=1000_1_cut579": ['C244', 'C504', 'C169'],
        "salbp_instance_n=1000_1_cut586": ['C094', 'C134'],
        "salbp_instance_n=1000_1_cut591": ['C082'],
        "salbp_instance_n=1000_1_cut595": ['C302', 'C112', 'C580'],
        "salbp_instance_n=1000_1_cut612": ['C193', 'C540'],
        "salbp_instance_n=1000_1_cut624": ['C613', 'C154'],
        "salbp_instance_n=1000_1_cut625": ['C490'],
        "salbp_instance_n=1000_1_cut637": ['C094', 'C235'],
        "salbp_instance_n=1000_1_cut639": ['C364'],
        "salbp_instance_n=1000_1_cut644": ['C608'],
        "salbp_instance_n=1000_1_cut65": ['C055'],
        "salbp_instance_n=1000_1_cut654": ['C323', 'C175'],
        "salbp_instance_n=1000_1_cut664": ['C461', 'C437'],
        "salbp_instance_n=1000_1_cut67": ['C006'],
        "salbp_instance_n=1000_1_cut671": ['C562', 'C596', 'C659'],
        "salbp_instance_n=1000_1_cut675": ['C497'],
        "salbp_instance_n=1000_1_cut676": ['C649', 'C652', 'C195'],
        "salbp_instance_n=1000_1_cut684": ['C368', 'C433', 'C487'],
        "salbp_instance_n=1000_1_cut686": ['C447', 'C299', 'C249'],
        "salbp_instance_n=1000_1_cut698": ['C348'],
        "salbp_instance_n=1000_1_cut707": ['C556', 'C092', 'C691'],
        "salbp_instance_n=1000_1_cut713": ['C159'],
        "salbp_instance_n=1000_1_cut720": ['C631'],
        "salbp_instance_n=1000_1_cut722": ['C309', 'C486', 'C237'],
        "salbp_instance_n=1000_1_cut725": ['C626', 'C631'],
        "salbp_instance_n=1000_1_cut735": ['C306', 'C681'],
        "salbp_instance_n=1000_1_cut739": ['C112', 'C538', 'C638'],
        "salbp_instance_n=1000_1_cut742": ['C351', 'C420', 'C336'],
        "salbp_instance_n=1000_1_cut754": ['C444'],
        "salbp_instance_n=1000_1_cut767": ['C249'],
        "salbp_instance_n=1000_1_cut768": ['C759', 'C165', 'C766'],
        "salbp_instance_n=1000_1_cut775": ['C410', 'C392', 'C566'],
        "salbp_instance_n=1000_1_cut779": ['C608', 'C688'],
        "salbp_instance_n=1000_1_cut78": ['C009', 'C040', 'C068'],
        "salbp_instance_n=1000_1_cut780": ['C383', 'C758'],
        "salbp_instance_n=1000_1_cut798": ['C750', 'C323'],
        "salbp_instance_n=1000_1_cut802": ['C437', 'C595'],
        "salbp_instance_n=1000_1_cut807": ['C237', 'C776', 'C437'],
        "salbp_instance_n=1000_1_cut812": ['C365', 'C314'],
        "salbp_instance_n=1000_1_cut815": ['C502'],
        "salbp_instance_n=1000_1_cut846": ['C638', 'C167'],
        "salbp_instance_n=1000_1_cut847": ['C545'],
        "salbp_instance_n=1000_1_cut849": ['C009', 'C701', 'C832'],
        "salbp_instance_n=1000_1_cut862": ['C351'],
        "salbp_instance_n=1000_1_cut870": ['C195', 'C065', 'C562'],
        "salbp_instance_n=1000_1_cut875": ['C497', 'C717', 'C447'],
        "salbp_instance_n=1000_1_cut884": ['C876', 'C724'],
        "salbp_instance_n=1000_1_cut888": ['C533'],
        "salbp_instance_n=1000_1_cut896": ['C775', 'C761'],
        "salbp_instance_n=1000_1_cut903": ['C481', 'C710'],
        "salbp_instance_n=1000_1_cut908": ['C789'],
        "salbp_instance_n=1000_1_cut911": ['C663'],
        "salbp_instance_n=1000_1_cut92": ['C068'],
        "salbp_instance_n=1000_1_cut927": ['C915', 'C626', 'C872'],
        "salbp_instance_n=1000_1_cut934": ['C207', 'C930'],
        "salbp_instance_n=1000_1_cut94": ['C011', 'C076'],
        "salbp_instance_n=1000_1_cut965": ['C940', 'C909'],
        "salbp_instance_n=1000_1_cut968": ['C900', 'C167'],
        "salbp_instance_n=1000_1_cut980": ['C231', 'C087', 'C471'],
        "salbp_instance_n=1000_201_cut373": ['C335'],
        "salbp_instance_n=1000_202_cut361": ['C066', 'C115'],
        "salbp_instance_n=1000_203_cut259": ['C164', 'C244'],
        "salbp_instance_n=1000_204_cut387": ['C089', 'C366', 'C250'],
        "salbp_instance_n=1000_205_cut214": ['C052'],
        "salbp_instance_n=1000_206_cut278": ['C221', 'C093'],
        "salbp_instance_n=1000_207_cut324": ['C321', 'C268', 'C135'],
        "salbp_instance_n=1000_208_cut318": ['C030'],
        "salbp_instance_n=1000_209_cut360": ['C030'],
        "salbp_instance_n=1000_20_cut228": ['C122', 'C052', 'C080'],
        "salbp_instance_n=1000_210_cut335": ['C332'],
        "salbp_instance_n=1000_240_cut350": ['C170', 'C348', 'C329'],
        "salbp_instance_n=1000_241_cut391": ['C378', 'C386', 'C304'],
        "salbp_instance_n=1000_242_cut361": ['C355'],
        "salbp_instance_n=1000_243_cut356": ['C260', 'C204'],
        "salbp_instance_n=1000_244_cut214": ['C135', 'C212'],
        "salbp_instance_n=1000_245_cut373": ['C186', 'C333', 'C051'],
        "salbp_instance_n=1000_246_cut252": ['C192', 'C245', 'C139'],
        "salbp_instance_n=1000_247_cut251": ['C242'],
        "salbp_instance_n=1000_248_cut358": ['C350'],
        "salbp_instance_n=1000_249_cut225": ['C122', 'C214'],
        "salbp_instance_n=1000_250_cut299": ['C230'],
        "salbp_instance_n=100_1": ['C045', 'C093', 'C058'],
        "salbp_instance_n=100_10": ['C085'],
        "salbp_instance_n=100_100": ['C096', 'C083'],
        "salbp_instance_n=100_20": ['C084', 'C089', 'C026'],
        "salbp_instance_n=100_21": ['C052'],
        "salbp_instance_n=100_22": ['C075', 'C096', 'C085'],
        "salbp_instance_n=100_23": ['C013', 'C094', 'C096'],
        "salbp_instance_n=100_24": ['C060', 'C029', 'C068'],
        "salbp_instance_n=100_25": ['C032', 'C076'],
        "salbp_instance_n=100_26": ['C072', 'C040'],
        "salbp_instance_n=100_27": ['C098', 'C085', 'C095'],
        "salbp_instance_n=100_28": ['C019'],
        "salbp_instance_n=100_29": ['C095', 'C061', 'C031'],
        "salbp_instance_n=100_30": ['C086', 'C063'],
        "salbp_instance_n=100_31": ['C057'],
        "salbp_instance_n=100_32": ['C070', 'C092'],
        "salbp_instance_n=100_33": ['C088'],
        "salbp_instance_n=100_34": ['C048'],
        "salbp_instance_n=100_35": ['C099'],
        "salbp_instance_n=100_36": ['C095', 'C062'],
        "salbp_instance_n=100_37": ['C092'],
        "salbp_instance_n=100_38": ['C037', 'C084'],
        "salbp_instance_n=100_39": ['C063'],
        "salbp_instance_n=100_40": ['C057', 'C097'],
        "salbp_instance_n=100_41": ['C071', 'C050', 'C036'],
        "salbp_instance_n=100_42": ['C097'],
        "salbp_instance_n=100_43": ['C042', 'C063', 'C099'],
        "salbp_instance_n=100_44": ['C086', 'C049'],
        "salbp_instance_n=100_45": ['C031', 'C097', 'C029'],
        "salbp_instance_n=100_46": ['C079', 'C067', 'C080'],
        "salbp_instance_n=100_47": ['C050', 'C026'],
        "salbp_instance_n=100_48": ['C052', 'C047', 'C059'],
        "salbp_instance_n=100_49": ['C074', 'C093'],
        "salbp_instance_n=100_50": ['C054'],
        "salbp_instance_n=100_51": ['C097'],
        "salbp_instance_n=100_52": ['C096', 'C095'],
        "salbp_instance_n=100_53": ['C080', 'C099', 'C019'],
        "salbp_instance_n=100_54": ['C089', 'C087'],
        "salbp_instance_n=100_55": ['C095', 'C003'],
        "salbp_instance_n=100_56": ['C095', 'C044', 'C069'],
        "salbp_instance_n=100_57": ['C013', 'C074'],
        "salbp_instance_n=100_58": ['C025', 'C084', 'C078'],
        "salbp_instance_n=20_1": ['C002', 'C019', 'C016'],
        "salbp_instance_n=20_10": ['C013', 'C012', 'C019'],
        "salbp_instance_n=20_100": ['C015'],
        "salbp_instance_n=20_63": ['C009', 'C005'],
        "salbp_instance_n=20_64": ['C012', 'C008', 'C018'],
        "salbp_instance_n=20_65": ['C018', 'C015', 'C009'],
        "salbp_instance_n=20_66": ['C017', 'C016'],
        "salbp_instance_n=20_67": ['C018', 'C016', 'C015'],
        "salbp_instance_n=20_68": ['C019'],
        "salbp_instance_n=20_69": ['C016', 'C019', 'C017'],
        "salbp_instance_n=20_70": ['C015'],
        "salbp_instance_n=20_71": ['C017', 'C016'],
        "salbp_instance_n=20_72": ['C017'],
        "salbp_instance_n=20_73": ['C016', 'C017', 'C012'],
        "salbp_instance_n=20_74": ['C011', 'C019', 'C010'],
        "salbp_instance_n=20_75": ['C019', 'C015'],
        "salbp_instance_n=20_76": ['C019', 'C015', 'C014'],
        "salbp_instance_n=20_77": ['C017'],
        "salbp_instance_n=20_78": ['C018'],
        "salbp_instance_n=20_79": ['C015', 'C017'],
        "salbp_instance_n=20_80": ['C016', 'C012', 'C018'],
        "salbp_instance_n=20_81": ['C009', 'C016'],
        "salbp_instance_n=20_82": ['C019'],
        "salbp_instance_n=20_83": ['C014', 'C010'],
        "salbp_instance_n=20_84": ['C017', 'C015', 'C019'],
        "salbp_instance_n=20_85": ['C018', 'C019', 'C015'],
        "salbp_instance_n=20_86": ['C011', 'C014'],
        "salbp_instance_n=20_87": ['C013', 'C005'],
        "salbp_instance_n=20_88": ['C019', 'C018', 'C012'],
        "salbp_instance_n=20_89": ['C019', 'C004'],
        "salbp_instance_n=20_90": ['C019'],
        "salbp_instance_n=20_91": ['C011'],
        "salbp_instance_n=20_92": ['C015', 'C014'],
        "salbp_instance_n=20_93": ['C009', 'C017', 'C019'],
        "salbp_instance_n=20_94": ['C019'],
        "salbp_instance_n=20_95": ['C016', 'C017', 'C014'],
        "salbp_instance_n=20_96": ['C019', 'C013'],
        "salbp_instance_n=20_97": ['C018', 'C008'],
        "salbp_instance_n=20_98": ['C016'],
        "salbp_instance_n=20_99": ['C015', 'C014', 'C018'],
        "salbp_instance_n=50_1": ['C040'],
        "salbp_instance_n=50_10": ['C031', 'C019'],
        "salbp_instance_n=50_100": ['C046', 'C043', 'C049'],
        "salbp_instance_n=50_100p10": ['C048'],
        "salbp_instance_n=50_100p2": ['C046'],
        "salbp_instance_n=50_100p3": ['C042', 'C043', 'C033'],
        "salbp_instance_n=50_1p10": ['C040'],
        "salbp_instance_n=50_1p2": ['C039', 'C031', 'C040'],
        "salbp_instance_n=50_1p3": ['C043'],
        "salbp_instance_n=50_1p4": ['C031'],
        "salbp_instance_n=50_1p5": ['C043', 'C039', 'C041'],
        "salbp_instance_n=50_1p6": ['C039', 'C047'],
        "salbp_instance_n=50_1p7": ['C009', 'C023', 'C049'],
        "salbp_instance_n=50_1p8": ['C046', 'C043'],
        "salbp_instance_n=50_1p9": ['C031'],
        "salbp_instance_n=50_28p10": ['C045', 'C047', 'C011'],
        "salbp_instance_n=50_28p9": ['C049'],
        "salbp_instance_n=50_29": ['C040'],
        "salbp_instance_n=50_29p10": ['C044', 'C034'],
        "salbp_instance_n=50_29p2": ['C040', 'C014', 'C034'],
        "salbp_instance_n=50_29p3": ['C040', 'C041', 'C047'],
        "salbp_instance_n=50_29p4": ['C016', 'C026', 'C039'],
        "salbp_instance_n=50_29p5": ['C018'],
        "salbp_instance_n=50_29p6": ['C018','C044'],
        "salbp_instance_n=50_29p7": ['C035', 'C018', 'C022'],
        "salbp_instance_n=50_29p8": ['C044', 'C048'],
        "salbp_instance_n=50_29p9": ['C040', 'C046', 'C018'],
        "salbp_instance_n=50_52p10": ['C031', 'C026', 'C044'],
        "salbp_instance_n=50_52p5": ['C044'],
        "salbp_instance_n=50_52p6": ['C049'],
        "salbp_instance_n=50_52p7": ['C049', 'C023'],
        "salbp_instance_n=50_52p8": ['C015', 'C024'],
        "salbp_instance_n=50_52p9": ['C031', 'C015', 'C048'],
        "salbp_instance_n=50_53": ['C016', 'C015', 'C038'],
        "salbp_instance_n=50_53p2": ['C049', 'C040', 'C027'],
        "salbp_instance_n=50_53p3": ['C038', 'C048'],
        "salbp_instance_n=50_53p4": ['C027'],
        "salbp_instance_n=50_53p5": ['C044', 'C027', 'C047'],
        "scholl_arc111_n=111": ['C110'],
        "scholl_arc83_n=83": ['C082'],
        "scholl_barthol2_n=148": ['C093'],
        "scholl_barthold_n=148": ['C125'],
        "scholl_bowman8_n=8": ['C006'],
        "scholl_buxey_n=29": ['C028'],
        "scholl_gunther_n=35": ['C028'],
        "scholl_hahn_n=53": ['C052'],
        "scholl_heskia_n=28": ['C027'],
        "scholl_jackson_n=11": ['C010'],
        "scholl_jaeschke_n=9": ['C008'],
        "scholl_kilbrid_n=45": ['C044'],
        "scholl_lutz1_n=32": ['C031'],
        "scholl_lutz2_n=89": ['C047', 'C087'],
        "scholl_lutz3_n=89": ['C020', 'C040', 'C050'],
        "scholl_mansoor_n=11": ['C010'],
        "scholl_mertens_n=7": ['C005'],
        "scholl_mitchell_n=21": ['C019', 'C020', 'C018'],
        "scholl_mukherje_n=94": ['C086'],
        "scholl_roszieg_n=25": ['C021'],
        "scholl_sawyer30_n=30": ['C018'],
        "scholl_scholl_n=297": ['C293', 'C294', 'C295'],
        "scholl_tonge70_n=70": ['C010', 'C054'],
        "scholl_warnecke_n=58": ['C028'],
        "scholl_wee-mag_n=75": ['C069'],
        "small_40": ['C001', 'C032'],
        "structured_graph_100": ['C050'],
        "structured_graph_200": ['C020', 'C150'],
        "dagtest": ['N22'],
    }
    return target_map.get(instance_name, [])

def main():
    parser = argparse.ArgumentParser(description="Compare GRASP and MILP on the same instances")
    parser.add_argument("--instance", type=str, help="Instance name to process (without extension)")
    parser.add_argument("--batch", action="store_true", help="Process all available instances in batch")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for GRASP reproducibility")
    parser.add_argument("--targets", type=str, nargs='+', help="Target nodes for selective mode")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per GRASP algorithm")
    parser.add_argument("--time_limit", type=int, default=7200, help="Time limit for MILP in seconds (default: 2h, 0=no limit)")
    parser.add_argument("--gap", type=float, default=0.0001, help="MILP stopping criterion (single-phase). Set very small (0.001) for long runs.")
    parser.add_argument("--gap_schedule", type=str, help="List of successive gaps (e.g., 0.05,0.03,0.02,0.01,0.005) for MILP phases. Replaces --gap.")
    parser.add_argument("--no_time_limit", action="store_true", help="Completely disable time limit for MILP (equivalent to --time_limit 0)")
    parser.add_argument("--outdir", type=str, default="results/compare", help="Output directory for comparison CSV")
    parser.add_argument("--no_dijkstra", action="store_true", help="Do not include Dijkstra baseline (included by default)")

    parser.add_argument("--grasp_iterations", type=int, default=100, help="GRASP reactive iterations (construction)")
    parser.add_argument("--grasp_time_budget", type=int, default=None, help="Time budget (s) for each heuristic (global fallback)")
    parser.add_argument("--no_grasp_early_stop", action="store_true", help="Disable early stopping (stagnation) in GRASP")
    parser.add_argument("--tabu_iter", type=int, default=20, help="Internal Tabu Search iterations (intensification)")
    parser.add_argument("--tabu_time_budget", type=int, default=None, help="Dedicated Tabu time budget (priority over grasp_time_budget)")
    parser.add_argument("--vnd_time_budget", type=int, default=None, help="Dedicated VND time budget")
    parser.add_argument("--mns_time_budget", type=int, default=None, help="Dedicated MNS time budget")
    args = parser.parse_args()
    
    # If no_time_limit option is enabled, disable time limit
    if args.no_time_limit:
        args.time_limit = 0

    #
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Global seed (optional)
    if args.seed is not None:
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))

    # Batch mode: process all available instances
    if args.batch:
        instance_names = discover_instances(base_dir)
        if not instance_names:
            print("No instances detected (JSON+MILP)")
            return
        print(f"Batch: {len(instance_names)} instances found")
        all_rows = []
        for name in instance_names:
            json_path = os.path.join(base_dir, "data", "instances_base", f"{name}.json")
            milp_path = os.path.join(base_dir, "data", "instances_milp", f"{name}_selectif.txt")
            targets = args.targets if args.targets else default_targets_for(name)
            run_id = f"batch_{name}_{datetime.now().strftime('%H%M%S')}"
            tracker = PipelineTracker(run_id, args.outdir)
            # Validate target existence in graph
            try:
                G, _, _ = load_adaptive_graph(json_path)
                tracker.log_step('load_instance', status='ok', instance=name, nodes=G.number_of_nodes())
            except Exception as e:
                print(f"\u21aa Skip {name}: graph loading error ({e})")
                tracker.log_step('load_instance', status='error', instance=name, error=str(e))
                continue
            if not targets:
                print(f"\u21aa Skip {name}: no known default target and --targets not provided")
                tracker.log_step('load_instance', status='error', instance=name, error='no_targets')
                continue
            if not all(t in G for t in targets):
                fallback = default_targets_for(name)
                if fallback and all(t in G for t in fallback):
                    print(f"\u21aa {name}: provided targets invalid for this graph, using default targets {fallback}")
                    targets = fallback
                else:
                    print(f"\u21aa Skip {name}: invalid targets for graph and no valid fallback")
                    tracker.log_step('load_instance', status='error', instance=name, error='invalid_targets')
                    continue

            # --- Pipeline robustness patch ---
            milp_data = None
            milp_results = {
                "milp_sequence": None,
                "milp_profit": float('-inf'),
                "milp_makespan": None,
                "milp_time": 0,
                "milp_status": "Error: MILP not executed",
                "milp_gap": None
            }
            # Try to execute MILP, but continue if it crashes!
            try:
                milp_data = load_data(milp_path)
                milp_results = run_milp(
                    milp_path,
                    time_limit=args.time_limit,
                    gap_limit=(None if (args.gap_schedule and args.gap_schedule != []) else (args.gap if args.gap and args.gap > 0 else None)),
                    gap_schedule=([float(x) for x in args.gap_schedule.split(',') if x.strip()] if args.gap_schedule else None)
                )
                tracker.log_step('run_milp', status=milp_results.get('milp_status','unknown'), gap=milp_results.get('milp_gap'))
            except Exception as e:
                print(f"[WARN] MILP failed on {name}: {e}")
                tracker.log_step('run_milp', status='error', error=str(e))
            # GRASP/Dijkstra always executed, even if MILP crashes
            try:
                include_dij = not args.no_dijkstra
                grasp_results = run_grasp_algorithms(
                    json_path, targets, runs=args.runs, milp_data=milp_data, seed=args.seed,
                    include_dijkstra=include_dij, tracker=tracker,
                    grasp_iterations=args.grasp_iterations,
                    grasp_time_budget=args.grasp_time_budget,
                    grasp_early_stop=(not args.no_grasp_early_stop),
                    tabu_iter=args.tabu_iter,
                    tabu_time_budget=args.tabu_time_budget,
                    vnd_time_budget=args.vnd_time_budget,
                    mns_time_budget=args.mns_time_budget
                )
            except Exception as e:
                print(f"[WARN] GRASP/Dijkstra failed on {name}: {e}")
                grasp_results = {}
            # Generate CSV even if everything crashed
            try:
                df = compare_solutions(grasp_results, milp_results, name, targets, G)
                tracker.log_step('aggregate', status='ok', rows=len(df))
                all_rows.append(df)
            except Exception as e:
                print(f"[WARN] CSV generation failed on {name}: {e}")
            tracker.finalize_report()
        if not all_rows:
            print("\u26a0 No results to save")
            return
        full_df = pd.concat(all_rows, ignore_index=True)
        os.makedirs(args.outdir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(args.outdir, f"compare_batch_{timestamp}.csv")
        print("\n=== GLOBAL SUMMARY (batch) ===")
        print(full_df.to_string(index=False))
        full_df.to_csv(out, index=False)
        print(f"\nBatch results saved to: {out}")
        return

    # Single-instance mode (default)
    if not args.instance:
        print("\u26a0 Please provide --instance <name> or use --batch")
        return

    # File paths
    instance_name = args.instance
    json_path = os.path.join(base_dir, "data", "instances_base", f"{instance_name}.json")
    milp_path = os.path.join(base_dir, "data", "instances_milp", f"{instance_name}_selectif.txt")

    
    if not os.path.exists(json_path):
        print(f"\u26a0 JSON file not found: {json_path}")
        return

    if not os.path.exists(milp_path):
        print(f"\u26a0 MILP file not found: {milp_path}")
        print("Run json_to_milp.py first to create the MILP instance")
        return

    # Determine target nodes with validation
    targets = args.targets if args.targets else default_targets_for(instance_name)
    try:
        G_check, _, _ = load_adaptive_graph(json_path)
    except Exception as e:
        print(f"\u26a0 Graph loading error: {e}")
        return
    if not targets:
        print("\u26a0 No targets specified")
        return
    if not all(t in G_check for t in targets):
        fallback = default_targets_for(instance_name)
        if fallback and all(t in G_check for t in fallback):
            print(f"\u21aa Provided targets invalid for this graph, using default targets {fallback}")
            targets = fallback
        else:
            print("\u26a0 Invalid targets for this graph and no valid fallback")
            return

    print(f"Instance: {instance_name}")
    print(f"Targets: {targets}")

    # Load MILP data (for GRASP profit/makespan)
    milp_data = load_data(milp_path)
    # Load JSON data (for order_rules)
    with open(json_path, "r") as f:
        json_data = json.load(f)
    # Merge MILP and JSON data for GRASP/Dijkstra computation
    milp_data_enriched = milp_data.copy()
    if "order_rules" in json_data:
        milp_data_enriched["order_rules"] = json_data["order_rules"]
    # Execute algorithms
    include_dij = not args.no_dijkstra
    run_id = f"single_{instance_name}_{datetime.now().strftime('%H%M%S')}"
    tracker = PipelineTracker(run_id, args.outdir)
    tracker.log_step('load_instance', status='ok', instance=instance_name)
    gap_schedule = None
    if args.gap_schedule:
        try:
            gap_schedule = [float(x) for x in args.gap_schedule.split(',') if x.strip()]
        except ValueError:
            print("\u26a0 Invalid format for --gap_schedule, ignored.")
    grasp_results = run_grasp_algorithms(
        json_path, targets, runs=args.runs, milp_data=milp_data_enriched, seed=args.seed,
        include_dijkstra=include_dij, tracker=tracker,
        grasp_iterations=args.grasp_iterations,
        grasp_time_budget=args.grasp_time_budget,
        grasp_early_stop=(not args.no_grasp_early_stop),
        tabu_iter=args.tabu_iter,
        tabu_time_budget=args.tabu_time_budget,
        vnd_time_budget=args.vnd_time_budget,
        mns_time_budget=args.mns_time_budget
    )
    milp_results = run_milp(
        milp_path,
        time_limit=args.time_limit,
        gap_limit=(None if gap_schedule else (args.gap if args.gap and args.gap > 0 else None)),
        gap_schedule=gap_schedule
    )
    tracker.log_step('run_milp', status=milp_results.get('milp_status','unknown'), gap=milp_results.get('milp_gap'))

    # Comparison
    df = compare_solutions(grasp_results, milp_results, instance_name, targets, G_check)
    tracker.log_step('aggregate', status='ok', rows=len(df))

    # Display results
    print("\n=== GRASP vs MILP COMPARISON ===")
    print(df.to_string(index=False))

    # Save results
    os.makedirs(args.outdir, exist_ok=True)
    


    # Force all outputs to directory passed by --outdir
    # Create a subfolder per instance to harmonize with benchmark2
    instance_dir = os.path.join(args.outdir, instance_name)
    os.makedirs(instance_dir, exist_ok=True)

    # Generate unique timestamp for all output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save comparison CSV
    output_file = os.path.join(instance_dir, f"compare_{instance_name}_{timestamp}.csv")
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Save validation report in instance subfolder
    if hasattr(tracker, 'validation_report') and tracker.validation_report:
        validation_filename = f"validation_single_{instance_name}_{timestamp}.json"
        validation_path = os.path.join(instance_dir, validation_filename)
        with open(validation_path, "w", encoding="utf-8") as f:
            json.dump(tracker.validation_report, f, indent=2, ensure_ascii=False)
        abs_validation_path = os.path.abspath(validation_path)
        print(f"[TRACE] Validation report saved to: {abs_validation_path}")
    tracker.finalize_report()

    # Save complete terminal log (stdout)
    try:
        import sys
        if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'getvalue'):
            terminal_log = sys.stdout.getvalue()
        else:
            terminal_log = None
        log_file = os.path.join(instance_dir, f"terminal_log_{instance_name}_{timestamp}.txt")
        if terminal_log:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(terminal_log)
            abs_log_file = os.path.abspath(log_file)
            print(f"[TRACE] Terminal log saved to: {abs_log_file}")
        else:
            # Even if content not retrievable, show expected path
            abs_log_file = os.path.abspath(log_file)
            print(f"[TRACE] (No content retrieved) Terminal log expected at: {abs_log_file}")
    except Exception as e:
        print(f"[WARN] Terminal log save failed: {e}")

    # Copy CPLEX log to instance folder (if available)
    try:
        milp_log_path = milp_results.get('milp_log_path') if 'milp_results' in locals() else None
        if milp_log_path and os.path.exists(milp_log_path):
            import shutil
            log_filename = os.path.basename(milp_log_path)
            cplex_copy_path = os.path.join(instance_dir, f"solver_{log_filename}_{timestamp}")
            shutil.copy2(milp_log_path, cplex_copy_path)
            print(f"CPLEX log copied to: {cplex_copy_path}")
    except Exception as e:
        print(f"[WARN] CPLEX log copy failed: {e}")

import sys, io
from contextlib import redirect_stdout, redirect_stderr

def run_with_terminal_log():
    # Default output directory (will be adjusted dynamically)
    outdir = "results/compare"
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        main()
    # Only save global terminal log if batch mode (multiple instances)
    import sys
    if '--batch' in sys.argv:
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(outdir, f"terminal_log_{timestamp}.txt")
        os.makedirs(outdir, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(buffer.getvalue())
        print(f"[INFO] Global terminal log saved to: {log_file}")

if __name__ == "__main__":
    run_with_terminal_log()
