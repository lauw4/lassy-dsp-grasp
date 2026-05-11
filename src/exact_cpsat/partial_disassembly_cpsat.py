"""CP-SAT (Google OR-Tools) formulation for partial disassembly.

Same time-discounted objective as the position-based MILP (slide 9 of the
IFAC presentation):
    max Σ v_i · x_i  −  λ · Σ C_i

where v_i = r_i − c_i and C_i is the completion time of part i (0 if not
selected). CP-SAT handles single-machine sequencing natively via optional
IntervalVars + NoOverlap, which gives a tighter implicit relaxation than
positional MILP formulations on this 1|prec|ΣwjCj-style problem.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from ortools.sat.python import cp_model


def build_model_cpsat(data: Dict[str, Any], scale: int = 10000):
    """Build the CP-SAT model for the time-discounted partial-disassembly problem.

    Returns (model, vars) where vars holds:
      x[i]    : BoolVar — select part i
      start[i], end[i] : IntVar — start / end on the single machine
      interval[i] : OptionalIntervalVar — present iff x[i] is true
      c_obj[i]    : IntVar — completion time used in the objective (0 if not selected)
      scale       : int — coefficient scale used to integerise the objective
      lam         : float — λ value used (auto-calibrated if not on data)
    """
    V = data['V']
    E = data['E']
    T_targets = data['T']
    r, c, p = data['r'], data['c'], data['p']

    if 'lambda_discount' in data and data['lambda_discount'] is not None:
        lam = float(data['lambda_discount'])
    else:
        pos_vals = [max(r[i] - c[i], 0.0) for i in V]
        total_p = sum(p[i] for i in V)
        lam = 0.05 * sum(pos_vals) / total_p if total_p > 0 and sum(pos_vals) > 0 else 0.05
    data['lambda_discount'] = lam

    v = {i: r[i] - c[i] for i in V}

    # CP-SAT requires integer coefficients in the objective AND integer durations.
    v_int   = {i: int(round(v[i] * scale)) for i in V}
    lam_int = int(round(lam * scale))
    p_int   = {i: max(1, int(round(p[i]))) for i in V}
    M_int   = sum(p_int[i] for i in V)

    model = cp_model.CpModel()

    x = {i: model.NewBoolVar(f'x_{i}') for i in V}
    start = {i: model.NewIntVar(0, M_int, f'start_{i}') for i in V}
    end   = {i: model.NewIntVar(0, M_int, f'end_{i}')   for i in V}
    interval = {
        i: model.NewOptionalIntervalVar(start[i], p_int[i], end[i], x[i], f'iv_{i}')
        for i in V
    }

    # Single machine: optional intervals can't overlap. Optionality is handled.
    model.AddNoOverlap([interval[i] for i in V])

    # Targets must be selected.
    for t in T_targets:
        model.Add(x[t] == 1)

    # Precedence on selection: x[j] ⇒ x[i] for each (i,j) ∈ E.
    # Precedence on time:     end[i] ≤ start[j] when both selected.
    for (i, j) in E:
        model.AddImplication(x[j], x[i])
        model.Add(end[i] <= start[j]).OnlyEnforceIf([x[i], x[j]])

    # c_obj[i] = end[i] if x[i] else 0  (used only in the objective)
    c_obj = {i: model.NewIntVar(0, M_int, f'c_{i}') for i in V}
    for i in V:
        model.Add(c_obj[i] == end[i]).OnlyEnforceIf(x[i])
        model.Add(c_obj[i] == 0).OnlyEnforceIf(x[i].Not())

    model.Maximize(
        sum(v_int[i] * x[i] for i in V)
        - lam_int * sum(c_obj[i] for i in V)
    )

    return model, {
        'x': x, 'start': start, 'end': end,
        'interval': interval, 'c_obj': c_obj,
        'scale': scale, 'lam': lam,
    }


def solve_cpsat(
    model: cp_model.CpModel,
    vars_: Dict[str, Any],
    time_limit: Optional[float] = None,
    num_workers: int = 12,
    log_path: Optional[str] = None,
    instance_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Solve a CP-SAT model and return a structured result.

    Result dict:
      status        : 'OPTIMAL' | 'FEASIBLE' | 'INFEASIBLE' | 'MODEL_INVALID' | 'UNKNOWN'
      obj           : real-valued objective (descaled)
      bound         : best dual bound (descaled), None if no feasible solution
      gap_pct       : 100·|bound − obj| / max(|obj|, ε), None if no feasible solution
      solve_time    : seconds elapsed
      x_values      : {i: 0/1} selected parts
      sequence      : list[int] of selected parts in completion-time order
      log_path      : path to the (text) solver log if requested
    """
    solver = cp_model.CpSolver()
    if time_limit is not None and time_limit > 0:
        solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = num_workers
    solver.parameters.log_search_progress = log_path is not None

    log_lines: List[str] = []
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
        solver.log_callback = log_lines.append

    t0 = time.perf_counter()
    status = solver.Solve(model)
    solve_time = time.perf_counter() - t0
    status_name = solver.StatusName(status)

    scale = vars_['scale']
    obj_real:   Optional[float] = None
    bound_real: Optional[float] = None
    gap_pct:    Optional[float] = None
    x_values: Dict[int, int] = {}
    sequence: List[int] = []

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        obj_real   = solver.ObjectiveValue() / scale
        bound_real = solver.BestObjectiveBound() / scale
        if abs(obj_real) > 1e-9:
            gap_pct = abs(bound_real - obj_real) / abs(obj_real) * 100.0
        else:
            gap_pct = abs(bound_real - obj_real) * 100.0

        # Decode selection and order by end-time
        x = vars_['x']
        end = vars_['end']
        ordered: List[Tuple[int, int]] = []
        for i, xi in x.items():
            v = solver.Value(xi)
            x_values[i] = int(v)
            if v == 1:
                ordered.append((i, solver.Value(end[i])))
        ordered.sort(key=lambda t: t[1])
        sequence = [i for i, _ in ordered]

    if log_path is not None and log_lines:
        try:
            with open(log_path, 'w', encoding='utf-8') as fh:
                fh.write('\n'.join(log_lines))
        except OSError:
            pass

    return {
        'status':     status_name,
        'obj':        obj_real,
        'bound':      bound_real,
        'gap_pct':    gap_pct,
        'solve_time': solve_time,
        'x_values':   x_values,
        'sequence':   sequence,
        'log_path':   log_path,
    }
