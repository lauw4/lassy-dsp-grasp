import pulp
from pulp import CPLEX_CMD, PULP_CBC_CMD
from typing import Dict, List, Tuple, Any, Optional
import pprint
import re
import sys
import os
from datetime import datetime
import io
import math
import json

# -- Data verification ------------------------------------------------

def verify_data(data: Dict[str, Any]) -> bool:
    required = ['V', 'E', 'T', 'r', 'c', 'p']
    missing = [k for k in required if k not in data or not data[k]]
    if missing:
        raise ValueError(f"Missing data: {', '.join(missing)}")
    V = set(data['V'])
    for (i, j) in data['E']:
        if i not in V or j not in V:
            raise ValueError(f"Precedence arc ({i},{j}) outside V")
    if not set(data['T']).issubset(V):
        raise ValueError("T (targets) must be a subset of V")
    for dico, name in [(data['r'], 'r'), (data['c'], 'c'), (data['p'], 'p')]:
        if any(i not in V for i in dico):
            raise ValueError(f"Some indices in {name} are not in V")
    return True

# -- Instance file loading ----------------------------------------

_SECTION_PATTERN = re.compile(r"^#\s*(.+)\s*$")

def _parse_value(val: str) -> Any:
    val = val.strip()
    if re.fullmatch(r"-?\d+", val):
        return int(val)
    try:
        return float(val)
    except ValueError:
        return val

def load_data(file_path: str) -> Dict[str, Any]:
    data = {'V': [], 'E': [], 'T': [], 'r': {}, 'c': {}, 'p': {}, 'C_max': None}
    section = None
    data['label_map'] = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('//'):
                continue
            m = _SECTION_PATTERN.match(line)
            if m:
                title = m.group(1).strip().lower().replace(' ', '')
                if 'targetcomponent' in title or 'targetcomponents' in title or 'target' in title:
                    section = 'T'
                elif 'component' in title:
                    section = 'V'
                elif 'precedence' in title:
                    section = 'E'
                elif 'revenue' in title or title.startswith('r'):
                    section = 'r'
                elif 'cost' in title or title.startswith('c'):
                    section = 'c'
                elif 'duration' in title or title.startswith('p'):
                    section = 'p'
                elif 'c_max' in title or 'horizon' in title:
                    section = 'C_max'
                else:
                    print(f"Unknown section: {title}", file=sys.stderr)
                    section = None
                continue
            if section == 'V':
                data['V'].extend([_parse_value(tok) for tok in re.split(r'[,;\s]+', line) if tok])
            elif section == 'E':
                parts = [int(tok) for tok in line.split()]
                if len(parts) != 2:
                    raise ValueError(f"Malformed arc: {line}")
                data['E'].append(tuple(parts))
            elif section == 'T':
                data['T'].extend([int(tok) for tok in re.split(r'[,;\s]+', line) if tok])
            elif section in {'r', 'c', 'p'}:
                parts = line.split()
                if len(parts) < 2:
                    continue  # Ignore empty or incomplete lines
                i, val = parts[:2]
                i = int(i)
                val = _parse_value(val)
                data[section][i] = val
            elif section == 'C_max':
                data['C_max'] = _parse_value(line)
    # Merge 'trap' attributes from JSON if available
    json_path = file_path.replace('_selectif.txt', '.json').replace('.txt', '.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as jf:
                json_data = json.load(jf)
            trap_map = {}
            for node in json_data.get('nodes', []):
                nid = node.get('id')
                if nid is not None:
                    trap_map[int(nid)] = node.get('trap', False)
            if trap_map:
                data['trap'] = trap_map
                data['nodes'] = json_data.get('nodes', [])
        except Exception as e:
            print(f"[WARN] JSON/MILP merge for 'trap' failed: {e}")
    verify_data(data)
    return data

# -- MILP model building ---------------------------------------------

def build_model(data: Dict[str, Any], use_time_aware: bool = False) -> Tuple[pulp.LpProblem, Dict[str, Any]]:
    """Build the partial-disassembly MILP.

    use_time_aware = False (default — Option B):
        Selection-only variant.  Objective `max Σ v_i · x_i` — knapsack with
        precedence, closes gap = 0 % on n ≤ 100 in seconds.  No y/s/sequence
        variables are created so the model stays small even on n ≈ 1000.

    use_time_aware = True:
        Full IFAC slide-9 objective `max Σ v_i x_i − λ Σ C_i`.  Adds y[i,k]
        position assignments, makespan, and the dormant Z_k machinery.
        Empirically does NOT close gap beyond n ≈ 25-35 with off-the-shelf
        solvers — kept for research extension only.  See
        analysis_for_amine/notes_objectif_milp.md for the full study.
    """
    V: List[int] = data['V']
    E: List[Tuple[int, int]] = data['E']
    T: List[int] = data['T']
    r, c, p = data['r'], data['c'], data['p']
    C_max: Optional[float] = data.get('C_max')

    model = pulp.LpProblem("Partial_Disassembly", pulp.LpMaximize)

    x = pulp.LpVariable.dicts('x', V, cat='Binary')

    # y[i,k] and the makespan variable s only exist when scheduling matters.
    K: List[int] = list(range(1, len(V) + 1))
    y = pulp.LpVariable.dicts('y', (V, K), cat='Binary') if use_time_aware else None
    s = pulp.LpVariable('makespan', lowBound=0) if use_time_aware else None

    # ── Objective 1 parameters (IWSPE 2026) ──────────────────────────
    # max Σ(v_i·x_i) - λ·Σ W_i
    # v_i = r_i - c_i  (net recovery margin)
    # W_i = completion time of part i (linearised below)
    # λ   = lambda_discount — auto-calibrated per instance (5 % of mean value rate)
    #       so the time-discount scales with the instance's own value density.
    if 'lambda_discount' in data and data['lambda_discount'] is not None:
        lam = float(data['lambda_discount'])
    else:
        # Auto-calibrate: λ = 0.05 × Σmax(v_i,0) / Σp_i
        _pos_vals = [max(r[i] - c[i], 0.0) for i in V]
        _total_p  = sum(p[i] for i in V)
        if _total_p > 0 and sum(_pos_vals) > 0:
            lam = 0.05 * sum(_pos_vals) / _total_p
        else:
            lam = 0.05
    data['lambda_discount'] = lam   # store so caller can read it back
    # Budget constraint parameters (managerial extension — dormant unless
    # budget_max is set on the instance; same h default as the GRASP side).
    B_max_eur = data.get('budget_max', None)
    h = data.get('overhead_rate', 0.05)

    # Net recovery margin v_i = r_i - c_i
    v = {i: r[i] - c[i] for i in V}

    # ── Selection constraints (always active) ───────────────────
    # Precedence on x: selecting j forces selecting i for each (i,j) ∈ E.
    # In the time-aware variant, this also tightens the LP relaxation.
    for (i, j) in E:
        model += x[j] <= x[i], f'x_prec_{i}_{j}'
    for i in T:
        model += x[i] == 1, f'target_{i}'

    # ── Sequence/makespan constraints (only when scheduling matters) ──
    if use_time_aware:
        for i in V:
            model += pulp.lpSum(y[i][k] for k in K) == x[i], f'assign_{i}'
        for k in K:
            model += pulp.lpSum(y[i][k] for i in V) <= 1, f'serial_{k}'
        for (i, j) in E:
            for k in K:
                model += y[j][k] <= pulp.lpSum(y[i][k_] for k_ in K if k_ < k), f'prec_{i}_{j}_{k}'
        # Tightening: positions filled contiguously from k=1 (u_k ≤ u_{k-1}).
        for k in K:
            if k == 1:
                continue
            model += (
                pulp.lpSum(y[i][k] for i in V)
                <= pulp.lpSum(y[i][k - 1] for i in V),
                f'u_monotone_{k}'
            )
        model += s >= pulp.lpSum(p[i] * y[i][k] for i in V for k in K), 'makespan_def'
        if C_max is not None:
            model += s <= C_max, 'horizon_limit'

    # ── Time-aware machinery (Z_k = CT[k]·u_k via McCormick) ─────
    # Active only when use_time_aware=True. Adds the cumulative-time
    # variables CT[k], the linearised completion-time-per-position Z[k],
    # and the precedence-based valid inequalities that tighten the LP.
    # See analysis_for_amine/notes_objectif_milp.md for the formulation
    # study and why this is hard to close beyond n ≈ 25-35.
    if use_time_aware:
        # --- CT[k]: cumulative processing time through position k ---
        CT = pulp.LpVariable.dicts('CT', K, lowBound=0)
        for k in K:
            slot_time = pulp.lpSum(p[j] * y[j][k] for j in V)
            if k == 1:
                model += CT[k] == slot_time, f'CT_def_{k}'
            else:
                model += CT[k] == CT[k - 1] + slot_time, f'CT_def_{k}'

        # --- Position-dependent upper bound on CT[k] ---
        p_sorted_desc = sorted((p[j] for j in V), reverse=True)
        M_lb = {}
        cumul_p = 0.0
        for idx, k in enumerate(K):
            cumul_p += p_sorted_desc[idx] if idx < len(p_sorted_desc) else 0.0
            M_lb[k] = cumul_p

        # --- Z_k = CT[k]·u_k via McCormick envelope ---
        Z = pulp.LpVariable.dicts('Z', K, lowBound=0)
        for k in K:
            u_k = pulp.lpSum(y[i][k] for i in V)
            model += Z[k] >= CT[k] - M_lb[k] * (1 - u_k), f'Z_lb_{k}'
            model += Z[k] <= M_lb[k] * u_k,                f'Z_ub_u_{k}'
            model += Z[k] <= CT[k],                         f'Z_ub_ct_{k}'

        # --- Precedence-based valid inequalities on Σ Z[k] ---
        ancestors = {i: set() for i in V}
        for (a, b) in E:
            ancestors[b].add(a)
        changed = True
        while changed:
            changed = False
            for i in V:
                new = set()
                for a in ancestors[i]:
                    new |= ancestors[a]
                if not new.issubset(ancestors[i]):
                    ancestors[i] |= new
                    changed = True
        min_ct = {i: sum(p[j] for j in ancestors[i]) + p[i] for i in V}
        model += (
            pulp.lpSum(Z[k] for k in K)
            >= pulp.lpSum(min_ct[i] * x[i] for i in V),
            'agg_pred_lb'
        )
        for k in K:
            model += (
                CT[k] >= pulp.lpSum(min_ct[i] * y[i][k] for i in V),
                f'CT_pred_lb_{k}'
            )
            model += (
                Z[k] >= pulp.lpSum(min_ct[i] * y[i][k] for i in V),
                f'Z_pred_lb_{k}'
            )

    # ── OBJECTIVE ────────────────────────────────────────────────
    # Selection-only variant: max Σ v_i · x_i (knapsack with precedence,
    # LP-tight, closes on n ≤ 100 in seconds).
    # Time-aware variant (slide 9): max Σ v_i · x_i − λ · Σ Z[k]
    # (1|prec|ΣwjCj — NP-hard with weak LP, doesn't close beyond n ≈ 25-35).
    if use_time_aware:
        model += (
            pulp.lpSum(v[i] * x[i] for i in V)
            - lam * pulp.lpSum(Z[k] for k in K),
            'Obj1_Discounted_Recovery_Profit'
        )
    else:
        model += (
            pulp.lpSum(v[i] * x[i] for i in V),
            'Obj1_Net_Recovery_Profit'
        )

    # ── Budget constraint (EUR) ─────────────────────────────────
    if B_max_eur is not None:
        model += pulp.lpSum(
            (c[i] + h * p[i]) * x[i] for i in V
        ) <= B_max_eur, 'budget_eur'

    # Add order constraints (order_rules) if present in data — sequence-only.
    order_rules = data.get('order_rules', [])
    if use_time_aware and order_rules:
        label_to_num = {str(i): i for i in V}
        for idx, rule in enumerate(order_rules):
            rule_if = rule.get('if', [])
            cond = rule.get('condition', '').lower()
            if len(rule_if) == 2:
                n1, n2 = rule_if[0], rule_if[1]
                n1_num = label_to_num.get(n1, n1) if not isinstance(n1, int) else n1
                n2_num = label_to_num.get(n2, n2) if not isinstance(n2, int) else n2
                if n1_num in V and n2_num in V:
                    if 'avant' in cond:
                        for k in K:
                            model += pulp.lpSum(y[n1_num][k_] for k_ in K if k_ < k) >= y[n2_num][k], f'order_avant_{n1_num}_{n2_num}_{k}_{idx}'
                    if 'apres' in cond:
                        for k in K:
                            model += pulp.lpSum(y[n2_num][k_] for k_ in K if k_ < k) >= y[n1_num][k], f'order_apres_{n1_num}_{n2_num}_{k}_{idx}'

    return model, {'x': x, 'y': y, 's': s}

# -- Resolution --------------------------------------------------------------

def solve_model(model: pulp.LpProblem, time_limit: Optional[int] = None, use_cplex: bool = True, gap_limit: Optional[float] = None, instance_name: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    solver = None
    time_limit_msg = f" (limite: {time_limit}s)" if time_limit else " (sans limite de temps)"
    solver_log_file: Optional[str] = None
    
    if use_cplex:
        try:
            # Operating system detection for CPLEX path
            if os.name == 'nt':  # Windows
                cplex_path = "C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio2211\\cplex\\bin\\x64_win64\\cplex.exe"
            else:
                import platform
                possible_paths = []
                if platform.system() == 'Darwin':  # macOS
                    possible_paths = [
                        "/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex",   # macOS Apple Silicon
                        "/Applications/CPLEX_Studio2211/cplex/bin/x86-64_osx/cplex",  # macOS Intel
                        "/Applications/CPLEX_Studio221/cplex/bin/arm64_osx/cplex",
                        "/Applications/CPLEX_Studio221/cplex/bin/x86-64_osx/cplex",
                        os.path.expanduser("~/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex"),
                        "cplex",
                    ]
                else:  # Linux/Unix
                    possible_paths = [
                        "/opt/ibm/ILOG/CPLEX_Studio2211/cplex/bin/x86-64_linux/cplex",
                        "/usr/local/bin/cplex",
                        "/usr/bin/cplex",
                        os.path.expanduser("~/cplex_studio/cplex/bin/x86-64_linux/cplex"),
                        "cplex",
                    ]
                cplex_path = None
                for path in possible_paths:
                    if path == "cplex" and os.system("which cplex > /dev/null 2>&1") == 0:
                        cplex_path = "cplex"
                        break
                    elif os.path.isfile(path) and os.access(path, os.X_OK):
                        cplex_path = path
                        break

                if cplex_path is None:
                    raise FileNotFoundError(f"CPLEX not found ({platform.system()})")
            
            if not os.path.isfile(cplex_path) and cplex_path != "cplex":
                raise FileNotFoundError(f"CPLEX not found at location: {cplex_path}")
            if cplex_path != "cplex" and not os.access(cplex_path, os.X_OK):
                raise PermissionError(f"CPLEX is not executable at: {cplex_path}")
            
            # Create a dedicated log file to capture interactive CPLEX output
            logs_dir = os.path.join("results", "solver_logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            # Extract instance size from name
            size_info = ""
            if instance_name:
                if "n=20" in instance_name:
                    size_info = "_n20"
                elif "n=50" in instance_name:
                    size_info = "_n50"
                elif "n=100" in instance_name and "1000" not in instance_name:
                    size_info = "_n100"
                elif "n=1000" in instance_name:
                    if "cut" in instance_name:
                        # Extract the number after cut
                        import re
                        match = re.search(r'cut(\d+)', instance_name)
                        if match:
                            cut_size = match.group(1)
                            size_info = f"_n{cut_size}"
                    else:
                        size_info = "_n1000"
            
            solver_log_file = os.path.join(logs_dir, f"cplex{size_info}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
            cplex_options = [f"set logfile {solver_log_file}"]
            if gap_limit is not None and gap_limit > 0:
                cplex_options.append(f"set mip tolerances mipgap {gap_limit}")
            # CPLEX tuning: aggressive cuts + emphasis on proving optimality
            cplex_options.append("set mip strategy variableselect 3")   # strong branching
            cplex_options.append("set mip cuts all 2")                  # aggressive cuts
            cplex_options.append("set emphasis mip 2")                  # emphasis optimality

            print(f"Using CPLEX{time_limit_msg} ({cplex_path})")
            solver = CPLEX_CMD(path=cplex_path, msg=True, timeLimit=time_limit, keepFiles=True, options=cplex_options)
        except Exception as e:
            print(f"CPLEX error: {e} → switching to CBC", file=sys.stderr)
    if solver is None:
        print(f"Using CBC{time_limit_msg}")
        if gap_limit is not None and gap_limit > 0:
            solver = PULP_CBC_CMD(msg=True, timeLimit=time_limit, gapRel=gap_limit)
        else:
            solver = PULP_CBC_CMD(msg=True, timeLimit=time_limit)
        solver_log_file = None
    

    result = model.solve(solver)
    return pulp.LpStatus[model.status] == 'Optimal', solver_log_file

# -- CPLEX parsers: gap, status, best integer --------------------------------
def _parse_last_gap_from_cplex_log(log_file: Optional[str]) -> Optional[float]:
    if not log_file or not os.path.exists(log_file):
        return None
    try:
        text = ''
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        lines = text.splitlines()
        for line in reversed(lines):
            if 'gap =' in line:
                m = re.search(r'([0-9.]+)%', line)
                if m:
                    try:
                        return float(m.group(1))
                    except Exception:
                        return None
        return None
    except Exception:
        return None

def _parse_last_status_from_cplex_log(log_file: Optional[str]) -> str:
    """
    Parse the final solver status from the CPLEX log (Optimal, TimeLimit, etc.)
    Returns a string: 'Optimal', 'TimeLimit', 'Feasible', 'Infeasible', etc.
    IMPORTANT: Ignores "dual simplex - optimal" which is just the LP resolution after MIP.
    """
    if not log_file or not os.path.exists(log_file):
        return "Unknown"
    # Optional debug trace — to identify which branch matched, drop a print()
    # before any `return ...` below. The matching pattern is described in the
    # surrounding `# Priority N:` comment.
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().lower()
            lines = text.splitlines()

        # Priority 1: TimeLimit - if log contains 'mip - time limit exceeded'
        if "mip - time limit exceeded" in text:
            return "TimeLimit"

        # Search MIP statuses from end (most recent first)
        for line in reversed(lines):
            l = line.strip().lower()

            # Priority 2: MIP optimal statuses (the real statuses to consider)
            if "mip - integer optimal, tolerance" in l:
                return "Optimal"
            if "mip - integer optimal solution" in l:
                return "Optimal"
            if "mip - integer optimal" in l and "tolerance" not in l:
                return "Optimal"

            # Priority 3: TimeLimit MIP patterns (more specific first)
            if "mip - time limit exceeded, integer feasible" in l:
                return "TimeLimit"
            if "mip - time limit exceeded" in l:
                return "TimeLimit"

            # Priority 4: Other MIP patterns
            if "mip - integer infeasible" in l or "mip - no integer solution" in l:
                return "Infeasible"
            if "mip - integer feasible" in l:
                return "Feasible"

            # Priority 5: Generic patterns (fallback)
            if "integer optimal solution" in l:
                return "Optimal"

            # Priority 6: Solution status
            if "solution status" in l:
                if "optimal" in l:
                    return "Optimal"
                elif "infeasible" in l:
                    return "Infeasible"
                elif "feasible" in l:
                    return "Feasible"

            # Priority 7: Generic TimeLimit patterns
            if "time limit exceeded, integer feasible" in l:
                return "TimeLimit"
            if "time limit exceeded" in l:
                return "TimeLimit"
            # Note: Avoid "timelimit" alone as it captures "CPXPARAM_TimeLimit"

            # Priority 8: Gap tolerance
            if "gap tolerance" in l and "reached" in l:
                return "GapReached"

            # IGNORE: "dual simplex - optimal" because it's just the LP resolution after MIP
            # This line appears systematically and does not indicate the actual MIP status

        # Global fallback patterns (if no specific pattern found)
        if "time limit exceeded" in text:
            return "TimeLimit"
        if "gap tolerance" in text:
            return "GapReached"
        if "integer feasible" in text:
            return "Feasible"

        return "Unknown"
    except Exception:
        return "Unknown"

def _parse_best_integer_from_cplex_log(log_file: Optional[str], fallback_terminal_log: Optional[str] = None) -> Optional[float]:
    """
    Extract the last 'Best Integer' value (best feasible solution) from CPLEX log.
    Looks for 'Found incumbent of value X' and 'MIP - Integer optimal solution: Objective = X'.
    If nothing is found, attempts to parse fallback_terminal_log (e.g., instance terminal.log).
    """
    import re
    def extract_incumbent(lines):
        best_integer = None
        for line in lines:
            if 'Found incumbent of value' in line:
                m = re.search(r'Found incumbent of value\s+([0-9]+\.?[0-9]*)', line)
                if m:
                    best_integer = float(m.group(1))
        for line in lines:
            if 'MIP - Integer optimal solution:' in line and 'Objective =' in line:
                m = re.search(r'Objective\s*=\s*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)', line)
                if m:
                    best_integer = float(m.group(1))
        return best_integer

    # 1. Try the main CPLEX log
    if log_file and os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            best_integer = extract_incumbent(lines)
            if best_integer is not None:
                return best_integer
        except Exception:
            pass
    # 2. Fallback to terminal.log if provided
    if fallback_terminal_log and os.path.exists(fallback_terminal_log):
        try:
            with open(fallback_terminal_log, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            best_integer = extract_incumbent(lines)
            if best_integer is not None:
                return best_integer
        except Exception:
            pass
    return None

# -- Result display ------------------------------------------------------

def _format_duration_human(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    try:
        secs = float(seconds)
    except Exception:
        return None
    if secs < 60:
        return f"{secs:.2f} s"
    total = int(round(secs))
    days = total // 86400
    rem = total % 86400
    hours = rem // 3600
    rem %= 3600
    minutes = rem // 60
    s_rem = rem % 60
    parts = []
    if days > 0:
        parts.append(f"{days} j")
    if hours > 0 or days > 0:
        parts.append(f"{hours} h")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{minutes} min")
    if s_rem > 0 and days == 0:  # avoid too fine detail if already displaying days
        parts.append(f"{s_rem} s")
    return " ".join(parts)

def display_solution(model: pulp.LpProblem, data: Dict[str, Any], vars: Dict[str, Any], print_zero: bool = False) -> None:
    V = data['V']
    x = vars['x']
    y = vars['y']
    s = vars['s']

    print("\n=== Result ===")
    print("Status:", pulp.LpStatus[model.status])
    print(f"Net profit = {pulp.value(model.objective):.2f}")
    if s:
        sp = s.value()
        human = _format_duration_human(sp)
        if human:
            print(f"Makespan   = {sp:.2f} s ({human})\n")
        else:
            print(f"Makespan   = {sp:.2f} s\n")

    rows = []
    for i in V:
        xi = x.get(i)
        if xi is None:
            continue
        if xi.value() > 0.5 or print_zero:
            rang = next((k for k in y[i] if y[i][k].value() > 0.5), None)
            rows.append((i, rang, xi.value()))
    rows.sort(key=lambda row: row[1] if row[1] is not None else 1e9)
    for i, rang, _ in rows:
        print(f"• Part {i} removed at position {rang}")

# -- Solution saving ------------------------------------------------      

def save_solution(output_path: str, model: pulp.LpProblem, data: Dict[str, Any], vars: Dict[str, Any], instance_path: Optional[str] = None, criteria: Optional[str] = None) -> None:
    V = data['V']
    T = data['T']
    x = vars['x']
    y = vars['y']
    s = vars['s']

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=== MILP Result ===\n")
        # Instance identification
        if instance_path:
            instance_name = os.path.splitext(os.path.basename(instance_path))[0]
            f.write(f"Instance: {instance_name}\n")
            f.write(f"File: {instance_path}\n")
        if criteria:
            f.write(f"Criteria: {criteria}\n")
        f.write(f"Status: {pulp.LpStatus[model.status]}\n")
        f.write(f"Net profit = {pulp.value(model.objective):.2f}\n")
        if s:
            sp = s.value()
            human = _format_duration_human(sp)
            if human:
                f.write(f"Makespan   = {sp:.2f} s ({human})\n\n")
            else:
                f.write(f"Makespan   = {sp:.2f} s\n\n")

        f.write(f"Target components: {', '.join(map(str, T))}\n\n")

        rows = []
        for i in V:
            xi = x.get(i)
            if xi is None or xi.value() is None:
                continue
            if xi.value() > 0.5:
                rang = next((k for k in y[i] if y[i][k].value() > 0.5), None)
                rows.append((i, rang, xi.value()))
        rows.sort(key=lambda row: row[1] if row[1] is not None else 1e9)
        for i, rang, _ in rows:
            tag = " [TARGET]" if i in T else ""
            f.write(f"• Part {i} removed at position {rang}{tag}\n")


# -- Importable entry point -----------------------------------------------

def extract_sequence(model: pulp.LpProblem, data: Dict[str, Any], vars: Dict[str, Any]) -> Tuple[List[int], float]:
    """Extract the ordered sequence of selected parts and compute the serial makespan.

    Returns (sequence, estimated_makespan).

    If the model has y[i,k] (time-aware variant), the sequence is read from
    those positional variables.  Otherwise (selection-only variant) we return
    a topological order over the selected x's, which is a valid sequence
    respecting precedence (and the order minimising completion times has to
    be a topological order anyway).
    """
    V = data['V']
    E = data['E']
    x = vars['x']
    y = vars.get('y')
    p = data['p']

    selected = [i for i in V
                if x.get(i) is not None
                and x[i].value() is not None
                and x[i].value() > 0.5]

    if y is not None:
        # Time-aware: positional variables tell us the order
        ranked: List[Tuple[int, int]] = []
        for i in selected:
            rang = next((k for k in y[i] if y[i][k].value() > 0.5), None)
            if rang is not None:
                ranked.append((i, rang))
        ranked.sort(key=lambda t: t[1])
        seq = [i for i, _ in ranked]
    else:
        # Selection-only: topological sort on the selected sub-graph
        sel_set = set(selected)
        in_deg = {i: 0 for i in selected}
        succs: Dict[int, List[int]] = {i: [] for i in selected}
        for (a, b) in E:
            if a in sel_set and b in sel_set:
                in_deg[b] += 1
                succs[a].append(b)
        roots = [i for i, d in in_deg.items() if d == 0]
        # Stable order by id for reproducibility
        roots.sort()
        seq = []
        while roots:
            i = roots.pop(0)
            seq.append(i)
            for j in succs[i]:
                in_deg[j] -= 1
                if in_deg[j] == 0:
                    roots.append(j)
            roots.sort()

    makespan_series = sum(p[i] for i in seq)
    print("[DIAG] extract_sequence - raw extracted sequence:", seq)
    return seq, makespan_series

def main(instance_path: str, time_limit: Optional[int] = 7200, gap_limit: Optional[float] = None) -> None:
    print(f"→ Loading instance: {instance_path}")
    data = load_data(instance_path)
    print("→ Data loaded:")
    pprint.pp(data)

    print("→ Building model...")
    model, variables = build_model(data)

    if time_limit is None or time_limit == 0:
        print("→ Solving (no time limit)...")
        time_limit_value = None
    else:
        print(f"→ Solving (limit: {time_limit}s)...")
        time_limit_value = time_limit
    if gap_limit is not None and gap_limit > 0:
        print(f"→ Stop criterion: gap ≤ {gap_limit*100:.2f}%")
    
    # Adaptive mode only if enabled by environment variable and no explicit gap
    adaptive_mode = ((gap_limit is None) or (gap_limit <= 0)) and os.getenv('DSP_ADAPTIVE_GAP', '0') == '1'
    logs_to_append: list[tuple[str, str]] = []  # (title, path)

    # Console redirection -> buffer for context
    stdout_orig, stderr_orig = sys.stdout, sys.stderr
    class _Tee:
        def __init__(self, *streams): self._streams = streams
        def write(self, data):
            for s in self._streams:
                try: s.write(data); s.flush()
                except Exception: pass
        def flush(self):
            for s in self._streams:
                try: s.flush()
                except Exception: pass
    buffer = io.StringIO()
    sys.stdout = _Tee(stdout_orig, buffer)
    sys.stderr = _Tee(stderr_orig, buffer)

    try:
        if adaptive_mode:
            relax_after = int(os.getenv('DSP_GAP_RELAX_AFTER', '3600'))  # 1h default
            # Step 1: gap 3%
            stage1_time = relax_after if time_limit_value is None else max(1, min(relax_after, time_limit_value))
            print(f"→ Adaptive mode: Step 1 mipgap=3% for {stage1_time if stage1_time else '∞'}s")
            optimal1, log1 = solve_model(model, time_limit=stage1_time, use_cplex=True, gap_limit=0.03)
            gap1 = _parse_last_gap_from_cplex_log(log1)
            if log1: logs_to_append.append(("Log CPLEX (Etape 1, 3%)", log1))
            stop_after_stage1 = (gap1 is not None and gap1 <= 0.03) or optimal1

            do_stage2 = not stop_after_stage1
            stage2_time = None
            if time_limit_value is not None:
                # Remaining time for step 2
                stage2_time = max(0, time_limit_value - stage1_time)
                if stage2_time == 0:
                    do_stage2 = False
            if do_stage2:
                print("→ Adaptive mode: Step 2 mipgap=5%")
                optimal2, log2 = solve_model(model, time_limit=stage2_time, use_cplex=True, gap_limit=0.05)
                if log2: logs_to_append.append(("Log CPLEX (Etape 2, 5%)", log2))
        else:
            # Simple mode: single run with requested gap
            optimal1, log1 = solve_model(model, time_limit=time_limit_value, use_cplex=True, gap_limit=gap_limit)
            if log1:
                gap_label = f"gap {gap_limit*100:.2f}%" if (gap_limit is not None and gap_limit > 0) else "gap N/A"
                logs_to_append.append((f"Log CPLEX ({gap_label})", log1))
    finally:
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig

    solve_log = buffer.getvalue()

    # Display the solution obtained after resolution(s)
    if pulp.LpStatus[model.status] == 'Optimal':
        display_solution(model, data, variables)
    else:
        print("No optimal solution found (stopped by gap or time limit).")

    # Determine save directory
    is_selective = '_selectif' in instance_path
    save_dir = "results/milp_save_selectif" if is_selective else "results/milp_save"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_name = f"{save_dir}/solution_milp_{timestamp}.txt"
    os.makedirs(save_dir, exist_ok=True)

    # Build criteria string
    crit_bits = []
    if time_limit is not None and time_limit > 0:
        crit_bits.append(f"time_limit={time_limit}s")
    if gap_limit is not None and gap_limit > 0:
        crit_bits.append(f"gap≤{gap_limit*100:.2f}%")
    criteria_str = ", ".join(crit_bits) if crit_bits else None

    # Build CPLEX summary from last log if available
    cplex_summary_text = None
    last_log_path = logs_to_append[-1][1] if logs_to_append else None
    if last_log_path and os.path.exists(last_log_path):
        try:
            info = _parse_cplex_log_summary(last_log_path)
            if info:
                cplex_summary_text = _format_cplex_summary(info, gap_limit)
        except Exception:
            cplex_summary_text = None

    # Save main result (with CPLEX Summary if available)
    save_solution_with_summary(output_name, model, data, variables, instance_path=instance_path, criteria=criteria_str, cplex_summary=cplex_summary_text)

    # Add logs (context + CPLEX steps)
    try:
        with open(output_name, 'a', encoding='utf-8') as f:
            if solve_log.strip():
                f.write("\n=== Execution context ===\n")
                f.write(solve_log)
            for title, path in logs_to_append:
                if path and os.path.exists(path):
                    f.write(f"\n=== {title} ===\n")
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as lf:
                            f.write(lf.read())
                    except Exception as e:
                        f.write(f"[Could not read log: {e}]\n")
    except Exception as e:
        print(f"Could not add logs to file: {e}")

    print(f" Solution saved to: {output_name}")

# -- CPLEX summary: parsing and formatting --------------------------------------

def _parse_cplex_log_summary(log_file: str) -> Optional[Dict[str, Any]]:
    if not log_file or not os.path.exists(log_file):
        return None
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        info: Dict[str, Any] = {}

        # Version
        m = re.search(r"^Version identifier:\s*(.+)$", text, flags=re.MULTILINE)
        if m:
            info['version'] = m.group(1).strip()

        # Parameters
        m = re.search(r"^CPXPARAM_TimeLimit\s+([0-9]+)\b", text, flags=re.MULTILINE)
        if m:
            try:
                info['time_limit'] = int(m.group(1))
            except Exception:
                info['time_limit'] = m.group(1).strip()
        m = re.search(r"^CPXPARAM_MIP_Tolerances_MIPGap\s+([0-9eE\.+-]+)", text, flags=re.MULTILINE)
        if m:
            try:
                info['mip_gap_param'] = float(m.group(1))
            except Exception:
                info['mip_gap_param'] = m.group(1).strip()

        # Parallel mode
        m = re.search(r"^Parallel mode:\s*([^,]+),\s*using up to\s*(\d+)\s*threads", text, flags=re.MULTILINE)
        if m:
            info['parallel_mode'] = m.group(1).strip()
            info['threads'] = int(m.group(2))

        # MIP emphasis / search method (facultatif)
        m = re.search(r"^MIP emphasis:\s*(.+)$", text, flags=re.MULTILINE)
        if m:
            info['mip_emphasis'] = m.group(1).strip()
        m = re.search(r"^MIP search method:\s*(.+)$", text, flags=re.MULTILINE)
        if m:
            info['mip_search'] = m.group(1).strip()

        # Objective result (last)
        m = None
        for match in re.finditer(r"MIP\s*-\s*.*?objective\s*=\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", text, flags=re.IGNORECASE):
            m = match
        if m:
            try:
                info['objective'] = float(m.group(1))
            except Exception:
                info['objective'] = m.group(1)

        # Solution time / Nodes (dernier bloc)
        m = None
        for match in re.finditer(r"Solution time\s*=\s*([0-9\.]+)\s*sec\.[^\n]*", text):
            m = match
        if m:
            info['solution_time_sec'] = float(m.group(1))
            # Extract Nodes on the same line if possible
            line = m.group(0)
            m_nodes = re.search(r"Nodes\s*=\s*([^\s\n]+(?:\s*\([^\)]*\))?)", line)
            if m_nodes:
                info['nodes'] = m_nodes.group(1)

        # Deterministic time
        m = None
        for match in re.finditer(r"Deterministic time\s*=\s*([0-9\.]+)\s*ticks", text):
            m = match
        if m:
            info['deterministic_ticks'] = float(m.group(1))

        # Best bound (if available)
        m = None
        for match in re.finditer(r"best bound\s*=?\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", text, flags=re.IGNORECASE):
            m = match
        if m:
            try:
                info['best_bound'] = float(m.group(1))
            except Exception:
                info['best_bound'] = m.group(1)

        # Last explicit mention of gap X%
        gap = _parse_last_gap_from_cplex_log(log_file)
        if gap is not None:
            info['last_gap'] = gap

        return info
    except Exception:
        return None


def _format_cplex_summary(info: Dict[str, Any], gap_limit: Optional[float]) -> str:
    parts: List[str] = []

    # Version / mode / threads line
    meta_bits = []
    if 'version' in info:
        meta_bits.append(f"Version: {info['version']}")
    if 'parallel_mode' in info:
        meta_bits.append(f"Mode: {info['parallel_mode']}")
    if 'threads' in info:
        meta_bits.append(f"Threads: {info['threads']}")
    if meta_bits:
        parts.append("; ".join(meta_bits))

    # Parameters
    param_bits = []
    if 'time_limit' in info and info['time_limit']:
        param_bits.append(f"TimeLimit={info['time_limit']}s")
    if gap_limit is not None and gap_limit > 0:
        param_bits.append(f"MIPGap≈{gap_limit*100:.2f}%")
    elif 'mip_gap_param' in info and info['mip_gap_param'] is not None:
        try:
            param_bits.append(f"MIPGap≈{float(info['mip_gap_param'])*100:.2f}%")
        except Exception:
            param_bits.append(f"MIPGap={info['mip_gap_param']}")
    if param_bits:
        parts.append("Parameters: " + "; ".join(param_bits))

    # Search
    search_bits = []
    if 'mip_search' in info:
        search_bits.append(f"Search: {info['mip_search']}")
    if 'mip_emphasis' in info:
        search_bits.append(f"Emphasis: {info['mip_emphasis']}")
    if search_bits:
        parts.append("; ".join(search_bits))

    # Results
    res_bits = []
    if 'objective' in info:
        try:
            res_bits.append(f"Objective={info['objective']:.6f}")
        except Exception:
            res_bits.append(f"Objective={info['objective']}")
    if 'best_bound' in info:
        try:
            res_bits.append(f"BestBound={info['best_bound']:.6f}")
        except Exception:
            res_bits.append(f"BestBound={info['best_bound']}")
    if 'last_gap' in info:
        res_bits.append(f"Gap≈{info['last_gap']*100:.2f}%")
    if 'solution_time_sec' in info:
        res_bits.append(f"Time={info['solution_time_sec']:.2f}s")
    if 'nodes' in info:
        res_bits.append(f"Nodes={info['nodes']}")
    if 'deterministic_ticks' in info:
        res_bits.append(f"DetTicks={info['deterministic_ticks']:.2f}")
    if res_bits:
        parts.append("Results: " + "; ".join(res_bits))

    return "\n".join(parts)


# -- Save variant with summary -------------------------------------

def save_solution_with_summary(output_path: str, model: pulp.LpProblem, data: Dict[str, Any], vars: Dict[str, Any], instance_path: Optional[str] = None, criteria: Optional[str] = None, cplex_summary: Optional[str] = None) -> None:
    """Enriches the top of the report with a compact CPLEX Summary if provided."""
    V = data['V']
    T = data['T']
    x = vars['x']
    y = vars['y']
    s = vars['s']

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=== MILP Result ===\n")
        if instance_path:
            instance_name = os.path.splitext(os.path.basename(instance_path))[0]
            f.write(f"Instance: {instance_name}\n")
            f.write(f"File: {instance_path}\n")
        if criteria:
            f.write(f"Criteria: {criteria}\n")

        # CPLEX Summary inserted early for quick reading
        if cplex_summary and cplex_summary.strip():
            f.write("\n=== CPLEX Summary ===\n")
            f.write(cplex_summary.strip() + "\n\n")

        f.write(f"Status: {pulp.LpStatus[model.status]}\n")
        f.write(f"Net profit = {pulp.value(model.objective):.2f}\n")
        if s:
            sp = s.value()
            human = _format_duration_human(sp)
            if human:
                f.write(f"Makespan   = {sp:.2f} s ({human})\n\n")
            else:
                f.write(f"Makespan   = {sp:.2f} s\n\n")

        f.write(f"Target components: {', '.join(map(str, T))}\n\n")

        rows = []
        for i in V:
            xi = x.get(i)
            if xi is None or xi.value() is None:
                continue
            if xi.value() > 0.5:
                rang = next((k for k in y[i] if y[i][k].value() > 0.5), None)
                rows.append((i, rang, xi.value()))
        rows.sort(key=lambda row: row[1] if row[1] is not None else 1e9)
        for i, rang, _ in rows:
            tag = " [TARGET]" if i in T else ""
            f.write(f"• Part {i} removed at position {rang}{tag}\n")
