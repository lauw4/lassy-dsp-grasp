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
                data['V'].extend([_parse_value(tok) for tok in re.split('[,;\s]+', line) if tok])
            elif section == 'E':
                parts = [int(tok) for tok in line.split()]
                if len(parts) != 2:
                    raise ValueError(f"Malformed arc: {line}")
                data['E'].append(tuple(parts))
            elif section == 'T':
                data['T'].extend([int(tok) for tok in re.split('[,;\s]+', line) if tok])
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

def build_model(data: Dict[str, Any]) -> Tuple[pulp.LpProblem, Dict[str, Any]]:
    V: List[int] = data['V']
    E: List[Tuple[int, int]] = data['E']
    T: List[int] = data['T']
    r, c, p = data['r'], data['c'], data['p']
    C_max: Optional[float] = data.get('C_max')

    K: List[int] = list(range(1, len(V) + 1))

    model = pulp.LpProblem("Partial_Disassembly", pulp.LpMaximize)

    x = pulp.LpVariable.dicts('x', V, cat='Binary')
    y = pulp.LpVariable.dicts('y', (V, K), cat='Binary')
    s = pulp.LpVariable('makespan', lowBound=0)

    # Add trap penalty if present in data
    trap_penalty = -20.0
    traps = data.get('trap', {})  # Dictionary {i: bool}
    # If data['trap'] doesn't exist, try to build it from nodes
    if not traps:
        traps = {}
        # If data['nodes'] exists (enriched case from JSON)
        for node in data.get('nodes', []):
            nid = node.get('id')
            if nid is not None:
                traps[int(nid)] = node.get('trap', False)
    model += pulp.lpSum((r[i] - c[i] + (trap_penalty if traps.get(i, False) else 0.0)) * x[i] for i in V), 'Net_profit'

    for i in V:
        model += pulp.lpSum(y[i][k] for k in K) == x[i], f'assign_{i}'
    for k in K:
        model += pulp.lpSum(y[i][k] for i in V) <= 1, f'serial_{k}'
    for (i, j) in E:
        for k in K:
            model += y[j][k] <= pulp.lpSum(y[i][k_] for k_ in K if k_ < k), f'prec_{i}_{j}_{k}'
    for i in T:
        model += x[i] == 1, f'target_{i}'
    model += s >= pulp.lpSum(p[i] * y[i][k] for i in V for k in K), 'makespan_def'
    if C_max is not None:
        model += s <= C_max, 'horizon_limit'

    # Add order constraints (order_rules) if present in data
    order_rules = data.get('order_rules', [])
    label_to_num = {str(i): i for i in V}
    for idx, rule in enumerate(order_rules):
        rule_if = rule.get('if', [])
        cond = rule.get('condition', '').lower()
        if len(rule_if) == 2:
            n1, n2 = rule_if[0], rule_if[1]
            # Convert label -> number if needed
            n1_num = label_to_num.get(n1, n1) if not isinstance(n1, int) else n1
            n2_num = label_to_num.get(n2, n2) if not isinstance(n2, int) else n2
            print(f"[DIAG] Adding order constraint: rule={rule}, n1={n1} -> {n1_num}, n2={n2} -> {n2_num}, condition={cond}")
            if n1_num in V and n2_num in V:
                # 'avant' rule: n1 must be before n2
                if 'avant' in cond:
                    for k in K:
                        print(f"[DIAG] Constraint: {n1_num} before {n2_num} (rank {k}): sum(y[{n1_num}][<k]) >= y[{n2_num}][{k}]")
                        model += pulp.lpSum(y[n1_num][k_] for k_ in K if k_ < k) >= y[n2_num][k], f'order_avant_{n1_num}_{n2_num}_{k}_{idx}'
                # 'apres' rule: n1 must be after n2
                if 'apres' in cond:
                    for k in K:
                        print(f"[DIAG] Constraint: {n1_num} after {n2_num} (rank {k}): sum(y[{n2_num}][<k]) >= y[{n1_num}][{k}]")
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
            else:  # Linux/Unix
                # Try several common locations on Linux
                possible_paths = [
                    "/opt/ibm/ILOG/CPLEX_Studio2211/cplex/bin/x86-64_linux/cplex",
                    "/usr/local/bin/cplex",
                    "/usr/bin/cplex",
                    os.path.expanduser("~/cplex_studio/cplex/bin/x86-64_linux/cplex"),
                    "cplex"  # If in PATH
                ]
                cplex_path = None
                for path in possible_paths:
                    if os.path.isfile(path) and os.access(path, os.X_OK):
                        cplex_path = path
                        break
                    elif path == "cplex" and os.system("which cplex > /dev/null 2>&1") == 0:
                        cplex_path = "cplex"
                        break
                
                if cplex_path is None:
                    raise FileNotFoundError("CPLEX not found on this Linux system")
            
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

            print(f"Using CPLEX{time_limit_msg} ({cplex_path})")
            solver = CPLEX_CMD(path=cplex_path, msg=True, timeLimit=time_limit, keepFiles=True, options=cplex_options)
        except Exception as e:
            print(f"CPLEX error: {e} → switching to CBC", file=sys.stderr)
    if solver is None:
        print(f"Using CBC{time_limit_msg}")
        if gap_limit is not None and gap_limit > 0:
            solver = PULP_CBC_CMD(msg=True, timeLimit=time_limit, fracGap=gap_limit)
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
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().lower()
            lines = text.splitlines()
        
        # print(f"[DEBUG STATUS] Parsing log file: {log_file}")
        
        # Priority 1: TimeLimit - if log contains 'mip - time limit exceeded'
        if "mip - time limit exceeded" in text:
            # print("[DEBUG STATUS] Found 'mip - time limit exceeded' -> TimeLimit")
            return "TimeLimit"
        
        # Search MIP statuses from end (most recent first)
        for i, line in enumerate(reversed(lines)):
            l = line.strip().lower()
            line_num = len(lines) - i
            
            # Priority 2: MIP optimal statuses (the real statuses to consider)
            if "mip - integer optimal, tolerance" in l:
                # print(f"[DEBUG STATUS] Line {line_num}: Found 'mip - integer optimal, tolerance' -> Optimal")
                return "Optimal"
            if "mip - integer optimal solution" in l:
                # print(f"[DEBUG STATUS] Line {line_num}: Found 'mip - integer optimal solution' -> Optimal")
                return "Optimal" 
            if "mip - integer optimal" in l and "tolerance" not in l:
                # print(f"[DEBUG STATUS] Line {line_num}: Found 'mip - integer optimal' -> Optimal")
                return "Optimal"
            
            # Priority 3: TimeLimit MIP patterns (more specific first)
            if "mip - time limit exceeded, integer feasible" in l:
                # print(f"[DEBUG STATUS] Line {line_num}: Found 'mip - time limit exceeded, integer feasible' -> TimeLimit")
                return "TimeLimit"
            if "mip - time limit exceeded" in l:
                # print(f"[DEBUG STATUS] Line {line_num}: Found 'mip - time limit exceeded' -> TimeLimit")
                return "TimeLimit"
            
            # Priority 4: Other MIP patterns
            if "mip - integer infeasible" in l or "mip - no integer solution" in l:
                # print(f"[DEBUG STATUS] Line {line_num}: Found MIP infeasible -> Infeasible")
                return "Infeasible"
            if "mip - integer feasible" in l:
                # print(f"[DEBUG STATUS] Line {line_num}: Found 'mip - integer feasible' -> Feasible")
                return "Feasible"
            
            # Priority 5: Generic patterns (fallback)
            if "integer optimal solution" in l:
                # print(f"[DEBUG STATUS] Line {line_num}: Found 'integer optimal solution' -> Optimal")
                return "Optimal"
            
            # Priority 6: Solution status
            if "solution status" in l:
                # print(f"[DEBUG STATUS] Line {line_num}: Found 'solution status': {l}")
                if "optimal" in l:
                    # print("[DEBUG STATUS] -> Optimal")
                    return "Optimal"
                elif "infeasible" in l:
                    # print("[DEBUG STATUS] -> Infeasible")
                    return "Infeasible"
                elif "feasible" in l:
                    # print("[DEBUG STATUS] -> Feasible")
                    return "Feasible"
            
            # Priority 7: Generic TimeLimit patterns
            if "time limit exceeded, integer feasible" in l:
                # print(f"[DEBUG STATUS] Line {line_num}: Found 'time limit exceeded, integer feasible' -> TimeLimit")
                return "TimeLimit"
            if "time limit exceeded" in l:
                # print(f"[DEBUG STATUS] Line {line_num}: Found 'time limit exceeded' -> TimeLimit")
                return "TimeLimit"
            # Note: Avoid "timelimit" alone as it captures "CPXPARAM_TimeLimit"
            
            # Priority 8: Gap tolerance
            if "gap tolerance" in l and "reached" in l:
                # print(f"[DEBUG STATUS] Line {line_num}: Found 'gap tolerance reached' -> GapReached")
                return "GapReached"
            
            # IGNORE: "dual simplex - optimal" because it's just the LP resolution after MIP
            # This line appears systematically and does not indicate the actual MIP status
        
        # Global fallback patterns (if no specific pattern found)
        if "time limit exceeded" in text:
            # print("[DEBUG STATUS] Found 'time limit exceeded' in text -> TimeLimit")
            return "TimeLimit"
        if "gap tolerance" in text:
            # print("[DEBUG STATUS] Found 'gap tolerance' in text -> GapReached")
            return "GapReached"
        if "integer feasible" in text:
            # print("[DEBUG STATUS] Found 'integer feasible' in text -> Feasible")
            return "Feasible"
        
        # print("[DEBUG STATUS] No status found -> Unknown")
        return "Unknown"
    except Exception as e:
        # print(f"[DEBUG STATUS] Exception: {e} -> Unknown")
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

    Returns (sequence, estimated_makespan)
    Note: the makespan returned here is the sum of p[i] for selected parts in order, consistent
    with the 'serial line' assumption used in the comparison.
    For the exact makespan as modeled, use the 's' variable from the model.
    """
    V = data['V']
    x = vars['x']
    y = vars['y']
    p = data['p']
    selected: List[Tuple[int,int]] = []
    for i in V:
        xi = x.get(i)
        if xi and xi.value() is not None and xi.value() > 0.5:
            # Find the associated rank k
            rang = next((k for k in y[i] if y[i][k].value() > 0.5), None)
            if rang is not None:
                selected.append((i, rang))
    selected.sort(key=lambda t: t[1])
    seq = [i for i,_ in selected]
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