#!/usr/bin/env python3
"""
Generate STRESS-TEST scenarios for the adaptive fuzzy pipeline.

Creates 5 difficulty levels beyond the current simple/intermediate/complex:
  - stress_4:     4 failures, mixed critical/non-critical
  - stress_6:     6 failures, multiple cascading on critical path
  - stress_8:     8 failures, heavy load with limited recovery
  - stress_10:   10 failures, near-saturation (10%+ of components fail)
  - cascade:      Chain of 3-5 adjacent failures (parent→child cascading)

Each scenario ensures at least partial recoverability (realistic industrial
setting where operators have SOME tools/options but the situation is severe).

Recovery difficulty is calibrated:
  - stress_4:  ~80% of failures have a recovery option
  - stress_6:  ~65% of failures have a recovery option
  - stress_8:  ~55% of failures have a recovery option
  - stress_10: ~45% of failures have a recovery option
  - cascade:   ~60% but concentrated on the same path → cascading effects
"""

import os, sys, json, glob, random
import networkx as nx
import numpy as np

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, BASE)
os.chdir(BASE)

FAILURES_DIR = "data/adaptive/generated_failures"
INSTANCES_DIR = "data/instances_base"

random.seed(2026)
np.random.seed(2026)

# ── Constants ──

SYMPTOMS = [
    "OVER_TORQUE_NO_ADVANCE",
    "ACCESS_BLOCKED",
    "NO_SEPARATION",
    "LOW_VALUE_BOTTLENECK",
]

SIGNAL_TEMPLATES = {
    "OVER_TORQUE_NO_ADVANCE": {"force": (50, 95), "torque": (60, 100), "over_torque": 1},
    "ACCESS_BLOCKED":         {"force": (40, 80), "torque": (20, 60), "access_blocked": 1},
    "NO_SEPARATION":          {"force": (70, 100), "torque": (30, 70), "joint_opened": 0},
    "LOW_VALUE_BOTTLENECK":   {"force": (20, 50), "torque": (10, 40), "low_value": 1},
}

OP_TYPES = ["unscrew", "pull", "cut", "pry", "detach"]
TOOL_IDS = ["T_hex", "T_torx", "T_pneumatic", "T_impact", "T_pliers"]
ALT_TOOLS = [["T_pneumatic"], ["T_impact"], ["T_power"], ["T_alt"], ["T_electric"]]

# ── Graph loading ──

def load_graph(instance_name):
    path = os.path.join(INSTANCES_DIR, f"{instance_name}.json")
    if not os.path.exists(path):
        return None, [], {}
    with open(path) as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    node_data = {}
    for node in data.get("nodes", []):
        if isinstance(node, dict):
            nid = node["id"]
            attrs = {k: v for k, v in node.items() if k != "id"}
            G.add_node(nid, **attrs)
            node_data[nid] = attrs
        else:
            G.add_node(str(node))
            node_data[str(node)] = {}
    for edge in data.get("edges", []):
        if isinstance(edge, list) and len(edge) == 2:
            G.add_edge(edge[0], edge[1])
        elif isinstance(edge, dict):
            G.add_edge(edge["from"], edge["to"])
    
    targets = data.get("targets", [])
    return G, targets, node_data


def ancestors_of_targets(G, targets):
    """Get all nodes on critical path to targets."""
    critical = set(targets)
    for t in targets:
        try:
            critical |= nx.ancestors(G, t)
        except nx.NodeNotFound:
            pass
    return critical


# ── Failure event generation ──

def make_signals(symptom):
    """Generate realistic signal values for a given symptom."""
    template = SIGNAL_TEMPLATES[symptom]
    signals = {}
    for key, val in template.items():
        if isinstance(val, tuple):
            signals[key] = round(random.uniform(val[0], val[1]), 1)
        else:
            signals[key] = val
    return signals


def make_failure(comp_id, G, node_data, targets, recovery_prob=0.7):
    """
    Generate a single failure event for a component.
    
    Args:
        comp_id: Component to fail
        G: Graph
        node_data: Node attributes
        targets: Target components
        recovery_prob: Probability that this failure has a recovery option
    
    Returns:
        Failure dict
    """
    symptom = random.choice(SYMPTOMS)
    signals = make_signals(symptom)
    
    # Get node profit
    profit = node_data.get(comp_id, {}).get('profit', random.randint(5, 95))
    
    # Compute profit cutoff (20th percentile)
    all_profits = [node_data.get(n, {}).get('profit', 0) for n in G.nodes]
    cutoff = float(np.percentile(all_profits, 20)) if all_profits else 20.0
    
    # Build context with calibrated recovery options
    has_recovery = random.random() < recovery_prob
    
    # Check if on critical path
    critical = ancestors_of_targets(G, targets) | set(targets)
    on_critical = comp_id in critical
    
    if has_recovery:
        # Choose recovery mechanism
        r = random.random()
        if r < 0.40:
            # Bypass
            alt_paths = True
            alt_tools = None
            no_destroy = random.choice([True, False])
        elif r < 0.65:
            # Change tool
            alt_paths = random.choice([True, False])
            alt_tools = random.choice(ALT_TOOLS)
            no_destroy = random.choice([True, False])
        elif r < 0.85:
            # Destroy (low value + destroyable)
            alt_paths = random.choice([True, False])
            alt_tools = None
            no_destroy = False
            profit = min(profit, max(int(cutoff * 0.7), 3))  # Force low value
        else:
            # Multiple options (easiest)
            alt_paths = True
            alt_tools = random.choice(ALT_TOOLS)
            no_destroy = False
    else:
        # No easy recovery: hard scenario
        alt_paths = False
        alt_tools = None
        # High value + non-destroyable = very hard
        no_destroy = True if random.random() < 0.6 else False
        profit = max(profit, int(cutoff * 1.5))  # Force high value
    
    context = {
        "value_piece": profit,
        "no_destroy": no_destroy,
        "alt_paths": alt_paths,
    }
    if alt_tools:
        context["alt_tools_available"] = alt_tools
    
    return {
        "comp_id": comp_id,
        "op_type": random.choice(OP_TYPES),
        "tool_id": random.choice(TOOL_IDS),
        "symptom": symptom,
        "signals": signals,
        "context": context,
        "history": {
            "failure_count": random.randint(0, 3)
        }
    }


# ── Scenario generators ──

def select_failure_components(G, targets, n_failures, mode="mixed"):
    """
    Select components to fail.
    
    Modes:
      - mixed: Mix of critical path and non-critical nodes
      - critical: Prefer critical path nodes (harder)
      - cascade: Select adjacent nodes for cascading failures
    """
    critical = ancestors_of_targets(G, targets) | set(targets)
    non_critical = set(G.nodes) - critical
    all_nodes = list(G.nodes)
    
    if len(all_nodes) < n_failures:
        n_failures = len(all_nodes)
    
    if mode == "mixed":
        # ~60% critical, ~40% non-critical
        n_crit = min(int(n_failures * 0.6) + 1, len(critical))
        n_non = n_failures - n_crit
        if n_non > len(non_critical):
            n_non = len(non_critical)
            n_crit = n_failures - n_non
        
        selected = set()
        crit_list = list(critical - set(targets))  # Don't fail targets directly (usually)
        if len(crit_list) < n_crit:
            crit_list = list(critical)
        selected.update(random.sample(crit_list, min(n_crit, len(crit_list))))
        
        remaining = list(non_critical - selected)
        if remaining and n_non > 0:
            selected.update(random.sample(remaining, min(n_non, len(remaining))))
        
        # Fill up if needed
        while len(selected) < n_failures:
            pool = list(set(all_nodes) - selected)
            if not pool:
                break
            selected.add(random.choice(pool))
        
        return list(selected)[:n_failures]
    
    elif mode == "critical":
        # All on critical path
        crit_list = list(critical - set(targets))
        if len(crit_list) < n_failures:
            crit_list = list(critical)
        if len(crit_list) < n_failures:
            crit_list = all_nodes
        return random.sample(crit_list, min(n_failures, len(crit_list)))
    
    elif mode == "cascade":
        # Find a chain of adjacent nodes (parent → child → grandchild...)
        # Start from a random critical node and walk down
        crit_list = list(critical - set(targets))
        if not crit_list:
            crit_list = list(critical)
        
        start = random.choice(crit_list)
        chain = [start]
        current = start
        
        for _ in range(n_failures - 1):
            successors = list(G.successors(current))
            if not successors:
                # Try predecessors instead
                predecessors = list(G.predecessors(current))
                if predecessors:
                    current = random.choice(predecessors)
                else:
                    # Pick random neighbor
                    remaining = list(set(all_nodes) - set(chain))
                    if remaining:
                        current = random.choice(remaining)
                    else:
                        break
            else:
                # Prefer successors not already in chain
                available = [s for s in successors if s not in chain]
                if available:
                    current = random.choice(available)
                else:
                    current = random.choice(successors)
            
            if current not in chain:
                chain.append(current)
        
        return chain[:n_failures]
    
    return random.sample(all_nodes, min(n_failures, len(all_nodes)))


def generate_stress_scenario(G, targets, node_data, n_failures, recovery_prob, mode="mixed"):
    """Generate a stress-test scenario."""
    comps = select_failure_components(G, targets, n_failures, mode=mode)
    failures = []
    for comp in comps:
        fail = make_failure(comp, G, node_data, targets, recovery_prob=recovery_prob)
        failures.append(fail)
    return failures


# ── Scenario configurations ──

SCENARIO_CONFIGS = {
    "stress_4":  {"n_failures": 4,  "recovery_prob": 0.80, "mode": "mixed"},
    "stress_6":  {"n_failures": 6,  "recovery_prob": 0.65, "mode": "mixed"},
    "stress_8":  {"n_failures": 8,  "recovery_prob": 0.55, "mode": "critical"},
    "stress_10": {"n_failures": 10, "recovery_prob": 0.45, "mode": "critical"},
    "cascade":   {"n_failures": 5,  "recovery_prob": 0.60, "mode": "cascade"},
}


# ── Instance selection ──

def select_instances(min_n=20):
    """Select instances suitable for stress testing (enough nodes)."""
    selected = []
    for f in sorted(glob.glob(os.path.join(INSTANCES_DIR, "*.json"))):
        name = os.path.basename(f).replace('.json', '')
        with open(f) as fh:
            data = json.load(fh)
        n = len(data.get('nodes', []))
        if n >= min_n:
            selected.append(name)
    return selected


# ── Main ──

def main():
    print("=" * 80)
    print("  STRESS-TEST SCENARIO GENERATOR")
    print("  Generating challenging scenarios for adaptive fuzzy pipeline")
    print("=" * 80)
    
    # Select instances: all with n >= 20
    all_instances = select_instances(min_n=20)
    
    # For large benchmark: use a representative subset
    # Pick 15 diverse instances (small/medium/large)
    instance_sizes = {}
    for name in all_instances:
        path = os.path.join(INSTANCES_DIR, f"{name}.json")
        with open(path) as f:
            data = json.load(f)
        instance_sizes[name] = len(data.get('nodes', []))
    
    # Sort by size and pick evenly
    sorted_instances = sorted(instance_sizes.items(), key=lambda x: x[1])
    
    # Pick ~15 instances spread across sizes
    n_pick = min(15, len(sorted_instances))
    step = max(1, len(sorted_instances) // n_pick)
    selected = [sorted_instances[i * step][0] for i in range(n_pick)]
    
    # Also ensure we include dagtest and gearpump for comparison
    for extra in ['dagtest', 'gearpump']:
        if extra in instance_sizes and extra not in selected:
            selected.append(extra)
    
    print(f"\n  Selected {len(selected)} instances for stress testing")
    for name in selected:
        print(f"    {name:50s} n={instance_sizes[name]}")
    
    # Generate scenarios
    total_scenarios = 0
    total_failures = 0
    
    for inst_name in selected:
        G, targets, node_data = load_graph(inst_name)
        if G is None or len(G.nodes) == 0:
            print(f"\n  SKIP {inst_name}: could not load graph")
            continue
        
        n = len(G.nodes)
        
        # Create output directory
        out_dir = os.path.join(FAILURES_DIR, inst_name)
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"\n  {inst_name} (n={n}, targets={len(targets)}):")
        
        for scen_name, config in SCENARIO_CONFIGS.items():
            n_fail = config["n_failures"]
            
            # Adapt failure count to instance size
            # Don't have more failures than 30% of nodes
            max_fail = max(2, int(n * 0.30))
            actual_fail = min(n_fail, max_fail)
            
            # Skip stress_8/10 for small instances
            if n < 30 and actual_fail > 5:
                continue
            
            scenario = generate_stress_scenario(
                G, targets, node_data,
                n_failures=actual_fail,
                recovery_prob=config["recovery_prob"],
                mode=config["mode"]
            )
            
            # Save
            out_path = os.path.join(out_dir, f"scenario_{scen_name}_{inst_name}.json")
            with open(out_path, 'w') as f:
                json.dump(scenario, f, indent=2)
            
            total_scenarios += 1
            total_failures += len(scenario)
            
            # Show recovery stats
            n_recoverable = sum(1 for fail in scenario if
                fail['context'].get('alt_paths') is True or
                bool(fail['context'].get('alt_tools_available')) or
                (not fail['context'].get('no_destroy', False) and 
                 fail['context'].get('value_piece', 100) <= 30))
            
            print(f"    {scen_name:12s}: {len(scenario)} failures "
                  f"({n_recoverable}/{len(scenario)} recoverable, "
                  f"mode={config['mode']})")
    
    print(f"\n{'=' * 80}")
    print(f"  TOTAL: {total_scenarios} scenarios, {total_failures} failure events")
    print(f"  Saved to: {FAILURES_DIR}/*/scenario_stress_*|cascade_*.json")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
