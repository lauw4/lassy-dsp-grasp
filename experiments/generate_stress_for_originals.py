#!/usr/bin/env python3
"""
Generate stress & extreme scenarios for the 28 original benchmark instances.
Adds 2 new levels:
  - stress:   4-5 failures, ~65% recovery probability (harder than complex)
  - extreme:  6-8 failures, ~50% recovery probability (near breaking point)

These are generated for the SAME 28 instances that already have
simple/intermediate/complex scenarios.
"""
import os, sys, json, glob, random
import networkx as nx
import numpy as np

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
    critical = set(targets)
    for t in targets:
        try:
            critical |= nx.ancestors(G, t)
        except nx.NodeNotFound:
            pass
    return critical


def make_signals(symptom):
    template = SIGNAL_TEMPLATES[symptom]
    signals = {}
    for key, val in template.items():
        if isinstance(val, tuple):
            signals[key] = round(random.uniform(val[0], val[1]), 1)
        else:
            signals[key] = val
    return signals


def make_failure(comp_id, G, node_data, targets, recovery_prob=0.7):
    symptom = random.choice(SYMPTOMS)
    signals = make_signals(symptom)
    profit = node_data.get(comp_id, {}).get('profit', random.randint(5, 95))
    all_profits = [node_data.get(n, {}).get('profit', 0) for n in G.nodes]
    cutoff = float(np.percentile(all_profits, 20)) if all_profits else 20.0
    
    critical = ancestors_of_targets(G, targets) | set(targets)
    on_critical = comp_id in critical
    has_recovery = random.random() < recovery_prob
    
    if has_recovery:
        r = random.random()
        if r < 0.40:
            alt_paths, alt_tools, no_destroy = True, None, random.choice([True, False])
        elif r < 0.65:
            alt_paths = random.choice([True, False])
            alt_tools = random.choice(ALT_TOOLS)
            no_destroy = random.choice([True, False])
        elif r < 0.85:
            alt_paths = random.choice([True, False])
            alt_tools, no_destroy = None, False
            profit = min(profit, max(int(cutoff * 0.7), 3))
        else:
            alt_paths, alt_tools, no_destroy = True, random.choice(ALT_TOOLS), False
    else:
        alt_paths, alt_tools = False, None
        no_destroy = True if random.random() < 0.6 else False
        profit = max(profit, int(cutoff * 1.5))
    
    context = {"value_piece": profit, "no_destroy": no_destroy, "alt_paths": alt_paths}
    if alt_tools:
        context["alt_tools_available"] = alt_tools
    
    return {
        "comp_id": comp_id,
        "op_type": random.choice(OP_TYPES),
        "tool_id": random.choice(TOOL_IDS),
        "symptom": symptom,
        "signals": signals,
        "context": context,
        "history": {"failure_count": random.randint(0, 3)}
    }


def select_components(G, targets, n_failures, mode="mixed"):
    critical = ancestors_of_targets(G, targets) | set(targets)
    non_critical = set(G.nodes) - critical
    all_nodes = list(G.nodes)
    n_failures = min(n_failures, len(all_nodes))
    
    if mode == "mixed":
        n_crit = min(int(n_failures * 0.6) + 1, len(critical))
        n_non = n_failures - n_crit
        if n_non > len(non_critical):
            n_non = len(non_critical)
            n_crit = n_failures - n_non
        selected = set()
        crit_list = list(critical - set(targets))
        if len(crit_list) < n_crit:
            crit_list = list(critical)
        selected.update(random.sample(crit_list, min(n_crit, len(crit_list))))
        remaining = list(non_critical - selected)
        if remaining and n_non > 0:
            selected.update(random.sample(remaining, min(n_non, len(remaining))))
        while len(selected) < n_failures:
            pool = list(set(all_nodes) - selected)
            if not pool:
                break
            selected.add(random.choice(pool))
        return list(selected)[:n_failures]
    else:
        return random.sample(all_nodes, min(n_failures, len(all_nodes)))


def generate_scenario(G, targets, node_data, n_failures, recovery_prob, mode="mixed"):
    comps = select_components(G, targets, n_failures, mode=mode)
    return [make_failure(comp, G, node_data, targets, recovery_prob=recovery_prob) for comp in comps]


# ── Get original 28 instances ──
def get_original_instances():
    instances = []
    for d in sorted(glob.glob(os.path.join(FAILURES_DIR, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        if name in ('dagtest', 'gearpump'):
            continue
        has_simple = len(glob.glob(os.path.join(d, "scenario_simple_*.json"))) > 0
        if has_simple:
            instances.append(name)
    return instances


def main():
    instances = get_original_instances()
    print("=" * 70)
    print(f"  GENERATING STRESS + EXTREME SCENARIOS")
    print(f"  For {len(instances)} original benchmark instances")
    print("=" * 70)
    
    # Scenario configurations
    configs = {
        "stress":  {"recovery_prob": 0.65, "mode": "mixed"},
        "extreme": {"recovery_prob": 0.50, "mode": "mixed"},
    }
    
    total_scenarios = 0
    total_failures = 0
    
    for inst_name in instances:
        G, targets, node_data = load_graph(inst_name)
        if G is None:
            print(f"  SKIP {inst_name}: graph not found")
            continue
        
        n = len(G.nodes)
        out_dir = os.path.join(FAILURES_DIR, inst_name)
        
        # Determine failure counts based on instance size
        # stress: ~5-8% of n, min 4, max 6
        # extreme: ~8-12% of n, min 5, max 10
        n_stress = max(4, min(6, int(n * 0.06) + 1))
        n_extreme = max(5, min(10, int(n * 0.10) + 1))
        
        # Don't exceed 30% of nodes
        n_stress = min(n_stress, int(n * 0.30))
        n_extreme = min(n_extreme, int(n * 0.30))
        
        print(f"  {inst_name:45s} (n={n:4d}): ", end="")
        
        for scen_name, config in configs.items():
            nf = n_stress if scen_name == "stress" else n_extreme
            scenario = generate_scenario(G, targets, node_data, nf, config["recovery_prob"], config["mode"])
            
            out_path = os.path.join(out_dir, f"scenario_{scen_name}_{inst_name}.json")
            with open(out_path, 'w') as f:
                json.dump(scenario, f, indent=2)
            
            total_scenarios += 1
            total_failures += len(scenario)
            
            n_rec = sum(1 for fail in scenario if
                fail['context'].get('alt_paths') is True or
                bool(fail['context'].get('alt_tools_available')) or
                (not fail['context'].get('no_destroy', False) and 
                 fail['context'].get('value_piece', 100) <= 30))
            
            print(f"{scen_name}={len(scenario)}f({n_rec}rec) ", end="")
        
        print()
    
    print(f"\n{'=' * 70}")
    print(f"  Generated {total_scenarios} scenarios, {total_failures} failure events")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
