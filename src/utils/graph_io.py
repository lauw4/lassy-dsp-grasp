import networkx as nx
import json
import os
import sys
from src.grasp.config import (
    OVERHEAD_RATE, DESTRUCTION_THRESHOLD, RISK_FACTOR,
    VALUE_EXPOSURE_RATE, BUDGET_RATIO,
)


def auto_lambda(G, alpha: float = 0.05) -> float:
    """Compute per-instance λ = alpha × Σmax(v_i, 0) / Σp_i.

    Ensures the time-discount term scales with the instance's own value density.
    For instances with large processing times (lutz, arc, SALBP), λ grows
    proportionally so λ·C_i never dominates trivially.

    alpha = 0.05  →  5 % of the average net-value rate (EUR/s or EUR/unit).
    """
    pos_vals = [
        max(G.nodes[n].get('profit', 0.0) - G.nodes[n].get('cost', 0.0), 0.0)
        for n in G.nodes()
    ]
    total_p = sum(
        G.nodes[n].get('time', G.nodes[n].get('duration', 1.0))
        for n in G.nodes()
    )
    total_v = sum(pos_vals)
    if total_p > 0 and total_v > 0:
        return alpha * total_v / total_p
    return 0.05  # safe fallback


def load_graph_from_txt(txt_path: str, alpha_lambda: float = 0.05):
    """Load a MILP TXT instance (*_selectif.txt) into a NetworkX DiGraph for GRASP.

    Uses **exactly the same** profits, costs, times and precedences as the MILP
    model so that GRASP vs MILP comparisons are fair.

    Node IDs are converted to strings 'C{i:03d}' (consistent with the JSON
    convention used by load_adaptive_graph).

    λ is auto-calibrated:  λ = alpha_lambda × Σmax(v_i,0) / Σp_i
    so the time penalty is proportional to the instance's own value scale.

    Args:
        txt_path:      absolute (or relative to cwd) path to *_selectif.txt file
        alpha_lambda:  fraction parameter for auto-λ (default 0.05 = 5 %)

    Returns:
        DiGraph with node attrs: profit, cost, time
        G.graph keys: targets, lambda_discount, source_file, overhead_rate, budget_max
    """
    # Import MILP parser (safe even when called from project root)
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from src.exact_milp_dijkstra.partial_disassembly_model import load_data

    data = load_data(txt_path)
    G = nx.DiGraph()

    # Build string-ID map: integer → 'C{i:03d}'
    id_map = {v: f"C{v:03d}" for v in data['V']}

    for v in data['V']:
        sid = id_map[v]
        G.add_node(sid,
                   profit=float(data['r'].get(v, 0.0)),
                   cost=float(data['c'].get(v, 0.0)),
                   time=float(data['p'].get(v, 1.0)))

    for (pred, succ) in data['E']:
        if pred in id_map and succ in id_map:
            G.add_edge(id_map[pred], id_map[succ])

    # Store targets (string IDs)
    G.graph['targets'] = [id_map[t] for t in data['T'] if t in id_map]
    G.graph['source_file'] = txt_path

    # Standard managerial params (budget_max, overhead_rate, …)
    _set_managerial_params(G)

    # TXT instances do NOT specify a budget constraint — remove the artificial
    # B_max imposed by _set_managerial_params so the greedy construction is free
    # to include any node (matching MILP's unconstrained scope).
    G.graph['budget_max'] = None

    # Auto-calibrate λ per instance
    lam = auto_lambda(G, alpha=alpha_lambda)
    G.graph['lambda_discount'] = lam

    return G


def _set_managerial_params(G, data=None):
    """Inject managerial parameters into G.graph from JSON or defaults.

    Parameters stored:
        overhead_rate (h), destruction_threshold (θ), risk_factor (α),
        value_exposure_rate (ρ), budget_max (B_max).
    """
    h = (data or {}).get('overhead_rate', OVERHEAD_RATE)
    G.graph['overhead_rate'] = h
    G.graph['destruction_threshold'] = (data or {}).get(
        'destruction_threshold', DESTRUCTION_THRESHOLD)
    G.graph['risk_factor'] = (data or {}).get(
        'risk_factor', RISK_FACTOR)
    G.graph['value_exposure_rate'] = (data or {}).get(
        'value_exposure_rate', VALUE_EXPOSURE_RATE)

    # B_max: explicit in JSON  >  ratio × Σ(c_i + h·p_i)
    if data and 'budget_max' in data:
        G.graph['budget_max'] = data['budget_max']
    else:
        ratio = (data or {}).get('budget_ratio', BUDGET_RATIO)
        total_budget = sum(
            G.nodes[n].get('cost', 0.0) + h * G.nodes[n].get(
                'time', G.nodes[n].get('duration', 1.0))
            for n in G.nodes()
        )
        G.graph['budget_max'] = ratio * total_budget


def example_graph():
    """
    Toy graph for testing GRASP/VND.
    
    Returns:
        DiGraph: Simple directed graph with dependencies
    """
    G = nx.DiGraph()
    G.add_nodes_from([
        ("A", {"time": 1.0}),
        ("B", {"time": 2.0}),
        ("C", {"time": 1.5}),
        ("D", {"time": 3.0}),
        ("E", {"time": 1.2}),
    ])
    G.add_edges_from([
        ("A", "C"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),
    ])
    return G

def load_graph_from_json(json_path):
    """
    Load directed graph from custom JSON file.
    
    Format: {"nodes": [{"id": "A", "time": 1.0}, ...], "edges": [["A", "B"], ...]}
    
    Returns:
        DiGraph: NetworkX graph
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    G = nx.DiGraph()
    for node in data["nodes"]:
        t = node.get("time", node.get("duration", 1.0))
        G.add_node(node["id"], time=t)

    for u, v in data["edges"]:
        G.add_edge(u, v)

    _set_managerial_params(G, data)
    return G

def load_enhanced_graph_from_json(json_path):
    """
    Load directed graph with zone and tau attributes from extended JSON.
    
    Based on DSP literature:
    Frizziero, L.; Liverani, A. "Disassembly Sequence Planning (DSP) Applied to a 
    Gear Box". Applied Sciences 2020, 10(13), 4591.

    - zone: Disassembly levels (Yi/Mitrouchev algorithms)
    - tau: Accessibility factor (disassembly wave propagation)
    
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    G = nx.DiGraph()
    for node in data["nodes"]:
        t = node.get("time", node.get("duration", 1.0))
        G.add_node(
            node["id"], 
            time=t,
            zone=node.get("zone", "default"),
            tau=node.get("tau", 1.0)
        )

    for u, v in data["edges"]:
        G.add_edge(u, v)

    _set_managerial_params(G, data)
    return G


def load_adaptive_graph(json_path):
    """
    Load graph with automatic attribute detection (zone, tau).
    
    Returns:
        tuple: (DiGraph, has_zones: bool, has_tau: bool)
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    has_zones = "zone" in data["nodes"][0] if data["nodes"] else False
    has_tau = "tau" in data["nodes"][0] if data["nodes"] else False
    
    for node in data["nodes"]:
        t = node.get("time", node.get("duration", 1.0))
        attrs = {
            "time": t,
            "profit": node.get("profit", 0.0),
            "cost": node.get("cost", 0.0),
            "trap": node.get("trap", False)
        }
        if has_zones:
            attrs["zone"] = node.get("zone", "default")
        if has_tau:
            attrs["tau"] = node.get("tau", 1.0)
        G.add_node(node["id"], **attrs)
    
    for edge in data["edges"]:
        G.add_edge(edge[0], edge[1])

    _set_managerial_params(G, data)
    return G, has_zones, has_tau