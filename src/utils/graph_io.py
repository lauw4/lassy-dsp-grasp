import networkx as nx
import json

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
    
    return G, has_zones, has_tau