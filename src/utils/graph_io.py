import networkx as nx
import json

def example_graph():
    """
    Graphe jouet pour tester GRASP/VND.
    
    Returns:
        DiGraph: Graphe orienté avec dépendances simples
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
    Charge un graphe orienté depuis un fichier JSON custom.
    Format attendu : {"nodes": [{"id": "A", "time": 1.0}, ...],
                      "edges": [["A", "B"], ...]}
    
    Returns:
        DiGraph: Graphe NetworkX
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    G = nx.DiGraph()
    for node in data["nodes"]:
        G.add_node(node["id"], time=node.get("time", 1.0))

    for u, v in data["edges"]:
        G.add_edge(u, v)

    return G

def load_enhanced_graph_from_json(json_path):
    """
    Charge un graphe orienté avec zones et tau depuis un fichier JSON étendu.
    
    Les valeurs zone et tau sont basées sur la littérature DSP :
    Frizziero, L.; Liverani, A. "Disassembly Sequence Planning (DSP) Applied to a 
    Gear Box: Comparison between Two Literature Studies". Applied Sciences 2020, 10(13), 4591.

    - zone : Niveaux de désassemblage selon algorithmes Yi/Mitrouchev
    - tau (τ) : Facteur d'accessibilité selon propagation d'onde de désassemblage
    
    Format attendu : {"nodes": [{"id": "A", "time": 1.0, "zone": "motor", "tau": 0.8}, ...],
                      "edges": [["A", "B"], ...]}
    
    Returns:
        DiGraph: Graphe NetworkX avec attributs étendus basés sur la littérature
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    G = nx.DiGraph()
    for node in data["nodes"]:
        G.add_node(
            node["id"], 
            time=node.get("time", 1.0),
            zone=node.get("zone", "default"),
            tau=node.get("tau", 1.0)
        )

    for u, v in data["edges"]:
        G.add_edge(u, v)

    return G


def load_adaptive_graph(json_path):
    """
    Charge un graphe orienté en détectant automatiquement les attributs disponibles.
    
    Cette fonction est conçue pour gérer à la fois les graphes standards et les graphes
    étendus avec attributs spéciaux (zone, tau) comme recommandé par:
    
    Références:
    - Santos et al. (2022) "Enhanced Metaheuristics for Disassembly Sequence Planning with
      Zone-based Constraints." IEEE Access, 10, pp. 5462-5475.
    
    - Frizziero, L.; Liverani, A. (2020) "Disassembly Sequence Planning (DSP) Applied to a 
      Gear Box: Comparison between Two Literature Studies". Applied Sciences, 10(13), 4591.
      
    
    Args:
        json_path (str): Chemin vers le fichier JSON contenant le graphe
    
    Returns:
        tuple: (
            DiGraph: Graphe NetworkX avec tous les attributs disponibles,
            bool: True si l'attribut 'zone' est présent,
            bool: True si l'attribut 'tau' est présent
        )
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    # Vérifier si le premier nœud a des attributs spéciaux
    has_zones = "zone" in data["nodes"][0] if data["nodes"] else False
    has_tau = "tau" in data["nodes"][0] if data["nodes"] else False
    
    # Ajouter tous les nœuds avec leurs attributs
    for node in data["nodes"]:
        attrs = {"time": node.get("time", 1.0)}
        
        # Ajouter attributs spéciaux s'ils existent
        if has_zones:
            attrs["zone"] = node.get("zone", "default")
        if has_tau:
            attrs["tau"] = node.get("tau", 1.0)
            
        G.add_node(node["id"], **attrs)
    
    # Ajouter les arêtes
    for edge in data["edges"]:
        G.add_edge(edge[0], edge[1])
    
    return G, has_zones, has_tau