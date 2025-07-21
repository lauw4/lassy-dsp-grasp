# Opérateur swap sélectif adapté à la littérature
def swap_selectif(seq, G, target_nodes=None):
    # Génère des voisins en permutant deux cibles, uniquement si la séquence reste valide
    neighbors = []
    n = len(seq)
    for i in range(n):
        for j in range(i+1, n):
            new_seq = list(seq)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            # Vérifie la validité locale : tous les prédécesseurs d'un noeud sont avant lui
            valid = True
            for idx, node in enumerate(new_seq):
                preds = list(G.predecessors(node))
                for p in preds:
                    if p in new_seq and new_seq.index(p) > idx:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                neighbors.append(new_seq)
    return neighbors
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import networkx as nx
import json
from src.grasp.constructive import grasp_with_vnd, greedy_randomized_construction, enhanced_mns_solver
# On retire les imports d'opérateurs non utilisés en mode sélectif
from src.utils.metrics import score

def load_graph_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    G = nx.DiGraph()
    # Ajout des noeuds avec attributs
    for node in data["nodes"]:
        G.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
    # Ajout des arcs
    for edge in data["edges"]:
        G.add_edge(edge[0], edge[1])
    return G


base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
instances = [
    ("structured_graph_100", os.path.join(base_data_dir, "structured_graph_100.json")),
    ("gearbox_enhanced_118", os.path.join(base_data_dir, "gearbox_enhanced_118_fixed.json"))
]


# Cibles conformes au benchmark
benchmark_targets = {
    "structured_graph_100": ["C050"],
    "gearbox_enhanced_118": ["GB025", "GB073", "GB081"]
}


# Wrapper local pour le mode selectif
def grasp_with_vnd_selectif(G, target_nodes=None, run_idx=0):
    # Mode sélectif : uniquement l'opérateur conforme à la littérature
    neighborhoods = [swap_selectif]
    return grasp_with_vnd(G, neighborhoods=neighborhoods, target_nodes=target_nodes)
algos = [
    # Tous les algos sont forcés en mode sélectif (séquence de cibles uniquement)
    ("GRASP simple (selectif)", lambda G, t, run_idx=0: greedy_randomized_construction(G, alpha=0.5, target_nodes=t)),
    ("GRASP+VND selectif", lambda G, t, run_idx=0: grasp_with_vnd_selectif(G, target_nodes=t, run_idx=run_idx)),
    ("MNS (selectif)", lambda G, t, run_idx=0: enhanced_mns_solver(G, target_nodes=t)),
    # Si tu ajoutes Tabu ou autre, il faut l'adapter pour ne traiter que les cibles
]


for name, path in instances:
    print(f"\n===== INSTANCE: {name} =====")
    G = load_graph_from_json(path)
    # Utilise les cibles du benchmark si disponibles, sinon les 5 premiers noeuds
    target_nodes = benchmark_targets.get(name, list(G.nodes)[:5])
    print(f"Cibles utilisées : {target_nodes}")
    for algo_name, algo_func in algos:
        print(f"\n--- {algo_name} ---")
        all_seqs = []
        corrections = 0
        for i in range(5):
            seq = algo_func(G, target_nodes, i)
            all_seqs.append(tuple(seq))
            s = score(seq, G)
            if s == float('inf'):
                corrections += 1
        unique_seqs = set(all_seqs)
        print(f"Diversité : {len(unique_seqs)} séquences uniques sur 5 runs")
        print(f"Corrections nécessaires : {corrections} / 5 runs")
        best_seq = min(all_seqs, key=lambda seq: score(seq, G))
        print(f"Chemin final choisi ({algo_name}) : {list(best_seq)} | score={score(best_seq, G)}")
