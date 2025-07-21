import os
import pandas as pd
from datetime import datetime



def save_multiple_runs(scores_dict, graph_name="example_graph", folder="results", mode="complet"):
    """
    Sauvegarde plusieurs listes de scores (GRASP, VND, MNS) dans des fichiers séparés.

    Args:
        scores_dict (dict): {method_name: [scores...]}
        graph_name (str): Nom du graphe utilisé
        folder (str): Dossier racine
    """
    os.makedirs(os.path.join(folder, mode, graph_name), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for method, scores in scores_dict.items():
        df = pd.DataFrame({method: scores})
        filename = f"scores_{method}_{timestamp}.csv"
        full_path = os.path.join(folder, mode, graph_name, filename)
        df.to_csv(full_path, index=False)
        print(f" Sauvegardé :{full_path}")

def save_single_run(sequence, score_val, graph_name="example_graph", method="grasp", folder="results", mode="complet"):
    """
    Sauvegarde un seul run (méthode, score, séquence).

    Args:
        sequence (list): Séquence obtenue
        score_val (float): Score de la séquence
        graph_name (str): Nom du graphe
        method (str): Méthode utilisée
        folder (str): Dossier de sauvegarde
    """
    os.makedirs(os.path.join(folder, mode, graph_name), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame({
        "method": [method],
        "score": [score_val],
        "sequence": [sequence],
        "timestamp": [timestamp]
    })
    filename = f"single_run_{method}_{timestamp}.csv"
    full_path = os.path.join(folder, mode, graph_name, filename)
    df.to_csv(full_path, index=False)
    print(f" Run sauvegardé :{full_path}")
