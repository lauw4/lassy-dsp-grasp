import os
import pandas as pd
from datetime import datetime



def save_multiple_runs(scores_dict, graph_name="example_graph", folder="results", mode="complet"):
    """
    Save multiple score lists (GRASP, VND, MNS) to separate files.

    Args:
        scores_dict (dict): {method_name: [scores...]}
        graph_name (str): Name of the graph used
        folder (str): Root folder
    """
    os.makedirs(os.path.join(folder, mode, graph_name), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for method, scores in scores_dict.items():
        df = pd.DataFrame({method: scores})
        filename = f"scores_{method}_{timestamp}.csv"
        full_path = os.path.join(folder, mode, graph_name, filename)
        df.to_csv(full_path, index=False)
        print(f" Saved: {full_path}")

def save_single_run(sequence, score_val, graph_name="example_graph", method="grasp", folder="results", mode="complet"):
    """
    Save a single run (method, score, sequence).

    Args:
        sequence (list): Obtained sequence
        score_val (float): Sequence score
        graph_name (str): Graph name
        method (str): Method used
        folder (str): Save folder
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
    print(f" Run saved: {full_path}")
