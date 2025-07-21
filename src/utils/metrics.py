"""
Fonctions d'évaluation pour les séquences de désassemblage

"""
def score(sequence, G):
    from src.grasp.local_search import is_valid
    """
    Calcul du score d'une séquence de désassemblage.
    Plus le score est PETIT, meilleure est la séquence.
    
    Référence: Frizziero et al. (2020) "DSP applied to a Gear Box."
    
    Section 4.3: "Évaluation des séquences de désassemblage"
    
    La fonction calcule la somme pondérée temps × position:
    score = Σ(temps_i × position_i)
    
    Cette métrique pénalise les séquences qui démontent tard les composants longs,
    comme recommandé par Ye et al. (2022) pour optimiser l'efficacité du désassemblage.
    
    Args:
        sequence: Liste ordonnée des composants (séquence de désassemblage)
        G: Graphe NetworkX avec attributs "time" sur les nœuds
        
    Returns:
        Score total (plus petit = meilleur)
    """
    if not is_valid(sequence, G):
        return float('inf')  # Pénalise fortement les séquences invalides
    total = 0.0
    for i, node in enumerate(sequence):
        t = G.nodes[node].get("time", 1.0)
        total += t * (i + 1)  # i + 1 = rang de démontage (1-based)
    return total  # Score brut


def adaptive_score(sequence, G):
    """
    Calcule le score en tenant compte des attributs spéciaux si disponibles
    """
    total = 0.0
    for i, node in enumerate(sequence):
        t = G.nodes[node].get("time", 1.0)
        
        # Utiliser tau s'il existe, sinon 1.0
        tau = G.nodes[node].get("tau", 1.0)
        
        # Position dans la séquence (i+1) multipliée par temps et facteur tau
        total += t * tau * (i + 1)
    return total

def selective_score(sequence, G, target_nodes):
    """
    Score de la séquence jusqu'à la dernière cible démontée.
    
    Args:
        sequence: Liste ordonnée des composants (séquence de désassemblage)
        G: Graphe NetworkX avec attributs "time" sur les nœuds
        target_nodes: Liste des nœuds cibles à démonter
    
    Returns:
        Score total jusqu'à la dernière cible
    """
    last_target_idx = max(sequence.index(t) for t in target_nodes)
    partial_seq = sequence[:last_target_idx + 1]
    return score(partial_seq, G)