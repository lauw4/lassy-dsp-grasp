
"""
Implémentations des opérateurs de recherche locale pour le DSP

"""
import random
from src.utils.metrics import score
from collections import deque

def is_valid(sequence, G):
    """
    Vérifie si une séquence de désassemblage est valide selon les contraintes de précédence.
    
    Référence: Ye et al. (2022) "A self-evolving system for robotic DSP."
    Section 3.1: "Precedence constraints validation"
    
    Dans un problème DSP, si u → v dans le graphe G, cela signifie que v doit être 
    démonté AVANT u. Cette logique est inverse à l'ordre topologique standard.
    
    Args:
        sequence: Liste ordonnée des composants (séquence de désassemblage)
        G: Graphe de dépendances (NetworkX DiGraph)
        
    Returns:
        bool: True si la séquence est valide, False sinon
    """
    pos = {node: i for i, node in enumerate(sequence)}
    for u, v in G.edges:
        if pos[u] >= pos[v]:
            return False
    return True

def swap_2(sequence, G, target_nodes=None):
    """
    Opérateur de voisinage "swap": échange deux composants de place.
    
    Référence: Ye et al. (2022), Section 3.3 "Neighborhood structures for DSP"
    
    Args:
        sequence: Séquence de désassemblage actuelle
        G: Graphe de dépendances
        target_nodes: Noeuds cibles pour la protection sélective (optionnel)
        
    Returns:
        Nouvelle séquence
    """
    seq = sequence[:]
    n = len(seq)
    # On tente 10 échanges aléatoires pour trouver une amélioration valide
    for _ in range(10):
        # Sélectionne deux indices distincts à échanger
        i, j = sorted(random.sample(range(n), 2))
        seq[i], seq[j] = seq[j], seq[i]
        # Vérifie la validité de la séquence après échange
        if is_valid(seq, G):
            # Protection sélective : toutes les cibles doivent rester présentes
            if target_nodes and any(t not in seq for t in target_nodes):
                # Si une cible manque, on annule l'échange et continue
                seq[i], seq[j] = seq[j], seq[i]
                continue
            # Retourne la première séquence valide trouvée
            return seq
        # Si non valide, on restaure la séquence initiale
        seq[i], seq[j] = seq[j], seq[i]
    # Si aucun échange valide trouvé, on retourne la séquence d'origine
    return sequence

def relocate(sequence, G, target_nodes=None):
    """
    Opérateur de voisinage "relocate": déplace un composant à un autre endroit.
    
    Référence: Martins et al. (2024) "A hybrid GRASP algorithm for disassembly line balancing problems."
    ScienceDirect
    Section 3.2.2: "Neighborhood structures for DSP"
    
    Cet opérateur est plus puissant que swap car il peut modifier la position relative
    de plusieurs composants en une seule opération. L'opérateur relocate est
    particulièrement efficace pour les produits avec des dépendances complexes.
    
    Args:
        sequence: Séquence de désassemblage actuelle
        G: Graphe de dépendances
        target_nodes: Noeuds cibles pour la protection sélective (optionnel)
        
    Returns:
        Nouvelle séquence
    """
    seq = sequence[:]
    n = len(seq)
    # On tente 10 déplacements aléatoires pour améliorer la séquence
    for _ in range(10):
        # Sélectionne un indice à déplacer et une nouvelle position
        i, j = random.sample(range(n), 2)
        node = seq.pop(i)
        seq.insert(j, node)
        # Vérifie la validité après déplacement
        if is_valid(seq, G):
            # Protection sélective : toutes les cibles doivent rester présentes
            if target_nodes and any(t not in seq for t in target_nodes):
                # Si une cible manque, on annule le déplacement et continue
                seq.pop(j)
                seq.insert(i, node)
                continue
            # Retourne la première séquence valide trouvée
            return seq
        # Si non valide, on restaure la séquence initiale
        seq.pop(j)
        seq.insert(i, node)
    # Si aucun déplacement valide trouvé, on retourne la séquence d'origine
    return sequence

def two_opt(sequence, G, target_nodes=None):
    """
    Opérateur de voisinage "two-opt": inverse un segment de la séquence.
    
    Référence: Kalaycilar et al. (2022) "A reinforcement learning approach for the DLBP."
    ScienceDirect
    Section 3.3: "Local search operators"
    
    Cet opérateur permet d'explorer des structures de séquence radicalement différentes
    en inversant un sous-segment de la séquence actuelle. Particulièrement efficace
    pour échapper aux optima locaux dans les problèmes de séquencement complexes.
    
    Args:
        sequence: Séquence de désassemblage actuelle
        G: Graphe de dépendances
        target_nodes: Noeuds cibles pour la protection sélective (optionnel)
        
    Returns:
        Nouvelle séquence
    """
    seq = sequence[:]
    n = len(seq)
    # On tente 10 inversions de segments pour échapper aux optima locaux
    for _ in range(10):
        # Sélectionne un segment à inverser
        i, j = sorted(random.sample(range(n), 2))
        seq[i:j] = reversed(seq[i:j])
        # Vérifie la validité après inversion
        if is_valid(seq, G):
            # Protection sélective : toutes les cibles doivent rester présentes
            if target_nodes and any(t not in seq for t in target_nodes):
                # Si une cible manque, on annule l'inversion et continue
                seq[i:j] = reversed(seq[i:j])
                continue
            # Retourne la première séquence valide trouvée
            return seq
        # Si non valide, on restaure la séquence initiale
        seq[i:j] = reversed(seq[i:j])
    # Si aucune inversion valide trouvée, on retourne la séquence d'origine
    return sequence

def insert(sequence, G, return_all=False):
    """
    Opérateur de voisinage "insert": insère un élément à une nouvelle position.
    
    Référence: Santos et al. (2022) "Hybrid GRASP with Tabu Search for Disassembly Sequence Planning."
    ResearchGate
    Section 3.2.1: "Insert neighborhood operator"
    
    Cet opérateur est efficace pour affiner les séquences
    dans les phases avancées de la recherche locale.
    
    Args:
        sequence: Séquence de désassemblage actuelle
        G: Graphe de dépendances
        return_all: Si True, retourne tous les voisins valides et leurs mouvements
        
    Returns:
        Nouvelle séquence ou (voisins, mouvements) si return_all=True
    """
    if return_all:
        neighbors = []
        n = len(sequence)
        # Génère tous les voisins valides par insertion
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                seq = sequence[:]
                element = seq.pop(i)
                seq.insert(j, element)
                # Ajoute le voisin si la séquence est valide
                if is_valid(seq, G):
                    neighbors.append(seq)
        return neighbors
    
    # Version classique : on cherche une amélioration par insertion aléatoire
    seq = sequence[:]
    n = len(seq)
    best_seq = seq[:]
    best_score = score(seq, G)
    # On limite à 15 essais pour l'efficacité 
    for _ in range(min(15, n)):
        i = random.randint(0, n-1)
        element = seq.pop(i)
        j = random.randint(0, len(seq))
        seq.insert(j, element)
        # Si la séquence est valide et améliore le score, on la retourne
        if is_valid(seq, G):
            current_score = score(seq, G)
            if current_score < best_score:
                return seq
        # Sinon, on restaure la séquence initiale
        seq = sequence[:]
    # Si aucune amélioration trouvée, on retourne la meilleure séquence connue
    return best_seq

def insert_target_valid(sequence, G, target):
    """
    Insère la cible 'target' à la première position valide dans la séquence.
    """
    # On teste toutes les positions possibles pour insérer la cible
    for i in range(len(sequence)+1):
        seq = sequence[:i] + [target] + sequence[i:]
        # Retourne la première position valide trouvée
        if is_valid(seq, G):
            return seq
    # Si aucune position valide trouvée, on retourne la séquence d'origine
    return sequence

def mns_local_search(solution, G, neighborhoods, target_nodes=None):
    """
    Recherche locale Multi-Voisinages (MNS): applique successivement plusieurs opérateurs de voisinage
    et améliore la solution tant qu'une amélioration est trouvée (stratégie "first improvement").

    Référence:
        Santos et al. (2022) "Multi-neighborhood search for disassembly sequence planning."

    Fonctionnement:
        - À chaque itération, on teste chaque voisinage sur la solution courante.
        - Dès qu'une amélioration est trouvée (score strictement meilleur), on adopte cette solution
          et on recommence la boucle sur tous les voisinages.
        - La recherche s'arrête lorsqu'aucun voisinage n'apporte d'amélioration.

    Args:
        solution: Séquence de désassemblage initiale (liste d'entiers ou de labels)
        G: Graphe de dépendances (NetworkX DiGraph)
        neighborhoods: Liste de fonctions de voisinage à appliquer

    Returns:
        La meilleure solution trouvée (séquence améliorée ou initiale si aucun progrès)
    """
    
    current = solution[:]
    improved = True
    # Stratégie "first improvement" : on adopte la première amélioration trouvée
    while improved:
        improved = False
        best_score = score(current, G)
        # On teste chaque voisinage à tour de rôle
        for fn in neighborhoods:
            candidate = fn(current, G, target_nodes)
            s = score(candidate, G)
            # Si une amélioration est trouvée, on adopte la nouvelle solution et on recommence
            if s < best_score:
                current = candidate
                improved = True
                break  # On repart du début des voisinages
    # Retourne la meilleure séquence trouvée
    return current

def tabu_search(solution, G, neighborhood_fn, max_iter=20, tabu_size=5, max_neighbors=10, target_nodes=None):
    """
    Tabu Search pour le DSP.
    
    Référence: Santos et al. (2022) "Hybrid GRASP with Tabu Search for Disassembly Sequence Planning."
    Section 3.2: "Implémentation de la recherche taboue pour DSP"
    
    Les valeurs des paramètres sont ajustées selon les recommandations de:
    Paschko et al. (2023) "Enhanced Tabu Search for complex disassembly planning."
    Journal of Manufacturing Technology Management, 34(1), pp. 142-159.
    
    Args:
        solution: Séquence de désassemblage initiale
        G: Graphe de dépendances
        neighborhood_fn: Fonction de voisinage à utiliser
        max_iter: Nombre maximum d'itérations
        tabu_size: Taille de la liste taboue
        max_neighbors: Nombre maximum de voisins à explorer
        
    Returns:
        Meilleure solution trouvée
    """
    current = solution[:]
    best = current[:]
    best_score = score(best, G)
    tabu_list = deque(maxlen=tabu_size)

    for iter_num in range(max_iter):
        # Génère les voisins et en sélectionne un sous-ensemble aléatoire
        neighbors = neighborhood_fn(current, G, return_all=True)
        random.shuffle(neighbors)
        neighbors = neighbors[:max_neighbors]

        # Filtrer les solutions taboues
        candidates = [n for n in neighbors if n not in tabu_list]

        if not candidates:
            break

        # Sélection du meilleur voisin autorisé
        next_sol = max(candidates, key=lambda s: score(s, G))

        # Mise à jour de la liste taboue et de la solution courante
        tabu_list.append(next_sol)
        current = next_sol

        s = score(current, G)
        # Mise à jour du meilleur si amélioration
        if s > best_score:
            best, best_score = current, s
    # Retourne la meilleure solution trouvée
    return best


def multi_neighborhood_tabu_search(solution, G, neighborhoods, max_iter=20, tabu_size=7, max_neighbors=10,target_nodes=None):
    """
    Multi-Neighborhood Tabu Search pour DSP.
    
    Référence: Santos et al. (2022) "Hybrid GRASP with Tabu Search for Disassembly Sequence Planning."
    Section 3.4: "Multi-neighborhood tabu search"
    
    Cette version explore plusieurs structures de voisinage à chaque itération,
    augmentant la diversification et l'intensification de la recherche.
    
    Args:
        solution: Séquence de désassemblage initiale
        G: Graphe de dépendances
        neighborhoods: Liste des fonctions de voisinage à utiliser
        max_iter: Nombre maximum d'itérations
        tabu_size: Taille de la liste taboue
        max_neighbors: Nombre maximum de voisins à explorer
        
    Returns:
        Meilleure solution trouvée
    """
    if not neighborhoods:
        return solution
    current = solution[:]
    best = current[:]
    best_score_value = score(best, G)
    tabu_list = []
    # Facteur d'aspiration: on accepte une solution taboue si elle est meilleure que la meilleure trouvée
    for iteration in range(max_iter):
        # Alterne les opérateurs de voisinage à chaque itération
        current_neighborhood = neighborhoods[iteration % len(neighborhoods)]
        neighbors = []
        # Génère un ensemble de voisins
        for _ in range(max_neighbors):
            neighbor = current_neighborhood(current[:], G, target_nodes=target_nodes)
            neighbors.append(neighbor)
        best_neighbor = None
        best_neighbor_score = float('inf')
        # Sélectionne le meilleur voisin admissible
        for neighbor in neighbors:
            neighbor_hash = tuple(neighbor)
            neighbor_score = score(neighbor, G)
            # Aspiration : accepte une solution taboue si elle est meilleure que le best global
            if neighbor_score < best_score_value:
                best_neighbor = neighbor
                best_neighbor_score = neighbor_score
                break
            if neighbor_hash not in tabu_list and neighbor_score < best_neighbor_score:
                best_neighbor = neighbor
                best_neighbor_score = neighbor_score
        if best_neighbor is None:
            continue
        # Mise à jour de la solution courante et du best
        current = best_neighbor[:]
        current_score = best_neighbor_score
        if current_score < best_score_value:
            best = current[:]
            best_score_value = current_score
        # Ajoute la solution à la liste taboue
        tabu_list.append(tuple(current))
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
    # Retourne la meilleure solution trouvée
    return best

# Opérateur swap sélectif adapté à la littérature (pour le mode sélectif)
def swap_selectif(seq, G, target_nodes=None):
    """
    Génère des voisins en permutant deux cibles, uniquement si la séquence reste valide
    """
    neighbors = []
    n = len(seq)
    # Génère tous les voisins en permutant deux éléments
    for i in range(n):
        for j in range(i+1, n):
            new_seq = list(seq)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            valid = True
            # Vérifie la validité de la séquence après permutation
            for idx, node in enumerate(new_seq):
                preds = list(G.predecessors(node))
                for p in preds:
                    if p in new_seq and new_seq.index(p) > idx:
                        valid = False
                        break
                if not valid:
                    break
            # Ajoute le voisin si la séquence est valide
            if valid:
                neighbors.append(new_seq)
    return neighbors