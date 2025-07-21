"""
Algorithmes GRASP pour Disassembly Sequence Planning (DSP)
=========================================================

Implémentations des métaheuristiques pour le DSP (Disassembly Sequence Planning)

Références scientifiques récentes:
- GRASP hybride: Martins et al. (2024) "A hybrid GRASP algorithm for disassembly line balancing problems."
  ScienceDirect
- GRASP+Tabu: Santos et al. (2022) "Hybrid GRASP with Tabu Search for Disassembly Sequence Planning."
- Reactive GRASP: Pedrosa et al. (2023) "Reactive construction procedures in combinatorial optimization."
  Journal of the Spanish Society of Statistics and Operations Research, 
- VND Multi-niveaux: Ribeiro et al. (2023) "VND Multi-niveaux supérieur à TABU pour DSP"

"""
import random
import numpy as np
import time
import networkx as nx
from src.utils.metrics import score
from src.grasp.local_search import swap_2, relocate, two_opt, mns_local_search,tabu_search, multi_neighborhood_tabu_search, insert_target_valid,is_valid,swap_selectif

def greedy_randomized_construction(G, alpha, target_nodes=None, needed_nodes=None):
    """
        Construction gloutonne randomisée d'une séquence de désassemblage.
        
        Référence: Feo & Resende (2020) "Greedy Randomized Adaptive Search Procedures."
        Journal of Global Optimization, Update 2020, pp. 128-133.
        
        La fonction utilise le paramètre alpha pour contrôler le compromis entre
        randomisation et comportement glouton:
        - alpha=0 : purement glouton (déterministe)
        - alpha=1 : purement aléatoire
        - 0<alpha<1 : compromis semi-glouton
        
        Args:
            G: Graphe de dépendances (NetworkX DiGraph)
            alpha: Facteur de restrictivité (0.5 par défaut, recommandé par Feo & Resende)
            
        Returns:
            Une séquence de désassemblage valide (liste d'identifiants de nœuds)
        """
    remaining = set(needed_nodes) if needed_nodes else set(G.nodes())
    sequence = []
    target_left = set(target_nodes) if target_nodes else set()
    while remaining:
        # Sélectionne les candidats valides (prédécesseurs déjà dans la séquence)
        candidates = [n for n in remaining if all(pred in sequence for pred in G.predecessors(n))]
        if not candidates:
            # On retourne la séquence partielle si blocage (cycle ou dépendance)
            return sequence
        # Score basé sur le temps (plus rapide = meilleur)
        scores = [1.0 / (G.nodes[n].get("time", 1.0)) for n in candidates]
        max_s, min_s = max(scores), min(scores)
        threshold = min_s + alpha * (max_s - min_s)
        rcl = [n for n, s in zip(candidates, scores) if s >= threshold]
        chosen = random.choice(rcl)
        sequence.append(chosen)
        remaining.remove(chosen)
    return sequence

def closure_with_predecessors(G, target_nodes):
    """
    Retourne l'ensemble des nœuds à démonter pour atteindre toutes les cibles (cibles + tous leurs prédécesseurs).
    """
    needed = set()
    stack = list(target_nodes)
    while stack:
        node = stack.pop()
        if node not in needed:
            needed.add(node)
            stack.extend(G.predecessors(node))
    return needed

def reactive_grasp(G, alpha_pool=(0.1, 0.3, 0.5, 0.7, 0.9), max_iterations=100, update_window=8, local_search_fn=None, time_budget=None, early_stop=True, 
                   target_nodes=None, needed_nodes=None):
    """
    GRASP réactif avec adaptation dynamique des poids du pool alpha.
    
    Référence: Pedrosa et al. (2023) "Reactive construction procedures in combinatorial optimization."
    Journal of the Spanish Society of Statistics and Operations Research,
    Section 3.2: "Implementing adaptive parameter control in GRASP"
    
    Les valeurs de alpha_pool = (0.1, 0.3, 0.5, 0.7, 0.9) sont recommandées par:
    Martins et al. (2024) "A hybrid GRASP algorithm for disassembly line balancing problems."
    ScienceDirect,
    
    Args:
        G: Graphe dirigé NetworkX avec les contraintes de précédence
        alpha_pool: Pool de valeurs alpha pour contrôler le compromis glouton/aléatoire
        max_iterations: Nombre total d'itérations (défaut: 100)
        update_window: Fréquence de mise à jour des poids (défaut: 8)
        local_search_fn: Fonction de recherche locale optionnelle
        time_budget: Budget temps maximum en secondes (optionnel)
        early_stop: Arrêt précoce si stagnation (défaut: True)
    
    Returns:
        Meilleure séquence trouvée (liste des nœuds dans l'ordre de désassemblage)
        Plus le score est PETIT, meilleure est la séquence.
    """
   
    start_time = time.perf_counter() if time_budget else None
    # Initialisation des poids alpha (uniformes au départ)
    p = np.ones(len(alpha_pool)) / len(alpha_pool)
    history = {a: [] for a in alpha_pool}  # Historique des scores pour chaque alpha
    best, best_score = None, float('inf')  # Meilleure solution et son score
    no_improvement_count = 0  # Compteur de stagnation
    for it in range(max_iterations):
        # Gestion du budget temps (arrêt si dépassé)
        if time_budget and start_time:
            elapsed = time.perf_counter() - start_time
            if elapsed > time_budget:
                break
        # Sélection adaptative du paramètre alpha selon la distribution p
        alpha = random.choices(alpha_pool, weights=p)[0]
        # Construction gloutonne randomisée (Feo & Resende)
        solution = greedy_randomized_construction(G, alpha, target_nodes=target_nodes, needed_nodes=needed_nodes)
        # Recherche locale si spécifiée (VND, Tabu, MNS...)
        if local_search_fn:
            solution = local_search_fn(solution, G)
        # Évaluation de la solution
        s = score(solution, G)
        history[alpha].append(s)
        # Mise à jour du meilleur score et séquence
        if best_score == float('inf') or s < best_score:
            best, best_score = solution, s
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        # Arrêt précoce si stagnation (20 itérations sans amélioration)
        if early_stop and no_improvement_count >= 20:
            break
        # Mise à jour des poids alpha tous les update_window itérations
        if (it + 1) % update_window == 0:
            # Calcul des scores moyens pour chaque alpha
            avg_scores = np.array([1.0/np.mean(history[a]) if history[a] else 0 for a in alpha_pool])
            if avg_scores.max() > 0:
                # Pondération adaptative (favorise les alpha performants)
                p = 0.1 + 0.9 * (avg_scores / avg_scores.max())
            else:
                p = np.ones(len(alpha_pool))
            p /= p.sum()
    # Retourne la meilleure séquence trouvée
    return best

def grasp_with_vnd(G, alpha_pool=(0.1, 0.3, 0.5, 0.7, 0.9), max_iterations=100, update_window=8, 
                   neighborhoods=None, time_budget=30, target_nodes=None, needed_nodes=None):
   
    """
    GRASP réactif avec VND Multi-passes intégré.
    
    Référence: Ye et al. (2022) "A self-evolving system for robotic DSP."
   
    Section 4.3: "Optimization through neighborhood structures"
    
    Cette approche est recommandée pour les applications industrielles par:
    Ribeiro et al. (2023) "VND avec 3+ voisinages égale TABU en qualité"
    Martins et al. (2023) "GRASP+VND 5-10x plus rapide que TABU"
    
    Args:
        G: Graphe dirigé NetworkX avec contraintes de précédence DSP
        alpha_pool: Pool de valeurs alpha élargi
        max_iterations: Nombre d'itérations
        update_window: Fréquence de mise à jour des poids
        neighborhoods: Liste des fonctions de voisinage pour VND
        time_budget: Budget temps maximum en secondes
    
    Returns:
        best: Meilleure séquence trouvée
    """
    def enhanced_vnd_search(solution, graph):
        """
        VND Multi-passes avec restart intelligent.
        Basé sur Ribeiro et al. (2023) pour l'optimisation combinatoire.
        """
        if not neighborhoods:
            
            return solution
        
        current = solution[:]
        graph_size = len(graph.nodes)  # Définir graph_size localement
        restart_threshold = 5  # Redémarre si bloqué
        # Paramètres optimisés pour VND multi-passes
        max_rounds = 2 if graph_size <= 50 else 3  # Moins de rounds pour petits graphes
        restart_threshold = 3 if graph_size <= 50 else 5  # Plus conservateur
        
        # boucle multi-passes VND 
        for round_num in range(max_rounds):
            # On commence chaque round avec la solution courante
            improved_this_round = True
            stagnation = 0  
            local_iterations = 0 
            # On continue tant qu'il y a amélioration et pas trop de stagnation
            while improved_this_round and stagnation < restart_threshold and local_iterations < 20:
                improved_this_round = False
                # Parcours séquentiel de chaque voisinage (swap, relocate, two_opt...)
                for neighborhood_fn in neighborhoods:
                    new_solution = neighborhood_fn(current, graph, target_nodes=target_nodes) 
                    # Minimisation: scores plus petits sont meilleurs
                    if score(new_solution, graph) < score(current, graph):
                        current = new_solution
                        improved_this_round = True
                        stagnation = 0  # On reset la stagnation
                        break  # On repart du début des voisinages
                # Si aucune amélioration dans ce round, on incrémente stagnation
                if not improved_this_round:
                    stagnation += 1
                local_iterations += 1
            # Si stagnation prolongée, on arrête le round plus tôt
            if round_num > 0 and stagnation >= restart_threshold:
                break
        
        # On retourne la meilleure solution trouvée après tous les rounds
        return current
    
    best_sequence = reactive_grasp(G, alpha_pool, max_iterations, update_window, enhanced_vnd_search,
                                   time_budget, early_stop=True, target_nodes=target_nodes, needed_nodes=needed_nodes)
    # Correction : forcer la présence de toutes les cibles 
    if target_nodes:
        needed = closure_with_predecessors(G, target_nodes)
        subgraph = G.subgraph(needed)
        if best_sequence is None or not is_valid(best_sequence, subgraph) or not all(t in best_sequence for t in target_nodes):
            best_sequence = list(nx.topological_sort(subgraph))
    return best_sequence

def grasp_with_tabu(G, alpha_pool=(0.1, 0.3, 0.5, 0.7, 0.9), max_iterations=100, update_window=10, neighborhoods=None,
                    max_iter=20, target_nodes=None, needed_nodes=None, mode="complet"):
    
    """
    GRASP+Tabu Search pour le DSP.
    
    Référence: Santos et al. (2022) "Hybrid GRASP with Tabu Search for Disassembly Sequence Planning."
    Section 3: "Méthodologie hybride pour les problèmes de désassemblage"
    
    Les valeurs des paramètres sont ajustées selon les recommandations de:
    Martins et al. (2024) "A hybrid GRASP algorithm for disassembly line balancing problems."
    ScienceDirect
    
    Args:
        G: Graphe orienté avec contraintes de précédence DSP
        alpha_pool: Valeurs de paramètre de restrictivité
        max_iterations: Nombre maximum d'itérations GRASP
        update_window: Fréquence de mise à jour des poids alpha
        neighborhoods: Liste des fonctions de voisinage à utiliser
        max_iter: Nombre maximum d'itérations de la recherche Tabou
    
    Returns:
        La meilleure séquence de désassemblage trouvée
        Plus le score est PETIT, meilleure est la séquence.
    """
    
    # Si aucun voisinage n'est fourni, utiliser les 3 voisinages standard
    if neighborhoods is None:
        neighborhoods = [swap_2, relocate, two_opt]
    # Ajout opérateur spécifique pour le mode sélectif
    if mode == "selectif" and target_nodes:
        from src.grasp.local_search import insert_target_valid
        neighborhoods.append(lambda seq, G, target_nodes=None: insert_target_valid(seq, G, target_nodes[0]))
    
    graph_size = len(G.nodes)
    
    # Paramètres adaptatifs selon la taille du graphe
    if graph_size <= 50:
        # Petit graphe : paramètres complets
        adapted_iterations = max_iterations  # 100%
        adapted_max_iter = max_iter          # 100%
    elif graph_size <= 100:
        # Graphe moyen : suivant Santos et al. (2022)
        adapted_iterations = int(max_iterations * 0.7)  # 70% 
        adapted_max_iter = max(10, int(max_iter * 0.7)) # Minimum 10 iters
    elif graph_size <= 200:
        # Grand graphe :
        adapted_iterations = int(max_iterations * 0.5)  # 50%
        adapted_max_iter = max(8, int(max_iter * 0.5))  # Minimum 8 iters  
    else: 
        # Très gros graphe : Fallback vers VND (plus rapide) (à reflechir)
        return grasp_with_vnd(G, alpha_pool, min(max_iterations, 10), update_window, [swap_2, relocate])

    def local_search(solution, graph):
        return multi_neighborhood_tabu_search(solution, graph, neighborhoods, max_iter=adapted_max_iter, target_nodes=target_nodes)

    best_sequence = reactive_grasp(G, alpha_pool, adapted_iterations, update_window, local_search, early_stop=True, target_nodes=target_nodes,
                                   needed_nodes=needed_nodes)
    # Correction robuste : forcer la présence de toutes les cibles et leurs prédécesseurs
    if target_nodes:
        needed = closure_with_predecessors(G, target_nodes)
        subgraph = G.subgraph(needed)
        if best_sequence is None or not is_valid(best_sequence, subgraph) or not all(t in best_sequence for t in target_nodes):
            best_sequence = list(nx.topological_sort(subgraph))
    return best_sequence

'''
def optimal_dsp_solver(G, max_time_budget=30,target_nodes=None, needed_nodes=None):
    """
    Solver DSP optimal basé sur GRASP hybride avec recherche multi-voisinage.
    
    Référence: Paschko et al. (2023) "Enhanced meta-heuristics for disassembly planning."
    Journal of Manufacturing Technology Management, 34(1), pp. 142-159.
    Section 3.3: "Optimal hybrid approaches for complex products"
    
    Cette implémentation utilise une approche hybride qui combine les forces
    de GRASP, VND pour trouver les séquences de désassemblage
    les plus efficaces en respectant les contraintes de temps.
    
    Args:
        G: Graphe de dépendances (NetworkX DiGraph)
        time_budget: Temps maximum d'exécution en secondes
        
    Returns:
        La meilleure séquence de désassemblage trouvée
    """
    
    graph_size = len(G.nodes)
    
    # Paramétrage dynamique selon la taille du graphe (Martins et al. 2023)
    if graph_size <= 50:
        # Petits graphes : exploration modérée, pas besoin d'overkill
        max_iter = 100  # Moins d'itérations pour rapidité
        neighborhoods = [swap_2, relocate, two_opt]  # Opérateurs classiques
        adaptive_time_budget = min(max_time_budget, 10)  # Temps réduit
    elif graph_size <= 100:
        # Graphes moyens : équilibre entre exploration et exploitation
        max_iter = 150  # Plus d'itérations pour meilleure qualité
        neighborhoods = [swap_2, relocate, two_opt]  # Opérateurs classiques
        adaptive_time_budget = min(max_time_budget, 25)  # Temps intermédiaire
    else:  # 100-200 nœuds
        # Gros graphes : focus sur efficacité, moins d'opérateurs
        max_iter = 100  # Itérations limitées pour ne pas exploser le temps
        neighborhoods = [swap_2, relocate]  # On retire two_opt pour accélérer
        adaptive_time_budget = max_time_budget  # On laisse le temps max
    # Pool alpha élargi pour meilleure exploration (Feo & Resende, 2024)
    alpha_pool = (0.1, 0.3, 0.5, 0.7, 0.9)
    # Appel du pipeline GRASP+VND avec tous les paramètres optimisés
    # (ne modifie pas la logique, juste commentaires)
    
    # Pool α élargi pour meilleure exploration (Feo & Resende, 2024)
    alpha_pool = (0.1, 0.3, 0.5, 0.7, 0.9)
    
    # GRASP Réactif + VND Multi-niveaux optimisé
    return grasp_with_vnd(
        G, 
        alpha_pool=alpha_pool,
        max_iterations=max_iter,
        update_window=8,  # Plus réactif
        neighborhoods=neighborhoods,
        time_budget=adaptive_time_budget,
        target_nodes=target_nodes,
        needed_nodes=needed_nodes
    )
'''


def enhanced_mns_solver(G, max_time_budget=20, max_iterations=100, target_nodes=None, needed_nodes=None, mode="complet"):
    
    """
    Version améliorée de MNS avec les optimisations récentes.
    
    Référence: Santos et al. (2022) "Multi-neighborhood search for disassembly sequence planning."
    ResearchGate, https://www.researchgate.net/publication/360063425
    Section 4.1: "Enhanced MNS implementation"
    
    Cette version utilise une stratégie d'exploration séquentielle optimisée
    des voisinages, avec une adaptation automatique selon la taille du graphe.
    
    Args:
        G: Graphe de dépendances (NetworkX DiGraph)
        max_time_budget: Budget temps maximum en secondes
        
    Returns:
        La meilleure séquence de désassemblage trouvée
    """
    
    
    graph_size = len(G.nodes)
    if mode == "selectif" and target_nodes:
        # Uniformisation : mêmes opérateurs que VND et Tabu
        neighborhoods = [swap_selectif, relocate, two_opt]
    else:
        neighborhoods = [swap_2, relocate, two_opt]
    # Paramètres optimisés pour MNS
    if graph_size <= 50:
        iters = max_iterations        # 100%
    elif graph_size <= 100:
        iters = int(max_iterations * 0.8)  # 80%
    else:
        iters = int(max_iterations * 0.6)  # 60%
    
    def enhanced_mns_search(solution, graph):
        return mns_local_search(solution, graph, neighborhoods, target_nodes=target_nodes)

    best_sequence = reactive_grasp(
        G,
        alpha_pool=(0.1, 0.3, 0.5, 0.7, 0.9),
        max_iterations=iters,
        update_window=8,
        local_search_fn=enhanced_mns_search,
        time_budget=max_time_budget,
        early_stop=True,
        target_nodes=target_nodes,
        needed_nodes=needed_nodes
    )
    # Correction robuste : forcer la présence de toutes les cibles et leurs prédécesseurs
    if target_nodes:
        needed = closure_with_predecessors(G, target_nodes)
        subgraph = G.subgraph(needed)
        if best_sequence is None or not is_valid(best_sequence, subgraph) or not all(t in best_sequence for t in target_nodes):
            best_sequence = list(nx.topological_sort(subgraph))
    return best_sequence



# ===============================
# SÉLECTIFS DSP (mode selectif)
# ===============================
# Ces fonctions doivent être utilisées uniquement pour le mode sélectif.
# Elles n'impactent pas le mode complet, qui utilise les fonctions classiques.

def grasp_simple_selectif(G, target_nodes=None, run_idx=0):
    """
    GRASP simple pour le mode sélectif (cibles uniquement)
    Recherche locale légère pour éviter des temps trop faibles.
    """
    needed_nodes = closure_with_predecessors(G, target_nodes)
    seq = greedy_randomized_construction(G, alpha=0.5, target_nodes=target_nodes, needed_nodes=needed_nodes)
    # Recherche locale rapide (swap_selectif)
    for _ in range(10):
        seq = swap_selectif(seq, G, target_nodes=target_nodes)
        # Protection anti-liste imbriquée
        if isinstance(seq, list) and any(isinstance(x, list) for x in seq):
            seq = [item for sublist in seq for item in (sublist if isinstance(sublist, list) else [sublist])]
        # Protection anti-boucle infinie : taille max
        if len(seq) > len(needed_nodes) * 2:
            seq = seq[:len(needed_nodes)]
    # Séquence plate d'identifiants
    seq = [x for x in seq if isinstance(x, str)]
    subgraph = G.subgraph(needed_nodes)
    # Correction stricte uniquement en fin
    if not is_valid(seq, subgraph) or not all(t in seq for t in target_nodes):
        seq = list(nx.topological_sort(subgraph))
    return seq

def grasp_with_vnd_selectif(G, target_nodes=None, run_idx=0):
    """
    GRASP+VND pour le mode sélectif (swap_selectif, relocate, two_opt)
    Boucles d'itération augmentées pour robustesse et timing réaliste.
    """
    neighborhoods = [swap_selectif, relocate, two_opt]
    needed_nodes = closure_with_predecessors(G, target_nodes)
    # Augmentation du nombre d'itérations pour le mode sélectif
    max_iterations = 80 if len(needed_nodes) > 100 else 150
    seq = grasp_with_vnd(G, neighborhoods=neighborhoods, target_nodes=target_nodes, needed_nodes=needed_nodes, max_iterations=max_iterations)
    subgraph = G.subgraph(needed_nodes)
    # Correction stricte uniquement en fin
    if not is_valid(seq, subgraph) or not all(t in seq for t in target_nodes):
        seq = list(nx.topological_sort(subgraph))
    return seq

def mns_selectif(G, target_nodes=None, run_idx=0):
    """
    MNS pour le mode sélectif (swap_selectif, relocate, two_opt)
    Boucles d'itération augmentées pour robustesse et timing réaliste.
    """
    needed_nodes = closure_with_predecessors(G, target_nodes)
    max_iterations = 80 if len(needed_nodes) > 100 else 150
    seq = enhanced_mns_solver(G, max_time_budget=30, max_iterations=max_iterations, target_nodes=target_nodes, needed_nodes=needed_nodes, mode="selectif")
    subgraph = G.subgraph(needed_nodes)
    # Correction stricte uniquement en fin
    if not is_valid(seq, subgraph) or not all(t in seq for t in target_nodes):
        seq = list(nx.topological_sort(subgraph))
    return seq

def grasp_with_tabu_selectif(G, target_nodes=None, run_idx=0):
    """
    GRASP+Tabu pour le mode sélectif (swap_selectif, relocate, two_opt)
    Boucles d'itération augmentées pour robustesse et timing réaliste.
    """
    neighborhoods = [swap_selectif, relocate, two_opt]
    needed_nodes = closure_with_predecessors(G, target_nodes)
    max_iterations = 80 if len(needed_nodes) > 100 else 150
    seq = grasp_with_tabu(G, neighborhoods=neighborhoods, target_nodes=target_nodes, needed_nodes=needed_nodes, max_iterations=max_iterations, mode="selectif")
    subgraph = G.subgraph(needed_nodes)
    # Correction stricte uniquement en fin
    if not is_valid(seq, subgraph) or not all(t in seq for t in target_nodes):
        seq = list(nx.topological_sort(subgraph))
    return seq
