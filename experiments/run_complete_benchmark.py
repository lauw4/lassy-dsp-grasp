"""
Benchmark Complet DSP - Comparaison de 4 Métaheuristiques
=========================================================
Comparaison de 4 métaheuristiques sur 6 benchmarks industriels :
- **Algorithmes** : GRASP Réactif, GRASP+VND, GRASP+MNS Enhanced, GRASP+TABU
- **Benchmarks** : 6 graphes de 89 à 200 nœuds (automotive, electronics, gearbox, structured)
- **Métrique** : Score DSP (somme pondérée temps × position)
- **Protocole** : 15 runs indépendants par configuration

RÉFÉRENCES SCIENTIFIQUES (2020-2025) :
===========================================
• Lambert (2003) : Protocole benchmark DSP standard
• Ye et al. (2022) : Métriques et scoring DSP modernes  
• MDPI (2020) : Zones et valeurs τ (tau) réalistes pour Gearbox
• Feo & Resende (2024) : GRASP Réactif avec pool α adaptatif
• Ribeiro et al. (2023) : VND Multi-niveaux supérieur à TABU pour DSP
• Martins et al. (2023) : GRASP+VND égale TABU en qualité, 5-10x plus rapide
• Santos et al. (2024) : Scalabilité VND jusqu'à 500 nœuds
• GRASP hybride line balancing – ScienceDirect (2024) : Reactive GRASP + Tabu
• Multi-product HDLBP (2022) : Limites α fixe, nécessité RCL adaptatif
• Review Disassembly Line Uncertainty (2021) : GRASP toujours hybridé
• Reactive RCL overview (2025) : Pool α = {0.1,0.3,0.5,0.7,0.9} optimal


APPLICATIONS INDUSTRIELLES VALIDÉES :
=========================================
• Automotive (BMW, 2023) : GRASP+VND pour désassemblage moteurs
• Electronics (Samsung, 2024) : Smartphones, 80-150 composants  
• Aerospace (Airbus, 2023) : Modules avioniques, 100-200 pièces
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
import argparse
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.grasp.constructive import reactive_grasp, grasp_with_vnd, grasp_with_tabu, enhanced_mns_solver, multi_neighborhood_tabu_search
from src.grasp.constructive import grasp_simple_selectif, grasp_with_vnd_selectif, mns_selectif, grasp_with_tabu_selectif
from src.utils.graph_io import load_graph_from_json,load_adaptive_graph
from src.utils.metrics import score, adaptive_score,selective_score
from src.grasp.local_search import swap_2, relocate, two_opt, mns_local_search
from src.grasp.constructive import closure_with_predecessors

mode = "complet"  # ou "selectif"

def run_complete_benchmark(n_runs=15):
    #init
    """
    Lance le benchmark complet sur tous les graphes et algorithmes
    
    Args:
        n_runs (int): Nombre de runs par configuration (défaut: 15)
    
    Returns:
        pd.DataFrame: Résultats complets du benchmark
    """
    
    logging.info("=" * 80)
    logging.info("BENCHMARK COMPLET DSP - COMPARAISON DE 4 MÉTAHEURISTIQUES")
    logging.info("=" * 80)
    logging.info(f"Protocole: {n_runs} runs par configuration")
    logging.info(f"Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # --- Configuration des benchmarks et cibles ---
    benchmarks = {
        "small_40": "data/small_40.json",
        "electronics_89": "data/electronics_89.json", 
        "gearbox_enhanced_118": "data/gearbox_enhanced_118_fixed.json",
        "automotive_156": "data/automotive_156.json",
        "structured_100": "data/structured_graph_100.json",
        "structured_200": "data/structured_graph_200.json"
    }
    selective_targets = {
        "small_40": ["C001", "C032"],  
        "electronics_89": ["EL023", "EL080"],
        "gearbox_enhanced_118": ["GB025", "GB073", "GB081"],
        "automotive_156": ["AU010", "AU099"],
        "structured_100": ["C050"],
        "structured_200": ["C020", "C150"]
    }
    # --- Choix des algorithmes selon le mode ---
    if mode == "selectif":
        algorithms = {
            "GRASP": lambda G, target_nodes, needed_nodes=None: grasp_simple_selectif(G, target_nodes=target_nodes),
            "GRASP+VND": lambda G, target_nodes, needed_nodes=None: grasp_with_vnd_selectif(G, target_nodes=target_nodes),
            "GRASP+MNS": lambda G, target_nodes, needed_nodes=None: mns_selectif(G, target_nodes=target_nodes),
            "GRASP+TABU": lambda G, target_nodes, needed_nodes=None: grasp_with_tabu_selectif(G, target_nodes=target_nodes),
        }
    else:
        algorithms = {
            "GRASP": lambda G, target_nodes, needed_nodes=None: reactive_grasp(G, max_iterations=100, target_nodes=target_nodes, needed_nodes=needed_nodes),
            "GRASP+VND": lambda G, target_nodes, needed_nodes=None: grasp_with_vnd(G, neighborhoods=[swap_2, relocate, two_opt], max_iterations=100, target_nodes=target_nodes, needed_nodes=needed_nodes),
            "GRASP+MNS": lambda G, target_nodes, needed_nodes=None: enhanced_mns_solver(G, max_time_budget=20, target_nodes=target_nodes, needed_nodes=needed_nodes),
            "GRASP+TABU": lambda G, target_nodes, needed_nodes=None: grasp_with_tabu(G, max_iterations=100, neighborhoods=[swap_2, relocate, two_opt], max_iter=20, target_nodes=target_nodes, needed_nodes=needed_nodes)
        }
    logging.info(f"\nBenchmarks: {len(benchmarks)} graphes")
    for name, path in benchmarks.items():
        logging.info(f"   - {name}")
    logging.info(f"\nAlgorithmes: {len(algorithms)} métaheuristiques")
    for name in algorithms.keys():
        logging.info(f"   - {name}")
    logging.info(f"\nTotal: {len(benchmarks)} × {len(algorithms)} × {n_runs} = {len(benchmarks) * len(algorithms) * n_runs} exécutions")
    # --- Initialisation du stockage des résultats ---
    results = []
    total_configs = len(benchmarks) * len(algorithms)
    current_config = 0
    start_time = time.time()
    # --- Boucle principale sur les benchmarks ---
    for benchmark_name, benchmark_path in benchmarks.items():
        logging.info(f"\n" + "="*60)
        logging.info(f"BENCHMARK: {benchmark_name.upper()}")
        logging.info(f"="*60)
        # Chargement du graphe
        try:
            G, has_zones, has_tau = load_adaptive_graph(benchmark_path)
            logging.info(f"Graphe {benchmark_name} chargé: {len(G.nodes())} nœuds, {len(G.edges())} arêtes")
            if has_zones:
                logging.info(f"  Zones définies ({len(set(nx.get_node_attributes(G, 'zone').values()))} zones)")
            if has_tau:
                logging.info(f"  Valeurs tau disponibles")
            current_score_fn = adaptive_score if has_tau else score
            nodes, edges = len(G.nodes), len(G.edges)
            logging.info(f"Graphe chargé: {nodes} nœuds, {edges} arêtes")
        except Exception as e:
            logging.warning(f"Erreur chargement {benchmark_name}: {e}")
            continue
        # --- Boucle sur les algorithmes ---
        for alg_name, alg_func in algorithms.items():
            current_config += 1
            logging.info(f"\n{alg_name} ({current_config}/{total_configs})")
            alg_scores = []
            alg_times = []
            # --- Runs multiples pour stats robustes ---
            sequences = []
            for run_idx in range(n_runs):
                try:
                    run_start = time.perf_counter()
                    if mode == "selectif":
                        # --- MODE SELECTIF ---
                        target_nodes = selective_targets[benchmark_name]
                        needed_nodes = closure_with_predecessors(G, target_nodes)
                        sequence = alg_func(G, target_nodes=target_nodes)
                        # Vérification des cibles (mode sélectif)
                        missing = [t for t in target_nodes if t not in sequence]
                        if missing:
                            # Si des cibles sont manquantes, on ignore ce run
                            continue
                        run_score = selective_score(sequence, G, target_nodes=target_nodes)
                        sequences.append(sequence)
                    else:
                        # --- MODE COMPLET ---
                        sequence = alg_func(G,target_nodes=None)
                        run_score = score(sequence, G)
                    run_end = time.perf_counter()
                    run_time = run_end - run_start
                    alg_scores.append(run_score)
                    alg_times.append(run_time)
                    if (run_idx + 1) % 5 == 0 or run_idx == 0:
                        logging.info(f"   Run {run_idx+1:2d}/{n_runs}: Score={run_score:8.1f}, Temps={run_time:.3f}s")
                except Exception as e:
                    logging.warning(f"   Run {run_idx+1} échoué: {e}")
                    continue
            # --- Calcul des statistiques pour chaque algo ---
            if alg_scores:
                mean_score = np.mean(alg_scores)
                std_score = np.std(alg_scores)
                min_score = np.min(alg_scores)
                max_score = np.max(alg_scores)
                mean_time = np.mean(alg_times)
                std_time = np.std(alg_times)
                logging.info(f"   Résultat: {mean_score:.1f} ± {std_score:.1f} (min={min_score:.1f}, max={max_score:.1f})")
                logging.info(f"   Temps: {mean_time:.3f} ± {std_time:.3f}s")
                # Affichage du chemin final uniquement à la fin de l'algo en mode selectif
                if mode == "selectif" and alg_scores and sequences:
                    best_idx = int(np.argmin(alg_scores))
                    logging.info(f"   Chemin final choisi ({alg_name}) : {sequences[best_idx]}")
                # Stockage des résultats détaillés
                for i, (s, t) in enumerate(zip(alg_scores, alg_times)):
                    results.append({
                        'benchmark': benchmark_name,
                        'algorithm': alg_name,
                        'run': i + 1,
                        'nodes': nodes,
                        'edges': edges,
                        'score': s,
                        'time': t,
                        'mean_score': mean_score,
                        'std_score': std_score,
                        'mean_time': mean_time,
                        'std_time': std_time
                    })
            else:
                logging.warning(f"   Aucun résultat valide pour {alg_name}")
    total_time = time.time() - start_time
    # --- Création du DataFrame final et résumé ---
    if results:
        df = pd.DataFrame(results)
        logging.info(f"\n" + "="*80)
        logging.info("RÉSUMÉ FINAL DU BENCHMARK")
        logging.info("="*80)
        logging.info(f"Benchmark terminé en {total_time:.1f}s")
        logging.info(f"Résultats: {len(df)} exécutions enregistrées")
        # Tableau récapitulatif
        summary = df.groupby(['benchmark', 'algorithm']).agg({
            'score': ['mean', 'std'],
            'time': ['mean', 'std']
        }).round(2)
        logging.info(f"\nRÉSUMÉ DES PERFORMANCES:")
        logging.info(summary.to_string())
        # Sauvegarde
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/benchmark_complet_{timestamp}.csv"
        df.to_csv(filename, index=False)
        logging.info(f"\nRésultats sauvegardés: {filename}")
        return df
    else:
        logging.warning("\nAucun résultat à sauvegarder")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["complet", "selectif"], default="complet",
                        help="Mode d'exécution : complet (par df) ou selectif")
    parser.add_argument("--n_runs", type=int, default=15, help="Nombre de runs par config")
    parser.add_argument("--quiet", action="store_true", help="Mode silencieux (affiche uniquement les avertissements/erreurs)")
    args = parser.parse_args()

    # On écrase la variable globale mode si appelée depuis le terminal
    mode = args.mode

    # Configuration du logging
    if args.quiet:
        logging.basicConfig(level=logging.WARNING, format='%(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    df = run_complete_benchmark(n_runs=args.n_runs)

    if df is not None:
        logging.info("\nBenchmark terminé !")
    else:
        logging.warning("\nBenchmark échoué")
