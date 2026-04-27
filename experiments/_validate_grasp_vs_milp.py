"""Quick validation: GRASP complet + prune vs MILP on small Scholl instances (same TXT data)."""
import sys, time
sys.path.insert(0, '.')
import pulp
from src.utils.graph_io import load_graph_from_txt
from src.grasp.constructive import run_grasp
from src.utils.metrics import score_obj1, prune_sequence
from src.exact_milp_dijkstra.partial_disassembly_model import load_data, build_model, solve_model

test_files = [
    ('mini_test',    'data/instances_milp/mini_test_selectif.txt'),
    ('mertens_n7',   'data/instances_milp/scholl_mertens_n=7_selectif.txt'),
    ('bowman8',      'data/instances_milp/scholl_bowman8_n=8_selectif.txt'),
    ('jaeschke_n9',  'data/instances_milp/scholl_jaeschke_n=9_selectif.txt'),
    ('jackson_n11',  'data/instances_milp/scholl_jackson_n=11_selectif.txt'),
    ('mansoor_n11',  'data/instances_milp/scholl_mansoor_n=11_selectif.txt'),
    ('roszieg_n25',  'data/instances_milp/scholl_roszieg_n=25_selectif.txt'),
    ('mitchell_n21', 'data/instances_milp/scholl_mitchell_n=21_selectif.txt'),
]

print(f"{'Instance':<18} {'n':>4} {'lambda':>8} {'GRASP+prune':>12} {'MILP':>10} {'Gap%':>7} {'Time':>6}")
print('-' * 72)
total_gap = 0.0
n_compare = 0
for name, txt in test_files:
    # GRASP complet (all nodes) + prune unprofitable nodes → same scope as MILP
    G = load_graph_from_txt(txt)
    targets = G.graph.get('targets', [])
    lam = G.graph['lambda_discount']
    n = G.number_of_nodes()
    best_g = float('-inf')
    t0 = time.perf_counter()
    for _ in range(20):
        seq, s = run_grasp(
            G, algorithm='vnd', mode='complet',
            runs=1, max_iterations=200, time_budget=30, early_stop=True,
        )
        # Protect mandatory targets when pruning (same constraint as MILP x_i=1 for i in T)
        seq_pruned = prune_sequence(seq, G, protected_nodes=set(targets))
        s_pruned = score_obj1(seq_pruned, G)
        if s_pruned > best_g:
            best_g = s_pruned
    grasp_time = time.perf_counter() - t0
    # MILP on same TXT data
    data = load_data(txt)
    model, _ = build_model(data)
    solve_model(model, time_limit=120, use_cplex=False, gap_limit=0.005)
    milp_obj = round(pulp.value(model.objective) or 0.0, 2)
    if milp_obj != 0:
        gap = (milp_obj - best_g) / abs(milp_obj) * 100.0
        total_gap += gap
        n_compare += 1
    else:
        gap = float('nan')
    flag = " ✓" if abs(gap) < 3 else " ✗"
    print(f"{name:<18} {n:>4} {lam:>8.4f} {best_g:>12.2f} {milp_obj:>10.2f} {gap:>7.1f}%{flag}  {grasp_time:.1f}s")

if n_compare > 0:
    print(f"\n  Average gap: {total_gap / n_compare:.2f} %  (target: < 3 %)")
