"""Debug buxey and sawyer: find why GRASP gets stuck far from MILP."""
import sys, time
sys.path.insert(0, '.')
import pulp
from src.utils.graph_io import load_graph_from_txt
from src.grasp.constructive import run_grasp
from src.utils.metrics import score_obj1, prune_sequence
from src.exact_milp_dijkstra.partial_disassembly_model import load_data, build_model, solve_model

for name, txt in [
    ('buxey_n29', 'data/instances_milp/scholl_buxey_n=29_selectif.txt'),
    ('sawyer_n30', 'data/instances_milp/scholl_sawyer30_n=30_selectif.txt'),
]:
    G = load_graph_from_txt(txt)
    targets = G.graph['targets']
    lam = G.graph['lambda_discount']
    n = G.number_of_nodes()
    print(f"\n=== {name}: n={n}, targets={targets}, lambda={lam:.4f} ===")

    # Show nodes
    vals = [(nid, d['profit'], d['cost'], d['time'], d['profit']-d['cost'])
            for nid, d in sorted(G.nodes(data=True))]
    for nid, p, c, t, v in sorted(vals, key=lambda x: -x[4])[:10]:
        print(f"  {nid}: profit={p:.0f}, cost={c:.0f}, time={t:.0f}, v={v:.0f}")

    # MILP
    data = load_data(txt)
    model, vars_ = build_model(data)
    solve_model(model, time_limit=120, use_cplex=False, gap_limit=0.001)
    milp_obj = pulp.value(model.objective)
    x_vals = vars_['x']
    sel = [f"C{i:03d}" for i in data['V'] if pulp.value(x_vals[i]) and pulp.value(x_vals[i]) > 0.5]
    W_sel = {f"C{i:03d}": round(pulp.value(vars_['W'][i]),1) for i in data['V'] if pulp.value(x_vals[i]) and pulp.value(x_vals[i]) > 0.5}
    print(f"  MILP obj={milp_obj:.2f}, selected={sel}")
    print(f"  MILP W_i={W_sel}")

    # GRASP many runs
    best_g = float('-inf')
    for _ in range(50):
        seq, _ = run_grasp(G, algorithm='vnd', mode='complet', runs=1, max_iterations=300, time_budget=60, early_stop=True)
        seq_p = prune_sequence(seq, G, protected_nodes=set(targets))
        s = score_obj1(seq_p, G)
        if s > best_g:
            best_g = s
            best_seq = seq_p
    print(f"  GRASP+prune best={best_g:.2f}, seq={best_seq}")
    print(f"  Gap: {(milp_obj - best_g)/abs(milp_obj)*100:.1f}%")
