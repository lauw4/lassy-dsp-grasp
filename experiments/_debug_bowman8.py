"""Debug bowman8: compare closure vs all nodes, check if MILP includes non-closure nodes."""
import sys
sys.path.insert(0, '.')
import pulp
from src.utils.graph_io import load_graph_from_txt
from src.grasp.constructive import closure_with_predecessors, run_grasp
from src.exact_milp_dijkstra.partial_disassembly_model import load_data, build_model, solve_model

txt = 'data/instances_milp/scholl_bowman8_n=8_selectif.txt'
G = load_graph_from_txt(txt)
targets = G.graph['targets']
lam = G.graph['lambda_discount']
needed = closure_with_predecessors(G, targets)
non_closure = set(G.nodes()) - needed

print(f"n_total={G.number_of_nodes()}, targets={targets}")
print(f"closure size={len(needed)}, non-closure={non_closure}")
print(f"lambda={lam:.4f}")
print("Node attrs:")
for nid, d in sorted(G.nodes(data=True)):
    v_i = d['profit'] - d['cost']
    print(f"  {nid}: profit={d['profit']}, cost={d['cost']}, time={d['time']}, v_i={v_i:.1f}  {'[TARGET]' if nid in targets else '[closure]' if nid in needed else '[EXCLUDED]'}")

# Run MILP without closure restriction
data = load_data(txt)
model, vars_ = build_model(data)
solve_model(model, time_limit=60, use_cplex=False, gap_limit=0.001)
milp_obj = pulp.value(model.objective)
x_vals = {i: pulp.value(vars_['x'][i]) for i in data['V']}
selected = [i for i, v in x_vals.items() if v and v > 0.5]
print(f"\nMILP obj={milp_obj:.2f}, selected={selected} -> string={[f'C{i:03d}' for i in selected]}")
W_vals = {i: pulp.value(vars_['W'][i]) for i in selected}
print(f"MILP W_i (completion times): {W_vals}")

# Run GRASP with full complet mode to see if it reaches MILP
best_g = float('-inf')
best_seq = None
for _ in range(30):
    seq, s = run_grasp(G, algorithm='vnd', mode='complet', runs=1, max_iterations=300, time_budget=30, early_stop=True)
    if s > best_g:
        best_g = s
        best_seq = seq
print(f"\nGRASP complet best={best_g:.2f}, seq={best_seq}")

# Selectif
best_s = float('-inf')
for _ in range(30):
    seq, s = run_grasp(G, algorithm='vnd', mode='selectif', target_nodes=targets, runs=1, max_iterations=300, time_budget=30, early_stop=True)
    if s > best_s:
        best_s = s
print(f"GRASP selectif best={best_s:.2f}")
