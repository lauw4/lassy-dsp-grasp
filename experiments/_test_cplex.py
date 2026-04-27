"""Test CPLEX detection and solve on mini_test."""
import sys, time
sys.path.insert(0, '.')
from src.exact_milp_dijkstra.partial_disassembly_model import load_data, build_model, solve_model
import pulp

data = load_data('data/instances_milp/mini_test_selectif.txt')
model, variables = build_model(data)
t0 = time.perf_counter()
opt, log = solve_model(model, time_limit=30, use_cplex=True, gap_limit=0.01, instance_name='mini_test')
elapsed = time.perf_counter() - t0
print(f"Status : {pulp.LpStatus[model.status]}")
print(f"Obj1   : {round(pulp.value(model.objective), 4)}")
print(f"Time   : {round(elapsed, 2)} s")
