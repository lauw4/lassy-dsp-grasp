"""Test: verify the DSP objective is now position-dependent (NP-hard)."""
import random
random.seed(42)

from src.utils.graph_io import load_adaptive_graph
from src.utils.metrics import score_profit_net, score
from src.grasp.constructive import closure_with_predecessors, run_grasp
import networkx as nx

# 1. Load graph
G, _, _ = load_adaptive_graph('data/instances_base/gearpump.json')
print(f"=== Gearpump: {len(G.nodes)} nodes ===")
print(f"C_max      = {G.graph.get('C_max', 'NOT SET'):.1f}")
print(f"dsp_lambda = {G.graph.get('dsp_lambda', 'NOT SET')}")
total_time = sum(
    G.nodes[n].get('time', G.nodes[n].get('duration', 1.0))
    for n in G.nodes()
)
print(f"Total time = {total_time},  C_max/total = {G.graph['C_max']/total_time:.0%}")

# 2. Position-dependent test
target = ['G20']
needed = closure_with_predecessors(G, target)
print(f"\nClosure size = {len(needed)}  (out of {len(G.nodes)})")

subG = G.subgraph(needed)
count = 0
topos = []
for t in nx.all_topological_sorts(subG):
    topos.append(list(t))
    count += 1
    if count >= 5:
        break

print(f"Sampled {len(topos)} topological sorts")

if len(topos) >= 2:
    scores = [score_profit_net(t, G) for t in topos]
    print("\n--- Position-dependent test ---")
    for i, (t, s) in enumerate(zip(topos, scores)):
        print(f"  Topo {i+1}: score = {s:.4f}")
    
    all_same = all(abs(s - scores[0]) < 1e-10 for s in scores)
    print(f"\nAll scores identical? {all_same}")
    if not all_same:
        print("YES - different scores for different orderings => NP-hard confirmed!")
    else:
        print("NO - still equivalent to topological sort (polynomial)")

# 3. GRASP test
print("\n--- Running GRASP ---")
seq, s = run_grasp(G, algorithm='vnd', mode='selectif', target_nodes=['G20'], runs=3)
print(f"GRASP best sequence: {seq}")
print(f"GRASP score_profit_net = {s:.4f}")
print(f"GRASP score (wct)      = {score(seq, G):.4f}")

# 4. Budget test
print("\n--- Budget (C_max) test ---")
total_seq_time = sum(
    G.nodes[n].get('time', G.nodes[n].get('duration', 1.0))
    for n in seq
)
print(f"Sequence total time = {total_seq_time}")
print(f"C_max = {G.graph['C_max']:.1f}")
print(f"Within budget? {total_seq_time <= G.graph['C_max']}")
