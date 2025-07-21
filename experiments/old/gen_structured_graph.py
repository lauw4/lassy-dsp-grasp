import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.generate_structured_graph import generate_structured_graph, save_graph_to_json


G = generate_structured_graph(n_nodes=40, n_levels=4, deps_per_node=2, seed=99)
save_graph_to_json(G, "data/small_40.json")


G = generate_structured_graph(n_nodes=100, n_levels=5, deps_per_node=3, seed=123)
save_graph_to_json(G, "data/structured_graph_100.json")

G = generate_structured_graph(n_nodes=200, n_levels=6, deps_per_node=3, seed=42)
save_graph_to_json(G, "data/structured_graph_200.json")