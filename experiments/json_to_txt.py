import os
import json
import argparse

def json_to_txt(json_path, txt_path=None):
	with open(json_path, 'r', encoding='utf-8') as f:
		data = json.load(f)

	nodes = data.get('nodes', [])
	edges = data.get('edges', [])
	targets = data.get('targets', [])
	order_rules = data.get('order_rules', [])

	# Verbose and commented format 
	node_ids = [n['id'] for n in nodes]
	# Use milp_index if present, otherwise fallback to order
	node_map = {n['id']: n['milp_index'] if 'milp_index' in n else i+1 for i, n in enumerate(nodes)}

	lines = []
	# Components (V)
	lines.append('# Components (V)')
	lines.append(' '.join(str(node_map[nid]) for nid in node_ids))

	# Precedence arcs (E)
	lines.append('\n# Precedence arcs (E)')
	for e in edges:
		lines.append(f"{node_map[e[0]]} {node_map[e[1]]}")

	# Targets (T)
	if targets:
		lines.append('\n# Target components (T)')
		lines.append(' '.join(str(node_map[t]) for t in targets))

	# Revenus (r)
	lines.append('\n# Revenues (r)')
	for n in nodes:
		lines.append(f"{node_map[n['id']]} {n['profit']}")


	# Costs (c) - read from each JSON node
	lines.append('\n# Costs (c)')
	for n in nodes:
		cost = n.get('cost', 0)
		lines.append(f"{node_map[n['id']]} {cost}")

	# Durations (p) - read from each JSON node
	lines.append('\n# Durations (p)')
	for n in nodes:
		duration = n.get('duration', 1)
		lines.append(f"{node_map[n['id']]} {duration}")

	# C_max if present
	if 'C_max' in data:
		lines.append('\n# C_max (optional)')
		lines.append(str(data['C_max']))

	# Section 'orderrules' 
	if order_rules:
		lines.append('\norderrules')
		for rule in order_rules:
			cond = rule.get('condition', '').lower()
			n1, n2 = rule.get('if', [None, None])
			if n1 in node_map and n2 in node_map:
				lines.append(f"{node_map[n1]} {node_map[n2]} {cond}")

	txt_content = '\n'.join(lines)

	if not txt_path:
		base = os.path.splitext(os.path.basename(json_path))[0]
		txt_path = os.path.join('data', 'instances_milp', f'{base}_selectif.txt')

	with open(txt_path, 'w', encoding='utf-8') as f:
		f.write(txt_content)
	print(f"[INFO] MILP file generated: {txt_path}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Convert a DSP JSON to MILP .txt")
	parser.add_argument('--json', required=True, help='Path to JSON instance file')
	parser.add_argument('--txt', help='Path to output TXT file (optional)')
	args = parser.parse_args()
	json_to_txt(args.json, args.txt)
