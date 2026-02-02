import json
import os
import random
from pathlib import Path

def parse_in2_file(file_path):
    """Parse an .IN2 file (Scholl format) and extract data"""
    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    n = int(lines[0])
    task_times = [int(lines[i+1]) for i in range(n)]
    # Precedence relations
    precedences = []
    for line in lines[n+1:]:
        if ',' in line:
            i, j = map(int, line.split(','))
            if i == -1 and j == -1:
                break
            precedences.append((i, j))
    return n, task_times, precedences

def generate_dsp_instance_scholl(n, task_times, precedences, instance_name):
    # Node generation
    nodes = []
    for task_id in range(1, n + 1):
        duration = task_times[task_id-1]
        # Realistic DSP attribute generation
        profit = random.randint(5, 30)
        cost = random.randint(1, 10)
        if duration >= 4:
            profit = random.randint(20, 30)
        elif duration <= 2:
            profit = random.randint(5, 15)
        node = {
            "id": f"C{task_id-1:03d}",
            "profit": profit,
            "cost": cost,
            "duration": duration,
            "milp_index": task_id
        }
        nodes.append(node)
    # Edge conversion (precedence relations)
    edges = []
    order_rules = []
    for pred, succ in precedences:
        pred_id = f"C{pred-1:03d}"
        succ_id = f"C{succ-1:03d}"
        edges.append([pred_id, succ_id])
        order_rules.append({
            "if": [pred_id, succ_id],
            "condition": f"{pred_id} avant {succ_id}"
        })
    # Target selection (identical to convert_salbp_to_dsp)
    successors = set(succ for _, succ in precedences)
    all_tasks = set(range(1, n + 1))
    leaves = all_tasks - set(pred for pred, _ in precedences)
    if leaves:
        target_count = min(random.randint(1, 3), len(leaves))
        target_tasks = random.sample(list(leaves), target_count)
    else:
        target_count = min(2, n)
        target_tasks = random.sample(range(1, n + 1), target_count)
    targets = [f"C{task-1:03d}" for task in target_tasks]
    dsp_instance = {
        "nodes": nodes,
        "edges": edges,
        "targets": targets,
        "order_rules": order_rules
    }
    return dsp_instance

def convert_dsp_to_txt(dsp_instance, output_path):
    with open(output_path, 'w') as f:
        milp_indices = [node['milp_index'] for node in dsp_instance['nodes']]
        f.write(f"# Components (V)\n")
        f.write(' '.join(map(str, milp_indices)) + '\n\n')
        f.write("# Precedence arcs (E)\n")
        for edge in dsp_instance['edges']:
            pred_idx = next(node['milp_index'] for node in dsp_instance['nodes'] if node['id'] == edge[0])
            succ_idx = next(node['milp_index'] for node in dsp_instance['nodes'] if node['id'] == edge[1])
            f.write(f"{pred_idx} {succ_idx}\n")
        f.write('\n')
        f.write("# Target components (T)\n")
        target_indices = []
        for target_id in dsp_instance['targets']:
            target_idx = next(node['milp_index'] for node in dsp_instance['nodes'] if node['id'] == target_id)
            target_indices.append(target_idx)
        f.write(' '.join(map(str, target_indices)) + '\n\n')
        f.write("# Revenues (r)\n")
        for node in dsp_instance['nodes']:
            f.write(f"{node['milp_index']} {node['profit']}\n")
        f.write('\n')
        f.write("# Costs (c)\n")
        for node in dsp_instance['nodes']:
            f.write(f"{node['milp_index']} {node['cost']}\n")
        f.write('\n')
        f.write("# Durations (p)\n")
        for node in dsp_instance['nodes']:
            f.write(f"{node['milp_index']} {node['duration']}\n")
        f.write('\n')
        f.write("orderrules\n")
        for rule in dsp_instance['order_rules']:
            pred_idx = next(node['milp_index'] for node in dsp_instance['nodes'] if node['id'] == rule['if'][0])
            succ_idx = next(node['milp_index'] for node in dsp_instance['nodes'] if node['id'] == rule['if'][1])
            pred_id = rule['if'][0].replace('C', 'au')
            succ_id = rule['if'][1].replace('C', 'au')
            f.write(f"{pred_idx} {succ_idx} {pred_id} avant {succ_id}\n")

def process_scholl_instances(input_dir, out_json_dir, out_txt_dir):
    input_dir = Path(input_dir)
    out_json_dir = Path(out_json_dir)
    out_txt_dir = Path(out_txt_dir)
    out_json_dir.mkdir(parents=True, exist_ok=True)
    out_txt_dir.mkdir(parents=True, exist_ok=True)
    for in2_file in input_dir.glob("*.IN2"):
        n, task_times, precedences = parse_in2_file(in2_file)
        instance_name = f"scholl_{in2_file.stem.lower()}_n={n}"
        print(f"Converting {in2_file.name} → {instance_name}.json/.txt ...")
        dsp_instance = generate_dsp_instance_scholl(n, task_times, precedences, instance_name)
        json_path = out_json_dir / f"{instance_name}.json"
        with open(json_path, 'w') as f:
            json.dump(dsp_instance, f, indent=2)
        txt_path = out_txt_dir / f"{instance_name}_selectif.txt"
        convert_dsp_to_txt(dsp_instance, txt_path)
        print(f"  ✓ Saved {json_path.name} and {txt_path.name}")

def main():
    process_scholl_instances(
        input_dir="c:/Users/Waele/OneDrive/Documents/Etude/Cours/semestre_9/STAGE/IMT STAGE 5A/articles_data_utiles/SALBP-data-sets/precedence graphs",
        out_json_dir="data/instances_base",
        out_txt_dir="data/instances_milp"
    )
    print("\n Scholl -> DSP Conversion completed! ")

if __name__ == "__main__":
    main()
