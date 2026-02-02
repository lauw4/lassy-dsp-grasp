import json
import os
import random
from pathlib import Path

def parse_alb_file(file_path):
    """Parse an .alb file and extract data"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    data = {}
    lines = content.strip().split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        if line == '<number of tasks>':
            data['num_tasks'] = int(lines[i+1])
            i += 2
        elif line == '<task times>':
            data['task_times'] = {}
            i += 1
            while i < len(lines) and not lines[i].startswith('<'):
                parts = lines[i].split()
                if len(parts) == 2:
                    task_id, time = int(parts[0]), int(parts[1])
                    data['task_times'][task_id] = time
                i += 1
        elif line == '<precedence relations>':
            data['precedences'] = []
            i += 1
            while i < len(lines) and not lines[i].startswith('<'):
                if ',' in lines[i]:
                    pred, succ = map(int, lines[i].split(','))
                    data['precedences'].append((pred, succ))
                i += 1
        else:
            i += 1
    
    return data

def generate_dsp_instance(alb_data, instance_name):
    """Convert SALBP data to DSP instance"""
    
    num_tasks = alb_data['num_tasks']
    task_times = alb_data['task_times']
    precedences = alb_data['precedences']
    
    # Node generation
    nodes = []
    for task_id in range(1, num_tasks + 1):
        duration = task_times.get(task_id, random.randint(1, 5))
        
        # Realistic DSP attribute generation
        profit = random.randint(5, 30)
        cost = random.randint(1, 10)
        
        # Adjust profit/cost based on duration (longer = more profitable)
        if duration >= 4:
            profit = random.randint(20, 30)
        elif duration <= 2:
            profit = random.randint(5, 15)
        
        node = {
            "id": f"C{task_id-1:03d}",  # C000, C001, etc.
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
    
    # Target selection (1-3 components)
    # Favor components without successors (graph leaves)
    successors = set(succ for _, succ in precedences)
    all_tasks = set(range(1, num_tasks + 1))
    leaves = all_tasks - set(pred for pred, _ in precedences)
    
    if leaves:
        target_count = min(random.randint(1, 3), len(leaves))
        target_tasks = random.sample(list(leaves), target_count)
    else:
        # If no leaves, pick random components
        target_count = min(2, num_tasks)
        target_tasks = random.sample(range(1, num_tasks + 1), target_count)
    
    targets = [f"C{task-1:03d}" for task in target_tasks]
    
    # Final DSP structure
    dsp_instance = {
        "nodes": nodes,
        "edges": edges,
        "targets": targets,
        "order_rules": order_rules
    }
    
    return dsp_instance

def convert_alb_to_txt(dsp_instance, output_path):
    """Convert DSP instance to .txt format"""
    
    with open(output_path, 'w') as f:
        # Components (V)
        milp_indices = [node['milp_index'] for node in dsp_instance['nodes']]
        f.write(f"# Components (V)\n")
        f.write(' '.join(map(str, milp_indices)) + '\n\n')
        
        # Precedence arcs (E)
        f.write("# Precedence arcs (E)\n")
        for edge in dsp_instance['edges']:
            pred_idx = next(node['milp_index'] for node in dsp_instance['nodes'] if node['id'] == edge[0])
            succ_idx = next(node['milp_index'] for node in dsp_instance['nodes'] if node['id'] == edge[1])
            f.write(f"{pred_idx} {succ_idx}\n")
        f.write('\n')
        
        # Target components (T)
        f.write("# Target components (T)\n")
        target_indices = []
        for target_id in dsp_instance['targets']:
            target_idx = next(node['milp_index'] for node in dsp_instance['nodes'] if node['id'] == target_id)
            target_indices.append(target_idx)
        f.write(' '.join(map(str, target_indices)) + '\n\n')
        
        # Revenues (r)
        f.write("# Revenues (r)\n")
        for node in dsp_instance['nodes']:
            f.write(f"{node['milp_index']} {node['profit']}\n")
        f.write('\n')
        
        # Costs (c)
        f.write("# Costs (c)\n")
        for node in dsp_instance['nodes']:
            f.write(f"{node['milp_index']} {node['cost']}\n")
        f.write('\n')
        
        # Durations (p)
        f.write("# Durations (p)\n")
        for node in dsp_instance['nodes']:
            f.write(f"{node['milp_index']} {node['duration']}\n")
        f.write('\n')
        
        # Order rules
        f.write("orderrules\n")
        for rule in dsp_instance['order_rules']:
            pred_idx = next(node['milp_index'] for node in dsp_instance['nodes'] if node['id'] == rule['if'][0])
            succ_idx = next(node['milp_index'] for node in dsp_instance['nodes'] if node['id'] == rule['if'][1])
            pred_id = rule['if'][0].replace('C', 'au')
            succ_id = rule['if'][1].replace('C', 'au')
            f.write(f"{pred_idx} {succ_idx} {pred_id} avant {succ_id}\n")

def process_dataset(dataset_name, input_path, max_instances=3):
    """Process a SALBP dataset and convert instances to DSP"""
    
    input_dir = Path(input_path)
    output_dir_json = Path("data/instances_base")
    output_dir_txt = Path("data/instances_milp")
    
    # Create output directories
    output_dir_json.mkdir(parents=True, exist_ok=True)
    output_dir_txt.mkdir(parents=True, exist_ok=True)
    
    # Process .alb files
    alb_files = list(input_dir.glob("*.alb"))
    selected_files = alb_files[:max_instances]  # Limit to max_instances
    
    print(f"\n=== Processing {dataset_name} ===")
    print(f"Found {len(alb_files)} .alb files, processing {len(selected_files)}")
    
    for alb_file in selected_files:
        print(f"Converting {alb_file.name}...")
        
        try:
            # Parse the .alb file
            alb_data = parse_alb_file(alb_file)
            
            # Generate instance name
            instance_name = f"salbp_{alb_file.stem}"
            
            # Convert to DSP
            dsp_instance = generate_dsp_instance(alb_data, instance_name)
            
            # Save JSON
            json_path = output_dir_json / f"{instance_name}.json"
            with open(json_path, 'w') as f:
                json.dump(dsp_instance, f, indent=2)
            
            # Save TXT  
            txt_path = output_dir_txt / f"{instance_name}_selectif.txt"
            convert_alb_to_txt(dsp_instance, txt_path)
            
            print(f"  ✓ Created {json_path.name} and {txt_path.name}")
            
        except Exception as e:
            print(f"  ✗ Error converting {alb_file.name}: {e}")

def main():
    """Convert SALBP instances of different sizes to DSP instances"""
    
    # Dataset configuration
    datasets = {
        "n=20": "c:/Users/Waele/OneDrive/Documents/Etude/Cours/semestre_9/STAGE/IMT STAGE 5A/articles_data_utiles/SALBP_benchmark/small data set_n=20/small data set_n=20",
        "n=50": "c:/Users/Waele/OneDrive/Documents/Etude/Cours/semestre_9/STAGE/IMT STAGE 5A/articles_data_utiles/SALBP_benchmark/medium data set_n=50",
        "n=50_permuted": "c:/Users/Waele/OneDrive/Documents/Etude/Cours/semestre_9/STAGE/IMT STAGE 5A/articles_data_utiles/SALBP_benchmark/medium data set_n=50permuted", 
        "n=100": "c:/Users/Waele/OneDrive/Documents/Etude/Cours/semestre_9/STAGE/IMT STAGE 5A/articles_data_utiles/SALBP_benchmark/large data set_n=100",
        "n=1000": "c:/Users/Waele/OneDrive/Documents/Etude/Cours/semestre_9/STAGE/IMT STAGE 5A/articles_data_utiles/SALBP_benchmark/very large data set_n=1000"
    }
    
    print("=== SALBP to DSP Conversion ===")
    print("Converting maximum 3 instances per dataset size")
    
    # Process each dataset
    for dataset_name, dataset_path in datasets.items():
        if Path(dataset_path).exists():
            process_dataset(dataset_name, dataset_path, max_instances=3)
        else:
            print(f" Dataset path not found: {dataset_path}")
    
    print("\n=== Conversion completed! ===")

if __name__ == "__main__":
    main()
