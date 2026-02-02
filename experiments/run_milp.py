#python experiments\run_milp.py --instance structured_graph_200 --mode selectif
#python experiments\run_milp.py --instance structured_graph_200 --mode selectif --time_limit 14400
#python experiments\run_milp.py --instance structured_graph_200 --mode selectif --gap 0.05

import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.exact_milp_dijkstra.partial_disassembly_model import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MILP model execution")
    parser.add_argument("--instance", type=str, default="electronics_89", help="Instance name (without extension)")
    parser.add_argument("--mode", choices=["complet", "selectif"], default="selectif", help="Execution mode")
    # Defaults "per literature": 2h and 3%
    parser.add_argument("--time_limit", type=int, default=7200, help="Time limit in seconds (0 for no limit). Default: 7200s (2h)")
    parser.add_argument("--gap_limit", type=float, default=0.03, help="Relative gap threshold (e.g., 0.03 for 3%%). Default: 0.03")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    instance_path = os.path.join(base_dir, '..', 'data', 'instances_milp', f"{args.instance}_{args.mode}.txt")
    
    # Verify file exists
    if not os.path.exists(instance_path):
        print(f" File not found: {instance_path}")
        print("Run json_to_milp.py first to create the MILP instance")
        sys.exit(1)
    
    print(f"Running MILP model on {instance_path}")
    print(f"Mode: {args.mode}, Time limit: {'none' if args.time_limit == 0 else f'{args.time_limit}s'}")
    if args.gap_limit is not None and args.gap_limit > 0:
        print(f"Gap criterion: {args.gap_limit*100:.2f}% (mipgap)")
    
    main(instance_path, time_limit=args.time_limit, gap_limit=(args.gap_limit if (args.gap_limit or 0) > 0 else None))
