#!/usr/bin/env python3
"""
Patch all generated scenario JSONs to ensure recoverability.

Root causes identified:
1. 0% of scenarios have alt_tools_available → change_tool never works
2. 48% have alt_paths=False → bypass blocked on nearly half
3. 46.5% have no_destroy=True → combined with above, many failures unrecoverable

Fix strategy:
- For each failure event, ensure at least one recovery path:
  a) If alt_paths=True → bypass is already feasible ✓
  b) If alt_paths=False → add alt_tools_available OR relax no_destroy
- Add alt_tools_available to ~30-40% of failures (realistic tool diversity)
- Critical path failures with no alt_paths MUST have either alt_tools or
  low-value + no_destroy=False

The fix maintains scenario difficulty levels:
 - simple (1 failure): generous recovery options
 - intermediate (2 failures): mixed recovery
 - complex (3 failures): some hard, but at least 1 recovery path each
"""

import os, sys, json, glob, random, copy
import networkx as nx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..')  # Project root
sys.path.insert(0, BASE)

FAILURES_DIR = os.path.join(BASE, "data/adaptive/generated_failures")
INSTANCES_DIR = os.path.join(BASE, "data/instances_base")
BACKUP_SUFFIX = ".bak"

random.seed(42)  # Reproducibility

# Tool alternatives pool (realistic for disassembly)
ALT_TOOLS_POOL = [
    ["T_pneumatic"],
    ["T_impact"],
    ["T_power"],
    ["T_alt"],
    ["T_electric"],
]


def load_graph(instance_name):
    """Load instance graph to check critical paths."""
    path = os.path.join(INSTANCES_DIR, f"{instance_name}.json")
    if not os.path.exists(path):
        return None, []
    with open(path) as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    for node in data.get("nodes", []):
        if isinstance(node, dict):
            nid = node["id"]
            G.add_node(nid, **{k: v for k, v in node.items() if k != "id"})
        else:
            G.add_node(str(node))
    for edge in data.get("edges", []):
        if isinstance(edge, list) and len(edge) == 2:
            G.add_edge(edge[0], edge[1])
        elif isinstance(edge, dict):
            G.add_edge(edge["from"], edge["to"])
    
    targets = data.get("targets", [])
    return G, targets


def is_on_critical_path(G, comp_id, targets):
    """Check if comp_id is an ancestor of any target."""
    if G is None:
        return True  # Conservative assumption
    for t in targets:
        if t == comp_id:
            return True
        try:
            if nx.has_path(G, comp_id, t):
                return True
        except nx.NodeNotFound:
            continue
    return False


def get_profit_cutoff(G, percentile=20):
    """Get the destroy profit threshold (20th percentile of all profits)."""
    if G is None:
        return 30.0  # Default
    import numpy as np
    profits = [G.nodes[n].get('profit', 0.0) for n in G.nodes]
    return float(np.percentile(profits, percentile)) if profits else 0.0


def patch_failure(fail, G, targets, scenario_type, fail_index, total_fails):
    """
    Patch a single failure event to ensure recoverability.
    
    Returns the patched failure dict.
    """
    ctx = fail.get('context', {})
    alt_paths = ctx.get('alt_paths', False)
    no_destroy = ctx.get('no_destroy', False)
    value_piece = ctx.get('value_piece', 50)
    comp_id = fail.get('comp_id', '')
    
    on_critical = is_on_critical_path(G, comp_id, targets)
    cutoff = get_profit_cutoff(G) if G else 30.0
    
    # Current recoverability assessment
    can_bypass = alt_paths is True
    has_alt_tools = bool(ctx.get('alt_tools_available'))
    can_destroy = (not no_destroy) and (value_piece <= cutoff)
    
    is_recoverable = can_bypass or has_alt_tools or can_destroy
    
    if is_recoverable:
        # Already recoverable, just optionally add alt_tools for realism
        if random.random() < 0.25:  # 25% chance to add alt tools
            ctx['alt_tools_available'] = random.choice(ALT_TOOLS_POOL)
        return fail
    
    # NOT recoverable → must fix
    # Strategy depends on scenario difficulty and position
    
    if on_critical:
        # CRITICAL PATH failure: MUST have at least one recovery option
        
        if scenario_type == 'simple':
            # Simple: generous - add bypass AND tool
            ctx['alt_paths'] = True
            if random.random() < 0.5:
                ctx['alt_tools_available'] = random.choice(ALT_TOOLS_POOL)
        
        elif scenario_type == 'intermediate':
            # Intermediate: one strong option
            choice = random.random()
            if choice < 0.4:
                ctx['alt_paths'] = True
            elif choice < 0.7:
                ctx['alt_tools_available'] = random.choice(ALT_TOOLS_POOL)
            else:
                # Relax destroy constraints
                ctx['no_destroy'] = False
                ctx['value_piece'] = min(value_piece, max(int(cutoff * 0.8), 5))
        
        elif scenario_type == 'complex':
            # Complex: harder but still one option
            choice = random.random()
            if choice < 0.35:
                ctx['alt_paths'] = True
            elif choice < 0.60:
                ctx['alt_tools_available'] = random.choice(ALT_TOOLS_POOL)
            else:
                # Destroy as last resort
                ctx['no_destroy'] = False
                ctx['value_piece'] = min(value_piece, max(int(cutoff * 0.6), 3))
        
        else:
            # Demo/other: add bypass
            ctx['alt_paths'] = True
    
    else:
        # NOT on critical path: can be less strict
        # Non-critical failures don't block targets, but may reduce profit
        # Add a recovery option for realism
        choice = random.random()
        if choice < 0.4:
            ctx['alt_paths'] = True
        elif choice < 0.7:
            ctx['alt_tools_available'] = random.choice(ALT_TOOLS_POOL)
        else:
            ctx['no_destroy'] = False
            ctx['value_piece'] = min(value_piece, max(int(cutoff * 0.7), 5))
    
    fail['context'] = ctx
    return fail


def patch_scenario_file(scenario_path, G, targets, instance_name):
    """Patch all failures in a scenario file."""
    with open(scenario_path) as f:
        failures = json.load(f)
    
    scenario_type = 'simple'
    base = os.path.basename(scenario_path)
    for lvl in ('simple', 'intermediate', 'complex'):
        if lvl in base:
            scenario_type = lvl
            break
    
    patched = []
    for idx, fail in enumerate(failures):
        patched_fail = patch_failure(
            copy.deepcopy(fail), G, targets, 
            scenario_type, idx, len(failures)
        )
        patched.append(patched_fail)
    
    # Backup original
    backup_path = scenario_path + BACKUP_SUFFIX
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            json.dump(failures, f, indent=2)
    
    # Write patched version
    with open(scenario_path, 'w') as f:
        json.dump(patched, f, indent=2)
    
    return len(failures), scenario_type


def main():
    print("=" * 70)
    print("SCENARIO PATCHER: Ensuring recoverability for all failure events")
    print("=" * 70)
    
    stats = {'total_files': 0, 'total_failures': 0, 'patched': 0, 'already_ok': 0}
    
    for inst_dir in sorted(glob.glob(os.path.join(FAILURES_DIR, "*"))):
        if not os.path.isdir(inst_dir):
            continue
        inst_name = os.path.basename(inst_dir)
        
        # Skip dagtest/gearpump demos (hand-crafted, already good)
        if inst_name in ('dagtest', 'gearpump'):
            print(f"  {inst_name:45s} → SKIP (hand-crafted demo)")
            continue
        
        G, targets = load_graph(inst_name)
        
        for scen_file in sorted(glob.glob(os.path.join(inst_dir, "scenario_*.json"))):
            n_fails, stype = patch_scenario_file(scen_file, G, targets, inst_name)
            stats['total_files'] += 1
            stats['total_failures'] += n_fails
            base = os.path.basename(scen_file)
            print(f"  {inst_name:45s} | {base:50s} | {n_fails} failures patched")
    
    print(f"\n{'=' * 70}")
    print(f"  Patched {stats['total_files']} scenario files ({stats['total_failures']} failure events)")
    print(f"  Backups saved as *.json.bak")
    print(f"{'=' * 70}")
    
    # Verification: re-analyze contexts
    print("\n  Verifying patched scenarios...")
    all_ctxs = []
    for inst_dir in sorted(glob.glob(os.path.join(FAILURES_DIR, "*"))):
        if not os.path.isdir(inst_dir):
            continue
        inst_name = os.path.basename(inst_dir)
        if inst_name in ('dagtest', 'gearpump'):
            continue
        for scen_file in sorted(glob.glob(os.path.join(inst_dir, "scenario_*.json"))):
            with open(scen_file) as f:
                failures = json.load(f)
            for fail in failures:
                ctx = fail.get('context', {})
                all_ctxs.append({
                    'alt_paths': ctx.get('alt_paths', False),
                    'alt_tools': bool(ctx.get('alt_tools_available')),
                    'no_destroy': ctx.get('no_destroy', False),
                    'value_piece': ctx.get('value_piece', 0),
                })
    
    total = len(all_ctxs)
    alt_paths_true = sum(1 for c in all_ctxs if c['alt_paths'] is True)
    alt_tools_true = sum(1 for c in all_ctxs if c['alt_tools'])
    no_destroy_true = sum(1 for c in all_ctxs if c['no_destroy'] is True)
    
    # Check recoverability
    recoverable = 0
    for c in all_ctxs:
        if c['alt_paths'] or c['alt_tools'] or (not c['no_destroy'] and c['value_piece'] <= 30):
            recoverable += 1
    
    print(f"\n  AFTER PATCH ({total} failure events):")
    print(f"    alt_paths = True:     {alt_paths_true:3d}/{total} ({alt_paths_true/total*100:.1f}%)")
    print(f"    alt_tools available:  {alt_tools_true:3d}/{total} ({alt_tools_true/total*100:.1f}%)")
    print(f"    no_destroy = True:    {no_destroy_true:3d}/{total} ({no_destroy_true/total*100:.1f}%)")
    print(f"    Recoverable (≥1 option): {recoverable:3d}/{total} ({recoverable/total*100:.1f}%)")


if __name__ == '__main__':
    main()
