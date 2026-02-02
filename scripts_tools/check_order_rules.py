def check_order_rules(sequence, order_rules):
    """
    Check if a sequence respects all order rules.
    
    Args:
        sequence (list): sequence of labels (e.g., ['A', 'B', ...])
        order_rules (list): list of order rules (JSON format)
    
    Returns:
        int: number of violations
    """
    violations = 0
    for rule in order_rules:
        cond = rule.get('condition', '').lower()
        rule_if = rule.get('if', [])
        if len(rule_if) == 2:
            n1, n2 = rule_if[0], rule_if[1]
            if n1 in sequence and n2 in sequence:
                idx_n1 = sequence.index(n1)
                idx_n2 = sequence.index(n2)
                if 'avant' in cond or 'before' in cond:
                    if idx_n1 > idx_n2:
                        print(f"[CHECK] Rule violated: {n1} must be before {n2} (idx {idx_n1} > {idx_n2})")
                        violations += 1
                if 'apres' in cond or 'after' in cond:
                    if idx_n1 < idx_n2:
                        print(f"[CHECK] Rule violated: {n1} must be after {n2} (idx {idx_n1} < {idx_n2})")
                        violations += 1
    return violations


if __name__ == "__main__":
    # Example usage
    sequence = ['A', 'B', 'D', 'C', 'E', 'F', 'G', 'I', 'H', 'J']
    order_rules = [
        {"if": ["J", "H"], "condition": "J before H"},
        {"if": ["E", "D"], "condition": "E before D"},
    ]
    violations = check_order_rules(sequence, order_rules)
    print(f"Total violations: {violations}")
