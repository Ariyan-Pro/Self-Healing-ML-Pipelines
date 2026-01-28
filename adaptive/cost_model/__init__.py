# adaptive/cost_model/__init__.py
"""
Cost model utilities - FOR REPORTING ONLY
Does not affect system behavior
"""

def calculate_potential_business_value(action, context=None):
    """
    Calculate POTENTIAL business value for reporting.
    Does not affect actual decision making.
    """
    # Reference values for reporting
    value_estimates = {
        'fallback': {
            'base': 0.3,
            'sla_benefit': 0.5,
            'customer_satisfaction': 0.2
        },
        'retrain': {
            'base': 1.5,
            'accuracy_recovery': 2.0,
            'long_term_value': 1.0
        },
        'rollback': {
            'base': 1.0,
            'downtime_prevention': 1.5,
            'risk_reduction': 0.8
        }
    }
    
    if action in value_estimates:
        return sum(value_estimates[action].values())
    return 0.0


def generate_cost_analysis_report(experiences):
    """
    Generate report showing system's cost optimization.
    For management presentation only.
    """
    total_cost = sum(exp.get('cost', 0) for exp in experiences)
    avg_cost = total_cost / len(experiences) if experiences else 0
    
    report = {
        "total_experiences": len(experiences),
        "average_cost_per_cycle": round(avg_cost, 3),
        "cost_distribution": {},
        "optimization_notes": []
    }
    
    # Show current optimization
    if avg_cost < 0.8:
        report["optimization_notes"].append(
            "System is operating in low-cost mode (test environment optimal)"
        )
    else:
        report["optimization_notes"].append(
            "System is balancing cost vs. performance"
        )
    
    return report
