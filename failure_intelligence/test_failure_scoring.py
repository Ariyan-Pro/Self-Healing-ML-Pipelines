"""Test the failure intelligence system"""

def test_failure_intelligence():
    print("Testing Failure Intelligence — The Crown Jewel")
    print("=" * 60)
    
    # Initialize
    fi = FailureIntelligence()
    scorer = PredictiveFailureScorer()
    fragility = FragilityIndex()
    
    # Log some failures
    print("\n1. Logging failure events...")
    failures = []
    for i in range(5):
        failure = fi.log_failure(
            state_before={'accuracy': 0.8 - (i * 0.1), 'drift': 0.1 + (i * 0.05)},
            action_taken=['retrain', 'rollback', 'fallback'][i % 3],
            outcome=['success', 'partial', 'failure'][i % 3],
            recovery_cost=10.0 + (i * 5.0)
        )
        failures.append(failure)
        print(f"  Logged failure {i+1}: {failure['outcome']} ()")
    
    # Test predictive scoring
    print("\n2. Predictive failure scoring...")
    risk_assessment = scorer.calculate_failure_likelihood(
        model_metrics={'accuracy': 0.85, 'drift_score': 0.15},
        data_quality={'missing_rate': 0.02, 'outlier_rate': 0.05},
        environment={'load_factor': 1.5, 'stability_score': 0.9}
    )
    
    print(f"  Failure likelihood: {risk_assessment['failure_likelihood']:.1%}")
    print(f"  Expected MTTR: {risk_assessment['expected_mttr']:.1f} minutes")
    print(f"  Expected cost: ")
    print(f"  Risk level: {risk_assessment['risk_level']}")
    
    # Test fragility index
    print("\n3. Fragility Index calculation...")
    fragility_result = fragility.calculate_fragility(
        model={'accuracy': 0.82, 'type': 'random_forest'},
        data_pipeline={'stability': 0.9},
        cost_model={'downtime_cost_per_minute': 1000}
    )
    
    print(f"  Fragility Index: {fragility_result['fragility_index']:.2f}")
    print(f"  Margin of safety: {fragility_result['margin_of_safety']:.1%}")
    print(f"  Worst case: {fragility_result['worst_case_scenario']['scenario']}")
    print(f"  Recommendation: {fragility_result['recommendation']}")
    
    print("\n" + "=" * 60)
    print("FAILURE INTELLIGENCE SYSTEM — OPERATIONAL")
    return {
        'failure_dataset_size': len(fi.failure_dataset),
        'risk_assessment': risk_assessment,
        'fragility_index': fragility_result['fragility_index']
    }

if __name__ == "__main__":
    results = test_failure_intelligence()