import numpy as np
import pandas as pd
import time
from scipy import stats
import json
from datetime import datetime
import sys

# Mock implementations for standalone testing
class MockDataDriftDetector:
    def detect_drift(self, ref_data, obs_data):
        return {
            'drift_detected': np.random.random() > 0.5,
            'drift_score': np.random.uniform(0, 0.5),
            'p_value': np.random.uniform(0, 1)
        }

class MockPolicyEngine:
    def __init__(self, config_path="configs/healing_policies.yaml"):
        self.config_path = config_path
    
    def evaluate(self, state):
        actions = ['retrain', 'rollback', 'fallback', 'none']
        return {
            'action': np.random.choice(actions, p=[0.4, 0.2, 0.3, 0.1]),
            'policy': 'mock_policy',
            'confidence': np.random.uniform(0.5, 1.0)
        }

class MockHealingActions:
    def __init__(self):
        pass
    
    def execute_action(self, action, state):
        time.sleep(0.01)  # Simulate work
        return {
            'status': 'success',
            'action': action,
            'execution_time': 0.01,
            'details': f'Mock {action} executed'
        }

# Use the mock implementations
DataDriftDetector = MockDataDriftDetector
PolicyEngine = MockPolicyEngine
HealingActions = MockHealingActions

class SyntheticDriftExperiment:
    """Generates and tests synthetic drift scenarios"""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        
        # Initialize components
        self.drift_detector = DataDriftDetector()
        self.policy_engine = PolicyEngine()
        self.healing = HealingActions()
        
        # Metrics tracking
        self.metrics = {
            'detection_latency': [],
            'decision_times': [],
            'healing_times': [],
            'actions_taken': [],
            'states_recorded': []
        }
    
    def generate_normal_data(self, n_samples=1000, n_features=5):
        """Generate reference (normal) data"""
        data = np.random.randn(n_samples, n_features)
        columns = [f'feature_{i}' for i in range(n_features)]
        return pd.DataFrame(data, columns=columns)
    
    def inject_drift(self, data, drift_type='covariate', severity=0.3):
        """Inject synthetic drift into data"""
        drifted = data.copy()
        
        if drift_type == 'covariate':
            # Shift mean of first feature
            drifted['feature_0'] += severity * np.random.randn(len(data))
        
        elif drift_type == 'concept':
            # Change relationship between features
            drifted['feature_1'] = 0.7 * drifted['feature_0'] + 0.3 * np.random.randn(len(data))
        
        elif drift_type == 'gradual':
            # Gradual drift over time
            for i in range(len(data)):
                if i > len(data) * 0.5:  # Start drift halfway
                    alpha = (i - len(data) * 0.5) / (len(data) * 0.5)
                    drifted.iloc[i, 0] += severity * alpha
        
        return drifted
    
    def run_drift_scenario(self, drift_type, severity=0.3):
        """Run complete drift detection and healing scenario"""
        
        # 1. Generate data
        print(f"\n{'='*60}")
        print(f"Running {drift_type} drift scenario (severity: {severity})")
        print('='*60)
        
        reference_data = self.generate_normal_data()
        drifted_data = self.inject_drift(reference_data, drift_type, severity)
        
        # 2. Detect drift (measure latency)
        start_time = time.time()
        drift_results = self.drift_detector.detect_drift(
            reference_data, 
            drifted_data
        )
        detection_latency = (time.time() - start_time) * 1000  # ms
        
        # 3. Make decision
        state = {
            'drift_score': drift_results.get('drift_score', 0),
            'accuracy_drop': np.random.uniform(0.1, 0.25) if drift_results['drift_detected'] else 0.05,
            'anomaly_rate': np.random.uniform(0.02, 0.1) if drift_results['drift_detected'] else 0.01,
            'time_since_last_action': 60  # minutes
        }
        
        start_time = time.time()
        decision = self.policy_engine.evaluate(state)
        decision_time = (time.time() - start_time) * 1000  # ms
        
        # 4. Execute healing
        start_time = time.time()
        healing_result = self.healing.execute_action(decision['action'], state)
        healing_time = (time.time() - start_time) * 1000  # ms
        
        # 5. Record metrics
        self.metrics['detection_latency'].append(detection_latency)
        self.metrics['decision_times'].append(decision_time)
        self.metrics['healing_times'].append(healing_time)
        self.metrics['actions_taken'].append(decision['action'])
        self.metrics['states_recorded'].append(state)
        
        # 6. Report results
        print(f"\n📊 Results:")
        print(f"  Detection: Drift {'DETECTED' if drift_results['drift_detected'] else 'not detected'}")
        print(f"  Drift Score: {drift_results.get('drift_score', 0):.3f}")
        print(f"  Decision: {decision['action']} (policy: {decision.get('policy', 'unknown')})")
        print(f"  Healing: {healing_result['status']}")
        print(f"\n⏱️  Timings (ms):")
        print(f"  Detection: {detection_latency:.2f}")
        print(f"  Decision: {decision_time:.2f}")
        print(f"  Healing: {healing_time:.2f}")
        
        return {
            'drift_type': drift_type,
            'severity': severity,
            'drift_detected': drift_results['drift_detected'],
            'drift_score': drift_results.get('drift_score', 0),
            'decision': decision,
            'healing_result': healing_result,
            'timings': {
                'detection_ms': detection_latency,
                'decision_ms': decision_time,
                'healing_ms': healing_time
            }
        }
    
    def run_all_scenarios(self):
        """Run comprehensive test suite"""
        scenarios = [
            ('covariate', 0.2),
            ('covariate', 0.4),
            ('concept', 0.3),
            ('gradual', 0.25),
            ('covariate', 0.1)  # Below threshold
        ]
        
        results = []
        for drift_type, severity in scenarios:
            result = self.run_drift_scenario(drift_type, severity)
            results.append(result)
        
        # Generate summary report
        self._generate_summary_report(results)
        return results
    
    def _generate_summary_report(self, results):
        """Generate comprehensive experiment report"""
        print(f"\n{'='*60}")
        print("📈 EXPERIMENT SUMMARY")
        print('='*60)
        
        detected = [r for r in results if r['drift_detected']]
        not_detected = [r for r in results if not r['drift_detected']]
        
        print(f"\nDetection Performance:")
        print(f"  Drift detected: {len(detected)}/{len(results)} scenarios")
        print(f"  False negatives: {len(not_detected)}")
        
        if detected:
            avg_detection = np.mean([r['timings']['detection_ms'] for r in detected])
            avg_decision = np.mean([r['timings']['decision_ms'] for r in detected])
            avg_healing = np.mean([r['timings']['healing_ms'] for r in detected])
            
            print(f"\nAverage Timings (detected scenarios):")
            print(f"  Detection: {avg_detection:.2f} ms")
            print(f"  Decision: {avg_decision:.2f} ms")
            print(f"  Healing: {avg_healing:.2f} ms")
            print(f"  Total: {avg_detection + avg_decision + avg_healing:.2f} ms")
        
        # Action distribution
        actions = {}
        for r in results:
            action = r['decision']['action']
            actions[action] = actions.get(action, 0) + 1
        
        print(f"\nAction Distribution:")
        for action, count in actions.items():
            print(f"  {action}: {count} scenarios")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"experiment_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'scenarios': len(results),
                'detected': len(detected),
                'results': results,
                'metrics_summary': {
                    'avg_detection_ms': float(avg_detection) if detected else 0,
                    'avg_decision_ms': float(avg_decision) if detected else 0,
                    'avg_healing_ms': float(avg_healing) if detected else 0,
                    'action_distribution': actions
                }
            }, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to: {output_file}")

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run synthetic drift experiments')
    parser.add_argument('--scenario', type=str, default='all',
                       choices=['covariate', 'concept', 'gradual', 'all'],
                       help='Drift scenario to run')
    parser.add_argument('--severity', type=float, default=0.3,
                       help='Drift severity (0.1 to 0.5)')
    
    args = parser.parse_args()
    
    experiment = SyntheticDriftExperiment()
    
    if args.scenario == 'all':
        print("🧪 Running all drift scenarios...")
        experiment.run_all_scenarios()
    else:
        print(f"🧪 Running {args.scenario} drift scenario...")
        experiment.run_drift_scenario(args.scenario, args.severity)
