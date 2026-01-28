import sys
import io

# Force UTF-8 encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Your existing code continues here...

# experiments/run_all_experiments.py
#!/usr/bin/env python3
"""
Comprehensive experiment runner for the Self-Healing ML Pipeline.
Runs all synthetic experiments and generates unified report.
"""

import sys
import os
import json
from datetime import datetime
import time

# Import experiment modules
from synthetic_drift import SyntheticDriftExperiment
from concept_shift_simulator import ConceptShiftSimulator
from noise_injection import NoiseInjectionExperiment

class ComprehensiveExperimentRunner:
    """Runs all experiments and generates unified report"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"experiment_results_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.all_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'system': 'Self-Healing ML Pipeline v0.1',
                'experiments': []
            },
            'synthetic_drift': None,
            'concept_shift': None,
            'noise_injection': None,
            'summary': {}
        }
    
    def run_synthetic_drift(self):
        """Run synthetic drift experiments"""
        print("\n" + "="*70)
        print("ðŸ§ª SYNTHETIC DRIFT EXPERIMENTS")
        print("="*70)
        
        start_time = time.time()
        experiment = SyntheticDriftExperiment()
        results = experiment.run_all_scenarios()
        
        # Save results
        drift_file = os.path.join(self.results_dir, "synthetic_drift.json")
        with open(drift_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        elapsed = time.time() - start_time
        
        self.all_results['synthetic_drift'] = {
            'file': drift_file,
            'elapsed_seconds': elapsed,
            'num_scenarios': len(results)
        }
        self.all_results['metadata']['experiments'].append('synthetic_drift')
        
        print(f"âœ… Synthetic drift experiments completed in {elapsed:.2f}s")
        return results
    
    def run_concept_shift(self):
        """Run concept shift experiments"""
        print(f"\n🧪 Running concept shift experiments...")
        
        try:
            from concept_shift_simulator import ConceptShiftSimulator
            experiment = ConceptShiftSimulator(seed=self.seed)
            results = experiment.run_comprehensive_test()
            return results
        except Exception as e:
            print(f"❌ Error during concept shift experiments: {e}")
            print("⚠️  Skipping concept shift experiments...")
            return {'error': str(e), 'status': 'failed', 'experiment': 'concept_shift'}
    def run_noise_injection(self):
        """Run noise injection experiments"""
        print("\n" + "="*70)
        print("ðŸ§ª NOISE INJECTION EXPERIMENTS")
        print("="*70)
        
        start_time = time.time()
        experiment = NoiseInjectionExperiment()
        experiment.run_all_scenarios()
        results = experiment.results
        
        # Save results
        noise_file = os.path.join(self.results_dir, "noise_injection.json")
        with open(noise_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        elapsed = time.time() - start_time
        
        self.all_results['noise_injection'] = {
            'file': noise_file,
            'elapsed_seconds': elapsed,
            'num_scenarios': len(results)
        }
        self.all_results['metadata']['experiments'].append('noise_injection')
        
        print(f"âœ… Noise injection experiments completed in {elapsed:.2f}s")
        return results
    

    def run_all(self):
        """Run all experiments"""
        print(f"{'='*60}")
        print(f"🚀 Starting comprehensive experiments for Self-Healing ML Pipeline")
        print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        all_results = {}
        
        # 1. Synthetic Drift Experiments
        try:
            print(f"\n{'='*60}")
            print(f"🧪 SYNTHETIC DRIFT EXPERIMENTS")
            print(f"{'='*60}")
            drift_results = self.run_synthetic_drift()
            all_results['synthetic_drift'] = drift_results
            print(f"✅ Synthetic drift experiments completed in {drift_results.get('total_time', 0):.2f}s")
        except Exception as e:
            print(f"❌ Synthetic drift experiments failed: {e}")
            all_results['synthetic_drift'] = {'error': str(e), 'status': 'failed'}
        
        # 2. Concept Shift Experiments
        try:
            print(f"\n{'='*60}")
            print(f"🧪 CONCEPT SHIFT EXPERIMENTS")
            print(f"{'='*60}")
            concept_results = self.run_concept_shift()
            all_results['concept_shift'] = concept_results
            print(f"✅ Concept shift experiments completed")
        except Exception as e:
            print(f"❌ Concept shift experiments failed: {e}")
            all_results['concept_shift'] = {'error': str(e), 'status': 'failed'}
        
        # 3. Noise Injection Experiments
        try:
            print(f"\n{'='*60}")
            print(f"🧪 NOISE INJECTION EXPERIMENTS")
            print(f"{'='*60}")
            noise_results = self.run_noise_injection()
            all_results['noise_injection'] = noise_results
            print(f"✅ Noise injection experiments completed")
        except Exception as e:
            print(f"❌ Noise injection experiments failed: {e}")
            all_results['noise_injection'] = {'error': str(e), 'status': 'failed'}
        
        # 4. Generate summary report
        final_results = self.generate_summary_report(all_results)
        
        return final_results
    def generate_summary_report(self, results=None):
        """Generate comprehensive experiment summary report"""
        from datetime import datetime
        import json
        
        if results is None:
            results = self.results
        
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE EXPERIMENT SUMMARY")
        print("="*60)
        
        # Safely extract data from results
        if not isinstance(results, dict):
            results = {}
        
        drift_data = results.get('synthetic_drift', {})
        concept_data = results.get('concept_shift', {})
        noise_data = results.get('noise_injection', {})
        
        # Handle different data structures safely
        drift_scenarios = 0
        if isinstance(drift_data, dict) and 'results' in drift_data:
            drift_scenarios = len(drift_data.get('results', []))
        
        concept_scenarios = 0
        if isinstance(concept_data, dict) and 'results' in concept_data:
            concept_scenarios = len(concept_data.get('results', []))
        
        noise_scenarios = 0
        if isinstance(noise_data, dict) and 'results' in noise_data:
            noise_scenarios = len(noise_data.get('results', []))
        
        total_scenarios = drift_scenarios + concept_scenarios + noise_scenarios
        
        print(f"\n📊 Experiment Coverage:")
        print(f"  Synthetic Drift: {drift_scenarios} scenarios")
        print(f"  Concept Shift: {concept_scenarios} scenarios")
        print(f"  Noise Injection: {noise_scenarios} scenarios")
        print(f"  Total: {total_scenarios} scenarios")
        
        # Check for errors
        errors = []
        for exp_name, data in results.items():
            if isinstance(data, dict) and data.get('status') == 'failed':
                errors.append(f"{exp_name}: {data.get('error', 'Unknown error')}")
        
        if errors:
            print(f"\n⚠️  Experiments with issues:")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"\n✅ All experiments completed successfully!")
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"comprehensive_experiment_summary_{timestamp}.json"
        
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'experiments': {
                'synthetic_drift': {
                    'scenarios': drift_scenarios,
                    'has_data': isinstance(drift_data, dict) and 'results' in drift_data
                },
                'concept_shift': {
                    'scenarios': concept_scenarios,
                    'has_data': isinstance(concept_data, dict) and 'results' in concept_data
                },
                'noise_injection': {
                    'scenarios': noise_scenarios,
                    'has_data': isinstance(noise_data, dict) and 'results' in noise_data
                }
            },
            'total_scenarios': total_scenarios,
            'errors': errors
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f"\n📁 Summary saved to: {summary_file}")
        
        return summary_data







def main():
    """Main entry point for the experiment suite"""
    runner = ComprehensiveExperimentRunner()
    print("🚀 Starting Self-Healing ML Pipeline Experiment Suite")
    print("="*60)
    results = runner.run_all()
    print("\n" + "="*60)
    print("✅ Experiment Suite Complete!")
    print(f"Results saved to: comprehensive_experiment_summary_*.json")
    return results

if __name__ == "__main__":
    main()
