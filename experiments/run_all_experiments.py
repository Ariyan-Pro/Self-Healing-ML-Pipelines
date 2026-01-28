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
        print("\n" + "="*70)
        print("ðŸ§ª CONCEPT SHIFT EXPERIMENTS")
        print("="*70)
        
        start_time = time.time()
        experiment = ConceptShiftSimulator()
        results = experiment.run_comprehensive_test()
        
        # Save results
        concept_file = os.path.join(self.results_dir, "concept_shift.json")
        with open(concept_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        elapsed = time.time() - start_time
        
        self.all_results['concept_shift'] = {
            'file': concept_file,
            'elapsed_seconds': elapsed,
            'num_scenarios': len(results)
        }
        self.all_results['metadata']['experiments'].append('concept_shift')
        
        print(f"âœ… Concept shift experiments completed in {elapsed:.2f}s")
        return results
    
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
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*70)
        print("ðŸ“Š COMPREHENSIVE EXPERIMENT SUMMARY")
        print("="*70)
        
        # Calculate overall statistics
        total_scenarios = 0
        total_detected = 0
        detection_times = []
        
        # Collect from all experiments
        experiments_data = []
        
        # Synthetic drift stats
        if self.all_results['synthetic_drift']:
            drift_file = self.all_results['synthetic_drift']['file']
            with open(drift_file, 'r') as f:
                drift_data = json.load(f)
            
            drift_scenarios = len(drift_data)
            drift_detected = sum(1 for r in drift_data if r['drift_detected'])
            total_scenarios += drift_scenarios
            total_detected += drift_detected
            
            experiments_data.append({
                'name': 'Synthetic Drift',
                'scenarios': drift_scenarios,
                'detected': drift_detected,
                'detection_rate': drift_detected / drift_scenarios if drift_scenarios else 0
            })
        
        # Concept shift stats
        if self.all_results['concept_shift']:
            concept_file = self.all_results['concept_shift']['file']
            with open(concept_file, 'r') as f:
                concept_data = json.load(f)
            
            concept_scenarios = concept_data.get('scenarios', 0)
            concept_detected = concept_data.get('detected', 0)
            total_scenarios += concept_scenarios
            total_detected += concept_detected
            
            experiments_data.append({
                'name': 'Concept Shift',
                'scenarios': concept_scenarios,
                'detected': concept_detected,
                'detection_rate': concept_detected / concept_scenarios if concept_scenarios else 0
            })
        
        # Noise injection stats
        if self.all_results['noise_injection']:
            noise_file = self.all_results['noise_injection']['file']
            with open(noise_file, 'r') as f:
                noise_data = json.load(f)
            
            noise_scenarios = len(noise_data.get('results', []))
            noise_detected = noise_data.get('summary', {}).get('detected', 0)
            total_scenarios += noise_scenarios
            total_detected += noise_detected
            
            experiments_data.append({
                'name': 'Noise Injection',
                'scenarios': noise_scenarios,
                'detected': noise_detected,
                'detection_rate': noise_detected / noise_scenarios if noise_scenarios else 0
            })
        
        # Print summary
        print(f"\nðŸ“ˆ Overall Performance:")
        print(f"  Total scenarios: {total_scenarios}")
        print(f"  Total detected: {total_detected}")
        print(f"  Overall detection rate: {total_detected/total_scenarios*100:.1f}%" if total_scenarios else "N/A")
        
        print(f"\nðŸ“‹ Experiment Breakdown:")
        for exp in experiments_data:
            print(f"  {exp['name']}:")
            print(f"    Scenarios: {exp['scenarios']}")
            print(f"    Detected: {exp['detected']}")
            print(f"    Detection rate: {exp['detection_rate']*100:.1f}%")
        
        # System capabilities summary
        print(f"\nðŸŽ¯ System Capabilities Demonstrated:")
        capabilities = [
            "âœ“ Covariate drift detection (feature distribution shifts)",
            "âœ“ Concept shift detection (relationship changes)",
            "âœ“ Noise/anomaly detection (various noise types)",
            "âœ“ Adaptive decision making (policy-based + bandit)",
            "âœ“ Healing action execution (retrain/rollback/fallback)",
            "âœ“ Performance monitoring (latency, accuracy tracking)",
            "âœ“ Safe fallback guarantees (deterministic safety)"
        ]
        
        for capability in capabilities:
            print(f"  {capability}")
        
        # Save comprehensive report
        summary_file = os.path.join(self.results_dir, "COMPREHENSIVE_SUMMARY.md")
        
        with open(summary_file, 'w') as f:
            f.write("# Self-Healing ML Pipeline: Experiment Summary\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**System Version:** v0.1-safe-autonomy\n\n")
            
            f.write("## Overall Performance\n\n")
            f.write(f"- **Total Scenarios:** {total_scenarios}\n")
            f.write(f"- **Total Detected:** {total_detected}\n")
            if total_scenarios:
                f.write(f"- **Detection Rate:** {total_detected/total_scenarios*100:.1f}%\n\n")
            
            f.write("## Experiment Details\n\n")
            for exp in experiments_data:
                f.write(f"### {exp['name']}\n")
                f.write(f"- Scenarios: {exp['scenarios']}\n")
                f.write(f"- Detected: {exp['detected']}\n")
                f.write(f"- Detection Rate: {exp['detection_rate']*100:.1f}%\n\n")
            
            f.write("## System Capabilities\n\n")
            for capability in capabilities:
                f.write(f"- {capability}\n")
            
            f.write("\n## Files Generated\n\n")
            for root, dirs, files in os.walk(self.results_dir):
                for file in files:
                    if file.endswith('.json') or file.endswith('.png'):
                        f.write(f"- {file}\n")
        
        # Save unified results
        unified_file = os.path.join(self.results_dir, "unified_results.json")
        with open(unified_file, 'w') as f:
            json.dump(self.all_results, f, indent=2, default=str)
        
        print(f"\nðŸ“ All results saved to directory: {self.results_dir}/")
        print(f"ðŸ“„ Summary report: {summary_file}")
        print(f"ðŸ“Š Unified results: {unified_file}")
        
        return self.all_results
    
    def run_all(self):
        """Run all experiments"""
        print("ðŸš€ Starting comprehensive experiments for Self-Healing ML Pipeline")
        print(f"ðŸ“… Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            self.run_synthetic_drift()
            self.run_concept_shift()
            self.run_noise_injection()
            
            final_results = self.generate_summary_report()
            
            print("\n" + "="*70)
            print("âœ… ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
            print("="*70)
            print(f"\nðŸŽ¯ Next steps:")
            print("  1. Review the generated reports")
            print("  2. Examine the JSON files for detailed results")
            print("  3. Use decision traces for audit and analysis")
            print(f"\nðŸ“ Results directory: {self.results_dir}/")
            
            return final_results
            
        except Exception as e:
            print(f"\nâŒ Error during experiments: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run comprehensive experiments for Self-Healing ML Pipeline'
    )
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['drift', 'concept', 'noise', 'all'],
                       help='Specific experiment to run')
    
    args = parser.parse_args()
    
    runner = ComprehensiveExperimentRunner()
    
    if args.experiment == 'all':
        runner.run_all()
    elif args.experiment == 'drift':
        runner.run_synthetic_drift()
        runner.generate_summary_report()
    elif args.experiment == 'concept':
        runner.run_concept_shift()
        runner.generate_summary_report()
    elif args.experiment == 'noise':
        runner.run_noise_injection()
        runner.generate_summary_report()
