# experiments/noise_injection.py
import numpy as np
import pandas as pd
from scipy import stats
import time
import json
from datetime import datetime

class NoiseInjectionExperiment:
    """Tests system resilience to various noise types"""
    
    def __init__(self):
        self.noise_types = ['gaussian', 'missing', 'outliers', 'correlated', 'adversarial']
        self.results = []
    
    def generate_clean_data(self, n_samples=1000, n_features=5):
        """Generate clean reference data"""
        np.random.seed(42)
        data = np.random.randn(n_samples, n_features)
        columns = [f'feature_{i}' for i in range(n_features)]
        return pd.DataFrame(data, columns=columns)
    
    def inject_noise(self, data, noise_type, intensity=0.1):
        """Inject specific noise type into data"""
        noisy_data = data.copy().values
        n_samples, n_features = noisy_data.shape
        
        if noise_type == 'gaussian':
            # Add Gaussian noise
            noise = np.random.randn(n_samples, n_features) * intensity
            noisy_data += noise
            
        elif noise_type == 'missing':
            # Randomly set values to NaN
            mask = np.random.random((n_samples, n_features)) < intensity
            noisy_data[mask] = np.nan
            
        elif noise_type == 'outliers':
            # Replace some values with extreme outliers
            n_outliers = int(n_samples * n_features * intensity)
            outlier_indices = np.random.choice(n_samples * n_features, n_outliers, replace=False)
            row_idx, col_idx = np.unravel_index(outlier_indices, (n_samples, n_features))
            noisy_data[row_idx, col_idx] *= 10  # 10x values
            
        elif noise_type == 'correlated':
            # Introduce correlation between features
            for i in range(1, n_features):
                noisy_data[:, i] = (1 - intensity) * noisy_data[:, i] + intensity * noisy_data[:, 0]
                
        elif noise_type == 'adversarial':
            # Adversarial noise targeting specific patterns
            for i in range(n_samples):
                if np.random.random() < intensity:
                    # Flip signs for adversarial samples
                    noisy_data[i] *= -1
        
        return pd.DataFrame(noisy_data, columns=data.columns)
    
    def calculate_anomaly_rate(self, clean_data, noisy_data):
        """Calculate anomaly rate using z-score method"""
        anomalies = 0
        total = 0
        
        for col in clean_data.columns:
            if col in noisy_data.columns:
                clean_vals = clean_data[col].dropna()
                noisy_vals = noisy_data[col].dropna()
                
                if len(clean_vals) > 10 and len(noisy_vals) > 10:
                    # Calculate statistics from clean data
                    mean = clean_vals.mean()
                    std = clean_vals.std()
                    
                    if std > 0:  # Avoid division by zero
                        # Calculate z-scores for noisy data
                        z_scores = np.abs((noisy_vals - mean) / std)
                        anomalies += np.sum(z_scores > 3.0)  # 3-sigma threshold
                        total += len(noisy_vals)
        
        return anomalies / total if total > 0 else 0
    
    def run_noise_scenario(self, noise_type, intensity=0.1):
        """Run single noise injection scenario"""
        print(f"\n{'='*60}")
        print(f"Testing Noise Type: {noise_type.upper()} (intensity: {intensity})")
        print('='*60)
        
        # Generate data
        clean_data = self.generate_clean_data()
        noisy_data = self.inject_noise(clean_data, noise_type, intensity)
        
        # Calculate anomaly rate
        start_time = time.time()
        anomaly_rate = self.calculate_anomaly_rate(clean_data, noisy_data)
        detection_time = (time.time() - start_time) * 1000  # ms
        
        # Simulate system state
        state = {
            'anomaly_rate': anomaly_rate,
            'drift_score': np.random.uniform(0.1, 0.3) if anomaly_rate > 0.05 else 0.05,
            'accuracy_drop': min(0.1 + anomaly_rate * 2, 0.5),  # Anomalies affect accuracy
            'time_since_last_action': 30
        }
        
        # Determine expected action
        if anomaly_rate > 0.05:
            action = 'fallback'  # High anomaly rate triggers fallback
        elif state['drift_score'] > 0.2:
            action = 'retrain'
        else:
            action = 'none'
        
        # Record results
        result = {
            'noise_type': noise_type,
            'intensity': intensity,
            'anomaly_rate': anomaly_rate,
            'detection_time_ms': detection_time,
            'system_state': state,
            'expected_action': action,
            'anomalies_detected': anomaly_rate > 0.05
        }
        
        # Print results
        print(f"\n📊 Results:")
        print(f"  Anomaly rate: {anomaly_rate:.3f}")
        print(f"  Detection time: {detection_time:.2f} ms")
        print(f"  Anomalies detected: {'YES' if anomaly_rate > 0.05 else 'NO'}")
        print(f"  Expected action: {action}")
        print(f"\n📈 System state:")
        for key, value in state.items():
            print(f"    {key}: {value:.3f}")
        
        self.results.append(result)
        return result
    
    def run_all_scenarios(self):
        """Run comprehensive noise injection tests"""
        intensities = [0.05, 0.1, 0.2, 0.3]
        
        for noise_type in self.noise_types:
            for intensity in intensities:
                self.run_noise_scenario(noise_type, intensity)
        
        # Generate summary
        self._generate_summary()
    
    def _generate_summary(self):
        """Generate comprehensive summary report"""
        print(f"\n{'='*60}")
        print("📋 NOISE INJECTION EXPERIMENT SUMMARY")
        print('='*60)
        
        # Calculate statistics
        detected = [r for r in self.results if r['anomalies_detected']]
        not_detected = [r for r in self.results if not r['anomalies_detected']]
        
        print(f"\nDetection Performance:")
        print(f"  Anomalies detected: {len(detected)}/{len(self.results)} scenarios")
        print(f"  Detection rate: {len(detected)/len(self.results)*100:.1f}%")
        
        if detected:
            avg_detection_time = np.mean([r['detection_time_ms'] for r in detected])
            avg_anomaly_rate = np.mean([r['anomaly_rate'] for r in detected])
            print(f"\n📈 Statistics (detected scenarios):")
            print(f"  Average detection time: {avg_detection_time:.2f} ms")
            print(f"  Average anomaly rate: {avg_anomaly_rate:.3f}")
        
        print(f"\n📊 By Noise Type:")
        for noise_type in self.noise_types:
            type_results = [r for r in self.results if r['noise_type'] == noise_type]
            if type_results:
                detected_count = sum(1 for r in type_results if r['anomalies_detected'])
                avg_rate = np.mean([r['anomaly_rate'] for r in type_results])
                print(f"  {noise_type}: {detected_count}/{len(type_results)} detected, "
                      f"avg anomaly rate: {avg_rate:.3f}")
        
        # Action recommendations
        print(f"\n🚨 Action Distribution (expected):")
        actions = {}
        for result in self.results:
            action = result['expected_action']
            actions[action] = actions.get(action, 0) + 1
        
        for action, count in actions.items():
            percentage = count / len(self.results) * 100
            print(f"  {action}: {count} scenarios ({percentage:.1f}%)")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"noise_injection_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'scenarios': len(self.results),
                'detected': len(detected),
                'results': self.results,
                'summary': {
                    'detection_rate': len(detected) / len(self.results) if self.results else 0,
                    'avg_detection_time_ms': float(avg_detection_time) if detected else 0,
                    'avg_anomaly_rate': float(avg_anomaly_rate) if detected else 0,
                    'action_distribution': actions
                }
            }, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    experiment = NoiseInjectionExperiment()
    print("🧪 Running comprehensive noise injection experiments...")
    experiment.run_all_scenarios()