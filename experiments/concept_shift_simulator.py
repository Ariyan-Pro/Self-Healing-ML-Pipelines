# experiments/concept_shift_simulator.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import json
from datetime import datetime

class ConceptShiftSimulator:
    """Simulates concept shift and evaluates detection/response"""
    
    def __init__(self, n_samples=1000, n_features=10):
        self.n_samples = n_samples
        self.n_features = n_features
        
        # Generate base data with linear relationship
        np.random.seed(42)
        self.X_base = np.random.randn(n_samples, n_features)
        self.coefficients = np.random.randn(n_features)
        self.y_base = (self.X_base @ self.coefficients + np.random.randn(n_samples) * 0.1 > 0).astype(int)
        
        self.models = {}
        self.results = []
    
    def induce_concept_shift(self, shift_type='abrupt', shift_time=0.5, magnitude=0.5):
        """Induce concept shift in the data"""
        
        X_shifted = self.X_base.copy()
        y_shifted = self.y_base.copy()
        
        split_point = int(self.n_samples * shift_time)
        
        if shift_type == 'abrupt':
            # Suddenly change coefficients
            new_coefficients = self.coefficients.copy()
            new_coefficients[:int(n_features * magnitude)] *= -1  # Flip signs
            
            # Generate new labels for shifted portion
            y_new = (X_shifted[split_point:] @ new_coefficients + 
                    np.random.randn(self.n_samples - split_point) * 0.1 > 0).astype(int)
            y_shifted[split_point:] = y_new
            
        elif shift_type == 'gradual':
            # Gradually change coefficients over time
            for i in range(split_point, self.n_samples):
                progress = (i - split_point) / (self.n_samples - split_point)
                mixed_coefficients = (1 - progress) * self.coefficients + progress * (-self.coefficients)
                y_shifted[i] = (X_shifted[i] @ mixed_coefficients + 
                               np.random.randn() * 0.1 > 0).astype(int)
        
        elif shift_type == 'recurring':
            # Cyclical concept shift
            y_shifted = self.y_base.copy()
            cycle_length = 200
            for i in range(self.n_samples):
                cycle_pos = (i % cycle_length) / cycle_length
                if cycle_pos > 0.5:
                    # Second half of cycle uses inverted relationship
                    y_shifted[i] = 1 - self.y_base[i]
        
        return X_shifted, y_shifted
    
    def evaluate_detection(self, X_train, y_train, X_test, y_test, window_size=100):
        """Evaluate concept shift detection using accuracy degradation"""
        
        # Train initial model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train[:window_size], y_train[:window_size])
        
        # Monitor accuracy over sliding windows
        accuracies = []
        detected_shifts = []
        
        for i in range(window_size, len(X_test), window_size//2):
            X_window = X_test[i:i+window_size]
            y_window = y_test[i:i+window_size]
            
            if len(X_window) < window_size:
                break
            
            # Predict and calculate accuracy
            y_pred = model.predict(X_window)
            accuracy = accuracy_score(y_window, y_pred)
            accuracies.append(accuracy)
            
            # Detect shift if accuracy drops significantly
            if len(accuracies) > 1:
                accuracy_drop = accuracies[-2] - accuracy
                if accuracy_drop > 0.15:  # 15% drop threshold
                    detected_shifts.append({
                        'window': i,
                        'accuracy_drop': accuracy_drop,
                        'previous_accuracy': accuracies[-2],
                        'current_accuracy': accuracy
                    })
        
        return {
            'accuracies': accuracies,
            'detected_shifts': detected_shifts,
            'average_accuracy': np.mean(accuracies) if accuracies else 0
        }
    
    def run_experiment(self, shift_type='abrupt', magnitude=0.5):
        """Run complete concept shift experiment"""
        
        print(f"\nðŸ” Running Concept Shift Experiment:")
        print(f"  Type: {shift_type}")
        print(f"  Magnitude: {magnitude}")
        
        # Generate shifted data
        X_shifted, y_shifted = self.induce_concept_shift(
            shift_type=shift_type,
            magnitude=magnitude
        )
        
        # Split into train/test (simulating time)
        split_idx = self.n_samples // 2
        X_train, y_train = X_shifted[:split_idx], y_shifted[:split_idx]
        X_test, y_test = X_shifted[split_idx:], y_shifted[split_idx:]
        
        # Evaluate detection
        results = self.evaluate_detection(X_train, y_train, X_test, y_test)
        
        # Generate report
        n_shifts = len(results['detected_shifts'])
        detection_rate = n_shifts > 0
        
        report = {
            'shift_type': shift_type,
            'magnitude': magnitude,
            'detected': detection_rate,
            'num_shifts_detected': n_shifts,
            'avg_accuracy': results['average_accuracy'],
            'detection_details': results['detected_shifts']
        }
        
        # Print results
        print(f"\nðŸ“Š Results:")
        print(f"  Shift detected: {'YES' if detection_rate else 'NO'}")
        print(f"  Number of shifts: {n_shifts}")
        print(f"  Average accuracy: {results['average_accuracy']:.3f}")
        
        if results['detected_shifts']:
            print(f"\nâš ï¸  Shift Events:")
            for i, shift in enumerate(results['detected_shifts'][:3]):  # Show first 3
                print(f"    Event {i+1}: Drop of {shift['accuracy_drop']:.3f} "
                      f"at window {shift['window']}")
        
        # Plot results
        self._plot_results(results, shift_type)
        
        self.results.append(report)
        return report
    
    def _plot_results(self, results, shift_type):
        """Visualize concept shift detection"""
        plt.figure(figsize=(10, 6))
        
        # Plot accuracy over time
        accuracies = results['accuracies']
        plt.plot(accuracies, label='Model Accuracy', linewidth=2)
        
        # Mark detected shifts
        for shift in results['detected_shifts']:
            window_idx = len(accuracies) // 2  # Approximate position
            plt.axvline(x=window_idx, color='red', alpha=0.5, linestyle='--')
            plt.text(window_idx, 0.5, 'Shift Detected', 
                    rotation=90, alpha=0.7)
        
        plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        plt.xlabel('Time Window')
        plt.ylabel('Accuracy')
        plt.title(f'Concept Shift Detection: {shift_type}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"concept_shift_{shift_type}_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"ðŸ“ˆ Plot saved to: {filename}")
    
    def run_comprehensive_test(self):
        """Run multiple concept shift scenarios"""
        scenarios = [
            ('abrupt', 0.3),
            ('abrupt', 0.7),
            ('gradual', 0.5),
            ('recurring', 0.4),
            ('abrupt', 0.1)  # Minor shift
        ]
        
        all_results = []
        for shift_type, magnitude in scenarios:
            result = self.run_experiment(shift_type, magnitude)
            all_results.append(result)
        
        # Generate summary
        self._generate_summary(all_results)
        return all_results
    
    def _generate_summary(self, results):
        """Generate comprehensive summary report"""
        print(f"\n{'='*60}")
        print("ðŸ“‹ CONCEPT SHIFT EXPERIMENT SUMMARY")
        print('='*60)
        
        detected = [r for r in results if r['detected']]
        not_detected = [r for r in results if not r['detected']]
        
        print(f"\nDetection Performance:")
        print(f"  Shifts detected: {len(detected)}/{len(results)} scenarios")
        
        if detected:
            avg_accuracy = np.mean([r['avg_accuracy'] for r in detected])
            avg_shifts = np.mean([r['num_shifts_detected'] for r in detected])
            print(f"  Average accuracy when detected: {avg_accuracy:.3f}")
            print(f"  Average shifts per scenario: {avg_shifts:.2f}")
        
        print(f"\nScenario Details:")
        for result in results:
            status = "âœ“" if result['detected'] else "âœ—"
            print(f"  {status} {result['shift_type']} (mag={result['magnitude']}): "
                  f"{result['num_shifts_detected']} shifts, acc={result['avg_accuracy']:.3f}")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"concept_shift_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'scenarios': len(results),
                'detected': len(detected),
                'results': results,
                'summary': {
                    'detection_rate': len(detected) / len(results) if results else 0,
                    'avg_accuracy_detected': float(avg_accuracy) if detected else 0,
                    'avg_shifts_per_scenario': float(avg_shifts) if detected else 0
                }
            }, f, indent=2, default=str)
        
        print(f"\nðŸ“ Detailed results saved to: {output_file}")

if __name__ == "__main__":
    simulator = ConceptShiftSimulator()
    
    # Run comprehensive test
    print("ðŸ§ª Running comprehensive concept shift experiments...")
    simulator.run_comprehensive_test()
