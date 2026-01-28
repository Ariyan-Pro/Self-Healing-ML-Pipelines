"""
Concept Shift Simulator for ML pipeline testing
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import json
from datetime import datetime

class ConceptShiftSimulator:
    """Simulates concept shift (relationship changes) in ML data"""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        self.results = []
        
    def generate_synthetic_data(self, n_samples=1000, n_features=10):
        """Generate synthetic classification data"""
        X = np.random.randn(n_samples, n_features)
        
        # Create meaningful coefficients
        coefficients = np.random.randn(n_features)
        coefficients = coefficients / np.linalg.norm(coefficients)
        
        # Generate labels with some noise
        logits = X @ coefficients
        probabilities = 1 / (1 + np.exp(-logits))
        y = (probabilities > 0.5).astype(int)
        
        # Add some label noise
        noise_mask = np.random.random(n_samples) < 0.1
        y[noise_mask] = 1 - y[noise_mask]
        
        return X, y, coefficients
    
    def induce_concept_shift(self, X, coefficients, shift_type='abrupt', magnitude=0.3):
        """Induce concept shift by changing feature relationships"""
        n_samples, n_features = X.shape
        
        if shift_type == 'abrupt':
            # Abrupt change: flip signs of some coefficients
            new_coefficients = coefficients.copy()
            # Fix: Use len(new_coefficients) instead of n_features
            new_coefficients[:int(len(new_coefficients) * magnitude)] *= -1
            
        elif shift_type == 'gradual':
            # Gradual change: interpolate between original and flipped
            new_coefficients = coefficients.copy()
            flip_indices = np.random.choice(
                n_features, 
                size=int(n_features * magnitude), 
                replace=False
            )
            for idx in flip_indices:
                new_coefficients[idx] = -coefficients[idx] * magnitude
        
        elif shift_type == 'recurring':
            # Recurring: oscillate between patterns
            new_coefficients = coefficients.copy()
            oscillation = np.sin(np.arange(n_features) * 0.5) * magnitude
            new_coefficients *= (1 + oscillation)
        
        # Generate new labels with shifted concept
        logits = X @ new_coefficients
        probabilities = 1 / (1 + np.exp(-logits))
        y_shifted = (probabilities > 0.5).astype(int)
        
        # Add noise
        noise_mask = np.random.random(n_samples) < 0.1
        y_shifted[noise_mask] = 1 - y_shifted[noise_mask]
        
        return X, y_shifted, new_coefficients
    
    def train_model(self, X_train, y_train):
        """Train a simple classifier"""
        model = LogisticRegression(random_state=self.seed, max_iter=1000)
        model.fit(X_train, y_train)
        return model
    
    def evaluate_shift(self, model, X_test, y_original, y_shifted):
        """Evaluate impact of concept shift"""
        # Predictions on original and shifted data
        y_pred_original = model.predict(X_test)
        y_pred_shifted = model.predict(X_test)
        
        # Calculate metrics
        acc_original = accuracy_score(y_original, y_pred_original)
        acc_shifted = accuracy_score(y_shifted, y_pred_shifted)
        
        accuracy_drop = acc_original - acc_shifted
        
        # Calculate drift score (difference in prediction distributions)
        drift_score = np.abs(y_pred_original - y_pred_shifted).mean()
        
        return {
            'accuracy_original': acc_original,
            'accuracy_shifted': acc_shifted,
            'accuracy_drop': accuracy_drop,
            'drift_score': drift_score,
            'shift_detected': accuracy_drop > 0.1  # Threshold for detection
        }
    
    def run_experiment(self, shift_type='abrupt', magnitude=0.3):
        """Run a complete concept shift experiment"""
        print(f"\n🔍 Running Concept Shift Experiment:")
        print(f"  Type: {shift_type}")
        print(f"  Magnitude: {magnitude}")
        
        # 1. Generate data
        X, y_original, coefficients = self.generate_synthetic_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_original, test_size=0.3, random_state=self.seed
        )
        
        # 2. Train initial model
        start_time = time.time()
        model = self.train_model(X_train, y_train)
        training_time = (time.time() - start_time) * 1000  # ms
        
        # 3. Induce concept shift in test data
        X_test_shifted, y_test_shifted, _ = self.induce_concept_shift(
            X_test, coefficients, shift_type, magnitude
        )
        
        # 4. Evaluate shift
        start_time = time.time()
        evaluation = self.evaluate_shift(model, X_test, y_test, y_test_shifted)
        evaluation_time = (time.time() - start_time) * 1000  # ms
        
        # 5. Record results
        result = {
            'shift_type': shift_type,
            'magnitude': magnitude,
            'training_time_ms': training_time,
            'evaluation_time_ms': evaluation_time,
            **evaluation
        }
        
        self.results.append(result)
        
        # 6. Print results
        print(f"📊 Results:")
        print(f"  Accuracy drop: {result['accuracy_drop']:.3f}")
        print(f"  Drift score: {result['drift_score']:.3f}")
        print(f"  Shift detected: {result['shift_detected']}")
        print(f"  Training time: {training_time:.2f} ms")
        print(f"  Evaluation time: {evaluation_time:.2f} ms")
        
        return result
    
    def run_comprehensive_test(self):
        """Run multiple concept shift scenarios"""
        print("🧪 Running comprehensive concept shift experiments...")
        
        scenarios = [
            ('abrupt', 0.1),
            ('abrupt', 0.3),
            ('abrupt', 0.5),
            ('gradual', 0.2),
            ('gradual', 0.4),
            ('recurring', 0.3)
        ]
        
        for shift_type, magnitude in scenarios:
            try:
                self.run_experiment(shift_type, magnitude)
            except Exception as e:
                print(f"❌ Error in scenario ({shift_type}, {magnitude}): {e}")
                # Continue with other scenarios
                continue
        
        return self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate comprehensive experiment report"""
        if not self.results:
            print("⚠️  No results to summarize")
            return None
        
        print("\n" + "="*60)
        print("📈 CONCEPT SHIFT EXPERIMENT SUMMARY")
        print("="*60)
        
        # Calculate statistics
        detected = [r for r in self.results if r['shift_detected']]
        not_detected = [r for r in self.results if not r['shift_detected']]
        
        print(f"\nDetection Performance:")
        print(f"  Shifts detected: {len(detected)}/{len(self.results)} scenarios")
        
        if detected:
            avg_accuracy_drop = np.mean([r['accuracy_drop'] for r in detected])
            avg_drift_score = np.mean([r['drift_score'] for r in detected])
            avg_eval_time = np.mean([r['evaluation_time_ms'] for r in detected])
            
            print(f"\n📊 Metrics (detected scenarios):")
            print(f"  Average accuracy drop: {avg_accuracy_drop:.3f}")
            print(f"  Average drift score: {avg_drift_score:.3f}")
            print(f"  Average evaluation time: {avg_eval_time:.2f} ms")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"concept_shift_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'scenarios': len(self.results),
                'detected': len(detected),
                'results': self.results,
                'summary': {
                    'detection_rate': len(detected)/len(self.results) if self.results else 0,
                    'avg_accuracy_drop': float(avg_accuracy_drop) if detected else 0,
                    'avg_drift_score': float(avg_drift_score) if detected else 0
                }
            }, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to: {output_file}")
        
        return {
            'detection_rate': len(detected)/len(self.results) if self.results else 0,
            'avg_accuracy_drop': avg_accuracy_drop if detected else 0,
            'avg_drift_score': avg_drift_score if detected else 0,
            'output_file': output_file
        }

def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run concept shift experiments')
    parser.add_argument('--scenario', type=str, default='comprehensive',
                       choices=['abrupt', 'gradual', 'recurring', 'comprehensive'],
                       help='Concept shift scenario to run')
    parser.add_argument('--magnitude', type=float, default=0.3,
                       help='Shift magnitude (0.1 to 0.5)')
    
    args = parser.parse_args()
    
    simulator = ConceptShiftSimulator()
    
    if args.scenario == 'comprehensive':
        print("🧪 Running comprehensive concept shift experiments...")
        simulator.run_comprehensive_test()
    else:
        print(f"🧪 Running {args.scenario} concept shift experiment...")
        simulator.run_experiment(args.scenario, args.magnitude)

if __name__ == "__main__":
    main()
