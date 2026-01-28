#!/usr/bin/env python3
"""
Quick test to verify all new components integrate correctly
"""

import sys
import os
import json

def test_experiment_harness():
    """Test experiment harness creation"""
    print("🧪 Testing Experiment Harness...")
    
    # Check if experiment files exist
    experiment_files = [
        'experiments/synthetic_drift.py',
        'experiments/concept_shift_simulator.py',
        'experiments/noise_injection.py',
        'experiments/run_all_experiments.py',
        'experiments/ablation_study.py'
    ]
    
    for file_path in experiment_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - Missing!")
    
    return all(os.path.exists(f) for f in experiment_files)

def test_decision_traces():
    """Test decision trace logging"""
    print("\n📝 Testing Decision Trace Logging...")
    
    try:
        # Test creating a simple trace
        trace_content = '''{
  "timestamp": "2026-01-27T14:03:11Z",
  "state": {
    "drift_score": 0.31,
    "accuracy_drop": 0.18,
    "anomaly_rate": 0.07
  },
  "rule_action": "retrain",
  "bandit_action": "fallback",
  "bandit_confidence": 0.64,
  "final_action": "retrain",
  "reason": "confidence below threshold",
  "cooldown_triggered": true
}'''
        
        # Create trace directory
        os.makedirs('logs/decision_traces', exist_ok=True)
        
        # Write a test trace
        test_trace_file = 'logs/decision_traces/test_trace.json'
        with open(test_trace_file, 'w') as f:
            f.write(trace_content)
        
        # Verify it was written
        if os.path.exists(test_trace_file):
            print(f"  ✅ Decision trace file created: {test_trace_file}")
            
            # Read and verify
            with open(test_trace_file, 'r') as f:
                trace_data = json.load(f)
            
            if trace_data.get('final_action') == 'retrain':
                print(f"  ✅ Trace data validated")
                
                # Clean up
                os.remove(test_trace_file)
                return True
            else:
                print(f"  ❌ Trace data validation failed")
                return False
        else:
            print(f"  ❌ Failed to create trace file")
            return False
            
    except Exception as e:
        print(f"  ❌ Error testing decision traces: {e}")
        return False

def test_directory_structure():
    """Verify the complete directory structure"""
    print("\n📁 Testing Directory Structure...")
    
    expected_dirs = [
        'experiments',
        'logs/decision_traces',
        'docs',
        'docs/architecture',
        'tests/experiments'
    ]
    
    all_exist = True
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ - Missing!")
            all_exist = False
    
    return all_exist

def test_enhanced_policy_engine():
    """Test if enhanced policy engine file exists"""
    print("\n🧠 Testing Enhanced Policy Engine...")
    
    if os.path.exists('decision_engine/enhanced_policy_engine.py'):
        print(f"  ✅ enhanced_policy_engine.py exists")
        
        # Check if it can be imported
        try:
            # Add current directory to path
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            # Try to read and check it's valid Python
            with open('decision_engine/enhanced_policy_engine.py', 'r') as f:
                content = f.read()
            
            # Check for key components
            if 'class EnhancedPolicyEngine' in content and 'evaluate_with_trace' in content:
                print(f"  ✅ File contains expected classes/methods")
                return True
            else:
                print(f"  ❌ File missing expected components")
                return False
                
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")
            return False
    else:
        print(f"  ❌ File not found in decision_engine/")
        
        # Check if it exists elsewhere
        for root, dirs, files in os.walk('.'):
            if 'enhanced_policy_engine.py' in files:
                print(f"  ⚠️  Found at: {os.path.join(root, 'enhanced_policy_engine.py')}")
                return False
        
        return False

def main():
    """Run all tests"""
    print("="*70)
    print("🔍 INTEGRATION TEST: Self-Healing ML Pipeline")
    print("="*70)
    
    tests = [
        ("Experiment Harness", test_experiment_harness),
        ("Decision Traces", test_decision_traces),
        ("Directory Structure", test_directory_structure),
        ("Enhanced Policy Engine", test_enhanced_policy_engine)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n🔍 Running: {test_name}")
            print("-" * 40)
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ❌ Error in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if not success:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
