#!/usr/bin/env python3
"""
Simple test of Enhanced Policy Engine
"""
import sys
import os

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "EnhancedPolicyEngine", 
    "decision_engine/enhanced_policy_engine.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Test
engine = module.EnhancedPolicyEngine()
print("🧠 Testing Enhanced Policy Engine")
print("="*60)

test_cases = [
    {"drift_score": 0.31, "accuracy_drop": 0.18, "anomaly_rate": 0.07},
    {"drift_score": 0.15, "accuracy_drop": 0.08, "anomaly_rate": 0.12},
    {"drift_score": 0.45, "accuracy_drop": 0.25, "anomaly_rate": 0.03}
]

for i, state in enumerate(test_cases, 1):
    print(f"\n🔍 Test {i}: State = {state}")
    
    try:
        decision = engine.evaluate_with_trace(state)
        print(f"  ✓ Decision made: {decision['action']}")
        print(f"  ✓ Incident ID: {decision['incident_id']}")
        
        if os.path.exists(decision['trace_file']):
            print(f"  ✓ Trace file created")
        else:
            print(f"  ✗ Trace file missing")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")

print(f"\n✅ Test completed!")
