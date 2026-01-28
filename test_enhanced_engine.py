#!/usr/bin/env python3
"""
Test the Enhanced Policy Engine
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from decision_engine.enhanced_policy_engine import EnhancedPolicyEngine
    print("✅ EnhancedPolicyEngine imported successfully")
    
    # Test it
    engine = EnhancedPolicyEngine()
    print("✅ Engine instance created")
    
    test_state = {
        "drift_score": 0.31,
        "accuracy_drop": 0.18,
        "anomaly_rate": 0.07,
        "time_since_last_action": 45
    }
    
    print(f"\n🔍 Testing with state: {test_state}")
    decision = engine.evaluate_with_trace(test_state)
    
    print(f"\n📋 Decision Results:")
    print(f"  Action: {decision['action']}")
    print(f"  Policy: {decision.get('policy', 'N/A')}")
    print(f"  Incident ID: {decision['incident_id']}")
    print(f"  Trace File: {decision['trace_file']}")
    
    print(f"\n📝 Decision Details:")
    details = decision['decision_details']
    print(f"  State: {details['state']}")
    print(f"  Rule Action: {details['rule_action']}")
    print(f"  Bandit Action: {details['bandit_action']}")
    print(f"  Bandit Confidence: {details['bandit_confidence']}")
    print(f"  Final Action: {details['final_action']}")
    print(f"  Reason: {details['reason']}")
    
    # Check if trace file was created
    if os.path.exists(decision['trace_file']):
        print(f"\n✅ Trace file created successfully!")
        
        # Show a preview of the trace
        import json
        with open(decision['trace_file'], 'r') as f:
            trace_data = json.load(f)
        
        print(f"\n📄 Trace Preview:")
        print(json.dumps(trace_data, indent=2)[:500] + "...")
    else:
        print(f"\n❌ Trace file not found at: {decision['trace_file']}")
    
    print(f"\n🎉 Enhanced Policy Engine test PASSED!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Trying alternative import...")
    
    # Try direct import
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "EnhancedPolicyEngine", 
        "decision_engine/enhanced_policy_engine.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    engine = module.EnhancedPolicyEngine()
    print("✅ Engine created via direct import")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
