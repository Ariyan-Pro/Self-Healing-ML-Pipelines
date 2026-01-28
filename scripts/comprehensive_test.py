# scripts/comprehensive_test.py
"""
Comprehensive test suite for the complete Self-Healing ML Pipelines system.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_module_imports():
    """Test that all critical modules can be imported."""
    print("🧪 Testing module imports...")
    
    modules_to_test = [
        ("utils.config_loader", "ConfigLoader"),
        ("monitoring.data_drift", "DataDriftDetector"),
        ("decision_engine.policy_engine", "PolicyEngine"),
        ("healing.healing_actions", "HealingActions"),
        ("orchestration.controller", "SelfHealingController"),
        ("adaptive.adaptive_controller", "AdaptiveHealingController"),
        ("adaptive_integration", "IntegratedHealingController"),
        ("orchestration.sla_simulator", "SLASimulator"),
        ("orchestration.canary_controller", "CanaryController"),
        ("adaptive.learning.shadow_learner", "ShadowLearner"),
    ]
    
    all_passed = True
    for module_path, class_name in modules_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"  ✅ {module_path}.{class_name}")
        except Exception as e:
            print(f"  ❌ {module_path}.{class_name}: {e}")
            all_passed = False
    
    return all_passed


def test_config_files():
    """Test that all configuration files exist and are valid."""
    print("\n📋 Testing configuration files...")
    
    config_files = [
        "configs/pipeline.yaml",
        "configs/healing_policies.yaml", 
        "configs/cost_model.yaml",
        "configs/sla_config.yaml",
        "configs/canary_config.yaml",
        "adaptive/cost_model/action_costs.yaml"
    ]
    
    all_passed = True
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                # Try to read the file
                with open(config_file, 'r') as f:
                    content = f.read()
                if len(content) > 10:
                    print(f"  ✅ {config_file}")
                else:
                    print(f"  ⚠️  {config_file}: File too small")
                    all_passed = False
            except Exception as e:
                print(f"  ❌ {config_file}: Read error - {e}")
                all_passed = False
        else:
            print(f"  ❌ {config_file}: Missing")
            all_passed = False
    
    return all_passed


def test_data_files():
    """Test that required data files exist."""
    print("\n📊 Testing data files...")
    
    data_files = [
        "adaptive/memory/experiences.json",
        "models/current_model.joblib",
        "models/current_metadata.json",
        "models/fallback/rule_based.joblib"
    ]
    
    all_passed = True
    for data_file in data_files:
        if os.path.exists(data_file):
            size = os.path.getsize(data_file)
            if size > 0:
                print(f"  ✅ {data_file} ({size} bytes)")
            else:
                print(f"  ⚠️  {data_file}: Empty file")
                all_passed = False
        else:
            print(f"  ❌ {data_file}: Missing")
            all_passed = False
    
    return all_passed


def test_system_integration():
    """Test the integrated system components."""
    print("\n🔗 Testing system integration...")
    
    from utils.config_loader import ConfigLoader
    from monitoring.data_drift import DataDriftDetector
    from decision_engine.policy_engine import PolicyEngine
    from healing.healing_actions import HealingActions
    
    all_passed = True
    
    try:
        # Test ConfigLoader
        config = ConfigLoader().load_config()
        if config and 'pipeline' in config:
            print("  ✅ ConfigLoader: Loaded configuration")
        else:
            print("  ❌ ConfigLoader: Failed to load config")
            all_passed = False
        
        # Test DataDriftDetector
        detector = DataDriftDetector()
        drift_result = detector.check_drift([0.1, 0.2, 0.3], [0.15, 0.25, 0.35])
        if drift_result is not None:
            print(f"  ✅ DataDriftDetector: drift_score={drift_result.get('drift_score', 'N/A')}")
        else:
            print("  ❌ DataDriftDetector: Failed")
            all_passed = False
        
        # Test PolicyEngine
        policy_engine = PolicyEngine()
        decision = policy_engine.evaluate({'data_drift': 0.25})
        if decision:
            print(f"  ✅ PolicyEngine: decision={decision.get('action', 'N/A')}")
        else:
            print("  ❌ PolicyEngine: Failed")
            all_passed = False
        
        # Test HealingActions
        healing = HealingActions()
        print(f"  ✅ HealingActions: Initialized")
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        all_passed = False
    
    return all_passed


def test_adaptive_system():
    """Test the adaptive learning components."""
    print("\n🧠 Testing adaptive system...")
    
    all_passed = True
    
    try:
        # Check experiences
        exp_path = "adaptive/memory/experiences.json"
        if os.path.exists(exp_path):
            with open(exp_path, 'r') as f:
                experiences = json.load(f)
            print(f"  ✅ Experiences: {len(experiences)} entries")
        else:
            print("  ⚠️  No experiences file")
            all_passed = False
        
        # Test Adaptive Controller
        from adaptive.adaptive_controller import AdaptiveHealingController
        adaptive = AdaptiveHealingController()
        context = {'data_drift': 0.15, 'anomaly_rate': 0.05}
        decision = adaptive.decide(context)
        if decision:
            print(f"  ✅ AdaptiveController: decision={decision}")
        else:
            print("  ❌ AdaptiveController: Failed")
            all_passed = False
        
        # Test Integrated Controller
        from adaptive_integration import IntegratedHealingController
        integrated = IntegratedHealingController()
        print(f"  ✅ IntegratedController: mode={integrated.mode}")
        
    except Exception as e:
        print(f"  ❌ Adaptive test failed: {e}")
        all_passed = False
    
    return all_passed


def test_phase3_features():
    """Test Phase 3 enterprise features."""
    print("\n🏢 Testing Phase 3 features...")
    
    all_passed = True
    
    try:
        # Test SLA Simulator
        from orchestration.sla_simulator import SLASimulator
        from datetime import datetime, timedelta
        
        sla = SLASimulator()
        outcome = {'performance_change': -0.1, 'recovery_time': 150.0}
        impact = sla.simulate_sla_impact(
            action='retrain',
            outcome=outcome,
            start_time=datetime.now() - timedelta(minutes=5),
            end_time=datetime.now()
        )
        print(f"  ✅ SLASimulator: {impact.get('action', 'N/A')}")
        
        # Test Canary Controller
        from orchestration.canary_controller import CanaryController
        
        canary = CanaryController()
        status = canary.get_rollout_status()
        print(f"  ✅ CanaryController: stage={status.get('current_stage', 'N/A')}")
        
        # Test Shadow Learner
        from adaptive.learning.shadow_learner import ShadowLearner
        
        shadow = ShadowLearner()
        context = {'data_drift': 0.25}
        shadow_exp = shadow.simulate_decision(context, 'retrain', 'fallback')
        print(f"  ✅ ShadowLearner: created experience")
        
        # Test Portfolio Demo
        from portfolio.interview_demo import demonstrate_system_capabilities
        print(f"  ✅ Portfolio demo: import successful")
        
    except Exception as e:
        print(f"  ❌ Phase 3 test failed: {e}")
        all_passed = False
    
    return all_passed


def run_comprehensive_test():
    """Run all comprehensive tests."""
    print("=" * 80)
    print("🧪 COMPREHENSIVE TEST SUITE - SELF-HEALING ML PIPELINES")
    print("=" * 80)
    
    results = {}
    
    # Run all tests
    results['imports'] = test_module_imports()
    results['configs'] = test_config_files()
    results['data'] = test_data_files()
    results['integration'] = test_system_integration()
    results['adaptive'] = test_adaptive_system()
    results['phase3'] = test_phase3_features()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📈 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for production.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review issues above.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
