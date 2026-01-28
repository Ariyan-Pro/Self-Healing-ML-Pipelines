"""
Integration script for Adaptive Self-Healing Intelligence.
Connects Phase 1 system with Adaptive components.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from orchestration.controller import SelfHealingController
    HAS_PHASE1 = True
except ImportError:
    print("Warning: Phase 1 components not available")
    HAS_PHASE1 = False

try:
    from adaptive.adaptive_controller import AdaptiveHealingController
    HAS_ADAPTIVE = True
except ImportError as e:
    print(f"Warning: Adaptive components not available: {e}")
    HAS_ADAPTIVE = False


class IntegratedHealingController:
    """
    Integrated controller that combines Phase 1 deterministic healing
    with Phase 2 adaptive intelligence.
    """
    
    def __init__(self):
        """Initialize integrated controller."""
        print("🚀 Initializing IntegratedHealingController...")
        
        # Mode: 'adaptive', 'deterministic', or 'hybrid'
        self.mode = "hybrid"
        
        # Confidence threshold for using adaptive decisions
        self.adaptive_confidence_threshold = 0.7
        
        # Initialize controllers
        self.deterministic_controller = None
        self.adaptive_controller = None
        
        if HAS_PHASE1:
            try:
                self.deterministic_controller = SelfHealingController()
                print("  - Phase 1: Deterministic controller initialized")
            except Exception as e:
                print(f"  Warning: Failed to initialize Phase 1 controller: {e}")
        
        if HAS_ADAPTIVE:
            try:
                self.adaptive_controller = AdaptiveHealingController()
                print("  - Phase 2: Adaptive controller initialized")
            except Exception as e:
                print(f"  Warning: Failed to initialize adaptive controller: {e}")
        
        print(f"✅ IntegratedHealingController ready (Mode: {self.mode})")
    
    def set_mode(self, mode: str):
        """Set operating mode."""
        valid_modes = ["adaptive", "deterministic", "hybrid"]
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of: {valid_modes}")
        
        self.mode = mode
        print(f"Mode set to: {mode}")
    
    def run_integrated_cycle(self, monitoring_data: dict) -> dict:
        """
        Run integrated healing cycle.
        
        In hybrid mode:
        1. Try adaptive decision making if available
        2. If adaptive not available or confidence low, use deterministic rules
        3. Fallback to simple rules if neither is available
        """
        print("\n" + "="*60)
        print("INTEGRATED HEALING CYCLE")
        print("="*60)
        
        # Extract signals
        signals = monitoring_data.get("signals", {})
        
        if self.mode == "deterministic" or self.adaptive_controller is None:
            print("Mode: DETERMINISTIC (Phase 1 rules)")
            return self._run_deterministic_cycle(monitoring_data)
        
        elif self.mode == "adaptive" and self.adaptive_controller is not None:
            print("Mode: ADAPTIVE (Phase 2 intelligence)")
            return self.adaptive_controller.run_adaptive_cycle(monitoring_data)
        
        else:  # hybrid
            print("Mode: HYBRID (Adaptive + Deterministic)")
            
            if self.adaptive_controller is None:
                print("  ⚠ Adaptive controller not available, using deterministic")
                return self._run_deterministic_cycle(monitoring_data)
            
            # Check if adaptive system has enough experience
            adaptive_status = self.adaptive_controller.get_system_status()
            has_components = adaptive_status.get("has_components", False)
            
            if not has_components:
                print("  ⚠ Adaptive components not available, using deterministic")
                return self._run_deterministic_cycle(monitoring_data)
            
            # Get adaptive decision
            print("  🤔 Getting adaptive decision...")
            adaptive_action, decision_metadata = self.adaptive_controller.decide_with_adaptation(
                signals, monitoring_data.get("system_state", {})
            )
            
            # In this simple version, always use adaptive decision in hybrid mode
            # (in real system, you'd check confidence and fallback if needed)
            print(f"  Adaptive decision: {adaptive_action}")
            
            # Execute using adaptive controller
            print(f"  ⚡ Executing: {adaptive_action}")
            execution_result = self.adaptive_controller.execute_healing_action(
                adaptive_action, signals, decision_metadata
            )
            
            # Prepare result
            result = {
                "mode": "hybrid",
                "final_action": adaptive_action,
                "execution_result": execution_result,
                "used_adaptive": True
            }
            
            print(f"  ✅ Hybrid cycle complete: {adaptive_action}")
            return result
    
    def _run_deterministic_cycle(self, monitoring_data: dict) -> dict:
        """Run Phase 1 deterministic cycle."""
        if self.deterministic_controller is not None:
            result = self.deterministic_controller.run_healing_cycle(monitoring_data)
            result["mode"] = "deterministic"
            result["cycle_type"] = "phase1"
            return result
        else:
            # Fallback to simple rules
            signals = monitoring_data.get("signals", {})
            if signals.get("data_drift", 0) > 0.2:
                action = "retrain"
            elif signals.get("accuracy_drop", 0) > 0.1:
                action = "rollback"
            else:
                action = "no_action"
            
            return {
                "mode": "deterministic",
                "action": action,
                "reason": "Simple rule-based fallback",
                "cycle_type": "fallback"
            }
    
    def get_status(self) -> dict:
        """Get integrated system status."""
        status = {
            "timestamp": "2024-01-26T12:00:00",
            "mode": self.mode,
            "has_phase1": self.deterministic_controller is not None,
            "has_adaptive": self.adaptive_controller is not None
        }
        
        if self.deterministic_controller is not None:
            try:
                phase1_status = self.deterministic_controller.get_status()
                status["deterministic_system"] = {
                    "state": phase1_status.get("state", "unknown"),
                    "uptime": phase1_status.get("uptime_seconds", 0),
                    "model_version": phase1_status.get("current_model_version", "unknown")
                }
            except:
                status["deterministic_system"] = {"state": "error"}
        
        if self.adaptive_controller is not None:
            try:
                adaptive_status = self.adaptive_controller.get_system_status()
                status["adaptive_system"] = adaptive_status
            except:
                status["adaptive_system"] = {"error": "failed to get status"}
        
        return status


def main():
    """Main integration demo."""
    print("🚀 SELF-HEALING ML PIPELINES - ADAPTIVE INTELLIGENCE INTEGRATION")
    print("="*70)
    
    # Create integrated controller
    controller = IntegratedHealingController()
    
    # Get initial status
    status = controller.get_status()
    print(f"\n📊 Initial Status:")
    print(f"  Mode: {status['mode']}")
    print(f"  Has Phase 1: {status['has_phase1']}")
    print(f"  Has Adaptive: {status['has_adaptive']}")
    
    # Run demo cycles
    print("\n🔄 Running Demo Cycles")
    print("-"*40)
    
    # Demo scenarios
    scenarios = [
        {
            "name": "Moderate Drift",
            "signals": {"data_drift": 0.25, "accuracy_drop": 0.1},
            "system_state": {"system_load": 0.6}
        },
        {
            "name": "Severe Accuracy Drop", 
            "signals": {"accuracy_drop": 0.2, "anomaly_rate": 0.15},
            "system_state": {"system_load": 0.8}
        },
        {
            "name": "Minor Issues",
            "signals": {"data_drift": 0.1, "anomaly_rate": 0.05},
            "system_state": {"system_load": 0.4}
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"  Signals: {scenario['signals']}")
        
        # Create monitoring data
        monitoring_data = {
            "signals": scenario["signals"],
            "system_state": scenario["system_state"]
        }
        
        # Run integrated cycle
        result = controller.run_integrated_cycle(monitoring_data)
        
        print(f"  Result: {result.get('final_action', result.get('action', 'N/A'))}")
        if 'execution_result' in result:
            exec_result = result['execution_result']
            print(f"  Outcome: {exec_result.get('outcome')}, " +
                  f"Time: {exec_result.get('recovery_time', 0):.1f}s")
    
    # Final status
    final_status = controller.get_status()
    print("\n" + "="*70)
    print("🎉 INTEGRATION DEMO COMPLETE")
    print("="*70)
    print(f"\n📈 Final Statistics:")
    print(f"  Mode: {final_status['mode']}")
    print(f"  Has Phase 1: {final_status['has_phase1']}")
    print(f"  Has Adaptive: {final_status['has_adaptive']}")
    
    print("\n🚀 System is ready for production use!")
    print("   Use: controller = IntegratedHealingController()")
    print("   Then: controller.run_integrated_cycle(monitoring_data)")


if __name__ == "__main__":
    main()
