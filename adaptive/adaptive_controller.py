"""
Adaptive Self-Healing Controller
Integrates all adaptive components into a unified system.
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

try:
    from adaptive.cost_model.cost_calculator import CostCalculator, HealingContext
    from adaptive.memory.experience_logger import ExperienceLogger
    from adaptive.learning.bandit_policy import ContextualBanditPolicy
    from adaptive.uncertainty.bayesian_drift import BayesianDriftDetector, ConfidenceEstimator
    HAS_ADAPTIVE_COMPONENTS = True
except ImportError as e:
    print(f"Warning: Some adaptive components failed to import: {e}")
    HAS_ADAPTIVE_COMPONENTS = False


class AdaptiveHealingController:
    """
    Main controller for adaptive self-healing.
    Integrates cost models, experience learning, bandit policies, and Bayesian uncertainty.
    """
    
    def __init__(self, config_path: str = "adaptive/cost_model/action_costs.yaml"):
        """Initialize adaptive controller with all components."""
        print("Initializing AdaptiveHealingController...")
        
        if not HAS_ADAPTIVE_COMPONENTS:
            print("Warning: Adaptive components not available, using minimal functionality")
            self.has_components = False
            return
        
        try:
            # Initialize all components
            self.cost_calculator = CostCalculator(config_path)
            print("  - Cost-aware decision making")
            
            self.experience_logger = ExperienceLogger(buffer_size=1000)
            print("  - Experience-based learning")
            
            self.bandit_policy = ContextualBanditPolicy(config_path)
            print("  - Contextual bandit optimization")
            
            self.bayesian_detector = BayesianDriftDetector()
            self.confidence_estimator = ConfidenceEstimator()
            print("  - Bayesian uncertainty quantification")
            
            self.has_components = True
            
            # State
            self.current_context = None
            self.last_decision_metadata = {}
            self.performance_history = []
            
            print("✅ Adaptive Self-Healing Controller initialized")
            
        except Exception as e:
            print(f"Warning: Failed to initialize adaptive components: {e}")
            print("Falling back to minimal functionality")
            self.has_components = False
    
    def create_healing_context(self, signals: Dict[str, float], 
                             system_state: Dict[str, Any]) -> Optional[HealingContext]:
        """Create healing context from current state."""
        if not self.has_components:
            return None
        
        # Estimate confidence for each signal
        confidence = {}
        for signal_name, signal_value in signals.items():
            # Simplified confidence estimation
            if signal_value < 0.1:
                confidence[signal_name] = 0.9
            elif signal_value < 0.3:
                confidence[signal_name] = 0.7
            else:
                confidence[signal_name] = 0.5
        
        # Create context
        now = datetime.now()
        context = HealingContext(
            signals=signals,
            confidence=confidence,
            system_state=system_state,
            time_of_day=now.hour,
            day_of_week=now.weekday(),
            recent_outcomes=[]
        )
        
        self.current_context = context
        return context
    
    def decide_with_adaptation(self, signals: Dict[str, float], 
                             system_state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Make adaptive healing decision using available components.
        
        Returns:
            (selected_action, decision_metadata)
        """
        if not self.has_components:
            # Fallback to simple rule-based decision
            if signals.get("data_drift", 0) > 0.2:
                return "retrain", {"reason": "High data drift", "method": "fallback"}
            elif signals.get("accuracy_drop", 0) > 0.1:
                return "rollback", {"reason": "Accuracy drop", "method": "fallback"}
            else:
                return "no_action", {"reason": "No significant issues", "method": "fallback"}
        
        try:
            # Create context
            context = self.create_healing_context(signals, system_state)
            
            # Get cost estimates
            cost_estimates = self.cost_calculator.get_all_action_costs(context)
            
            # Get bandit policy recommendation
            bandit_action, bandit_metadata = self.bandit_policy.decide(signals, system_state)
            
            # Get experience-based recommendation
            experience_recommendation = self.experience_logger.get_action_recommendation(signals)
            
            # Combine recommendations
            recommendations = {
                "cost_based": min(cost_estimates.items(), key=lambda x: x[1])[0] if cost_estimates else "no_action",
                "bandit_based": bandit_action,
                "experience_based": experience_recommendation.get("recommendation", "no_action")
            }
            
            # Simple voting mechanism
            action_votes = {}
            for source, action in recommendations.items():
                action_votes[action] = action_votes.get(action, 0) + 1
            
            # Select action with most votes
            if action_votes:
                selected_action = max(action_votes.items(), key=lambda x: x[1])[0]
            else:
                selected_action = "no_action"
            
            # Prepare metadata
            decision_metadata = {
                "selected_action": selected_action,
                "recommendations": recommendations,
                "action_votes": action_votes,
                "cost_estimates": cost_estimates,
                "bandit_metadata": bandit_metadata,
                "experience_recommendation": experience_recommendation,
                "decision_timestamp": datetime.now().isoformat()
            }
            
            self.last_decision_metadata = decision_metadata
            return selected_action, decision_metadata
            
        except Exception as e:
            print(f"Warning: Adaptive decision failed: {e}")
            # Fallback to simple rule
            if signals.get("data_drift", 0) > 0.2:
                return "retrain", {"reason": "Fallback: High data drift", "error": str(e)}
            else:
                return "no_action", {"reason": "Fallback: No action", "error": str(e)}
    
    def execute_healing_action(self, action: str, signals: Dict[str, float],
                             decision_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Execute healing action with monitoring and learning."""
        if not self.has_components:
            # Simulate execution
            import random
            recovery_time = random.uniform(1.0, 5.0)
            outcome = "success" if random.random() > 0.1 else "failure"
            
            return {
                "action_executed": action,
                "outcome": outcome,
                "recovery_time": recovery_time,
                "performance_change": 0.0,
                "simulated": True
            }
        
        try:
            # Start recovery timer
            self.experience_logger.start_recovery_timer()
            
            # Calculate action cost
            if self.current_context:
                action_cost = self.cost_calculator.calculate_context_cost(action, self.current_context)
            else:
                action_cost = 1.0
            
            # Simulate execution (in real system, this would call actual healing actions)
            import random
            import time
            
            # Simulate different actions
            if action == "retrain":
                recovery_time = random.uniform(30.0, 60.0)
                outcome = "success" if random.random() < 0.85 else "failure"
                performance_change = random.uniform(0.05, 0.25) if outcome == "success" else random.uniform(-0.1, 0.0)
            elif action == "rollback":
                recovery_time = random.uniform(5.0, 15.0)
                outcome = "success" if random.random() < 0.95 else "failure"
                performance_change = random.uniform(-0.05, 0.05)
            elif action == "fallback":
                recovery_time = random.uniform(0.5, 2.0)
                outcome = "partial_success"
                performance_change = random.uniform(-0.2, 0.0)
            else:  # no_action
                recovery_time = 0.0
                outcome = "no_change"
                performance_change = 0.0
            
            # Calculate reward for learning
            reward = 0.0
            if "bandit_metadata" in decision_metadata:
                try:
                    context_hash = decision_metadata["bandit_metadata"].get("context_hash")
                    if context_hash and hasattr(self.bandit_policy, 'calculate_reward'):
                        reward = self.bandit_policy.calculate_reward(
                            outcome, recovery_time, performance_change, action_cost
                        )
                        self.bandit_policy.update_policy(context_hash, action, reward)
                except:
                    pass
            
            # Log experience
            if hasattr(self.experience_logger, 'log_healing_action'):
                self.experience_logger.log_healing_action(
                    signals=signals,
                    action_taken=action,
                    action_cost=action_cost,
                    outcome=outcome,
                    performance_change=performance_change
                )
            
            # Update performance history
            self.performance_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "outcome": outcome,
                "recovery_time": recovery_time,
                "performance_change": performance_change,
                "reward": reward
            })
            
            # Keep history manageable
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            return {
                "action_executed": action,
                "outcome": outcome,
                "recovery_time": recovery_time,
                "performance_change": performance_change,
                "reward": reward,
                "action_cost": action_cost
            }
            
        except Exception as e:
            print(f"Warning: Action execution failed: {e}")
            return {
                "action_executed": action,
                "outcome": "failure",
                "recovery_time": 0.0,
                "performance_change": 0.0,
                "error": str(e)
            }
    
    def run_adaptive_cycle(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete adaptive healing cycle."""
        print("\n🔄 Starting adaptive healing cycle...")
        
        # Extract signals and system state
        signals = monitoring_data.get("signals", {})
        system_state = monitoring_data.get("system_state", {})
        
        # Make adaptive decision
        print("  🤔 Making adaptive decision...")
        action, decision_metadata = self.decide_with_adaptation(signals, system_state)
        
        # Execute healing action
        print(f"  ⚡ Executing: {action}")
        execution_result = self.execute_healing_action(action, signals, decision_metadata)
        
        # Prepare comprehensive result
        result = {
            "cycle_timestamp": datetime.now().isoformat(),
            "decision": {
                "action": action,
                "metadata": decision_metadata
            },
            "execution": execution_result,
            "system_state_after": self.get_system_status()
        }
        
        print(f"  ✅ Cycle complete: {action} → {execution_result['outcome']}")
        if execution_result.get('recovery_time', 0) > 0:
            print(f"     Recovery: {execution_result['recovery_time']:.1f}s")
        
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "has_components": self.has_components if hasattr(self, 'has_components') else False,
            "performance_history_count": len(self.performance_history) if hasattr(self, 'performance_history') else 0
        }
        
        if self.has_components:
            try:
                exp_stats = self.experience_logger.get_statistics()
                status["experience"] = {
                    "total_experiences": exp_stats.get("total_experiences", 0),
                    "overall_success_rate": exp_stats.get("overall_success_rate", 0.0)
                }
            except:
                pass
            
            try:
                bandit_stats = self.bandit_policy.get_statistics()
                status["bandit_policy"] = {
                    "total_contexts": bandit_stats.get("total_contexts", 0)
                }
            except:
                pass
        
        return status
    
    def save_state(self) -> None:
        """Save adaptive components state."""
        if self.has_components:
            try:
                self.bandit_policy.save_state()
                print("💾 Bandit policy state saved")
            except:
                pass
            
            try:
                self.experience_logger.buffer.save_experiences()
                print("💾 Experience buffer saved")
            except:
                pass


# Factory function
def create_adaptive_controller(config_path: str = None) -> AdaptiveHealingController:
    """Create and return an adaptive healing controller."""
    if config_path is None:
        config_path = "adaptive/cost_model/action_costs.yaml"
    
    return AdaptiveHealingController(config_path)


if __name__ == "__main__":
    # Test the adaptive controller
    print("🧪 Testing AdaptiveHealingController")
    print("=" * 50)
    
    controller = AdaptiveHealingController()
    
    # Create test monitoring data
    import numpy as np
    np.random.seed(42)
    
    test_data = {
        "signals": {
            "data_drift": 0.25,
            "accuracy_drop": 0.12
        },
        "system_state": {
            "system_load": 0.65,
            "memory_usage": 0.58
        }
    }
    
    # Run adaptive cycle
    result = controller.run_adaptive_cycle(test_data)
    
    print("\n📊 Cycle Result Summary:")
    print(f"  Action: {result['decision']['action']}")
    print(f"  Outcome: {result['execution']['outcome']}")
    
    # Get system status
    status = controller.get_system_status()
    print(f"\n📈 System Status:")
    print(f"  Has components: {status['has_components']}")
    print(f"  Performance cycles: {status['performance_history_count']}")
    
    print("\n✅ AdaptiveHealingController test complete!")
