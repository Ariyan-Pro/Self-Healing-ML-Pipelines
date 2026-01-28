# adaptive/learning/shadow_learner.py
"""
Shadow Mode Learning - Learn from decisions without executing them.
Enables safe exploration in production.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


class ShadowLearner:
    """
    Learn from hypothetical decisions without execution.
    Implements counterfactual learning for safe exploration.
    """
    
    def __init__(self, experiences_path: str = "adaptive/memory/experiences.json"):
        self.experiences_path = Path(experiences_path)
        self.shadow_experiences = []
        self.counterfactual_analysis = []
        
    def simulate_decision(self, context: Dict, proposed_action: str, 
                         actual_action: str) -> Dict:
        """
        Simulate a different decision than what was actually taken.
        
        Args:
            context: Decision context
            proposed_action: Action to simulate
            actual_action: Action actually taken
            
        Returns:
            Simulated outcome with estimated metrics
        """
        # Generate simulated outcome based on historical patterns
        simulated_outcome = self._estimate_outcome(context, proposed_action)
        
        # Calculate estimated cost
        estimated_cost = self._estimate_cost(proposed_action, simulated_outcome)
        
        # Calculate counterfactual value
        counterfactual_value = self._calculate_counterfactual_value(
            proposed_action, actual_action, simulated_outcome
        )
        
        shadow_experience = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "action_simulated": proposed_action,
            "action_actual": actual_action,
            "simulated_outcome": simulated_outcome,
            "estimated_cost": estimated_cost,
            "counterfactual_value": counterfactual_value,
            "learning_type": "shadow_mode"
        }
        
        self.shadow_experiences.append(shadow_experience)
        
        # Store for analysis
        self.counterfactual_analysis.append({
            "actual_action": actual_action,
            "simulated_action": proposed_action,
            "value_difference": counterfactual_value,
            "context_hash": hash(json.dumps(context, sort_keys=True))
        })
        
        return shadow_experience
    
    def _estimate_outcome(self, context: Dict, action: str) -> Dict:
        """Estimate outcome based on historical patterns."""
        # Load historical experiences
        historical_outcomes = self._get_historical_outcomes(action, context)
        
        if not historical_outcomes:
            # Default estimates if no history
            return {
                "success_probability": self._get_default_success_rate(action),
                "recovery_time": self._get_default_recovery_time(action),
                "performance_change": self._get_default_performance_change(action),
                "confidence": 0.5
            }
        
        # Calculate averages from historical data
        success_rates = [o.get("outcome") == "success" for o in historical_outcomes]
        recovery_times = [o.get("recovery_time", 0) for o in historical_outcomes]
        performance_changes = [o.get("performance_change", 0) for o in historical_outcomes]
        
        return {
            "success_probability": np.mean(success_rates) if success_rates else 0.5,
            "recovery_time": np.mean(recovery_times) if recovery_times else 0,
            "performance_change": np.mean(performance_changes) if performance_changes else 0,
            "confidence": min(0.9, len(historical_outcomes) / 10)
        }
    
    def _get_historical_outcomes(self, action: str, context: Dict) -> List[Dict]:
        """Get historical outcomes for similar contexts."""
        if not self.experiences_path.exists():
            return []
        
        with open(self.experiences_path, 'r') as f:
            all_experiences = json.load(f)
        
        # Filter by action
        action_experiences = [e for e in all_experiences 
                            if e.get("action_taken") == action]
        
        if not action_experiences:
            return []
        
        # Return recent experiences
        return action_experiences[-10:]
    
    def _estimate_cost(self, action: str, outcome: Dict) -> float:
        """Estimate cost of an action based on outcome."""
        # Base costs
        base_costs = {
            "retrain": 2.0,
            "rollback": 1.5,
            "fallback": 0.5,
            "no_action": 0.1
        }
        
        base_cost = base_costs.get(action, 1.0)
        
        # Adjust based on outcome
        success_probability = outcome.get("success_probability", 0.5)
        recovery_time = outcome.get("recovery_time", 0)
        performance_change = outcome.get("performance_change", 0)
        
        # Cost increases with longer recovery and negative performance impact
        time_penalty = recovery_time * 0.01
        performance_penalty = max(0, -performance_change) * 10
        
        # Success reduces cost
        success_bonus = success_probability * -0.5
        
        total_cost = base_cost + time_penalty + performance_penalty + success_bonus
        
        return max(0, total_cost)
    
    def _calculate_counterfactual_value(self, simulated_action: str, 
                                       actual_action: str,
                                       simulated_outcome: Dict) -> float:
        """Calculate value difference between simulated and actual actions."""
        # Load actual outcome if available
        actual_experiences = []
        if self.experiences_path.exists():
            with open(self.experiences_path, 'r') as f:
                actual_experiences = json.load(f)
        
        # Find most recent actual experience
        recent_actual = None
        for exp in reversed(actual_experiences):
            if exp.get("action_taken") == actual_action:
                recent_actual = exp
                break
        
        if not recent_actual:
            return 0.0
        
        # Calculate value of actual action
        actual_cost = recent_actual.get("action_cost", 1.0)
        actual_performance = recent_actual.get("performance_change", 0)
        
        actual_value = -actual_cost + (actual_performance * 5)
        
        # Calculate value of simulated action
        simulated_cost = self._estimate_cost(simulated_action, simulated_outcome)
        simulated_performance = simulated_outcome.get("performance_change", 0)
        
        simulated_value = -simulated_cost + (simulated_performance * 5)
        
        return simulated_value - actual_value
    
    def _get_default_success_rate(self, action: str) -> float:
        """Get default success rate for an action."""
        defaults = {
            "retrain": 0.7,
            "rollback": 0.9,
            "fallback": 0.95,
            "no_action": 1.0
        }
        return defaults.get(action, 0.8)
    
    def _get_default_recovery_time(self, action: str) -> float:
        """Get default recovery time for an action (in ms)."""
        defaults = {
            "retrain": 30000.0,
            "rollback": 5000.0,
            "fallback": 1000.0,
            "no_action": 0.0
        }
        return defaults.get(action, 10000.0)
    
    def _get_default_performance_change(self, action: str) -> float:
        """Get default performance change for an action."""
        defaults = {
            "retrain": 0.1,
            "rollback": 0.05,
            "fallback": -0.02,
            "no_action": 0.0
        }
        return defaults.get(action, 0.0)
    
    def analyze_shadow_learning(self) -> Dict:
        """Analyze shadow learning results."""
        if not self.shadow_experiences:
            return {"status": "no_shadow_experiences"}
        
        # Calculate potential improvements
        positive_counterfactuals = [
            exp for exp in self.counterfactual_analysis
            if exp["value_difference"] > 0
        ]
        
        negative_counterfactuals = [
            exp for exp in self.counterfactual_analysis
            if exp["value_difference"] < 0
        ]
        
        total_value_opportunity = sum(
            exp["value_difference"] for exp in positive_counterfactuals
        )
        
        return {
            "total_shadow_experiences": len(self.shadow_experiences),
            "positive_counterfactuals": len(positive_counterfactuals),
            "negative_counterfactuals": len(negative_counterfactuals),
            "total_value_opportunity": total_value_opportunity,
            "average_value_opportunity": (
                total_value_opportunity / len(positive_counterfactuals)
                if positive_counterfactuals else 0
            ),
            "most_promising_alternative": self._find_most_promising_alternative(),
            "recommendations": self._generate_shadow_recommendations()
        }
    
    def _find_most_promising_alternative(self) -> Optional[Dict]:
        """Find the most promising alternative action."""
        if not self.counterfactual_analysis:
            return None
        
        # Group by alternative action
        action_values = {}
        for analysis in self.counterfactual_analysis:
            alt_action = analysis["simulated_action"]
            value = analysis["value_difference"]
            
            if alt_action not in action_values:
                action_values[alt_action] = []
            action_values[alt_action].append(value)
        
        # Find action with highest average value
        best_action = None
        best_avg_value = -float('inf')
        
        for action, values in action_values.items():
            avg_value = np.mean(values) if values else 0
            if avg_value > best_avg_value:
                best_avg_value = avg_value
                best_action = action
        
        if best_action:
            return {
                "action": best_action,
                "average_value_improvement": best_avg_value,
                "sample_size": len(action_values[best_action])
            }
        
        return None
    
    def _generate_shadow_recommendations(self) -> List[str]:
        """Generate recommendations based on shadow learning."""
        recommendations = []
        analysis = self.analyze_shadow_learning()
        
        if analysis["positive_counterfactuals"] > 0:
            rec = (
                f"Found {analysis['positive_counterfactuals']} cases where "
                f"alternative actions would have been better (total value: "
                f"\)"
            )
            recommendations.append(rec)
            
            best_alt = analysis.get("most_promising_alternative")
            if best_alt:
                rec = (
                    f"Consider using '{best_alt['action']}' more often. "
                    f"Shadow learning shows average improvement of "
                    f"\ per decision "
                    f"(based on {best_alt['sample_size']} simulations)"
                )
                recommendations.append(rec)
        
        if analysis["negative_counterfactuals"] > 0:
            rec = (
                f"System is avoiding {analysis['negative_counterfactuals']} "
                f"potentially worse decisions"
            )
            recommendations.append(rec)
        
        return recommendations
    
    def save_shadow_experiences(self, output_path: Optional[str] = None):
        """Save shadow experiences to file."""
        if not output_path:
            output_path = "adaptive/memory/shadow_experiences.json"
        
        data = {
            "shadow_experiences": self.shadow_experiences,
            "counterfactual_analysis": self.counterfactual_analysis,
            "analysis_summary": self.analyze_shadow_learning(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)


# Example usage
if __name__ == "__main__":
    shadow_learner = ShadowLearner()
    
    # Simulate shadow learning
    context = {"data_drift": 0.25, "system_load": 0.6}
    
    shadow_exp = shadow_learner.simulate_decision(
        context=context,
        proposed_action="retrain",
        actual_action="fallback"
    )
    
    print("Shadow experience created")
    analysis = shadow_learner.analyze_shadow_learning()
    print("Analysis:", json.dumps(analysis, indent=2))
    
    shadow_learner.save_shadow_experiences()
    print("Shadow experiences saved")
