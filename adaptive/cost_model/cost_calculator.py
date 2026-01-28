"""
Cost Calculator for Adaptive Self-Healing
Computes expected costs for different healing actions based on context.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional
import yaml
import os


@dataclass
class ActionCost:
    """Cost model for a single healing action."""
    name: str
    compute_cost: float
    latency_cost: float
    risk_score: float
    recovery_time: float
    success_probability: float
    
    def total_cost(self, weights: Dict[str, float]) -> float:
        """Calculate weighted total cost."""
        components = {
            'compute': self.compute_cost,
            'latency': self.latency_cost,
            'risk': self.risk_score,
            'recovery_time': self.recovery_time / 100.0  # Normalize
        }
        
        total = 0.0
        for component, weight in weights.items():
            if component in components:
                total += components[component] * weight
        
        # Adjust for success probability (higher success = lower effective cost)
        effective_cost = total / (self.success_probability + 1e-10)
        return effective_cost


@dataclass
class HealingContext:
    """Context for healing decision making."""
    signals: Dict[str, float]  # e.g., {'data_drift': 0.25, 'accuracy_drop': 0.15}
    confidence: Dict[str, float]  # Confidence for each signal
    system_state: Dict[str, Any]  # Current system state
    time_of_day: int  # Hour of day (0-23)
    day_of_week: int  # Day of week (0-6)
    recent_outcomes: list  # Recent healing outcomes


class CostCalculator:
    """Main cost calculator for adaptive healing decisions."""
    
    def __init__(self, config_path: str = "adaptive/cost_model/action_costs.yaml"):
        self.default_config = self._create_default_config()
        self.load_config(config_path)
        self.initialize_actions()
        
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration if YAML fails."""
        return {
            'actions': {
                'retrain': {
                    'compute_cost': 5.0,
                    'latency_cost': 3.0,
                    'risk_score': 3.0,
                    'recovery_time': 60.0,
                    'success_probability': 0.85
                },
                'rollback': {
                    'compute_cost': 1.0,
                    'latency_cost': 2.0,
                    'risk_score': 1.0,
                    'recovery_time': 10.0,
                    'success_probability': 0.95
                },
                'fallback': {
                    'compute_cost': 0.5,
                    'latency_cost': 1.0,
                    'risk_score': 0.5,
                    'recovery_time': 1.0,
                    'success_probability': 1.0
                }
            },
            'cost_weights': {
                'compute': 0.3,
                'latency': 0.25,
                'risk': 0.25,
                'recovery_time': 0.2
            },
            'signal_weights': {
                'data_drift': 0.4,
                'accuracy_drop': 0.3,
                'anomaly_rate': 0.2,
                'latency_increase': 0.1
            }
        }
    
    def load_config(self, config_path: str):
        """Load cost configuration from YAML with fallback."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    if loaded_config:
                        self.config = loaded_config
                    else:
                        self.config = self.default_config
            else:
                self.config = self.default_config
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            print("Using default configuration")
            self.config = self.default_config
    
    def initialize_actions(self):
        """Initialize action cost models from config."""
        self.actions = {}
        actions_config = self.config.get('actions', {})
        
        # Ensure all required actions exist
        default_actions = self.default_config['actions']
        for action_name in default_actions.keys():
            if action_name in actions_config:
                action_config = actions_config[action_name]
            else:
                action_config = default_actions[action_name]
            
            self.actions[action_name] = ActionCost(
                name=action_name,
                compute_cost=float(action_config.get('compute_cost', 1.0)),
                latency_cost=float(action_config.get('latency_cost', 1.0)),
                risk_score=float(action_config.get('risk_score', 1.0)),
                recovery_time=float(action_config.get('recovery_time', 10.0)),
                success_probability=float(action_config.get('success_probability', 0.8))
            )
        
        self.cost_weights = self.config.get('cost_weights', self.default_config['cost_weights'])
        self.signal_weights = self.config.get('signal_weights', self.default_config['signal_weights'])
    
    def calculate_context_cost(self, action_name: str, context: HealingContext) -> float:
        """
        Calculate contextual cost for an action.
        
        Cost = Base Cost × Signal Severity × Time Factor × Recent Performance
        """
        if action_name not in self.actions:
            # If action not found, use no_action with high cost
            return 100.0
        
        action = self.actions[action_name]
        
        # 1. Base cost
        base_cost = action.total_cost(self.cost_weights)
        
        # 2. Signal severity factor
        signal_severity = self._calculate_signal_severity(context.signals)
        
        # 3. Time factor (certain actions cost more at peak times)
        time_factor = self._calculate_time_factor(context.time_of_day, action_name)
        
        # 4. Recent performance factor
        performance_factor = self._calculate_performance_factor(
            action_name, context.recent_outcomes
        )
        
        # 5. Confidence adjustment
        confidence_factor = self._calculate_confidence_factor(context.confidence)
        
        # Combine all factors
        contextual_cost = (
            base_cost * 
            signal_severity * 
            time_factor * 
            performance_factor *
            confidence_factor
        )
        
        return max(0.1, contextual_cost)  # Ensure minimum cost
    
    def _calculate_signal_severity(self, signals: Dict[str, float]) -> float:
        """Calculate weighted signal severity."""
        if not signals:
            return 1.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for signal, value in signals.items():
            weight = self.signal_weights.get(signal, 0.1)
            weighted_sum += weight * value
            total_weight += weight
        
        if total_weight == 0:
            return 1.0
        
        severity = 1.0 + (weighted_sum / total_weight)
        return min(severity, 2.0)  # Cap at 2x
    
    def _calculate_time_factor(self, hour: int, action_name: str) -> float:
        """Calculate time-based cost factor."""
        # Peak hours: 9 AM - 5 PM (higher cost for disruptive actions)
        is_peak_hour = 9 <= hour <= 17
        
        if action_name == "retrain":
            # Retraining is most expensive during peak hours
            return 1.5 if is_peak_hour else 1.0
        elif action_name == "rollback":
            # Rollback is slightly more expensive during peak
            return 1.2 if is_peak_hour else 1.0
        else:
            # Fallback is time-insensitive
            return 1.0
    
    def _calculate_performance_factor(self, action_name: str, recent_outcomes: list) -> float:
        """Calculate factor based on recent performance of this action."""
        if not recent_outcomes:
            return 1.0
        
        # Filter outcomes for this specific action
        action_outcomes = [
            outcome for outcome in recent_outcomes 
            if outcome.get('action') == action_name
        ]
        
        if not action_outcomes:
            return 1.0
        
        # Calculate success rate
        successes = sum(1 for outcome in action_outcomes if outcome.get('success', False))
        success_rate = successes / len(action_outcomes)
        
        # Adjust cost based on success rate (higher success = lower cost)
        return 1.0 / (success_rate + 0.5)  # +0.5 to avoid division by near-zero
    
    def _calculate_confidence_factor(self, confidence: Dict[str, float]) -> float:
        """Adjust cost based on confidence levels."""
        if not confidence:
            return 1.0
        
        avg_confidence = np.mean(list(confidence.values()))
        
        # Higher confidence reduces effective cost
        # 0% confidence = 2x cost, 100% confidence = 0.5x cost
        confidence_factor = 2.0 - (1.5 * avg_confidence)
        return max(0.5, min(2.0, confidence_factor))
    
    def get_all_action_costs(self, context: HealingContext) -> Dict[str, float]:
        """Calculate costs for all available actions."""
        costs = {}
        for action_name in self.actions.keys():
            costs[action_name] = self.calculate_context_cost(action_name, context)
        
        return costs
    
    def get_optimal_action(self, context: HealingContext) -> tuple:
        """
        Find optimal action based on cost minimization.
        Returns (action_name, cost, all_costs)
        """
        all_costs = self.get_all_action_costs(context)
        
        if not all_costs:
            return "no_action", 0.0, {}
        
        # Find minimum cost action
        optimal_action = min(all_costs.items(), key=lambda x: x[1])
        
        return optimal_action[0], optimal_action[1], all_costs


# Simple factory function for integration
def create_cost_calculator() -> CostCalculator:
    """Factory function to create cost calculator."""
    return CostCalculator()


if __name__ == "__main__":
    # Test the cost calculator
    calculator = CostCalculator()
    
    # Test context
    test_context = HealingContext(
        signals={"data_drift": 0.25, "accuracy_drop": 0.1},
        confidence={"data_drift": 0.8, "accuracy_drop": 0.6},
        system_state={"load": 0.7, "memory_usage": 0.6},
        time_of_day=14,  # 2 PM
        day_of_week=2,   # Wednesday
        recent_outcomes=[
            {"action": "retrain", "success": True, "recovery_time": 45.0},
            {"action": "rollback", "success": True, "recovery_time": 8.0}
        ]
    )
    
    # Calculate costs
    costs = calculator.get_all_action_costs(test_context)
    print("Action Costs:")
    for action, cost in costs.items():
        print(f"  {action}: {cost:.3f}")
    
    # Get optimal action
    optimal_action, optimal_cost, _ = calculator.get_optimal_action(test_context)
    print(f"\nOptimal Action: {optimal_action} (cost: {optimal_cost:.3f})")
