"""
Policy Engine for Self-Healing ML Pipeline
Evaluates system state against configured policies
"""
import yaml
from typing import Dict, Any, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyEngine:
    """Evaluates policies to determine healing actions"""
    
    def __init__(self, config_path: str = "configs/healing_policies.yaml"):
        self.config_path = config_path
        self.policies = self._load_policies()
        logger.info(f"PolicyEngine initialized with {len(self.policies)} policies")
    
    def _load_policies(self) -> List[Dict[str, Any]]:
        """Load policies from YAML configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            policies = config.get('policies', [])
            logger.info(f"Loaded {len(policies)} policies from {self.config_path}")
            return policies
            
        except Exception as e:
            logger.error(f"Error loading policies from {self.config_path}: {e}")
            return []
    
    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate state against all policies and return best matching action"""
        best_match = None
        best_score = -1
        
        for policy in self.policies:
            score = self._evaluate_policy(policy, state)
            if score > best_score:
                best_score = score
                best_match = policy
        
        if best_match:
            return {
                'action': best_match.get('action', 'fallback'),
                'policy': best_match.get('name', 'unknown'),
                'severity': best_match.get('severity', 'medium'),
                'confidence': best_score,
                'triggered_conditions': self._get_triggered_conditions(best_match, state)
            }
        else:
            # Default fallback
            return {
                'action': 'fallback',
                'policy': 'default_fallback',
                'severity': 'low',
                'confidence': 0.0,
                'triggered_conditions': []
            }
    
    def _evaluate_policy(self, policy: Dict[str, Any], state: Dict[str, Any]) -> float:
        """Evaluate a single policy against state, return match score (0-1)"""
        conditions = policy.get('conditions', {})
        score = 0.0
        total_conditions = len(conditions)
        
        if total_conditions == 0:
            return 0.0
        
        for condition, threshold in conditions.items():
            state_value = state.get(condition, 0)
            
            # Simple threshold evaluation
            if isinstance(threshold, dict):
                # Range check
                min_val = threshold.get('min', float('-inf'))
                max_val = threshold.get('max', float('inf'))
                if min_val <= state_value <= max_val:
                    score += 1
            else:
                # Simple threshold
                if state_value >= threshold:
                    score += 1
        
        return score / total_conditions if total_conditions > 0 else 0.0
    
    def _get_triggered_conditions(self, policy: Dict[str, Any], state: Dict[str, Any]) -> List[str]:
        """Get list of conditions that were triggered"""
        triggered = []
        conditions = policy.get('conditions', {})
        
        for condition, threshold in conditions.items():
            state_value = state.get(condition, 0)
            
            if isinstance(threshold, dict):
                min_val = threshold.get('min', float('-inf'))
                max_val = threshold.get('max', float('inf'))
                if min_val <= state_value <= max_val:
                    triggered.append(condition)
            else:
                if state_value >= threshold:
                    triggered.append(condition)
        
        return triggered

if __name__ == "__main__":
    # Test the policy engine
    engine = PolicyEngine()
    
    test_state = {
        'drift_score': 0.31,
        'accuracy_drop': 0.18,
        'anomaly_rate': 0.07,
        'time_since_last_action': 45
    }
    
    decision = engine.evaluate(test_state)
    print(f"Decision: {decision}")
