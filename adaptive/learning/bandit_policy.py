"""
Bandit Policy Engine for Adaptive Self-Healing
Contextual bandit approach for learning optimal healing policies.
"""

import numpy as np
import random
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
from pathlib import Path


@dataclass
class BanditArm:
    """Represents a healing action as a bandit arm."""
    name: str
    estimated_value: float  # Q-value estimate
    visit_count: int  # Number of times selected
    last_updated: str
    
    def ucb_score(self, total_visits: int, exploration_factor: float = 2.0) -> float:
        """Calculate UCB1 score for this arm."""
        if self.visit_count == 0:
            return float('inf')  # Never visited, explore!
        
        exploration = exploration_factor * np.sqrt(
            np.log(total_visits) / self.visit_count
        )
        return self.estimated_value + exploration
    
    def update_value(self, reward: float, learning_rate: float = 0.1) -> None:
        """Update Q-value estimate using incremental update."""
        if self.visit_count == 0:
            self.estimated_value = reward
        else:
            self.estimated_value += learning_rate * (reward - self.estimated_value)
        
        self.visit_count += 1
        self.last_updated = datetime.now().isoformat()


@dataclass
class ContextState:
    """Represents a specific context/state for bandits."""
    context_hash: str
    context_features: Dict[str, float]  # Normalized features
    arms: Dict[str, BanditArm]  # Arms available in this context
    total_visits: int
    
    def select_arm(self, exploration_rate: float = 0.1) -> Tuple[str, bool]:
        """
        Select an arm using epsilon-greedy with UCB fallback.
        
        Returns:
            (arm_name, was_exploratory)
        """
        # Epsilon-greedy: explore with probability epsilon
        if random.random() < exploration_rate:
            # Exploratory move: random selection among under-explored arms
            exploratory_arms = [
                name for name, arm in self.arms.items()
                if arm.visit_count < 5  # Under-explored
            ]
            
            if exploratory_arms:
                selected = random.choice(exploratory_arms)
                return selected, True
        
        # Exploitative move: UCB selection
        if not self.arms:
            return "no_action", False
        
        # Calculate UCB scores
        ucb_scores = {}
        for name, arm in self.arms.items():
            ucb_scores[name] = arm.ucb_score(self.total_visits)
        
        # Select arm with highest UCB score
        selected = max(ucb_scores.items(), key=lambda x: x[1])[0]
        return selected, False
    
    def update_arm(self, arm_name: str, reward: float, learning_rate: float = 0.1) -> None:
        """Update the selected arm with observed reward."""
        if arm_name in self.arms:
            self.arms[arm_name].update_value(reward, learning_rate)
            self.total_visits += 1


class ContextualBanditPolicy:
    """
    Contextual bandit policy engine for adaptive healing decisions.
    Learns which healing actions work best in which contexts.
    """
    
    def __init__(self, config_path: str = "adaptive/cost_model/action_costs.yaml"):
        self.contexts: Dict[str, ContextState] = {}
        self.config = self._load_config(config_path)
        self.exploration_rate = self.config.get('learning', {}).get('exploration_rate', 0.1)
        self.learning_rate = self.config.get('learning', {}).get('learning_rate', 0.01)
        self.memory_path = Path("adaptive/learning/bandit_memory.json")
        
        # Initialize with default actions
        self.default_actions = ["retrain", "rollback", "fallback", "no_action"]
        
        # Load saved state if exists
        self.load_state()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration."""
        import yaml
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {}
    
    def _create_context_hash(self, signals: Dict[str, float], 
                           system_state: Dict[str, Any]) -> str:
        """Create hash for context based on signals and system state."""
        # Discretize signals for context creation
        discretized = {}
        
        # Discretize signal values
        for sig, val in signals.items():
            if val < 0.1:
                discretized[sig] = "low"
            elif val < 0.3:
                discretized[sig] = "medium"
            else:
                discretized[sig] = "high"
        
        # Add time of day discretization
        hour = datetime.now().hour
        if hour < 6:
            discretized["time"] = "night"
        elif hour < 12:
            discretized["time"] = "morning"
        elif hour < 18:
            discretized["time"] = "afternoon"
        else:
            discretized["time"] = "evening"
        
        # Add system load discretization
        load = system_state.get("system_load", 0.5)
        if load < 0.3:
            discretized["load"] = "low"
        elif load < 0.7:
            discretized["load"] = "medium"
        else:
            discretized["load"] = "high"
        
        # Create hash from discretized features
        context_str = json.dumps(discretized, sort_keys=True)
        return str(hash(context_str))
    
    def _extract_features(self, signals: Dict[str, float], 
                         system_state: Dict[str, Any]) -> Dict[str, float]:
        """Extract and normalize features for context."""
        features = {}
        
        # Signal features (normalized)
        for sig, val in signals.items():
            features[f"signal_{sig}"] = min(val, 1.0)  # Cap at 1.0
        
        # System features
        features["system_load"] = system_state.get("system_load", 0.5)
        features["memory_usage"] = system_state.get("memory_usage", 0.5)
        
        # Time features
        now = datetime.now()
        features["hour_sin"] = np.sin(2 * np.pi * now.hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * now.hour / 24)
        features["day_of_week"] = now.weekday() / 6.0  # Normalized
        
        return features
    
    def get_or_create_context(self, signals: Dict[str, float], 
                            system_state: Dict[str, Any]) -> ContextState:
        """Get existing context or create new one."""
        context_hash = self._create_context_hash(signals, system_state)
        
        if context_hash in self.contexts:
            return self.contexts[context_hash]
        
        # Create new context
        features = self._extract_features(signals, system_state)
        
        # Initialize arms for this context
        arms = {}
        for action in self.default_actions:
            arms[action] = BanditArm(
                name=action,
                estimated_value=1.0,  # Optimistic initialization
                visit_count=0,
                last_updated=datetime.now().isoformat()
            )
        
        context = ContextState(
            context_hash=context_hash,
            context_features=features,
            arms=arms,
            total_visits=0
        )
        
        self.contexts[context_hash] = context
        return context
    
    def decide(self, signals: Dict[str, float], 
              system_state: Dict[str, Any],
              rule_based_fallback: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Decide on healing action using contextual bandits.
        
        Args:
            signals: Current monitoring signals
            system_state: Current system state
            rule_based_fallback: Whether to use rule-based fallback for safety
            
        Returns:
            (selected_action, metadata)
        """
        # Get or create context
        context = self.get_or_create_context(signals, system_state)
        
        # Select arm using bandit policy
        selected_action, was_exploratory = context.select_arm(self.exploration_rate)
        
        # Rule-based safety check (hybrid approach)
        if rule_based_fallback:
            selected_action = self._apply_safety_rules(selected_action, signals)
        
        # Prepare metadata
        metadata = {
            "context_hash": context.context_hash,
            "was_exploratory": was_exploratory,
            "arm_estimates": {name: arm.estimated_value for name, arm in context.arms.items()},
            "arm_visits": {name: arm.visit_count for name, arm in context.arms.items()},
            "total_context_visits": context.total_visits,
            "selected_arm_estimate": context.arms[selected_action].estimated_value,
            "selected_arm_visits": context.arms[selected_action].visit_count
        }
        
        return selected_action, metadata
    
    def _apply_safety_rules(self, suggested_action: str, signals: Dict[str, float]) -> str:
        """Apply safety rules to prevent catastrophic decisions."""
        # Rule 1: Never retrain during critical accuracy drops without high confidence
        if suggested_action == "retrain" and signals.get("accuracy_drop", 0) > 0.2:
            # Check if we have recent successful retrains
            retrain_success_rate = self._get_action_success_rate("retrain", signals)
            if retrain_success_rate < 0.7:  # Less than 70% success rate
                return "rollback"  # Safer alternative
        
        # Rule 2: Never fallback for severe data drift
        if suggested_action == "fallback" and signals.get("data_drift", 0) > 0.4:
            return "retrain"  # More aggressive action needed
        
        # Rule 3: Prefer rollback during peak hours (less disruptive)
        hour = datetime.now().hour
        if 9 <= hour <= 17 and suggested_action == "retrain":
            # Check if rollback has been successful recently
            rollback_success_rate = self._get_action_success_rate("rollback", signals)
            if rollback_success_rate > 0.8:
                return "rollback"  # Less disruptive during business hours
        
        return suggested_action
    
    def _get_action_success_rate(self, action: str, signals: Dict[str, float]) -> float:
        """Get success rate for an action in similar contexts."""
        # This is a simplified version - in reality would query experience buffer
        # For now, return default success probabilities from config
        action_config = self.config.get('actions', {}).get(action, {})
        return action_config.get('success_probability', 0.5)
    
    def update_policy(self, context_hash: str, action: str, 
                     reward: float, learning_rate: float = None) -> None:
        """Update policy with observed reward."""
        if context_hash not in self.contexts:
            # Context might have been created after decision but before update
            # This can happen in distributed systems
            return
        
        context = self.contexts[context_hash]
        
        if action not in context.arms:
            # Create arm if it doesn't exist
            context.arms[action] = BanditArm(
                name=action,
                estimated_value=1.0,
                visit_count=0,
                last_updated=datetime.now().isoformat()
            )
        
        # Update arm with reward
        lr = learning_rate if learning_rate is not None else self.learning_rate
        context.update_arm(action, reward, lr)
        
        # Auto-save periodically
        if context.total_visits % 10 == 0:
            self.save_state()
    
    def calculate_reward(self, outcome: str, recovery_time: float, 
                        performance_change: float, action_cost: float) -> float:
        """
        Calculate reward for a healing action.
        
        Higher reward is better.
        """
        # Base reward from outcome
        if outcome == "success":
            outcome_reward = 1.0
        elif outcome == "partial_success":
            outcome_reward = 0.5
        else:  # failure
            outcome_reward = -1.0
        
        # Time penalty (faster is better)
        time_penalty = min(recovery_time / 60.0, 1.0)  # Normalize to 0-1
        
        # Performance improvement reward
        performance_reward = performance_change * 2.0  # Scale performance changes
        
        # Cost penalty (lower cost is better)
        cost_penalty = action_cost / 10.0  # Normalize cost
        
        # Combined reward
        total_reward = (
            outcome_reward * 0.5 + 
            (1 - time_penalty) * 0.2 + 
            performance_reward * 0.2 + 
            (1 - cost_penalty) * 0.1
        )
        
        # Clip to reasonable range
        return max(-1.0, min(1.0, total_reward))
    
    def save_state(self) -> None:
        """Save bandit state to disk."""
        if not self.contexts:
            return
        
        # Ensure directory exists
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for serialization
        data = {
            "contexts": {},
            "exploration_rate": self.exploration_rate,
            "learning_rate": self.learning_rate,
            "saved_at": datetime.now().isoformat()
        }
        
        for ctx_hash, context in self.contexts.items():
            ctx_data = {
                "context_features": context.context_features,
                "total_visits": context.total_visits,
                "arms": {}
            }
            
            for arm_name, arm in context.arms.items():
                ctx_data["arms"][arm_name] = {
                    "estimated_value": arm.estimated_value,
                    "visit_count": arm.visit_count,
                    "last_updated": arm.last_updated
                }
            
            data["contexts"][ctx_hash] = ctx_data
        
        # Save to file
        with open(self.memory_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_state(self) -> None:
        """Load bandit state from disk."""
        if not self.memory_path.exists():
            return
        
        try:
            with open(self.memory_path, 'r') as f:
                data = json.load(f)
            
            self.exploration_rate = data.get("exploration_rate", 0.1)
            self.learning_rate = data.get("learning_rate", 0.01)
            
            for ctx_hash, ctx_data in data.get("contexts", {}).items():
                arms = {}
                for arm_name, arm_data in ctx_data.get("arms", {}).items():
                    arms[arm_name] = BanditArm(
                        name=arm_name,
                        estimated_value=arm_data["estimated_value"],
                        visit_count=arm_data["visit_count"],
                        last_updated=arm_data["last_updated"]
                    )
                
                context = ContextState(
                    context_hash=ctx_hash,
                    context_features=ctx_data["context_features"],
                    arms=arms,
                    total_visits=ctx_data["total_visits"]
                )
                
                self.contexts[ctx_hash] = context
            
            print(f"Loaded bandit policy with {len(self.contexts)} contexts")
            
        except Exception as e:
            print(f"Failed to load bandit state: {e}")
            self.contexts = {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get policy statistics."""
        stats = {
            "total_contexts": len(self.contexts),
            "exploration_rate": self.exploration_rate,
            "learning_rate": self.learning_rate,
            "context_details": {}
        }
        
        for ctx_hash, context in self.contexts.items():
            stats["context_details"][ctx_hash] = {
                "total_visits": context.total_visits,
                "arm_preferences": {
                    name: arm.estimated_value 
                    for name, arm in context.arms.items()
                },
                "most_visited_arm": max(
                    context.arms.items(), 
                    key=lambda x: x[1].visit_count
                )[0] if context.arms else None
            }
        
        return stats


# Factory function for integration
def create_bandit_policy(config_path: str = None) -> ContextualBanditPolicy:
    """Create and return a bandit policy engine."""
    if config_path is None:
        config_path = "adaptive/cost_model/action_costs.yaml"
    
    return ContextualBanditPolicy(config_path)


if __name__ == "__main__":
    # Test the bandit policy
    policy = ContextualBanditPolicy()
    
    # Test decision making
    test_signals = {"data_drift": 0.25, "accuracy_drop": 0.1}
    test_state = {"system_load": 0.7, "memory_usage": 0.6}
    
    action, metadata = policy.decide(test_signals, test_state)
    print(f"Selected action: {action}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")
    
    # Simulate update
    reward = policy.calculate_reward("success", 45.0, 0.15, 8.5)
    policy.update_policy(metadata["context_hash"], action, reward)
    
    # Get statistics
    stats = policy.get_statistics()
    print(f"\nPolicy statistics: {stats['total_contexts']} contexts")
