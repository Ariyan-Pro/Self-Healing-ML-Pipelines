"""
Experience Logger for Adaptive Self-Healing
Logs healing actions and outcomes for learning.
"""

import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd


@dataclass
class HealingExperience:
    """Single healing experience/outcome."""
    timestamp: str
    signals: Dict[str, float]  # Input signals
    action_taken: str  # What action was performed
    action_cost: float  # Calculated cost of action
    outcome: str  # 'success', 'partial_success', 'failure'
    recovery_time: float  # Seconds to recover
    performance_change: float  # Performance delta after action
    confidence_scores: Dict[str, float]  # Confidence for each signal
    context_hash: str  # Hash of context for deduplication
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class ExperienceBuffer:
    """
    Experience replay buffer for healing decisions.
    Stores experiences and provides sampling for learning.
    """
    
    def __init__(self, max_size: int = 1000, storage_path: str = "adaptive/memory/experiences.json"):
        self.max_size = max_size
        self.storage_path = Path(storage_path)
        self.experiences: List[HealingExperience] = []
        
        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing experiences if available
        self.load_experiences()
    
    def add_experience(self, experience: HealingExperience) -> None:
        """Add a new experience to the buffer."""
        self.experiences.append(experience)
        
        # Maintain max size (FIFO)
        if len(self.experiences) > self.max_size:
            self.experiences.pop(0)
        
        # Auto-save
        self.save_experiences()
    
    def create_experience(
        self,
        signals: Dict[str, float],
        action_taken: str,
        action_cost: float,
        outcome: str,
        recovery_time: float,
        performance_change: float = 0.0,
        confidence_scores: Optional[Dict[str, float]] = None
    ) -> HealingExperience:
        """Create and add a new experience."""
        
        if confidence_scores is None:
            confidence_scores = {sig: 0.8 for sig in signals.keys()}
        
        # Create context hash for deduplication
        context_data = {
            "signals": signals,
            "action_taken": action_taken,
            "confidence": confidence_scores
        }
        context_hash = str(hash(json.dumps(context_data, sort_keys=True)))
        
        experience = HealingExperience(
            timestamp=datetime.now().isoformat(),
            signals=signals,
            action_taken=action_taken,
            action_cost=action_cost,
            outcome=outcome,
            recovery_time=recovery_time,
            performance_change=performance_change,
            confidence_scores=confidence_scores,
            context_hash=context_hash
        )
        
        self.add_experience(experience)
        return experience
    
    def get_recent_experiences(self, n: int = 100) -> List[HealingExperience]:
        """Get the n most recent experiences."""
        return self.experiences[-n:] if self.experiences else []
    
    def get_experiences_by_action(self, action: str) -> List[HealingExperience]:
        """Get all experiences for a specific action."""
        return [exp for exp in self.experiences if exp.action_taken == action]
    
    def get_experiences_by_signal(self, signal_pattern: Dict[str, float]) -> List[HealingExperience]:
        """Get experiences with similar signal patterns."""
        similar_experiences = []
        
        for exp in self.experiences:
            similarity = self._calculate_signal_similarity(exp.signals, signal_pattern)
            if similarity > 0.7:  # 70% similarity threshold
                similar_experiences.append((exp, similarity))
        
        # Sort by similarity
        similar_experiences.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _ in similar_experiences]
    
    def _calculate_signal_similarity(self, signals1: Dict[str, float], 
                                   signals2: Dict[str, float]) -> float:
        """Calculate similarity between two signal sets."""
        if not signals1 or not signals2:
            return 0.0
        
        # Get union of all signals
        all_signals = set(signals1.keys()) | set(signals2.keys())
        
        if not all_signals:
            return 0.0
        
        similarities = []
        for signal in all_signals:
            val1 = signals1.get(signal, 0.0)
            val2 = signals2.get(signal, 0.0)
            
            # Calculate similarity for this signal (1 - normalized difference)
            diff = abs(val1 - val2)
            max_val = max(abs(val1), abs(val2), 1e-10)
            similarity = 1.0 - (diff / max_val)
            similarities.append(max(0.0, similarity))
        
        return sum(similarities) / len(similarities)
    
    def calculate_action_statistics(self, action: str) -> Dict[str, float]:
        """Calculate statistics for a specific action."""
        action_experiences = self.get_experiences_by_action(action)
        
        if not action_experiences:
            return {
                "count": 0,
                "success_rate": 0.0,
                "avg_recovery_time": 0.0,
                "avg_cost": 0.0,
                "avg_performance_change": 0.0
            }
        
        successes = sum(1 for exp in action_experiences if exp.outcome == "success")
        total = len(action_experiences)
        
        return {
            "count": total,
            "success_rate": successes / total if total > 0 else 0.0,
            "avg_recovery_time": sum(exp.recovery_time for exp in action_experiences) / total,
            "avg_cost": sum(exp.action_cost for exp in action_experiences) / total,
            "avg_performance_change": sum(exp.performance_change for exp in action_experiences) / total
        }
    
    def save_experiences(self) -> None:
        """Save experiences to disk."""
        if not self.experiences:
            return
        
        data = [exp.to_dict() for exp in self.experiences]
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_experiences(self) -> None:
        """Load experiences from disk."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.experiences = []
            for item in data:
                # Handle missing fields in old format
                if 'confidence_scores' not in item:
                    item['confidence_scores'] = {}
                if 'context_hash' not in item:
                    item['context_hash'] = str(hash(json.dumps(item.get('signals', {}), sort_keys=True)))
                
                self.experiences.append(HealingExperience(**item))
            
            print(f"Loaded {len(self.experiences)} experiences from {self.storage_path}")
            
        except Exception as e:
            print(f"Failed to load experiences: {e}")
            self.experiences = []
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert experiences to pandas DataFrame for analysis."""
        if not self.experiences:
            return pd.DataFrame()
        
        data = [exp.to_dict() for exp in self.experiences]
        return pd.DataFrame(data)


class ExperienceLogger:
    """
    High-level experience logger for easy integration.
    """
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer = ExperienceBuffer(max_size=buffer_size)
        self.current_recovery_start = None
    
    def start_recovery_timer(self) -> None:
        """Start timer for measuring recovery time."""
        self.current_recovery_start = time.time()
    
    def stop_recovery_timer(self) -> float:
        """Stop timer and return elapsed time."""
        if self.current_recovery_start is None:
            return 0.0
        
        elapsed = time.time() - self.current_recovery_start
        self.current_recovery_start = None
        return elapsed
    
    def log_healing_action(
        self,
        signals: Dict[str, float],
        action_taken: str,
        action_cost: float,
        outcome: str,
        performance_change: float = 0.0,
        confidence_scores: Optional[Dict[str, float]] = None
    ) -> HealingExperience:
        """
        Log a healing action with outcome.
        
        Args:
            signals: Input signals that triggered the action
            action_taken: Action performed
            action_cost: Calculated cost of the action
            outcome: 'success', 'partial_success', or 'failure'
            performance_change: Change in performance metrics
            confidence_scores: Confidence for each signal
            
        Returns:
            The logged experience
        """
        recovery_time = self.stop_recovery_timer()
        
        experience = self.buffer.create_experience(
            signals=signals,
            action_taken=action_taken,
            action_cost=action_cost,
            outcome=outcome,
            recovery_time=recovery_time,
            performance_change=performance_change,
            confidence_scores=confidence_scores
        )
        
        print(f"📝 Logged experience: {action_taken} → {outcome} (cost: {action_cost:.2f}, time: {recovery_time:.1f}s)")
        return experience
    
    def get_action_recommendation(self, signals: Dict[str, float]) -> Dict[str, Any]:
        """
        Get action recommendation based on similar past experiences.
        
        Returns:
            Dictionary with recommended action and statistics
        """
        similar_experiences = self.buffer.get_experiences_by_signal(signals)
        
        if not similar_experiences:
            return {"recommendation": "no_action", "confidence": 0.0, "similar_cases": 0}
        
        # Count outcomes by action
        action_outcomes = {}
        for exp in similar_experiences:
            if exp.action_taken not in action_outcomes:
                action_outcomes[exp.action_taken] = {"success": 0, "total": 0}
            
            action_outcomes[exp.action_taken]["total"] += 1
            if exp.outcome == "success":
                action_outcomes[exp.action_taken]["success"] += 1
        
        # Calculate success rates
        success_rates = {}
        for action, counts in action_outcomes.items():
            success_rates[action] = counts["success"] / counts["total"]
        
        # Recommend action with highest success rate
        if success_rates:
            best_action = max(success_rates.items(), key=lambda x: x[1])
            return {
                "recommendation": best_action[0],
                "confidence": best_action[1],
                "similar_cases": len(similar_experiences),
                "success_rates": success_rates
            }
        
        return {"recommendation": "no_action", "confidence": 0.0, "similar_cases": 0}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {}
        
        # Overall statistics
        stats["total_experiences"] = len(self.buffer.experiences)
        
        if self.buffer.experiences:
            stats["overall_success_rate"] = sum(
                1 for exp in self.buffer.experiences if exp.outcome == "success"
            ) / len(self.buffer.experiences)
        
        # Per-action statistics
        actions = ["retrain", "rollback", "fallback", "no_action"]
        stats["actions"] = {}
        for action in actions:
            stats["actions"][action] = self.buffer.calculate_action_statistics(action)
        
        return stats


# Factory function for easy integration
def create_experience_logger(buffer_size: int = 1000) -> ExperienceLogger:
    """Create and return an experience logger."""
    return ExperienceLogger(buffer_size=buffer_size)


if __name__ == "__main__":
    # Test the experience logger
    logger = ExperienceLogger(buffer_size=10)
    
    # Log some test experiences
    logger.start_recovery_timer()
    time.sleep(0.1)  # Simulate recovery time
    
    logger.log_healing_action(
        signals={"data_drift": 0.25, "accuracy_drop": 0.1},
        action_taken="retrain",
        action_cost=8.5,
        outcome="success",
        performance_change=0.15
    )
    
    logger.start_recovery_timer()
    time.sleep(0.05)
    
    logger.log_healing_action(
        signals={"anomaly_rate": 0.3},
        action_taken="fallback",
        action_cost=1.2,
        outcome="partial_success",
        performance_change=-0.05
    )
    
    # Get statistics
    stats = logger.get_statistics()
    print("\nStatistics:")
    print(json.dumps(stats, indent=2))
    
    # Get recommendation
    recommendation = logger.get_action_recommendation({"data_drift": 0.2})
    print(f"\nRecommendation for data_drift=0.2: {recommendation}")
