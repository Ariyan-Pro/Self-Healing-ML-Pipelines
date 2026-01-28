# orchestration/canary_controller.py
"""
Canary Rollout Controller for Gradual Production Deployment.
Implements progressive traffic shifting and validation.
"""

import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


class RolloutStage(Enum):
    """Stages of canary rollout."""
    SHADOW = "shadow"  # 0% traffic, only logging
    CANARY_1 = "canary_1"  # 1% traffic
    CANARY_5 = "canary_5"  # 5% traffic
    CANARY_25 = "canary_25"  # 25% traffic
    CANARY_50 = "canary_50"  # 50% traffic
    FULL = "full"  # 100% traffic


class CanaryController:
    """Controller for gradual rollout of healing decisions."""
    
    def __init__(self, config_path: str = "configs/canary_config.yaml"):
        self.config = self._load_config(config_path)
        self.current_stage = RolloutStage.SHADOW
        self.rollout_start_time = datetime.now()
        self.metrics = {
            "decisions_made": 0,
            "decisions_executed": 0,
            "successes": 0,
            "failures": 0,
            "rollbacks": 0
        }
        self.stage_history = []
    
    def _load_config(self, config_path: str) -> Dict:
        """Load canary rollout configuration."""
        default_config = {
            "stages": {
                "shadow": {
                    "traffic_percentage": 0.0,
                    "duration_hours": 24,
                    "success_threshold": 0.95,
                    "max_failures": 3
                },
                "canary_1": {
                    "traffic_percentage": 0.01,
                    "duration_hours": 12,
                    "success_threshold": 0.98,
                    "max_failures": 2
                },
                "canary_5": {
                    "traffic_percentage": 0.05,
                    "duration_hours": 6,
                    "success_threshold": 0.99,
                    "max_failures": 1
                },
                "canary_25": {
                    "traffic_percentage": 0.25,
                    "duration_hours": 4,
                    "success_threshold": 0.995,
                    "max_failures": 1
                },
                "canary_50": {
                    "traffic_percentage": 0.5,
                    "duration_hours": 2,
                    "success_threshold": 0.998,
                    "max_failures": 0
                },
                "full": {
                    "traffic_percentage": 1.0,
                    "duration_hours": 0,
                    "success_threshold": 1.0,
                    "max_failures": 0
                }
            },
            "validation_checks": [
                "accuracy_not_degraded",
                "latency_within_limits",
                "no_cascading_failures",
                "cost_within_budget"
            ],
            "rollback_triggers": [
                "consecutive_failures > threshold",
                "sla_violation_detected",
                "cost_exceeded_budget",
                "manual_override"
            ]
        }
        return default_config
    
    def should_execute_decision(self, decision: str, context: Dict) -> Tuple[bool, str]:
        """
        Determine if a decision should be executed based on canary stage.
        
        Args:
            decision: The healing decision
            context: Current context (drift level, system load, etc.)
            
        Returns:
            Tuple of (should_execute, reason)
        """
        self.metrics["decisions_made"] += 1
        
        # Always execute in shadow mode (for learning)
        if self.current_stage == RolloutStage.SHADOW:
            self.metrics["decisions_executed"] += 1
            return True, "shadow_mode_learning"
        
        # Check if we should progress to next stage
        if self._should_progress_stage():
            self._progress_stage()
        
        # Get traffic percentage for current stage
        traffic_pct = self.config["stages"][self.current_stage.value]["traffic_percentage"]
        
        # Randomly decide based on traffic percentage
        should_execute = random.random() < traffic_pct
        
        if should_execute:
            self.metrics["decisions_executed"] += 1
            reason = f"canary_stage_{self.current_stage.value}_traffic_{traffic_pct}"
        else:
            reason = f"traffic_throttled_stage_{self.current_stage.value}"
        
        return should_execute, reason
    
    def record_outcome(self, decision: str, outcome: Dict, executed: bool):
        """
        Record outcome of a decision for canary analysis.
        
        Args:
            decision: Healing decision
            outcome: Outcome dictionary
            executed: Whether decision was executed
        """
        if not executed:
            return
        
        success = outcome.get("status") == "success"
        
        if success:
            self.metrics["successes"] += 1
        else:
            self.metrics["failures"] += 1
            
            # Check if we need to rollback stage
            if self._should_rollback_stage():
                self._rollback_stage()
    
    def _should_progress_stage(self) -> bool:
        """Determine if we should progress to next stage."""
        current_stage_config = self.config["stages"][self.current_stage.value]
        
        # Check duration requirement
        stage_duration = timedelta(hours=current_stage_config["duration_hours"])
        time_in_stage = datetime.now() - self.rollout_start_time
        
        if time_in_stage < stage_duration:
            return False
        
        # Check success threshold
        total_executions = self.metrics["decisions_executed"]
        if total_executions == 0:
            return False
        
        success_rate = self.metrics["successes"] / total_executions
        if success_rate < current_stage_config["success_threshold"]:
            return False
        
        # Check failure count
        if self.metrics["failures"] > current_stage_config["max_failures"]:
            return False
        
        return True
    
    def _progress_stage(self):
        """Progress to next rollout stage."""
        stages = list(RolloutStage)
        current_index = stages.index(self.current_stage)
        
        if current_index < len(stages) - 1:
            new_stage = stages[current_index + 1]
            
            # Record stage transition
            transition = {
                "from": self.current_stage.value,
                "to": new_stage.value,
                "timestamp": datetime.now().isoformat(),
                "metrics_at_transition": self.metrics.copy()
            }
            self.stage_history.append(transition)
            
            # Update stage
            self.current_stage = new_stage
            self.rollout_start_time = datetime.now()
            
            # Reset stage-specific metrics
            self.metrics.update({
                "decisions_made": 0,
                "decisions_executed": 0,
                "successes": 0,
                "failures": 0
            })
            
            print(f"Progressed to stage: {new_stage.value}")
    
    def _should_rollback_stage(self) -> bool:
        """Determine if we should rollback to previous stage."""
        current_stage_config = self.config["stages"][self.current_stage.value]
        
        # Check consecutive failures
        if self.metrics["failures"] > current_stage_config["max_failures"]:
            return True
        
        # Check if we're in early stages and seeing issues
        if self.current_stage in [RolloutStage.CANARY_1, RolloutStage.CANARY_5]:
            failure_rate = self.metrics["failures"] / max(self.metrics["decisions_executed"], 1)
            if failure_rate > 0.1:  # 10% failure rate
                return True
        
        return False
    
    def _rollback_stage(self):
        """Rollback to previous stage."""
        stages = list(RolloutStage)
        current_index = stages.index(self.current_stage)
        
        if current_index > 0:
            new_stage = stages[current_index - 1]
            
            # Record rollback
            rollback = {
                "from": self.current_stage.value,
                "to": new_stage.value,
                "timestamp": datetime.now().isoformat(),
                "reason": "failure_threshold_exceeded",
                "metrics_at_rollback": self.metrics.copy()
            }
            self.stage_history.append(rollback)
            
            # Update stage
            self.current_stage = new_stage
            self.rollout_start_time = datetime.now()
            
            # Reset metrics
            self.metrics.update({
                "decisions_made": 0,
                "decisions_executed": 0,
                "successes": 0,
                "failures": 0
            })
            
            self.metrics["rollbacks"] += 1
            print(f"Rolled back to stage: {new_stage.value}")
    
    def get_rollout_status(self) -> Dict:
        """Get current rollout status."""
        current_config = self.config["stages"][self.current_stage.value]
        
        return {
            "current_stage": self.current_stage.value,
            "traffic_percentage": current_config["traffic_percentage"],
            "time_in_stage": (datetime.now() - self.rollout_start_time).total_seconds(),
            "stage_duration_hours": current_config["duration_hours"],
            "metrics": self.metrics,
            "stage_history": self.stage_history[-5:],
            "success_rate": (
                self.metrics["successes"] / max(self.metrics["decisions_executed"], 1)
                if self.metrics["decisions_executed"] > 0 else 0
            )
        }
    
    def generate_rollout_report(self, output_path: Optional[str] = None) -> Dict:
        """Generate comprehensive rollout report."""
        report = {
            "rollout_summary": {
                "current_stage": self.current_stage.value,
                "start_time": self.rollout_start_time.isoformat(),
                "total_duration_hours": (
                    datetime.now() - self.rollout_start_time
                ).total_seconds() / 3600,
                "total_decisions": self.metrics["decisions_made"],
                "execution_rate": (
                    self.metrics["decisions_executed"] / 
                    max(self.metrics["decisions_made"], 1)
                ),
                "success_rate": (
                    self.metrics["successes"] / 
                    max(self.metrics["decisions_executed"], 1)
                    if self.metrics["decisions_executed"] > 0 else 0
                ),
                "rollbacks": self.metrics["rollbacks"]
            },
            "stage_performance": [],
            "recommendations": self._generate_rollout_recommendations()
        }
        
        # Analyze each stage's performance
        for transition in self.stage_history:
            if "metrics_at_transition" in transition:
                metrics = transition["metrics_at_transition"]
                if metrics["decisions_executed"] > 0:
                    success_rate = metrics["successes"] / metrics["decisions_executed"]
                    report["stage_performance"].append({
                        "stage": transition.get("to", "unknown"),
                        "success_rate": success_rate,
                        "decisions_executed": metrics["decisions_executed"],
                        "timestamp": transition["timestamp"]
                    })
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _generate_rollout_recommendations(self) -> List[str]:
        """Generate recommendations based on rollout performance."""
        recommendations = []
        
        success_rate = (
            self.metrics["successes"] / max(self.metrics["decisions_executed"], 1)
            if self.metrics["decisions_executed"] > 0 else 0
        )
        
        if success_rate < 0.9:
            recommendations.append(
                f"Low success rate ({success_rate:.1%}): "
                "Consider adjusting confidence thresholds or cost models"
            )
        
        if self.metrics["rollbacks"] > 2:
            recommendations.append(
                f"Multiple rollbacks ({self.metrics['rollbacks']}): "
                "System may be too aggressive. Increase stage durations."
            )
        
        execution_rate = (
            self.metrics["decisions_executed"] / max(self.metrics["decisions_made"], 1)
        )
        if execution_rate < 0.1 and self.current_stage != RolloutStage.SHADOW:
            recommendations.append(
                f"Low execution rate ({execution_rate:.1%}): "
                "Consider progressing stages more quickly"
            )
        
        return recommendations


# Example usage
if __name__ == "__main__":
    canary = CanaryController()
    
    print("Initial rollout status:", canary.get_rollout_status())
    
    # Simulate some decisions
    for i in range(10):
        should_execute, reason = canary.should_execute_decision(
            "retrain", {"drift": 0.2}
        )
        
        if should_execute:
            # Simulate outcome
            outcome = {"status": "success" if random.random() > 0.1 else "failed"}
            canary.record_outcome("retrain", outcome, True)
    
    print("\nAfter 10 decisions:", canary.get_rollout_status())
    report = canary.generate_rollout_report("canary_report.json")
    print("\nCanary report generated")
