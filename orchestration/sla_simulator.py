# orchestration/sla_simulator.py
"""
SLA (Service Level Agreement) Simulation for Production Hardening
Simulates business impact of healing decisions.
"""

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class SLASimulator:
    """Simulate SLA impacts for production hardening."""
    
    def __init__(self, sla_config_path: str = "configs/sla_config.yaml"):
        self.sla_config = self._load_sla_config(sla_config_path)
        self.violations = []
        self.costs = []
        
    def _load_sla_config(self, config_path: str) -> Dict:
        """Load SLA configuration."""
        # Default SLA configuration
        default_config = {
            "availability": {
                "target": 0.999,  # 99.9% availability
                "penalty_per_minute": 100.0,  # $ per minute of downtime
                "measurement_window": timedelta(hours=1)
            },
            "latency": {
                "p95_target_ms": 100.0,
                "penalty_per_ms_over": 0.1  # $ per ms over target
            },
            "accuracy": {
                "minimum": 0.85,
                "degradation_penalty": 50.0  # $ per % point below minimum
            },
            "cost_limits": {
                "max_retrain_cost": 500.0,
                "max_rollback_cost": 200.0,
                "max_fallback_cost": 100.0
            }
        }
        
        # In production, load from YAML
        return default_config
    
    def simulate_sla_impact(self, action: str, outcome: Dict, 
                           start_time: datetime, end_time: datetime) -> Dict:
        """
        Simulate SLA impact of a healing action.
        
        Args:
            action: Healing action taken
            outcome: Action outcome dictionary
            start_time: When action started
            end_time: When action completed
            
        Returns:
            Dictionary with SLA impact metrics
        """
        duration = end_time - start_time
        
        # Calculate availability impact
        availability_impact = self._calculate_availability_impact(
            action, duration, outcome
        )
        
        # Calculate performance impact
        performance_impact = self._calculate_performance_impact(
            action, outcome
        )
        
        # Calculate financial impact
        financial_impact = self._calculate_financial_impact(
            action, availability_impact, performance_impact
        )
        
        # Record violation if any
        impact = {
            "action": action,
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "availability_impact": availability_impact,
            "performance_impact": performance_impact,
            "financial_impact": financial_impact,
            "sla_violation": any([
                availability_impact.get("violation", False),
                performance_impact.get("violation", False),
                financial_impact.get("violation", False)
            ])
        }
        
        if impact["sla_violation"]:
            self.violations.append(impact)
            self.costs.append(financial_impact.get("total_penalty", 0))
        
        return impact
    
    def _calculate_availability_impact(self, action: str, 
                                      duration: timedelta,
                                      outcome: Dict) -> Dict:
        """Calculate availability SLA impact."""
        downtime_minutes = duration.total_seconds() / 60
        
        # Different actions have different availability expectations
        availability_targets = {
            "retrain": 0.95,  # Retraining can take longer
            "rollback": 0.98,
            "fallback": 0.99,
            "no_action": 0.999
        }
        
        target = availability_targets.get(action, 0.99)
        actual_availability = 1.0 - (downtime_minutes / 60)  # Simplified
        
        violation = actual_availability < target
        penalty = 0
        
        if violation:
            penalty = (target - actual_availability) * 60 * \
                     self.sla_config["availability"]["penalty_per_minute"]
        
        return {
            "target": target,
            "actual": actual_availability,
            "downtime_minutes": downtime_minutes,
            "violation": violation,
            "penalty": penalty
        }
    
    def _calculate_performance_impact(self, action: str, 
                                     outcome: Dict) -> Dict:
        """Calculate performance SLA impact."""
        # Extract performance metrics from outcome
        accuracy_change = outcome.get("performance_change", 0)
        recovery_time = outcome.get("recovery_time", 0)
        
        accuracy_violation = accuracy_change < 0  # Negative = degradation
        latency_violation = recovery_time > \
            self.sla_config["latency"]["p95_target_ms"]
        
        penalty = 0
        
        if accuracy_violation:
            degradation = abs(accuracy_change) * 100  # Convert to percentage
            penalty += degradation * \
                      self.sla_config["accuracy"]["degradation_penalty"]
        
        if latency_violation:
            excess_latency = recovery_time - \
                           self.sla_config["latency"]["p95_target_ms"]
            penalty += excess_latency * \
                      self.sla_config["latency"]["penalty_per_ms_over"]
        
        return {
            "accuracy_change": accuracy_change,
            "recovery_time_ms": recovery_time,
            "accuracy_violation": accuracy_violation,
            "latency_violation": latency_violation,
            "penalty": penalty
        }
    
    def _calculate_financial_impact(self, action: str,
                                   availability_impact: Dict,
                                   performance_impact: Dict) -> Dict:
        """Calculate total financial impact."""
        total_penalty = (
            availability_impact.get("penalty", 0) +
            performance_impact.get("penalty", 0)
        )
        
        # Check if action cost exceeds limits
        cost_limit = self.sla_config["cost_limits"].get(
            f"max_{action}_cost", float("inf")
        )
        cost_violation = total_penalty > cost_limit
        
        return {
            "total_penalty": total_penalty,
            "cost_limit": cost_limit,
            "cost_violation": cost_violation,
            "violation": cost_violation
        }
    
    def generate_sla_report(self, output_path: Optional[str] = None) -> Dict:
        """Generate comprehensive SLA report."""
        report = {
            "summary": {
                "total_simulations": len(self.violations) + len(self.costs),
                "total_violations": len(self.violations),
                "total_penalty": sum(self.costs),
                "average_penalty": np.mean(self.costs) if self.costs else 0,
                "violation_rate": len(self.violations) / max(len(self.costs), 1)
            },
            "violations_by_action": {},
            "timeline": self.violations[-10:] if self.violations else [],  # Last 10
            "recommendations": self._generate_recommendations()
        }
        
        # Count violations by action type
        for violation in self.violations:
            action = violation.get("action", "unknown")
            report["violations_by_action"][action] = \
                report["violations_by_action"].get(action, 0) + 1
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on SLA violations."""
        recommendations = []
        
        if not self.violations:
            recommendations.append("No SLA violations detected in simulation")
            return recommendations
        
        # Analyze patterns
        if len(self.violations) > 5:
            recommendations.append(
                "High violation rate: Consider adjusting cost models or "
                "increasing confidence thresholds"
            )
        
        # Check for expensive actions
        avg_penalty = np.mean(self.costs) if self.costs else 0
        if avg_penalty > 50:
            recommendations.append(
                f"High average penalty (): "
                "Review action cost assumptions"
            )
        
        # Check specific actions
        actions_with_violations = set(
            v.get("action") for v in self.violations
        )
        for action in actions_with_violations:
            recommendations.append(
                f"Action '{action}' causing SLA violations: "
                "Consider alternative strategies or improved implementation"
            )
        
        return recommendations
    
    def reset(self):
        """Reset simulation state."""
        self.violations = []
        self.costs = []


# Example usage
if __name__ == "__main__":
    simulator = SLASimulator()
    
    # Simulate a retrain action
    outcome = {
        "performance_change": -0.1,  # 10% degradation
        "recovery_time": 150.0,  # 150ms recovery
        "status": "success"
    }
    
    impact = simulator.simulate_sla_impact(
        action="retrain",
        outcome=outcome,
        start_time=datetime.now() - timedelta(minutes=5),
        end_time=datetime.now()
    )
    
    print("SLA Impact:", json.dumps(impact, indent=2))
    report = simulator.generate_sla_report("sla_simulation_report.json")
    print("\nSLA Report generated")
