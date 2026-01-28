"""Explanation builder for self-healing ML system."""
import json
from typing import Dict, Any, List
from datetime import datetime

from decision_engine.decision_trace import DecisionTrace


class ExplanationBuilder:
    """Builds human-readable explanations from decision traces."""
    
    def __init__(self):
        """Initialize explanation builder."""
        pass
    
    def build_explanation(self, trace: DecisionTrace) -> Dict[str, Any]:
        """
        Build a comprehensive explanation from a decision trace.
        
        Args:
            trace: Decision trace object
            
        Returns:
            Dictionary with explanation details
        """
        explanation = {
            "summary": self._build_summary(trace),
            "detailed": self._build_detailed_explanation(trace),
            "recommendations": self._build_recommendations(trace),
            "visualization": self._build_visualization_data(trace),
            "timestamp": datetime.now().isoformat()
        }
        
        return explanation
    
    def _build_summary(self, trace: DecisionTrace) -> Dict[str, Any]:
        """Build a summary of the decision."""
        return {
            "decision": trace.decision,
            "severity": trace.severity,
            "reason": trace.reason,
            "policy": trace.policy_name,
            "confidence": trace.confidence,
            "execution_time_ms": trace.execution_time_ms
        }
    
    def _build_detailed_explanation(self, trace: DecisionTrace) -> Dict[str, Any]:
        """Build detailed explanation."""
        details = {
            "signals_analyzed": len(trace.signals),
            "signal_values": {
                key: round(float(value), 4) if isinstance(value, (int, float)) else str(value)
                for key, value in trace.signals.items()
            },
            "threshold_checks": self._extract_threshold_checks(trace),
            "decision_process": self._describe_decision_process(trace),
            "timestamp": trace.timestamp.isoformat() if hasattr(trace.timestamp, 'isoformat') else str(trace.timestamp),
            "trace_id": trace.trace_id
        }
        
        if trace.metadata:
            details["metadata"] = trace.metadata
        
        if trace.healing_action_result:
            details["healing_result"] = trace.healing_action_result
        
        return details
    
    def _extract_threshold_checks(self, trace: DecisionTrace) -> List[Dict[str, Any]]:
        """Extract threshold checks from signals."""
        checks = []
        
        # Common thresholds for ML monitoring
        common_thresholds = {
            "data_drift": {"warning": 0.1, "critical": 0.2},
            "accuracy_drop": {"warning": 0.05, "critical": 0.1},
            "anomaly_rate": {"warning": 0.03, "critical": 0.05},
            "latency_ms": {"warning": 100, "critical": 500}
        }
        
        for signal_name, signal_value in trace.signals.items():
            if isinstance(signal_value, (int, float)):
                check = {
                    "metric": signal_name,
                    "value": round(float(signal_value), 4),
                    "type": "monitoring_signal"
                }
                
                # Add threshold comparison if we have thresholds
                if signal_name in common_thresholds:
                    thresholds = common_thresholds[signal_name]
                    check["thresholds"] = thresholds
                    
                    if signal_value >= thresholds["critical"]:
                        check["status"] = "critical"
                    elif signal_value >= thresholds["warning"]:
                        check["status"] = "warning"
                    else:
                        check["status"] = "normal"
                
                checks.append(check)
        
        return checks
    
    def _describe_decision_process(self, trace: DecisionTrace) -> Dict[str, Any]:
        """Describe the decision-making process."""
        process = {
            "trigger": trace.reason or "Unknown trigger",
            "decision_maker": "policy_engine",
            "decision_type": "rule_based",
            "traceability": {
                "full_trace_available": True,
                "trace_id": trace.trace_id,
                "audit_ready": True
            }
        }
        
        # Add policy-specific details
        if trace.policy_name:
            process["policy"] = {
                "name": trace.policy_name,
                "applied": True
            }
        
        return process
    
    def _build_recommendations(self, trace: DecisionTrace) -> List[Dict[str, Any]]:
        """Build recommendations based on the decision."""
        recommendations = []
        
        # Base recommendation based on decision
        base_recommendation = {
            "decision": trace.decision,
            "priority": "high" if trace.severity in ["critical", "high"] else "medium",
            "action_required": trace.decision != "no_action"
        }
        
        if trace.decision == "retrain":
            base_recommendation.update({
                "recommendation": "Retrain the model with recent data",
                "steps": [
                    "Collect recent production data",
                    "Validate data quality",
                    "Retrain model with updated data",
                    "Validate new model performance",
                    "Deploy new model if performance improves"
                ],
                "estimated_time": "1-2 hours",
                "risk": "medium"
            })
        elif trace.decision == "rollback":
            base_recommendation.update({
                "recommendation": "Rollback to previous model version",
                "steps": [
                    "Identify stable previous version",
                    "Validate rollback candidate",
                    "Switch traffic to previous model",
                    "Monitor performance after rollback"
                ],
                "estimated_time": "5-10 minutes",
                "risk": "low"
            })
        elif trace.decision == "fallback":
            base_recommendation.update({
                "recommendation": "Switch to fallback model",
                "steps": [
                    "Activate fallback model",
                    "Redirect traffic to fallback",
                    "Investigate primary model issues",
                    "Plan recovery strategy"
                ],
                "estimated_time": "2-5 minutes",
                "risk": "low"
            })
        elif trace.decision == "no_action":
            base_recommendation.update({
                "recommendation": "Continue monitoring",
                "steps": [
                    "Maintain current configuration",
                    "Continue regular monitoring cycles",
                    "Review thresholds if needed"
                ],
                "estimated_time": "N/A",
                "risk": "none"
            })
        
        recommendations.append(base_recommendation)
        
        # Add additional recommendations based on signals
        if "data_drift" in trace.signals and trace.signals["data_drift"] > 0.15:
            recommendations.append({
                "type": "investigation",
                "recommendation": "Investigate data drift source",
                "priority": "medium",
                "details": f"Data drift score: {trace.signals['data_drift']:.3f}"
            })
        
        if "anomaly_rate" in trace.signals and trace.signals["anomaly_rate"] > 0.03:
            recommendations.append({
                "type": "investigation",
                "recommendation": "Investigate anomaly patterns",
                "priority": "medium",
                "details": f"Anomaly rate: {trace.signals['anomaly_rate']:.3f}"
            })
        
        return recommendations
    
    def _build_visualization_data(self, trace: DecisionTrace) -> Dict[str, Any]:
        """Build data for visualization."""
        return {
            "metrics": {
                key: float(value) if isinstance(value, (int, float)) else 0
                for key, value in trace.signals.items()
                if isinstance(value, (int, float))
            },
            "decision_timeline": {
                "timestamp": trace.timestamp.isoformat() if hasattr(trace.timestamp, 'isoformat') else str(trace.timestamp),
                "action": trace.decision
            },
            "severity_level": trace.severity
        }
    
    def to_json(self, trace: DecisionTrace) -> str:
        """Convert explanation to JSON string."""
        explanation = self.build_explanation(trace)
        return json.dumps(explanation, indent=2, default=str)
    
    def to_human_readable(self, trace: DecisionTrace) -> str:
        """Convert to human-readable text."""
        explanation = self.build_explanation(trace)
        
        lines = [
            "=" * 60,
            "SELF-HEALING ML SYSTEM - DECISION EXPLANATION",
            "=" * 60,
            f"Timestamp: {explanation['timestamp']}",
            f"Trace ID: {explanation['detailed']['trace_id']}",
            "",
            "SUMMARY",
            "-" * 40,
            f"Decision: {explanation['summary']['decision']}",
            f"Severity: {explanation['summary']['severity']}",
            f"Reason: {explanation['summary']['reason']}",
            f"Policy: {explanation['summary']['policy']}",
            "",
            "DETAILS",
            "-" * 40,
            f"Signals analyzed: {explanation['detailed']['signals_analyzed']}",
        ]
        
        # Add signal values
        for metric, value in explanation['detailed']['signal_values'].items():
            lines.append(f"  - {metric}: {value}")
        
        lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 40,
        ])
        
        # Add recommendations
        for i, rec in enumerate(explanation['recommendations'], 1):
            lines.append(f"{i}. {rec['recommendation']} (Priority: {rec['priority']})")
            if 'steps' in rec:
                for step in rec['steps']:
                    lines.append(f"   • {step}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# Legacy function for backward compatibility
def build_explanation(trace):
    """Legacy function for building explanations."""
    builder = ExplanationBuilder()
    return builder.build_explanation(trace)
