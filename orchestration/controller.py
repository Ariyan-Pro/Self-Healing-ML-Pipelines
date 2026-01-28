"""Main controller for self-healing ML system."""
import logging
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import time

from monitoring.data_drift import DataDriftDetector
from decision_engine.policy_engine import PolicyEngine
from healing.healing_actions import HealingActions
from explainability.explanation_builder import ExplanationBuilder

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Current state of the self-healing system."""
    status: str = "idle"
    last_check: Optional[datetime] = None
    active_alerts: Dict[str, Any] = None
    model_version: str = "unknown"
    healing_actions_count: Dict[str, int] = None
    
    def __post_init__(self):
        if self.active_alerts is None:
            self.active_alerts = {}
        if self.healing_actions_count is None:
            self.healing_actions_count = {}


class SelfHealingController:
    """
    Main controller orchestrating the self-healing loop.
    
    Coordinates monitoring, detection, decision, healing, and explanation.
    """
    
    def __init__(self, config_path: str = "configs/pipeline.yaml"):
        """
        Initialize the controller.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.state = SystemState()
        
        # Initialize components
        self.drift_detector = DataDriftDetector(
            method="ks",  # Hardcoded for now
            threshold=self.config["monitoring"]["drift"]["threshold"]
        )
        
        self.policy_engine = PolicyEngine(
            config_path="configs/healing_policies.yaml"
        )
        
        self.healing_actions = HealingActions(
            config=self.config.get("healing", {})
        )
        
        self.explanation_builder = ExplanationBuilder()
        
        # Metrics tracking
        self.metrics = {
            "cycles_completed": 0,
            "drift_detections": 0,
            "healing_actions": 0,
            "total_execution_time_ms": 0
        }
        
        logger.info("SelfHealingController initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def run_monitoring_cycle(self, inference_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single monitoring cycle.
        
        Args:
            inference_data: Current inference data with features and predictions
        
        Returns:
            Dictionary of monitoring signals
        """
        logger.debug("Running monitoring cycle")
        
        signals = {}
        
        # Monitor data drift
        if "features" in inference_data and "reference_features" in inference_data:
            drift_results = self.drift_detector.detect_batch_drift(
                inference_data["reference_features"],
                inference_data["features"]
            )
            
            # Aggregate drift scores
            if drift_results:
                avg_drift = sum(r.drift_score for r in drift_results.values()) / len(drift_results)
                max_drift = max(r.drift_score for r in drift_results.values())
                
                signals["data_drift"] = avg_drift
                signals["max_feature_drift"] = max_drift
                signals["drift_features"] = {
                    name: r.drift_score 
                    for name, r in drift_results.items() 
                    if r.is_drift
                }
                
                if any(r.is_drift for r in drift_results.values()):
                    self.metrics["drift_detections"] += 1
        
        # Monitor anomaly rate
        if "predictions" in inference_data:
            predictions = inference_data["predictions"]
            anomaly_threshold = self.config["monitoring"]["anomalies"]["threshold"]
            
            # Simple anomaly detection based on prediction distribution
            import numpy as np
            z_scores = (predictions - np.mean(predictions)) / np.std(predictions)
            anomaly_rate = np.mean(np.abs(z_scores) > anomaly_threshold)
            
            signals["anomaly_rate"] = float(anomaly_rate)
        
        # Monitor performance metrics
        if "performance" in inference_data:
            signals.update(inference_data["performance"])
        
        logger.debug(f"Monitoring signals: {signals}")
        return signals
    
    def run_healing_cycle(self, inference_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a complete self-healing cycle.
        
        Args:
            inference_data: Current inference data
        
        Returns:
            Complete cycle results including action and explanation
        """
        logger.info("Starting self-healing cycle")
        start_time = time.time()
        
        try:
            self.state.status = "monitoring"
            self.state.last_check = datetime.now()
            
            # 1. Monitoring & Detection
            signals = self.run_monitoring_cycle(inference_data)
            
            # 2. Decision
            self.state.status = "decision"
            action, decision_trace = self.policy_engine.decide(signals)
            
            # 3. Healing (if needed)
            healing_result = None
            if action != "no_action":
                self.state.status = "healing"
                
                # Execute healing action
                healing_result = self.healing_actions.execute_action(
                    action,
                    data=inference_data.get("features"),
                    labels=inference_data.get("labels")
                )
                
                # Update state
                if healing_result.get("status") == "success":
                    self.state.healing_actions_count[action] = \
                        self.state.healing_actions_count.get(action, 0) + 1
                    self.metrics["healing_actions"] += 1
                
                decision_trace.set_healing_result(healing_result)
            
            # 4. Explanation
            self.state.status = "explanation"
            explanation = self.explanation_builder.build_explanation(decision_trace)
            
            # Update metrics
            execution_time = (time.time() - start_time) * 1000
            self.metrics["cycles_completed"] += 1
            self.metrics["total_execution_time_ms"] += execution_time
            
            self.state.status = "idle"
            
            cycle_result = {
                "cycle_id": decision_trace.trace_id,
                "timestamp": datetime.now().isoformat(),
                "signals": signals,
                "action": action,
                "decision_trace": decision_trace.to_dict(),
                "healing_result": healing_result,
                "explanation": explanation,
                "execution_time_ms": execution_time,
                "system_state": {
                    "status": self.state.status,
                    "model_version": self.state.model_version,
                    "active_alerts": self.state.active_alerts
                }
            }
            
            logger.info(f"Cycle completed: action={action}, "
                       f"time={execution_time:.2f}ms")
            
            return cycle_result
            
        except Exception as e:
            logger.error(f"Healing cycle failed: {e}")
            self.state.status = "error"
            
            return {
                "cycle_id": str(datetime.now().timestamp()),
                "timestamp": datetime.now().isoformat(),
                "action": "error",
                "error": str(e),
                "system_state": {
                    "status": "error",
                    "last_check": self.state.last_check.isoformat() 
                    if self.state.last_check else None
                }
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        avg_cycle_time = (
            self.metrics["total_execution_time_ms"] / self.metrics["cycles_completed"]
            if self.metrics["cycles_completed"] > 0 else 0
        )
        
        return {
            "state": self.state.status,
            "last_check": self.state.last_check.isoformat() 
            if self.state.last_check else None,
            "metrics": {
                "cycles_completed": self.metrics["cycles_completed"],
                "drift_detections": self.metrics["drift_detections"],
                "healing_actions": self.metrics["healing_actions"],
                "avg_cycle_time_ms": avg_cycle_time
            },
            "policy_stats": self.policy_engine.get_policy_stats(),
            "healing_counts": dict(self.state.healing_actions_count)
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics = {
            "cycles_completed": 0,
            "drift_detections": 0,
            "healing_actions": 0,
            "total_execution_time_ms": 0
        }
        self.state.healing_actions_count = {}
        logger.info("Metrics reset")


