"""State management for self-healing ML system."""
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """System state representation."""
    status: str = "idle"  # idle, monitoring, healing, error
    last_cycle_time: Optional[datetime] = None
    last_healing_action: Optional[str] = None
    last_healing_time: Optional[datetime] = None
    model_version: str = "unknown"
    active_alerts: List[str] = None
    metrics: Dict[str, Any] = None
    uptime_seconds: float = 0.0
    
    def __post_init__(self):
        if self.active_alerts is None:
            self.active_alerts = []
        if self.metrics is None:
            self.metrics = {}


class StateManager:
    """Manages system state with thread safety."""
    
    def __init__(self, state_file: str = "state/system_state.json"):
        """
        Initialize state manager.
        
        Args:
            state_file: Path to state persistence file
        """
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._state = SystemState()
        self._lock = threading.RLock()
        self._start_time = datetime.now()
        
        # Load existing state if available
        self._load_state()
        
        logger.info("StateManager initialized")
    
    def _load_state(self) -> None:
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                # Convert string dates back to datetime
                if data.get("last_cycle_time"):
                    data["last_cycle_time"] = datetime.fromisoformat(data["last_cycle_time"])
                if data.get("last_healing_time"):
                    data["last_healing_time"] = datetime.fromisoformat(data["last_healing_time"])
                
                self._state = SystemState(**data)
                logger.info(f"Loaded state from {self.state_file}")
                
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
    
    def _save_state(self) -> None:
        """Save state to file."""
        try:
            with self._lock:
                state_dict = asdict(self._state)
                
                # Convert datetime to string for JSON serialization
                if state_dict["last_cycle_time"]:
                    state_dict["last_cycle_time"] = state_dict["last_cycle_time"].isoformat()
                if state_dict["last_healing_time"]:
                    state_dict["last_healing_time"] = state_dict["last_healing_time"].isoformat()
                
                # Update uptime
                state_dict["uptime_seconds"] = (datetime.now() - self._start_time).total_seconds()
                
                with open(self.state_file, 'w') as f:
                    json.dump(state_dict, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def update_status(self, status: str) -> None:
        """
        Update system status.
        
        Args:
            status: New status (idle, monitoring, healing, error)
        """
        with self._lock:
            self._state.status = status
            if status == "monitoring":
                self._state.last_cycle_time = datetime.now()
            self._save_state()
        
        logger.debug(f"System status updated to: {status}")
    
    def record_healing_action(
        self,
        action: str,
        success: bool = True,
        details: Optional[Dict] = None
    ) -> None:
        """
        Record a healing action.
        
        Args:
            action: Type of healing action (retrain, rollback, fallback)
            success: Whether the action was successful
            details: Additional action details
        """
        with self._lock:
            self._state.last_healing_action = action
            self._state.last_healing_time = datetime.now()
            
            # Update metrics
            metric_key = f"healing_{action}_{'success' if success else 'failure'}"
            self._state.metrics[metric_key] = self._state.metrics.get(metric_key, 0) + 1
            
            if details:
                self._state.metrics[f"last_{action}_details"] = details
            
            self._save_state()
        
        logger.info(f"Healing action recorded: {action} (success: {success})")
    
    def add_alert(self, alert: str, severity: str = "warning") -> None:
        """
        Add an alert to the system.
        
        Args:
            alert: Alert message
            severity: Alert severity (info, warning, error, critical)
        """
        with self._lock:
            alert_entry = {
                "message": alert,
                "severity": severity,
                "timestamp": datetime.now().isoformat()
            }
            self._state.active_alerts.append(alert_entry)
            
            # Keep only last 100 alerts
            if len(self._state.active_alerts) > 100:
                self._state.active_alerts = self._state.active_alerts[-100:]
            
            # Update alert metrics
            metric_key = f"alerts_{severity}"
            self._state.metrics[metric_key] = self._state.metrics.get(metric_key, 0) + 1
            
            self._save_state()
        
        logger.warning(f"Alert added: [{severity}] {alert}")
    
    def clear_alerts(self) -> None:
        """Clear all active alerts."""
        with self._lock:
            self._state.active_alerts = []
            self._save_state()
        
        logger.info("All alerts cleared")
    
    def update_model_version(self, version: str) -> None:
        """
        Update current model version.
        
        Args:
            version: New model version
        """
        with self._lock:
            self._state.model_version = version
            self._save_state()
        
        logger.info(f"Model version updated to: {version}")
    
    def update_metric(self, key: str, value: Any) -> None:
        """
        Update a metric value.
        
        Args:
            key: Metric key
            value: Metric value
        """
        with self._lock:
            self._state.metrics[key] = value
            self._save_state()
    
    def increment_metric(self, key: str, amount: int = 1) -> None:
        """
        Increment a metric value.
        
        Args:
            key: Metric key
            amount: Amount to increment
        """
        with self._lock:
            current = self._state.metrics.get(key, 0)
            self._state.metrics[key] = current + amount
            self._save_state()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current system state.
        
        Returns:
            Current state as dictionary
        """
        with self._lock:
            state_dict = asdict(self._state)
            
            # Convert datetime to string for response
            if state_dict["last_cycle_time"]:
                state_dict["last_cycle_time"] = state_dict["last_cycle_time"].isoformat()
            if state_dict["last_healing_time"]:
                state_dict["last_healing_time"] = state_dict["last_healing_time"].isoformat()
            
            # Calculate uptime
            state_dict["uptime_seconds"] = (datetime.now() - self._start_time).total_seconds()
            state_dict["uptime_human"] = str(timedelta(seconds=int(state_dict["uptime_seconds"])))
            
            # Calculate time since last cycle
            if self._state.last_cycle_time:
                time_since_last = datetime.now() - self._state.last_cycle_time
                state_dict["seconds_since_last_cycle"] = time_since_last.total_seconds()
            
            return state_dict
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get system health status.
        
        Returns:
            Health status dictionary
        """
        state = self.get_state()
        
        # Determine health based on various factors
        health_status = "healthy"
        issues = []
        
        # Check for recent errors
        if state["status"] == "error":
            health_status = "unhealthy"
            issues.append("System status is 'error'")
        
        # Check for stale state
        if "seconds_since_last_cycle" in state:
            if state["seconds_since_last_cycle"] > 3600:  # 1 hour
                health_status = "warning"
                issues.append(f"No cycle for {state['seconds_since_last_cycle']:.0f} seconds")
        
        # Check for too many alerts
        critical_alerts = [a for a in state["active_alerts"] 
                          if a["severity"] in ["error", "critical"]]
        if len(critical_alerts) > 5:
            health_status = "unhealthy"
            issues.append(f"Too many critical alerts: {len(critical_alerts)}")
        
        # Check for failed healing actions
        failed_healing = state["metrics"].get("healing_failures_total", 0)
        if failed_healing > 3:
            health_status = "warning"
            issues.append(f"Multiple healing failures: {failed_healing}")
        
        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "issues": issues if issues else None,
            "summary": {
                "uptime": state["uptime_human"],
                "model_version": state["model_version"],
                "system_status": state["status"],
                "active_alerts": len(state["active_alerts"]),
                "critical_alerts": len(critical_alerts)
            }
        }
    
    def reset(self) -> None:
        """Reset all state (except uptime)."""
        with self._lock:
            self._state = SystemState()
            self._state.uptime_seconds = (datetime.now() - self._start_time).total_seconds()
            self._save_state()
        
        logger.info("System state reset")
