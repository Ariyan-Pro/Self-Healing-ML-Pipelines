"""Healing actions for self-healing ML system."""
import logging
import json
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import joblib
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class HealingActions:
    """Orchestrates healing actions for ML pipeline recovery."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize healing actions.
        
        Args:
            config: Healing configuration dictionary
        """
        self.config = config
        self.models_dir = Path("models")
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.registry_dir = self.models_dir / "registry"
        self.fallback_model_path = Path(config.get("fallback_model_path", 
                                                   "models/fallback/rule_based.joblib"))
        
        # Ensure directories exist
        self.models_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.registry_dir.mkdir(exist_ok=True)
        
        logger.info("HealingActions initialized")
    
    def retrain(self, data: pd.DataFrame, labels: pd.Series, 
                model_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Trigger model retraining.
        
        Args:
            data: Training features
            labels: Training labels
            model_params: Optional model parameters
            
        Returns:
            Dictionary with retraining results
        """
        logger.info("Starting retraining process")
        
        try:
            # Import here to avoid circular dependencies
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, f1_score
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                data, labels, test_size=0.2, random_state=42
            )
            
            # Train new model
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                **(model_params or {})
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            # Save model
            model_version = f"model_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = self.checkpoints_dir / f"{model_version}.joblib"
            
            joblib.dump(model, model_path)
            
            # Update registry
            registry_entry = {
                "version": model_version,
                "path": str(model_path),
                "accuracy": accuracy,
                "f1_score": f1,
                "training_date": datetime.now().isoformat(),
                "sample_count": len(data),
                "features": list(data.columns)
            }
            
            registry_file = self.registry_dir / f"{model_version}.json"
            with open(registry_file, 'w') as f:
                json.dump(registry_entry, f, indent=2)
            
            result = {
                "action": "retrain",
                "status": "success",
                "model_version": model_version,
                "model_path": str(model_path),
                "metrics": {
                    "accuracy": accuracy,
                    "f1_score": f1
                },
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Retraining successful: accuracy={accuracy:.4f}, f1={f1:.4f}")
            
            return result
            
        except Exception as e:
            error_msg = f"Retraining failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "action": "retrain",
                "status": "failed",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def rollback(self, target_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Rollback to a previous model version.
        
        Args:
            target_version: Specific version to rollback to. If None,
                           rolls back to most recent stable version.
        
        Returns:
            Dictionary with rollback results
        """
        logger.info("Starting rollback process")
        
        try:
            # Find available model versions
            model_files = list(self.checkpoints_dir.glob("*.joblib"))
            
            if not model_files:
                raise ValueError("No model checkpoints found")
            
            # Sort by modification time (newest first)
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # If target version specified, find it
            if target_version:
                target_path = self.checkpoints_dir / f"{target_version}.joblib"
                if not target_path.exists():
                    raise ValueError(f"Target version {target_version} not found")
                selected_path = target_path
            else:
                # Use second newest (rollback from current)
                if len(model_files) > 1:
                    selected_path = model_files[1]  # Skip current
                else:
                    selected_path = model_files[0]  # Only one available
            
            # Load the model
            model = joblib.load(selected_path)
            
            # Update current model symlink or pointer
            current_model_path = self.models_dir / "current_model.joblib"
            joblib.dump(model, current_model_path)
            
            result = {
                "action": "rollback",
                "status": "success",
                "model_version": selected_path.stem,
                "model_path": str(selected_path),
                "rollback_date": datetime.now().isoformat(),
                "message": f"Successfully rolled back to {selected_path.stem}"
            }
            
            logger.info(f"Rollback successful to {selected_path.stem}")
            
            return result
            
        except Exception as e:
            error_msg = f"Rollback failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "action": "rollback",
                "status": "failed",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def fallback(self) -> Dict[str, Any]:
        """
        Switch to fallback model.
        
        Returns:
            Dictionary with fallback results
        """
        logger.info("Starting fallback process")
        
        try:
            # Check if fallback model exists
            if not self.fallback_model_path.exists():
                # Create a simple rule-based fallback if not exists
                self._create_fallback_model()
            
            # Load fallback model
            fallback_model = joblib.load(self.fallback_model_path)
            
            # Update current model pointer
            current_model_path = self.models_dir / "current_model.joblib"
            joblib.dump(fallback_model, current_model_path)
            
            result = {
                "action": "fallback",
                "status": "success",
                "fallback_type": "rule_based",
                "fallback_path": str(self.fallback_model_path),
                "activation_time": datetime.now().isoformat(),
                "message": "Successfully switched to fallback model"
            }
            
            logger.info("Fallback successful")
            
            return result
            
        except Exception as e:
            error_msg = f"Fallback failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "action": "fallback",
                "status": "failed",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def _create_fallback_model(self) -> None:
        """Create a simple rule-based fallback model."""
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        # Create a very simple model
        X_dummy = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
        y_dummy = np.array([0, 1, 1, 0])
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_dummy, y_dummy)
        
        # Ensure directory exists
        self.fallback_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model, self.fallback_model_path)
        
        logger.info(f"Created fallback model at {self.fallback_model_path}")
    
    def execute_action(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a healing action.
        
        Args:
            action: Action to execute ('retrain', 'rollback', 'fallback')
            **kwargs: Additional arguments for the action
        
        Returns:
            Dictionary with action results
        """
        action_map = {
            "retrain": self.retrain,
            "rollback": self.rollback,
            "fallback": self.fallback
        }
        
        if action not in action_map:
            raise ValueError(f"Unknown action: {action}. "
                           f"Available actions: {list(action_map.keys())}")
        
        logger.info(f"Executing healing action: {action}")
        return action_map[action](**kwargs)