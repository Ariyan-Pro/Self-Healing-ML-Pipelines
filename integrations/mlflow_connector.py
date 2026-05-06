#!/usr/bin/env python3
"""
MLflow Integration Connector for Self-Healing ML Pipelines

Provides experiment tracking, model registry, and artifact logging capabilities
through MLflow integration.

Features:
    - Automatic experiment tracking for healing actions
    - Model versioning and registry integration
    - Metrics logging for drift detection results
    - Artifact storage for model snapshots

Usage:
    from integrations.mlflow_connector import MLflowConnector
    
    connector = MLflowConnector(tracking_uri="http://localhost:5000")
    connector.start_experiment("healing-session-001")
    connector.log_metrics({"drift_score": 0.85, "confidence": 0.92})
    connector.register_model("my-model", model_path="./models/v1")
    connector.end_experiment()

Author: Self-Healing ML Pipelines Team
License: MIT
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class MLflowConnector:
    """
    Connector for MLflow experiment tracking and model registry.
    
    This connector provides a clean interface to MLflow for tracking
    self-healing experiments, logging metrics, and managing models.
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "self-healing-ml-pipelines",
        registry_uri: Optional[str] = None
    ):
        """
        Initialize MLflow connector.
        
        Args:
            tracking_uri: MLflow tracking server URI (default: local file store)
            experiment_name: Name of the experiment to track under
            registry_uri: Model registry URI (default: same as tracking_uri)
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.registry_uri = registry_uri or tracking_uri
        self._experiment_id: Optional[str] = None
        self._run_id: Optional[str] = None
        self._mlflow = None
        self._active_run = False
        
        # Try to import mlflow
        try:
            import mlflow
            import mlflow.sklearn
            import mlflow.pyfunc
            self._mlflow = mlflow
        except ImportError:
            print("⚠️  MLflow not installed. Install with: pip install mlflow")
            print("   Running in mock mode - no actual tracking will occur")
    
    def _ensure_mlflow_available(self) -> bool:
        """Check if MLflow is available."""
        if self._mlflow is None:
            return False
        return True
    
    def connect(self) -> 'MLflowConnector':
        """
        Establish connection to MLflow tracking server.
        
        Returns:
            Self for method chaining
        """
        if self._ensure_mlflow_available():
            if self.tracking_uri:
                self._mlflow.set_tracking_uri(self.tracking_uri)
            if self.registry_uri:
                self._mlflow.set_registry_uri(self.registry_uri)
            
            print(f"✅ Connected to MLflow tracking server: {self.tracking_uri or 'default'}")
        else:
            print("📝 MLflow not available - running in mock mode")
        
        return self
    
    def start_experiment(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> 'MLflowConnector':
        """
        Start a new MLflow experiment run.
        
        Args:
            experiment_name: Override default experiment name
            run_name: Name for this specific run
            tags: Additional tags to attach to the run
            
        Returns:
            Self for method chaining
        """
        exp_name = experiment_name or self.experiment_name
        
        if self._ensure_mlflow_available():
            # Set or create experiment
            experiment = self._mlflow.get_experiment_by_name(exp_name)
            if experiment:
                self._experiment_id = experiment.experiment_id
            else:
                self._experiment_id = self._mlflow.create_experiment(exp_name)
            
            self._mlflow.set_experiment(experiment_id=self._experiment_id)
            
            # Start run
            tags = tags or {}
            tags['created_at'] = datetime.now().isoformat()
            tags['system'] = 'self-healing-ml-pipelines'
            
            self._run = self._mlflow.start_run(run_name=run_name, tags=tags)
            self._run_id = self._run.info.run_id
            self._active_run = True
            
            print(f"🧪 Started MLflow experiment: {exp_name}")
            print(f"   Run ID: {self._run_id}")
            print(f"   Run name: {run_name or 'auto-generated'}")
        else:
            # Mock mode
            self._experiment_id = f"mock-exp-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self._run_id = f"mock-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self._active_run = True
            print(f"📝 [MOCK] Started experiment: {exp_name}, Run: {self._run_id}")
        
        return self
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> 'MLflowConnector':
        """
        Log metrics to the current run.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number for time-series metrics
            
        Returns:
            Self for method chaining
        """
        if not self._active_run:
            print("⚠️  No active run. Call start_experiment() first.")
            return self
        
        if self._ensure_mlflow_available():
            for name, value in metrics.items():
                if step is not None:
                    self._mlflow.log_metric(name, value, step=step)
                else:
                    self._mlflow.log_metric(name, value)
            print(f"📊 Logged {len(metrics)} metrics")
        else:
            print(f"📝 [MOCK] Logged metrics: {metrics}")
        
        return self
    
    def log_params(self, params: Dict[str, Any]) -> 'MLflowConnector':
        """
        Log parameters to the current run.
        
        Args:
            params: Dictionary of parameter names to values
            
        Returns:
            Self for method chaining
        """
        if not self._active_run:
            print("⚠️  No active run. Call start_experiment() first.")
            return self
        
        if self._ensure_mlflow_available():
            for name, value in params.items():
                self._mlflow.log_param(name, value)
            print(f"⚙️  Logged {len(params)} parameters")
        else:
            print(f"📝 [MOCK] Logged params: {params}")
        
        return self
    
    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None
    ) -> 'MLflowConnector':
        """
        Log a local file or directory as an artifact.
        
        Args:
            local_path: Path to the local file/directory
            artifact_path: Optional path within the artifact directory
            
        Returns:
            Self for method chaining
        """
        if not self._active_run:
            print("⚠️  No active run. Call start_experiment() first.")
            return self
        
        if self._ensure_mlflow_available():
            self._mlflow.log_artifact(local_path, artifact_path)
            print(f"📦 Logged artifact: {local_path}")
        else:
            print(f"📝 [MOCK] Logged artifact: {local_path}")
        
        return self
    
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        conda_env: Optional[str] = None,
        registered_model_name: Optional[str] = None
    ) -> 'MLflowConnector':
        """
        Log a model to MLflow.
        
        Args:
            model: The model object to log
            artifact_path: Path within the artifact directory
            conda_env: Path to conda environment file
            registered_model_name: If provided, register the model
            
        Returns:
            Self for method chaining
        """
        if not self._active_run:
            print("⚠️  No active run. Call start_experiment() first.")
            return self
        
        if self._ensure_mlflow_available():
            if registered_model_name:
                self._mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    conda_env=conda_env,
                    registered_model_name=registered_model_name
                )
                print(f"🤖 Logged and registered model: {registered_model_name}")
            else:
                self._mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    conda_env=conda_env
                )
                print(f"🤖 Logged model to {artifact_path}")
        else:
            print(f"📝 [MOCK] Logged model to {artifact_path}")
            if registered_model_name:
                print(f"   [MOCK] Registered as: {registered_model_name}")
        
        return self
    
    def register_model(
        self,
        model_name: str,
        model_uri: Optional[str] = None,
        model_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Register a model in the MLflow Model Registry.
        
        Args:
            model_name: Name for the registered model
            model_uri: URI of an existing logged model (e.g., "runs:/<run_id>/model")
            model_path: Local path to model (will be logged first)
            
        Returns:
            Model version string if successful
        """
        if self._ensure_mlflow_available():
            try:
                if model_path and not model_uri:
                    # Log model first
                    self.log_model(None, artifact_path="temp_model", registered_model_name=model_name)
                    model_uri = f"runs:/{self._run_id}/temp_model"
                
                if model_uri:
                    model_version = self._mlflow.register_model(model_uri, model_name)
                    print(f"✅ Registered model: {model_name} (version: {model_version.version})")
                    return model_version.version
            except Exception as e:
                print(f"⚠️  Error registering model: {e}")
        else:
            print(f"📝 [MOCK] Would register model: {model_name}")
            return "v1"
        
        return None
    
    def get_model(self, model_name: str, version: Optional[str] = "latest"):
        """
        Load a model from the MLflow Model Registry.
        
        Args:
            model_name: Name of the registered model
            version: Model version or "latest"
            
        Returns:
            The loaded model object
        """
        if self._ensure_mlflow_available():
            try:
                if version == "latest":
                    model = self._mlflow.sklearn.load_model(f"models:/{model_name}/latest")
                else:
                    model = self._mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
                print(f"✅ Loaded model: {model_name} (version: {version})")
                return model
            except Exception as e:
                print(f"⚠️  Error loading model: {e}")
        else:
            print(f"📝 [MOCK] Would load model: {model_name} (version: {version})")
            return None
    
    def end_experiment(
        self,
        status: str = "FINISHED",
        save_summary: bool = True
    ) -> 'MLflowConnector':
        """
        End the current experiment run.
        
        Args:
            status: Final status (FINISHED, FAILED, KILLED)
            save_summary: Whether to save a summary JSON file
            
        Returns:
            Self for method chaining
        """
        if not self._active_run:
            print("⚠️  No active run to end.")
            return self
        
        if self._ensure_mlflow_available():
            self._mlflow.end_run(status=status)
            print(f"✅ Ended MLflow run: {self._run_id}")
        else:
            print(f"📝 [MOCK] Ended run: {self._run_id}")
        
        # Save summary
        if save_summary:
            summary = {
                'experiment_id': self._experiment_id,
                'run_id': self._run_id,
                'status': status,
                'ended_at': datetime.now().isoformat()
            }
            summary_path = Path(f'logs/mlflow_run_{self._run_id}_summary.json')
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"📄 Run summary saved to {summary_path}")
        
        self._active_run = False
        return self
    
    def search_runs(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict]:
        """
        Search for runs in the experiment.
        
        Args:
            filter_string: MLflow filter query string
            max_results: Maximum number of runs to return
            
        Returns:
            List of run information dictionaries
        """
        if self._ensure_mlflow_available():
            runs = self._mlflow.search_runs(
                experiment_ids=[self._experiment_id],
                filter_string=filter_string,
                max_results=max_results
            )
            print(f"🔍 Found {len(runs)} runs")
            return runs.to_dict('records') if hasattr(runs, 'to_dict') else []
        else:
            print("📝 [MOCK] Would search runs")
            return []
    
    def __enter__(self):
        """Context manager entry."""
        return self.connect().start_experiment()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        status = "FAILED" if exc_type else "FINISHED"
        self.end_experiment(status=status)


def demo_usage():
    """Demonstrate MLflow connector usage."""
    print("\n" + "="*60)
    print("MLflow Connector Demo")
    print("="*60 + "\n")
    
    # Example 1: Basic usage with context manager
    print("Example 1: Using context manager\n")
    with MLflowConnector(experiment_name="demo-experiment") as connector:
        connector.log_metrics({
            'drift_score': 0.85,
            'accuracy': 0.92,
            'latency_ms': 45.2
        })
        connector.log_params({
            'model_type': 'random_forest',
            'n_estimators': 100,
            'max_depth': 10
        })
    
    print("\n" + "-"*60 + "\n")
    
    # Example 2: Manual lifecycle management
    print("Example 2: Manual lifecycle management\n")
    connector = MLflowConnector(experiment_name="healing-sessions")
    connector.connect()
    connector.start_experiment(run_name="healing-session-001")
    connector.log_metrics({'severity': 0.75, 'confidence': 0.88})
    connector.end_experiment()
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    demo_usage()
