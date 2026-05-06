#!/usr/bin/env python3
"""
Weights & Biases (W&B) Integration Connector for Self-Healing ML Pipelines

Provides experiment tracking, visualization, and collaboration capabilities
through Weights & Biases integration.

Features:
    - Real-time experiment tracking and visualization
    - Hyperparameter sweep integration
    - Model artifact versioning
    - Team collaboration dashboards
    - Automated report generation

Usage:
    from integrations.wandb_connector import WandBConnector
    
    connector = WandBConnector(project="self-healing-ml")
    connector.start_run(config={"learning_rate": 0.01})
    connector.log_metrics({"drift_score": 0.85, "accuracy": 0.92})
    connector.finish()

Author: Self-Healing ML Pipelines Team
License: MIT
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class WandBConnector:
    """
    Connector for Weights & Biases experiment tracking.
    
    This connector provides a clean interface to W&B for tracking
    self-healing experiments, logging metrics, and visualizing results.
    """
    
    def __init__(
        self,
        project: str = "self-healing-ml-pipelines",
        entity: Optional[str] = None,
        api_key: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Initialize W&B connector.
        
        Args:
            project: W&B project name
            entity: W&B username or team name
            api_key: W&B API key (or set WANDB_API_KEY env var)
            tags: Default tags for all runs
        """
        self.project = project
        self.entity = entity
        self.api_key = api_key
        self.tags = tags or []
        self._wandb = None
        self._run = None
        self._active = False
        
        # Try to import wandb
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            print("⚠️  Weights & Biases not installed. Install with: pip install wandb")
            print("   Running in mock mode - no actual tracking will occur")
    
    def _ensure_wandb_available(self) -> bool:
        """Check if W&B is available."""
        if self._wandb is None:
            return False
        return True
    
    def login(self, api_key: Optional[str] = None) -> 'WandBConnector':
        """
        Authenticate with W&B.
        
        Args:
            api_key: W&B API key (optional, can use env var)
            
        Returns:
            Self for method chaining
        """
        if self._ensure_wandb_available():
            key = api_key or self.api_key
            if key:
                self._wandb.login(key=key)
            else:
                print("⚠️  No API key provided. Set WANDB_API_KEY env var or pass api_key parameter.")
            print(f"✅ W&B authentication configured")
        else:
            print("📝 [MOCK] W&B login simulated")
        
        return self
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        group: Optional[str] = None
    ) -> 'WandBConnector':
        """
        Start a new W&B run.
        
        Args:
            run_name: Name for this run
            config: Configuration/hyperparameters dictionary
            tags: Additional tags for this run
            notes: Optional notes about the run
            group: Group name for organizing related runs
            
        Returns:
            Self for method chaining
        """
        if self._ensure_wandb_available():
            run_tags = self.tags + (tags or [])
            
            self._run = self._wandb.init(
                project=self.project,
                entity=self.entity,
                name=run_name,
                config=config,
                tags=run_tags,
                notes=notes,
                group=group,
                save_code=False
            )
            self._active = True
            
            print(f"🧪 Started W&B run: {run_name or 'auto-generated'}")
            print(f"   Project: {self.project}")
            print(f"   URL: {self._run.get_url()}")
        else:
            # Mock mode
            run_id = f"mock-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self._run = type('MockRun', (), {'id': run_id, 'name': run_name})()
            self._active = True
            print(f"📝 [MOCK] Started W&B run: {run_name or run_id}")
            print(f"   Project: {self.project}")
        
        return self
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        commit: bool = True
    ) -> 'WandBConnector':
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number
            commit: Whether to commit the metrics
            
        Returns:
            Self for method chaining
        """
        if not self._active:
            print("⚠️  No active run. Call start_run() first.")
            return self
        
        if self._ensure_wandb_available():
            self._wandb.log(metrics, step=step, commit=commit)
            print(f"📊 Logged {len(metrics)} metrics")
        else:
            print(f"📝 [MOCK] Logged metrics: {metrics}")
        
        return self
    
    def log_chart(
        self,
        title: str,
        data: Any,
        chart_type: str = "line"
    ) -> 'WandBConnector':
        """
        Log a chart/visualization to W&B.
        
        Args:
            title: Chart title
            data: Chart data (list, dict, or pandas DataFrame)
            chart_type: Type of chart (line, bar, scatter, etc.)
            
        Returns:
            Self for method chaining
        """
        if not self._active:
            print("⚠️  No active run. Call start_run() first.")
            return self
        
        if self._ensure_wandb_available():
            try:
                if chart_type == "line":
                    chart = self._wandb.plot.line_series(
                        xs=list(range(len(data))) if isinstance(data, list) else data,
                        ys=data if isinstance(data, list) else [],
                        keys=["value"],
                        title=title
                    )
                    self._wandb.log({f"{title}_chart": chart})
                print(f"📈 Logged chart: {title}")
            except Exception as e:
                print(f"⚠️  Error logging chart: {e}")
        else:
            print(f"📝 [MOCK] Logged chart: {title} ({chart_type})")
        
        return self
    
    def log_table(
        self,
        name: str,
        columns: List[str],
        data: List[List[Any]]
    ) -> 'WandBConnector':
        """
        Log a data table to W&B.
        
        Args:
            name: Table name
            columns: Column names
            data: Row data
            
        Returns:
            Self for method chaining
        """
        if not self._active:
            print("⚠️  No active run. Call start_run() first.")
            return self
        
        if self._ensure_wandb_available():
            table = self._wandb.Table(columns=columns, data=data)
            self._wandb.log({name: table})
            print(f"📋 Logged table: {name} ({len(data)} rows)")
        else:
            print(f"📝 [MOCK] Logged table: {name} ({len(data)} rows)")
        
        return self
    
    def log_artifact(
        self,
        local_path: str,
        artifact_type: str = "model",
        name: Optional[str] = None
    ) -> 'WandBConnector':
        """
        Log an artifact to W&B.
        
        Args:
            local_path: Path to local file/directory
            artifact_type: Type of artifact (model, dataset, etc.)
            name: Artifact name (default: filename)
            
        Returns:
            Self for method chaining
        """
        if not self._active:
            print("⚠️  No active run. Call start_run() first.")
            return self
        
        if self._ensure_wandb_available():
            artifact_name = name or Path(local_path).name
            artifact = self._wandb.Artifact(artifact_name, type=artifact_type)
            
            if Path(local_path).is_file():
                artifact.add_file(local_path)
            elif Path(local_path).is_dir():
                artifact.add_dir(local_path)
            
            self._wandb.log_artifact(artifact)
            print(f"📦 Logged artifact: {artifact_name}")
        else:
            print(f"📝 [MOCK] Logged artifact: {local_path}")
        
        return self
    
    def use_artifact(
        self,
        artifact_name: str,
        artifact_type: str = "model"
    ) -> Optional[Any]:
        """
        Use/download an artifact from W&B.
        
        Args:
            artifact_name: Name of the artifact
            artifact_type: Type of artifact
            
        Returns:
            The artifact object
        """
        if self._ensure_wandb_available():
            try:
                artifact = self._wandb.use_artifact(artifact_name, type=artifact_type)
                artifact_dir = artifact.download()
                print(f"✅ Downloaded artifact: {artifact_name} to {artifact_dir}")
                return artifact
            except Exception as e:
                print(f"⚠️  Error using artifact: {e}")
        else:
            print(f"📝 [MOCK] Would use artifact: {artifact_name}")
            return None
    
    def log_config(self, config: Dict[str, Any]) -> 'WandBConnector':
        """
        Update run configuration.
        
        Args:
            config: Configuration dictionary to update
            
        Returns:
            Self for method chaining
        """
        if not self._active:
            print("⚠️  No active run. Call start_run() first.")
            return self
        
        if self._ensure_wandb_available():
            if self._run.config:
                for key, value in config.items():
                    self._run.config[key] = value
            print(f"⚙️  Updated {len(config)} config values")
        else:
            print(f"📝 [MOCK] Updated config: {config}")
        
        return self
    
    def finish(
        self,
        exit_code: int = 0,
        save_summary: bool = True
    ) -> 'WandBConnector':
        """
        Finish the current run.
        
        Args:
            exit_code: Exit code (0 for success)
            save_summary: Whether to save a summary JSON file
            
        Returns:
            Self for method chaining
        """
        if not self._active:
            print("⚠️  No active run to finish.")
            return self
        
        if self._ensure_wandb_available():
            self._wandb.finish(exit_code=exit_code)
            print(f"✅ Finished W&B run")
        else:
            print(f"📝 [MOCK] Finished run")
        
        # Save summary
        if save_summary:
            summary = {
                'project': self.project,
                'run_id': self._run.id if self._run else 'unknown',
                'run_name': self._run.name if self._run else 'unknown',
                'exit_code': exit_code,
                'finished_at': datetime.now().isoformat()
            }
            summary_path = Path(f'logs/wandb_run_{summary["run_id"]}_summary.json')
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"📄 Run summary saved to {summary_path}")
        
        self._active = False
        return self
    
    def watch_model(
        self,
        model: Any,
        criterion: Optional[Any] = None,
        log: str = "all",
        log_freq: int = 100
    ) -> 'WandBConnector':
        """
        Watch a model for gradient and parameter histograms.
        
        Args:
            model: PyTorch model to watch
            criterion: Loss function
            log: What to log ("gradients", "parameters", "all")
            log_freq: Logging frequency in steps
            
        Returns:
            Self for method chaining
        """
        if not self._active:
            print("⚠️  No active run. Call start_run() first.")
            return self
        
        if self._ensure_wandb_available():
            self._wandb.watch(model, criterion=criterion, log=log, log_freq=log_freq)
            print(f"👁️ Watching model (log_freq={log_freq})")
        else:
            print(f"📝 [MOCK] Would watch model")
        
        return self
    
    def __enter__(self):
        """Context manager entry."""
        return self.login().start_run()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        exit_code = 1 if exc_type else 0
        self.finish(exit_code=exit_code)


def demo_usage():
    """Demonstrate W&B connector usage."""
    print("\n" + "="*60)
    print("Weights & Biases Connector Demo")
    print("="*60 + "\n")
    
    # Example 1: Basic usage with context manager
    print("Example 1: Using context manager\n")
    with WandBConnector(project="demo-healing") as connector:
        connector.log_metrics({
            'drift_score': 0.85,
            'accuracy': 0.92,
            'latency_ms': 45.2
        })
        connector.log_config({
            'model_type': 'random_forest',
            'n_estimators': 100
        })
    
    print("\n" + "-"*60 + "\n")
    
    # Example 2: Manual lifecycle management
    print("Example 2: Manual lifecycle management\n")
    connector = WandBConnector(project="healing-sessions")
    connector.login()
    connector.start_run(
        run_name="healing-session-001",
        config={'threshold': 0.8},
        tags=['production', 'critical']
    )
    connector.log_metrics({'severity': 0.75, 'confidence': 0.88})
    connector.finish()
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    demo_usage()
