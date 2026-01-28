"""Training pipeline for self-healing ML system."""
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import joblib
import json
import tempfile
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates model training pipeline."""
    
    def __init__(self, config_path: str = "configs/pipeline.yaml"):
        """
        Initialize training pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.models_dir = Path("models")
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.registry_dir = self.models_dir / "registry"
        
        # Ensure directories exist
        self.models_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.registry_dir.mkdir(exist_ok=True)
        
        logger.info("TrainingPipeline initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def load_data(self, data_path: Optional[str] = None, data: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load training data from file or DataFrame.
        
        Args:
            data_path: Path to training data CSV
            data: DataFrame containing training data
            
        Returns:
            Tuple of (features, target)
        """
        try:
            if data is not None:
                df = data
            elif data_path is not None:
                df = pd.read_csv(data_path)
            else:
                raise ValueError("Either data_path or data must be provided")
            
            # Extract features and target
            feature_cols = self.config["data"].get("feature_columns", [])
            target_col = self.config["data"].get("target_column", "target")
            
            if not feature_cols:
                # If not specified, use all columns except target
                feature_cols = [col for col in df.columns if col != target_col]
            
            X = df[feature_cols]
            y = df[target_col]
            
            logger.info(f"Loaded data: {len(X)} samples, {len(feature_cols)} features")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_params: Optional[Dict] = None
    ) -> RandomForestClassifier:
        """
        Train a Random Forest classifier.
        
        Args:
            X: Training features
            y: Training labels
            model_params: Model hyperparameters
            
        Returns:
            Trained model
        """
        logger.info(f"Training model on {len(X)} samples")
        
        # Default parameters
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": -1
        }
        
        # Update with provided parameters
        if model_params:
            default_params.update(model_params)
        
        # Train model
        model = RandomForestClassifier(**default_params)
        model.fit(X, y)
        
        logger.info("Model training completed")
        
        return model
    
    def evaluate_model(
        self,
        model: RandomForestClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        logger.info(f"Model evaluation: {metrics}")
        
        return metrics
    
    def save_model(
        self,
        model: RandomForestClassifier,
        metrics: Dict[str, float],
        X_train: pd.DataFrame,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save model and update registry.
        
        Args:
            model: Trained model
            metrics: Evaluation metrics
            X_train: Training features (for feature information)
            version: Model version (auto-generated if None)
            
        Returns:
            Dictionary with save information
        """
        # Generate version if not provided
        if not version:
            version = f"model_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save model
        model_path = self.checkpoints_dir / f"{version}.joblib"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            "version": version,
            "path": str(model_path),
            "training_date": datetime.now().isoformat(),
            "metrics": metrics,
            "features": list(X_train.columns),
            "feature_count": len(X_train.columns),
            "sample_count": len(X_train),
            "model_type": "RandomForestClassifier",
            "model_params": model.get_params()
        }
        
        metadata_path = self.registry_dir / f"{version}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update current model pointer
        current_model_path = self.models_dir / "current_model.joblib"
        joblib.dump(model, current_model_path)
        
        # Update current metadata
        current_metadata_path = self.models_dir / "current_metadata.json"
        with open(current_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved: {version} at {model_path}")
        
        return {
            "status": "success",
            "version": version,
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "metrics": metrics
        }
    
    def run(
        self,
        data_path: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        test_size: float = 0.2,
        model_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Args:
            data_path: Path to training data (optional if data is provided)
            data: DataFrame containing training data (optional if data_path is provided)
            test_size: Proportion of data to use for testing
            model_params: Model hyperparameters
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting training pipeline")
        
        try:
            # 1. Load data
            X, y = self.load_data(data_path, data)
            
            # 2. Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test")
            
            # 3. Train model
            model = self.train_model(X_train, y_train, model_params)
            
            # 4. Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test)
            
            # 5. Save model
            save_result = self.save_model(model, metrics, X_train)
            
            # Prepare final result
            result = {
                "pipeline": "training",
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "data_info": {
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "features": list(X_train.columns),
                    "feature_count": len(X_train.columns)
                },
                "metrics": metrics,
                "model_info": save_result
            }
            
            logger.info("Training pipeline completed successfully")
            
            return result
            
        except Exception as e:
            error_msg = f"Training pipeline failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "pipeline": "training",
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": error_msg
            }
