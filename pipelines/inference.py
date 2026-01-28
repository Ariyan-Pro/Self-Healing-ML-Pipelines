"""Inference pipeline for self-healing ML system."""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import joblib
import json

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Orchestrates model inference pipeline."""
    
    def __init__(self):
        """Initialize inference pipeline."""
        self.models_dir = Path("models")
        self.current_model_path = self.models_dir / "current_model.joblib"
        self.current_metadata_path = self.models_dir / "current_metadata.json"
        
        # Load current model
        self.model = None
        self.metadata = {}
        self._load_current_model()
        
        logger.info("InferencePipeline initialized")
    
    def _load_current_model(self) -> None:
        """Load current model from disk."""
        try:
            if self.current_model_path.exists():
                self.model = joblib.load(self.current_model_path)
                logger.info(f"Loaded current model from {self.current_model_path}")
            
            if self.current_metadata_path.exists():
                with open(self.current_metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info("Loaded current model metadata")
                
        except Exception as e:
            logger.error(f"Failed to load current model: {e}")
            self.model = None
            self.metadata = {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "loaded": self.model is not None,
            "metadata": self.metadata,
            "model_path": str(self.current_model_path),
            "features": self.metadata.get("features", []),
            "version": self.metadata.get("version", "unknown")
        }
    
    def predict(
        self,
        features: Union[pd.DataFrame, np.ndarray, List[List[float]]],
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Make predictions using the current model.
        
        Args:
            features: Input features
            return_proba: Whether to return probabilities
            
        Returns:
            Predictions or probabilities
        """
        if self.model is None:
            raise ValueError("No model loaded. Please ensure a model is trained.")
        
        # Convert to numpy array if needed
        if isinstance(features, pd.DataFrame):
            features_array = features.values
        elif isinstance(features, list):
            features_array = np.array(features)
        else:
            features_array = features
        
        # Validate feature dimensions
        expected_features = len(self.metadata.get("features", []))
        if features_array.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, "
                f"got {features_array.shape[1]}"
            )
        
        # Make predictions
        if return_proba:
            predictions = self.model.predict_proba(features_array)
        else:
            predictions = self.model.predict(features_array)
        
        logger.debug(f"Made predictions on {len(predictions)} samples")
        
        return predictions
    
    def batch_predict(
        self,
        data_path: str,
        output_path: Optional[str] = None,
        return_proba: bool = False
    ) -> Dict[str, Any]:
        """
        Make batch predictions from a data file.
        
        Args:
            data_path: Path to input data CSV
            output_path: Path to save predictions (optional)
            return_proba: Whether to return probabilities
            
        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Starting batch prediction on {data_path}")
        
        try:
            # Load data
            data = pd.read_csv(data_path)
            
            # Get feature columns from metadata
            feature_cols = self.metadata.get("features", [])
            
            if not feature_cols:
                # If no metadata, use all columns
                feature_cols = data.columns.tolist()
            
            # Extract features
            X = data[feature_cols]
            
            # Make predictions
            predictions = self.predict(X, return_proba)
            
            # Create results DataFrame
            results_df = data.copy()
            
            if return_proba:
                # Add probability columns
                n_classes = predictions.shape[1]
                for i in range(n_classes):
                    results_df[f'class_{i}_probability'] = predictions[:, i]
                results_df['prediction'] = np.argmax(predictions, axis=1)
            else:
                results_df['prediction'] = predictions
            
            # Save results if output path provided
            if output_path:
                results_df.to_csv(output_path, index=False)
                logger.info(f"Predictions saved to {output_path}")
            
            # Calculate prediction statistics
            if not return_proba:
                unique_preds, counts = np.unique(predictions, return_counts=True)
                pred_stats = {
                    str(pred): count for pred, count in zip(unique_preds, counts)
                }
            else:
                pred_stats = {
                    "probability_mean": float(np.mean(predictions)),
                    "probability_std": float(np.std(predictions))
                }
            
            result = {
                "pipeline": "inference",
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "model_version": self.metadata.get("version", "unknown"),
                "samples_processed": len(predictions),
                "prediction_stats": pred_stats,
                "features_used": feature_cols,
                "output_path": output_path
            }
            
            if output_path:
                result["output_path"] = output_path
            
            logger.info(f"Batch prediction completed: {len(predictions)} samples")
            
            return result
            
        except Exception as e:
            error_msg = f"Batch prediction failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "pipeline": "inference",
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": error_msg
            }
    
    def real_time_predict(
        self,
        feature_dict: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Make real-time prediction from a feature dictionary.
        
        Args:
            feature_dict: Dictionary mapping feature names to values
            
        Returns:
            Dictionary with prediction result
        """
        logger.debug(f"Real-time prediction for features: {feature_dict}")
        
        try:
            # Get expected feature order from metadata
            expected_features = self.metadata.get("features", [])
            
            if not expected_features:
                # If no metadata, use features from input
                expected_features = list(feature_dict.keys())
            
            # Create feature array in correct order
            feature_array = []
            for feat in expected_features:
                if feat in feature_dict:
                    feature_array.append(feature_dict[feat])
                else:
                    raise ValueError(f"Missing feature: {feat}")
            
            # Reshape for single prediction
            features = np.array(feature_array).reshape(1, -1)
            
            # Make prediction
            prediction = self.predict(features, return_proba=False)[0]
            probability = self.predict(features, return_proba=True)[0]
            
            result = {
                "prediction": int(prediction),
                "probabilities": probability.tolist(),
                "features": feature_dict,
                "model_version": self.metadata.get("version", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.debug(f"Real-time prediction result: {result}")
            
            return result
            
        except Exception as e:
            error_msg = f"Real-time prediction failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }