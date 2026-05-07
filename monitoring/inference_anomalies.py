import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime


class AnomalyDetector:
    """
    Anomaly detector for ML inference monitoring.
    
    Detects anomalies in predictions using statistical methods.
    """
    
    def __init__(self, z_thresh: float = 3.0, window_size: int = 100):
        """
        Initialize the anomaly detector.
        
        Args:
            z_thresh: Z-score threshold for anomaly detection (default: 3.0)
            window_size: Size of the rolling window for baseline calculation
        """
        self.z_thresh = z_thresh
        self.window_size = window_size
        self.history: List[float] = []
        self.anomaly_count = 0
        self.total_count = 0
    
    def add_prediction(self, value: float) -> bool:
        """
        Add a new prediction and check if it's anomalous.
        
        Args:
            value: The prediction value to check
            
        Returns:
            True if the value is anomalous, False otherwise
        """
        self.total_count += 1
        self.history.append(value)
        
        # Keep only the recent window
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Need at least some data points to detect anomalies
        if len(self.history) < 5:
            return False
        
        # Calculate z-score
        mean = np.mean(self.history)
        std = np.std(self.history)
        
        if std == 0:
            return False
        
        z_score = abs((value - mean) / std)
        is_anomaly = z_score > self.z_thresh
        
        if is_anomaly:
            self.anomaly_count += 1
        
        return is_anomaly
    
    def anomaly_rate(self) -> float:
        """
        Calculate the overall anomaly rate.
        
        Returns:
            The ratio of anomalies to total predictions
        """
        if self.total_count == 0:
            return 0.0
        return self.anomaly_count / self.total_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics.
        
        Returns:
            Dictionary with current stats
        """
        if not self.history:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "anomaly_rate": 0.0,
                "total_predictions": 0,
                "anomaly_count": 0
            }
        
        return {
            "mean": float(np.mean(self.history)),
            "std": float(np.std(self.history)),
            "min": float(np.min(self.history)),
            "max": float(np.max(self.history)),
            "anomaly_rate": self.anomaly_rate(),
            "total_predictions": self.total_count,
            "anomaly_count": self.anomaly_count
        }
    
    def reset(self):
        """Reset the detector state."""
        self.history = []
        self.anomaly_count = 0
        self.total_count = 0


def anomaly_rate(predictions, z_thresh=3.0):
    """
    Legacy function for anomaly rate calculation.
    Deprecated: Use AnomalyDetector class instead.
    
    Args:
        predictions: Array of prediction values
        z_thresh: Z-score threshold for anomaly detection
        
    Returns:
        The proportion of anomalous predictions
    """
    z_scores = (predictions - predictions.mean()) / predictions.std()
    anomalies = np.abs(z_scores) > z_thresh
    return anomalies.mean()
