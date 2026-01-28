"""Data drift detection module."""
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Data class for drift detection results."""
    drift_score: float
    p_value: Optional[float]
    is_drift: bool
    method: str
    threshold: float
    feature_name: Optional[str] = None
    statistic: Optional[float] = None
    sample_sizes: Optional[Dict[str, int]] = None
    

class DataDriftDetector:
    """Detects data drift between reference and current distributions."""
    
    def __init__(self, method: str = "ks", threshold: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            method: Detection method ('ks' for Kolmogorov-Smirnov, 
                    'psi' for Population Stability Index)
            threshold: Significance threshold
        """
        self.method = method
        self.threshold = threshold
        self.supported_methods = ["ks", "psi", "wasserstein"]
        
        if method not in self.supported_methods:
            raise ValueError(f"Method {method} not supported. "
                           f"Choose from {self.supported_methods}")
    
    def detect_drift(
        self,
        reference: Union[np.ndarray, List[float]],
        current: Union[np.ndarray, List[float]],
        feature_name: Optional[str] = None
    ) -> DriftResult:
        """
        Detect drift between reference and current data.
        
        Args:
            reference: Reference distribution data
            current: Current distribution data
            feature_name: Optional name of the feature being analyzed
            
        Returns:
            DriftResult containing detection results
        """
        # Convert to numpy arrays
        ref_array = np.array(reference).flatten()
        curr_array = np.array(current).flatten()
        
        # Record sample sizes
        sample_sizes = {
            "reference": len(ref_array),
            "current": len(curr_array)
        }
        
        if self.method == "ks":
            return self._ks_drift(ref_array, curr_array, feature_name, sample_sizes)
        elif self.method == "psi":
            return self._psi_drift(ref_array, curr_array, feature_name, sample_sizes)
        elif self.method == "wasserstein":
            return self._wasserstein_drift(ref_array, curr_array, feature_name, sample_sizes)
    
    def _ks_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: Optional[str],
        sample_sizes: Dict[str, int]
    ) -> DriftResult:
        """Kolmogorov-Smirnov test for drift detection."""
        stat, p_value = stats.ks_2samp(reference, current)
        drift_score = 1 - p_value
        is_drift = p_value < self.threshold
        
        logger.debug(
            f"KS test: statistic={stat:.4f}, p={p_value:.4f}, "
            f"drift_score={drift_score:.4f}, drift={is_drift}"
        )
        
        return DriftResult(
            drift_score=drift_score,
            p_value=p_value,
            is_drift=is_drift,
            method="kolmogorov_smirnov",
            threshold=self.threshold,
            feature_name=feature_name,
            statistic=stat,
            sample_sizes=sample_sizes
        )
    
    def _psi_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: Optional[str],
        sample_sizes: Dict[str, int]
    ) -> DriftResult:
        """Population Stability Index for drift detection."""
        # Create bins based on reference data (ensure at least 2 bins)
        n_bins = max(2, min(10, len(reference) // 20))
        bins = np.histogram_bin_edges(reference, bins=n_bins)
        
        # Calculate histograms
        ref_hist, _ = np.histogram(reference, bins=bins)
        curr_hist, _ = np.histogram(current, bins=bins)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        
        # Normalize to get proportions
        ref_prop = (ref_hist + epsilon) / (len(reference) + epsilon * n_bins)
        curr_prop = (curr_hist + epsilon) / (len(current) + epsilon * n_bins)
        
        # Calculate PSI
        psi_values = (curr_prop - ref_prop) * np.log((curr_prop + epsilon) / (ref_prop + epsilon))
        
        psi_total = np.sum(psi_values)
        
        # Convert PSI to drift score (0-1)
        drift_score = min(1.0, psi_total / 0.5)  # PSI > 0.25 indicates drift
        is_drift = psi_total > 0.25
        
        return DriftResult(
            drift_score=drift_score,
            p_value=None,
            is_drift=is_drift,
            method="population_stability_index",
            threshold=0.25,
            feature_name=feature_name,
            statistic=psi_total,
            sample_sizes=sample_sizes
        )
    
    def _wasserstein_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: Optional[str],
        sample_sizes: Dict[str, int]
    ) -> DriftResult:
        """Wasserstein distance for drift detection."""
        from scipy.stats import wasserstein_distance
        
        distance = wasserstein_distance(reference, current)
        
        # Normalize distance to 0-1 range (empirical threshold)
        drift_score = min(1.0, distance / 10.0)
        is_drift = distance > 1.0  # Empirical threshold
        
        return DriftResult(
            drift_score=drift_score,
            p_value=None,
            is_drift=is_drift,
            method="wasserstein_distance",
            threshold=1.0,
            feature_name=feature_name,
            statistic=distance,
            sample_sizes=sample_sizes
        )
    
    def detect_batch_drift(
        self,
        reference_data: Dict[str, np.ndarray],
        current_data: Dict[str, np.ndarray]
    ) -> Dict[str, DriftResult]:
        """
        Detect drift for multiple features.
        
        Args:
            reference_data: Dictionary of reference features
            current_data: Dictionary of current features
            
        Returns:
            Dictionary mapping feature names to drift results
        """
        results = {}
        
        for feature_name in reference_data.keys():
            if feature_name in current_data:
                result = self.detect_drift(
                    reference_data[feature_name],
                    current_data[feature_name],
                    feature_name
                )
                results[feature_name] = result
        
        return results


# Legacy function for backward compatibility
def detect_drift(reference, current, threshold=0.05):
    """Legacy function for simple drift detection."""
    detector = DataDriftDetector(method="ks", threshold=threshold)
    result = detector.detect_drift(reference, current)
    
    return {
        "drift_score": result.drift_score,
        "drift_detected": result.is_drift,
        "p_value": result.p_value,
        "statistic": result.statistic
    }
