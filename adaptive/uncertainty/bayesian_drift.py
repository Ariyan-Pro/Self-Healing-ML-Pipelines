"""
Bayesian Drift Detection with Uncertainty Quantification
Provides probability estimates instead of binary decisions.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import warnings


@dataclass
class BayesianDriftResult:
    """Result of Bayesian drift detection."""
    drift_probability: float  # P(drift | data)
    confidence_interval: Tuple[float, float]  # 95% CI for drift probability
    bayes_factor: float  # Evidence ratio: P(data | drift) / P(data | no_drift)
    posterior_mean: float  # Posterior mean of drift parameter
    posterior_std: float  # Posterior standard deviation
    decision: str  # 'drift', 'no_drift', or 'uncertain'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "drift_probability": self.drift_probability,
            "confidence_interval_lower": self.confidence_interval[0],
            "confidence_interval_upper": self.confidence_interval[1],
            "bayes_factor": self.bayes_factor,
            "posterior_mean": self.posterior_mean,
            "posterior_std": self.posterior_std,
            "decision": self.decision
        }


class BayesianDriftDetector:
    """
    Bayesian drift detector with uncertainty quantification.
    Uses Bayesian methods to estimate drift probability with confidence intervals.
    """
    
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """
        Initialize with prior parameters.
        
        Args:
            prior_alpha: Prior alpha parameter for Beta distribution
            prior_beta: Prior beta parameter for Beta distribution
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
        # Cache for likelihood computations
        self._likelihood_cache = {}
    
    def detect_drift_bayesian(self, reference_data: np.ndarray, 
                            current_data: np.ndarray,
                            method: str = "ks",
                            threshold: float = 0.05) -> BayesianDriftResult:
        """
        Detect drift using Bayesian methods with uncertainty quantification.
        
        Args:
            reference_data: Reference distribution data
            current_data: Current distribution data
            method: Detection method ('ks', 'psi', or 'wasserstein')
            threshold: Decision threshold for drift probability
            
        Returns:
            BayesianDriftResult with probability estimates
        """
        if method == "ks":
            return self._bayesian_ks_test(reference_data, current_data, threshold)
        elif method == "psi":
            return self._bayesian_psi_test(reference_data, current_data, threshold)
        elif method == "wasserstein":
            return self._bayesian_wasserstein_test(reference_data, current_data, threshold)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _bayesian_ks_test(self, ref_data: np.ndarray, curr_data: np.ndarray,
                         threshold: float) -> BayesianDriftResult:
        """
        Bayesian Kolmogorov-Smirnov test.
        Uses Bayesian bootstrap to estimate KS statistic distribution.
        """
        n_ref = len(ref_data)
        n_curr = len(curr_data)
        
        if n_ref < 10 or n_curr < 10:
            warnings.warn("Sample size too small for reliable Bayesian inference")
            return BayesianDriftResult(
                drift_probability=0.5,
                confidence_interval=(0.0, 1.0),
                bayes_factor=1.0,
                posterior_mean=0.0,
                posterior_std=1.0,
                decision="uncertain"
            )
        
        # Bayesian bootstrap to estimate KS statistic distribution
        ks_samples = []
        for _ in range(1000):  # Bootstrap samples
            # Sample weights from Dirichlet distribution
            ref_weights = np.random.dirichlet(np.ones(n_ref))
            curr_weights = np.random.dirichlet(np.ones(n_curr))
            
            # Compute weighted ECDFs
            ref_sorted = np.sort(ref_data)
            curr_sorted = np.sort(curr_data)
            
            # Compute KS statistic with weights
            ks_stat = self._weighted_ks_statistic(
                ref_sorted, ref_weights, curr_sorted, curr_weights
            )
            ks_samples.append(ks_stat)
        
        ks_samples = np.array(ks_samples)
        
        # Compute posterior statistics
        posterior_mean = np.mean(ks_samples)
        posterior_std = np.std(ks_samples)
        
        # Compute drift probability: P(KS > threshold)
        drift_prob = np.mean(ks_samples > threshold)
        
        # Compute 95% credible interval
        ci_lower = np.percentile(ks_samples, 2.5)
        ci_upper = np.percentile(ks_samples, 97.5)
        
        # Compute Bayes factor (approximate)
        # BF = P(data | drift) / P(data | no_drift)
        # Approximate using Savage-Dickey density ratio
        prior_at_threshold = stats.beta.pdf(threshold, self.prior_alpha, self.prior_beta)
        posterior_at_threshold = stats.gaussian_kde(ks_samples)(threshold)
        
        if posterior_at_threshold > 0:
            bayes_factor = posterior_at_threshold / (prior_at_threshold + 1e-10)
        else:
            bayes_factor = 1.0
        
        # Make decision with uncertainty awareness
        if drift_prob > 0.8 and (ci_upper - ci_lower) < 0.3:
            decision = "drift"
        elif drift_prob < 0.2 and (ci_upper - ci_lower) < 0.3:
            decision = "no_drift"
        else:
            decision = "uncertain"
        
        return BayesianDriftResult(
            drift_probability=float(drift_prob),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            bayes_factor=float(bayes_factor),
            posterior_mean=float(posterior_mean),
            posterior_std=float(posterior_std),
            decision=decision
        )
    
    def _weighted_ks_statistic(self, ref_sorted: np.ndarray, ref_weights: np.ndarray,
                              curr_sorted: np.ndarray, curr_weights: np.ndarray) -> float:
        """Compute KS statistic with weighted data."""
        # Compute weighted ECDFs at all unique points
        all_points = np.unique(np.concatenate([ref_sorted, curr_sorted]))
        
        # Compute weighted ECDF for reference
        ref_ecdf = np.zeros_like(all_points)
        cum_weight = 0.0
        ref_idx = 0
        
        for i, point in enumerate(all_points):
            while ref_idx < len(ref_sorted) and ref_sorted[ref_idx] <= point:
                cum_weight += ref_weights[ref_idx]
                ref_idx += 1
            ref_ecdf[i] = cum_weight
        
        # Compute weighted ECDF for current
        curr_ecdf = np.zeros_like(all_points)
        cum_weight = 0.0
        curr_idx = 0
        
        for i, point in enumerate(all_points):
            while curr_idx < len(curr_sorted) and curr_sorted[curr_idx] <= point:
                cum_weight += curr_weights[curr_idx]
                curr_idx += 1
            curr_ecdf[i] = cum_weight
        
        # KS statistic is maximum difference
        ks_stat = np.max(np.abs(ref_ecdf - curr_ecdf))
        return ks_stat
    
    def _bayesian_psi_test(self, ref_data: np.ndarray, curr_data: np.ndarray,
                          threshold: float) -> BayesianDriftResult:
        """
        Bayesian Population Stability Index test.
        Uses Dirichlet-Multinomial conjugate prior for bin probabilities.
        """
        # Create bins (10 equal-width bins)
        all_data = np.concatenate([ref_data, curr_data])
        bins = np.histogram_bin_edges(all_data, bins=10)
        
        # Count samples in bins
        ref_counts, _ = np.histogram(ref_data, bins=bins)
        curr_counts, _ = np.histogram(curr_data, bins=bins)
        
        # Add pseudocounts for stability
        ref_counts = ref_counts + 1
        curr_counts = curr_counts + 1
        
        # Compute proportions
        ref_props = ref_counts / ref_counts.sum()
        curr_props = curr_counts / curr_counts.sum()
        
        # Bayesian inference for proportion differences
        # Use Beta-Binomial model for each bin
        drift_probs = []
        
        for ref_prop, curr_prop in zip(ref_props, curr_props):
            # Posterior Beta parameters for reference
            ref_alpha = self.prior_alpha + ref_counts.sum() * ref_prop
            ref_beta = self.prior_beta + ref_counts.sum() * (1 - ref_prop)
            
            # Posterior Beta parameters for current
            curr_alpha = self.prior_alpha + curr_counts.sum() * curr_prop
            curr_beta = self.prior_beta + curr_counts.sum() * (1 - curr_prop)
            
            # Sample from posteriors
            ref_samples = np.random.beta(ref_alpha, ref_beta, 1000)
            curr_samples = np.random.beta(curr_alpha, curr_beta, 1000)
            
            # Compute PSI-like metric for samples
            psi_samples = (curr_samples - ref_samples) * np.log(
                (curr_samples + 1e-10) / (ref_samples + 1e-10)
            )
            
            # Probability that PSI > threshold
            drift_prob = np.mean(psi_samples > threshold)
            drift_probs.append(drift_prob)
        
        # Average across bins
        avg_drift_prob = np.mean(drift_probs)
        
        # Simplified result for now
        return BayesianDriftResult(
            drift_probability=float(avg_drift_prob),
            confidence_interval=(max(0.0, avg_drift_prob - 0.1), min(1.0, avg_drift_prob + 0.1)),
            bayes_factor=avg_drift_prob / (1 - avg_drift_prob + 1e-10),
            posterior_mean=float(avg_drift_prob),
            posterior_std=float(np.std(drift_probs)),
            decision="drift" if avg_drift_prob > 0.7 else "no_drift" if avg_drift_prob < 0.3 else "uncertain"
        )
    
    def _bayesian_wasserstein_test(self, ref_data: np.ndarray, curr_data: np.ndarray,
                                  threshold: float) -> BayesianDriftResult:
        """
        Bayesian Wasserstein distance test.
        Uses bootstrap to estimate distribution of Wasserstein distance.
        """
        n_ref = len(ref_data)
        n_curr = len(curr_data)
        
        # Bootstrap samples of Wasserstein distance
        wasserstein_samples = []
        
        for _ in range(1000):
            # Bootstrap resample
            ref_sample = np.random.choice(ref_data, size=n_ref, replace=True)
            curr_sample = np.random.choice(curr_data, size=n_curr, replace=True)
            
            # Compute Wasserstein distance
            ref_sorted = np.sort(ref_sample)
            curr_sorted = np.sort(curr_sample)
            
            # 1D Wasserstein distance (Earth mover's distance)
            if len(ref_sorted) == len(curr_sorted):
                wasserstein = np.mean(np.abs(ref_sorted - curr_sorted))
            else:
                # Interpolate to common length
                min_len = min(len(ref_sorted), len(curr_sorted))
                wasserstein = np.mean(np.abs(ref_sorted[:min_len] - curr_sorted[:min_len]))
            
            wasserstein_samples.append(wasserstein)
        
        wasserstein_samples = np.array(wasserstein_samples)
        
        # Compute posterior statistics
        posterior_mean = np.mean(wasserstein_samples)
        posterior_std = np.std(wasserstein_samples)
        
        # Drift probability
        drift_prob = np.mean(wasserstein_samples > threshold)
        
        # Credible interval
        ci_lower = np.percentile(wasserstein_samples, 2.5)
        ci_upper = np.percentile(wasserstein_samples, 97.5)
        
        # Decision
        if drift_prob > 0.8:
            decision = "drift"
        elif drift_prob < 0.2:
            decision = "no_drift"
        else:
            decision = "uncertain"
        
        return BayesianDriftResult(
            drift_probability=float(drift_prob),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            bayes_factor=drift_prob / (1 - drift_prob + 1e-10),
            posterior_mean=float(posterior_mean),
            posterior_std=float(posterior_std),
            decision=decision
        )
    
    def combine_multiple_tests(self, results: Dict[str, BayesianDriftResult],
                              weights: Optional[Dict[str, float]] = None) -> BayesianDriftResult:
        """
        Combine results from multiple drift detection methods.
        
        Args:
            results: Dictionary of method -> BayesianDriftResult
            weights: Optional weights for each method
            
        Returns:
            Combined result
        """
        if not results:
            raise ValueError("No results to combine")
        
        if weights is None:
            # Default weights
            weights = {"ks": 0.4, "psi": 0.3, "wasserstein": 0.3}
        
        # Weighted average of drift probabilities
        drift_probs = []
        weight_values = []
        
        for method, result in results.items():
            drift_probs.append(result.drift_probability)
            weight_values.append(weights.get(method, 0.1))
        
        drift_probs = np.array(drift_probs)
        weight_values = np.array(weight_values)
        weight_values = weight_values / weight_values.sum()  # Normalize
        
        combined_prob = np.sum(drift_probs * weight_values)
        
        # Compute combined credible interval (simplified)
        lower_bounds = [r.confidence_interval[0] for r in results.values()]
        upper_bounds = [r.confidence_interval[1] for r in results.values()]
        
        combined_lower = np.average(lower_bounds, weights=weight_values)
        combined_upper = np.average(upper_bounds, weights=weight_values)
        
        # Decision based on combined probability
        if combined_prob > 0.7:
            decision = "drift"
        elif combined_prob < 0.3:
            decision = "no_drift"
        else:
            decision = "uncertain"
        
        return BayesianDriftResult(
            drift_probability=float(combined_prob),
            confidence_interval=(float(combined_lower), float(combined_upper)),
            bayes_factor=combined_prob / (1 - combined_prob + 1e-10),
            posterior_mean=float(combined_prob),
            posterior_std=float(np.std(drift_probs)),
            decision=decision
        )


class ConfidenceEstimator:
    """
    Estimates confidence in drift detection results.
    """
    
    @staticmethod
    def estimate_confidence(result: BayesianDriftResult) -> float:
        """
        Estimate confidence score (0-1) for a drift detection result.
        
        Higher confidence means more certain about the decision.
        """
        # Factors affecting confidence:
        # 1. Width of confidence interval (narrower = more confident)
        ci_width = result.confidence_interval[1] - result.confidence_interval[0]
        ci_confidence = 1.0 - min(ci_width, 1.0)
        
        # 2. Bayes factor magnitude (further from 1 = more confident)
        bayes_factor_confidence = min(abs(np.log(result.bayes_factor + 1e-10)) / 5.0, 1.0)
        
        # 3. Distance from decision boundary (0.5)
        boundary_distance = abs(result.drift_probability - 0.5) * 2.0
        
        # 4. Posterior standard deviation (smaller = more confident)
        std_confidence = 1.0 - min(result.posterior_std, 1.0)
        
        # Combined confidence
        confidence = 0.4 * ci_confidence + 0.3 * bayes_factor_confidence + 0.2 * boundary_distance + 0.1 * std_confidence
        
        return float(max(0.0, min(1.0, confidence)))
    
    @staticmethod
    def get_confidence_level(confidence_score: float) -> str:
        """Convert confidence score to human-readable level."""
        if confidence_score >= 0.9:
            return "very_high"
        elif confidence_score >= 0.7:
            return "high"
        elif confidence_score >= 0.5:
            return "medium"
        elif confidence_score >= 0.3:
            return "low"
        else:
            return "very_low"


# Factory functions for integration
def create_bayesian_drift_detector(prior_alpha: float = 1.0, prior_beta: float = 1.0) -> BayesianDriftDetector:
    """Create a Bayesian drift detector."""
    return BayesianDriftDetector(prior_alpha, prior_beta)

def create_confidence_estimator() -> ConfidenceEstimator:
    """Create a confidence estimator."""
    return ConfidenceEstimator()


if __name__ == "__main__":
    # Test Bayesian drift detection
    detector = BayesianDriftDetector()
    
    # Generate test data
    np.random.seed(42)
    ref_data = np.random.randn(1000)
    curr_data = np.random.randn(1000) * 1.2 + 0.3  # Some drift
    
    # Test KS method
    result = detector.detect_drift_bayesian(ref_data, curr_data, method="ks", threshold=0.05)
    print("Bayesian KS Test Result:")
    print(f"  Drift probability: {result.drift_probability:.3f}")
    print(f"  Confidence interval: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")
    print(f"  Bayes factor: {result.bayes_factor:.3f}")
    print(f"  Decision: {result.decision}")
    
    # Test confidence estimation
    confidence = ConfidenceEstimator.estimate_confidence(result)
    confidence_level = ConfidenceEstimator.get_confidence_level(confidence)
    print(f"  Confidence: {confidence:.3f} ({confidence_level})")
    
    # Test multiple methods combination
    results = {}
    for method in ["ks", "psi", "wasserstein"]:
        try:
            results[method] = detector.detect_drift_bayesian(ref_data, curr_data, method=method, threshold=0.05)
        except:
            continue
    
    if results:
        combined = detector.combine_multiple_tests(results)
        print(f"\nCombined result (from {len(results)} methods):")
        print(f"  Combined drift probability: {combined.drift_probability:.3f}")
        print(f"  Combined decision: {combined.decision}")
