import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Tuple
import joblib
from datetime import datetime
import json

class FailureIntelligence:
    """C.1 — Failure as a First-Class Signal"""
    
    def __init__(self):
        self.failure_dataset = []
        self.fragility_scores = {}
        
    def log_failure(self, 
                   state_before: Dict,
                   action_taken: str,
                   outcome: str,
                   recovery_cost: float,
                   metadata: Dict = None):
        """Log failure event as F = (state_before, action_taken, outcome, recovery_cost)"""
        
        failure_record = {
            'timestamp': datetime.now().isoformat(),
            'state_before': state_before,
            'action_taken': action_taken,
            'outcome': outcome,
            'recovery_cost': recovery_cost,
            'metadata': metadata or {}
        }
        
        self.failure_dataset.append(failure_record)
        return failure_record

class PredictiveFailureScorer:
    """C.2 — Predictive Failure Scoring"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def calculate_failure_likelihood(self,
                                   model_metrics: Dict,
                                   data_quality: Dict,
                                   environment: Dict) -> Dict:
        """P(failure | model, data, environment)"""
        
        # Extract features
        features = self._extract_features(model_metrics, data_quality, environment)
        
        if self.is_trained:
            prob = self.model.predict_proba([features])[0]
            failure_prob = prob[1]  # Probability of failure class
        else:
            # Default heuristic scoring
            failure_prob = self._heuristic_failure_score(features)
        
        # Expected MTTR (Mean Time To Recovery)
        expected_mttr = self._estimate_mttr(failure_prob, features)
        
        # Expected cost
        expected_cost = self._estimate_cost(failure_prob, expected_mttr)
        
        return {
            'failure_likelihood': float(failure_prob),
            'expected_mttr': float(expected_mttr),
            'expected_cost': float(expected_cost),
            'risk_level': self._risk_classification(failure_prob),
            'features': features
        }
    
    def _extract_features(self, model_metrics, data_quality, environment):
        """Extract 20+ predictive features"""
        return [
            model_metrics.get('accuracy', 0.5),
            model_metrics.get('f1_score', 0.5),
            model_metrics.get('drift_score', 0),
            data_quality.get('missing_rate', 0),
            data_quality.get('outlier_rate', 0),
            environment.get('load_factor', 1.0),
            environment.get('stability_score', 1.0),
            # Add more features...
        ]
    
    def _heuristic_failure_score(self, features: List[float]) -> float:
        """Calculate failure probability using heuristic scoring when model is not trained."""
        if not features:
            return 0.5
        
        # Heuristic: lower accuracy/f1, higher drift/missing_rate/outlier_rate = higher failure prob
        accuracy = features[0] if len(features) > 0 else 0.5
        f1_score = features[1] if len(features) > 1 else 0.5
        drift_score = features[2] if len(features) > 2 else 0
        missing_rate = features[3] if len(features) > 3 else 0
        outlier_rate = features[4] if len(features) > 4 else 0
        
        # Weighted heuristic score
        failure_prob = (
            (1 - accuracy) * 0.25 +
            (1 - f1_score) * 0.25 +
            drift_score * 0.2 +
            missing_rate * 0.15 +
            outlier_rate * 0.15
        )
        
        return min(1.0, max(0.0, failure_prob))
    
    def _estimate_mttr(self, failure_prob: float, features: List[float]) -> float:
        """Estimate Mean Time To Recovery based on failure probability and features."""
        # Base MTTR in minutes
        base_mttr = 30.0
        
        # Higher failure probability might indicate more complex issues
        mttr = base_mttr * (1 + failure_prob * 0.5)
        
        return mttr
    
    def _estimate_cost(self, failure_prob: float, mttr: float) -> float:
        """Estimate expected cost of failure."""
        # Cost per minute of downtime
        cost_per_minute = 1000.0
        
        expected_cost = failure_prob * mttr * cost_per_minute
        
        return expected_cost
    
    def _risk_classification(self, failure_prob: float) -> str:
        """Classify risk level based on failure probability."""
        if failure_prob > 0.7:
            return 'CRITICAL'
        elif failure_prob > 0.4:
            return 'HIGH'
        elif failure_prob > 0.2:
            return 'MEDIUM'
        else:
            return 'LOW'

class FragilityIndex:
    """C.3 — Fragility Index (This Is Gold)"""
    
    def __init__(self):
        self.stress_scenarios = self._define_stress_scenarios()
    
    def _define_stress_scenarios(self) -> list:
        """Define stress test scenarios."""
        return [
            {'name': 'data_drift', 'severity': 0.5},
            {'name': 'concept_drift', 'severity': 0.7},
            {'name': 'load_spike', 'severity': 2.0},
            {'name': 'data_quality_degradation', 'severity': 0.3}
        ]
    
    def _apply_stress_scenario(self, model, data_pipeline, scenario) -> dict:
        """Apply stress scenario to model and return stressed metrics."""
        base_accuracy = model.get('accuracy', 0.85) if isinstance(model, dict) else 0.85
        severity = scenario.get('severity', 0.5)
        
        # Apply stress based on scenario type
        if scenario['name'] == 'data_drift':
            return {'accuracy': base_accuracy * (1 - severity * 0.2), 'drift': severity}
        elif scenario['name'] == 'concept_drift':
            return {'accuracy': base_accuracy * (1 - severity * 0.3), 'drift': severity * 1.5}
        elif scenario['name'] == 'load_spike':
            return {'accuracy': base_accuracy * (1 - severity * 0.1), 'latency': severity * 100}
        else:  # data_quality_degradation
            return {'accuracy': base_accuracy * (1 - severity * 0.15), 'missing_rate': severity * 0.2}
    
    def _estimate_failure_under_stress(self, stressed_metrics: dict) -> float:
        """Estimate failure probability under stressed conditions."""
        accuracy = stressed_metrics.get('accuracy', 0.85)
        drift = stressed_metrics.get('drift', 0)
        
        # Simple heuristic: lower accuracy + higher drift = higher failure prob
        failure_prob = (1 - accuracy) * 0.5 + drift * 0.5
        return min(1.0, max(0.0, failure_prob))
    
    def _calculate_scenario_cost(self, failure_prob: float, cost_model: dict, scenario: dict) -> float:
        """Calculate expected cost for a scenario."""
        downtime_cost = cost_model.get('downtime_cost_per_minute', 1000) if isinstance(cost_model, dict) else 1000
        avg_mttr = 30  # minutes
        
        return failure_prob * downtime_cost * avg_mttr
    
    def _rate_risk(self, failure_prob: float, cost: float) -> str:
        """Rate risk level based on failure probability and cost."""
        if failure_prob > 0.7 or cost > 50000:
            return 'CRITICAL'
        elif failure_prob > 0.4 or cost > 20000:
            return 'HIGH'
        elif failure_prob > 0.2 or cost > 10000:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_recommendation(self, fragility_index: float, margin_of_safety: float) -> str:
        """Generate recommendation based on fragility analysis."""
        if fragility_index > 0.7:
            return 'URGENT: System is highly fragile. Immediate remediation required.'
        elif fragility_index > 0.4:
            return 'WARNING: Consider implementing additional safeguards.'
        elif margin_of_safety < 0.3:
            return 'CAUTION: Low margin of safety. Monitor closely.'
        else:
            return 'System resilience is acceptable. Continue monitoring.'
    
    def calculate_fragility(self, model, data_pipeline, cost_model) -> Dict:
        """Fragility(model) = E[cost | stress scenarios]"""
        
        fragility_scores = []
        scenario_results = []
        
        for scenario in self.stress_scenarios:
            # Apply stress
            stressed_metrics = self._apply_stress_scenario(model, data_pipeline, scenario)
            
            # Calculate failure probability
            failure_prob = self._estimate_failure_under_stress(stressed_metrics)
            
            # Calculate expected cost
            scenario_cost = self._calculate_scenario_cost(failure_prob, cost_model, scenario)
            
            fragility_scores.append(scenario_cost)
            scenario_results.append({
                'scenario': scenario['name'],
                'failure_probability': failure_prob,
                'expected_cost': scenario_cost,
                'risk_rating': self._rate_risk(failure_prob, scenario_cost)
            })
        
        fragility_index = np.mean(fragility_scores)
        margin_of_safety = 1.0 - (np.std(fragility_scores) / fragility_index) if fragility_index > 0 else 0
        
        return {
            'fragility_index': float(fragility_index),
            'margin_of_safety': float(margin_of_safety),
            'worst_case_scenario': max(scenario_results, key=lambda x: x['expected_cost']),
            'scenario_breakdown': scenario_results,
            'recommendation': self._generate_recommendation(fragility_index, margin_of_safety)
        }