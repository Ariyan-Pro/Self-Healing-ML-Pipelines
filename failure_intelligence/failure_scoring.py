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

class FragilityIndex:
    """C.3 — Fragility Index (This Is Gold)"""
    
    def __init__(self):
        self.stress_scenarios = self._define_stress_scenarios()
        
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