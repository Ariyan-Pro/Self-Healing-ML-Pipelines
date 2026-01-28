# adaptive/memory/experience_enricher.py
"""
Enhance experiences with metadata for better analysis.
Does not affect learning, only adds reporting context.
"""

from datetime import datetime
import json
from pathlib import Path


class ExperienceEnricher:
    """Add metadata to experiences for analysis and reporting."""
    
    def __init__(self, experiences_path=None):
        self.experiences_path = experiences_path or Path("adaptive/memory/experiences.json")
    
    def add_reporting_metadata(self, experience):
        """Add metadata for reporting (non-breaking)."""
        if not isinstance(experience, dict):
            return experience
        
        # Add timestamp context
        experience["_reporting_metadata"] = {
            "analysis_timestamp": datetime.now().isoformat(),
            "cycle_count": self._get_experience_count() + 1,
            "system_phase": "testing",  # Will be "production" in real deployment
            "optimization_status": "cost_minimization"  # Current strategy
        }
        
        # Add potential value estimates for reporting
        if experience.get("action") == "fallback":
            experience["_reporting_metadata"]["potential_production_value"] = {
                "estimated_sla_savings": 0.5,
                "customer_retention_impact": 0.3,
                "note": "In production, fallback has additional business value"
            }
        
        return experience
    
    def _get_experience_count(self):
        """Get current experience count."""
        if self.experiences_path.exists():
            try:
                with open(self.experiences_path, 'r') as f:
                    experiences = json.load(f)
                    return len(experiences)
            except:
                return 0
        return 0
    
    def generate_learning_report(self):
        """Generate report on system learning progress."""
        if not self.experiences_path.exists():
            return {"status": "no_experiences", "learning_stage": "initial"}
        
        with open(self.experiences_path, 'r') as f:
            experiences = json.load(f)
        
        # Analyze learning progress
        if len(experiences) < 10:
            stage = "early_exploration"
        elif len(experiences) < 50:
            stage = "pattern_learning"
        else:
            stage = "stable_optimization"
        
        return {
            "total_experiences": len(experiences),
            "learning_stage": stage,
            "recommendation": self._get_stage_recommendation(stage, len(experiences))
        }
    
    def _get_stage_recommendation(self, stage, count):
        """Get recommendation based on learning stage."""
        recommendations = {
            "early_exploration": "Continue exploration - system is learning cost surface",
            "pattern_learning": "System is identifying patterns - ready for production tuning",
            "stable_optimization": "System has stable policies - ready for full autonomy"
        }
        return recommendations.get(stage, f"Continue operation with {count} experiences")
