"""
Decision trace for audit logging
"""
import json
from datetime import datetime
from typing import Dict, Any
import uuid
import os

class DecisionTrace:
    """Structured audit trail for every healing decision"""
    
    def __init__(self, incident_id: str = None):
        # Generate trace_id if not provided
        self.trace_id = str(uuid.uuid4())
        self.incident_id = incident_id or f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        self.timestamp = datetime.now().isoformat()
        self.state = {}
        self.rule_action = None
        self.bandit_action = None
        self.bandit_confidence = None
        self.final_action = None
        self.reason = ""
        self.cooldown_triggered = False
        self.system_version = "v0.1-safe-autonomy"
    
    def record_state(self, state: Dict[str, Any]):
        """Record system state at decision time"""
        self.state = state
    
    def record_rule_decision(self, action: str):
        """Record rule-based decision"""
        self.rule_action = action
    
    def record_bandit_decision(self, action: str, confidence: float):
        """Record bandit-based decision"""
        self.bandit_action = action
        self.bandit_confidence = confidence
    
    def record_final_decision(self, action: str, reason: str):
        """Record final hybrid decision"""
        self.final_action = action
        self.reason = reason
    
    def record_cooldown(self, triggered: bool):
        """Record cooldown status"""
        self.cooldown_triggered = triggered
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "trace_id": self.trace_id,
            "incident_id": self.incident_id,
            "timestamp": self.timestamp,
            "state": self.state,
            "rule_action": self.rule_action,
            "bandit_action": self.bandit_action,
            "bandit_confidence": self.bandit_confidence,
            "final_action": self.final_action,
            "reason": self.reason,
            "cooldown_triggered": self.cooldown_triggered,
            "system_version": self.system_version
        }
    
    def save(self, directory: str = "logs/decision_traces"):
        """Save trace to JSON file"""
        os.makedirs(directory, exist_ok=True)
        
        filename = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.trace_id[:8]}.json"
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str):
        """Load trace from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        trace = cls(data.get('incident_id'))
        trace.trace_id = data.get('trace_id', str(uuid.uuid4()))
        trace.timestamp = data.get('timestamp', trace.timestamp)
        trace.state = data.get('state', {})
        trace.rule_action = data.get('rule_action')
        trace.bandit_action = data.get('bandit_action')
        trace.bandit_confidence = data.get('bandit_confidence')
        trace.final_action = data.get('final_action')
        trace.reason = data.get('reason', '')
        trace.cooldown_triggered = data.get('cooldown_triggered', False)
        trace.system_version = data.get('system_version', 'v0.1-safe-autonomy')
        
        return trace

# Helper function to create test traces
def create_test_trace():
    """Create a test trace for verification"""
    trace = DecisionTrace("test_verification")
    trace.record_state({"drift_score": 0.25, "accuracy_drop": 0.1})
    trace.record_rule_decision("retrain")
    trace.record_bandit_decision("fallback", 0.75)
    trace.record_final_decision("retrain", "high drift detected")
    return trace.save()

if __name__ == "__main__":
    # Create a test trace
    test_file = create_test_trace()
    print(f"✅ Test trace created: {test_file}")
    
    # Verify it can be loaded
    loaded_trace = DecisionTrace.load(test_file)
    print(f"✅ Trace loaded successfully")
    print(f"   Trace ID: {loaded_trace.trace_id}")
    print(f"   Final action: {loaded_trace.final_action}")
