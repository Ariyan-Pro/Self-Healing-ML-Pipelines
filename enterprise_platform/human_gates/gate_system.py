#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# enterprise_platform/human_gates/gate_system.py
"""
Human-in-the-loop gate system for enterprise governance.
Provides veto power and approval workflows.
"""

import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class GateType(Enum):
    """Types of human gates."""
    VETO_ONLY = "veto_only"  # Human can veto, auto passes otherwise
    APPROVAL_REQUIRED = "approval_required"  # Explicit approval needed
    ACKNOWLEDGMENT = "acknowledgment"  # Human acknowledges, auto proceeds
    ESCALATION = "escalation"  # Escalates to higher authority


class GateStatus(Enum):
    """Status of a gate."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    EXPIRED = "expired"
    BYPASSED = "bypassed"


@dataclass
class GateRule:
    """Rule defining when a gate is triggered."""
    risk_score_threshold: float = 0.7
    action_types: List[str] = field(default_factory=lambda: ["retrain", "rollback"])
    cost_threshold: float = 1000.00
    model_tier: str = "tier1_critical"
    compliance_tags: List[str] = field(default_factory=list)
    
    def should_trigger(self, context: Dict[str, Any]) -> bool:
        """Determine if gate should be triggered based on context."""
        # Check risk score
        if context.get("risk_score", 0) > self.risk_score_threshold:
            return True
        
        # Check action type
        if context.get("action") in self.action_types:
            return True
        
        # Check cost
        if context.get("estimated_cost", 0) > self.cost_threshold:
            return True
        
        # Check model tier
        if context.get("model_tier") == self.model_tier:
            return True
        
        # Check compliance tags intersection
        context_tags = set(context.get("compliance_tags", []))
        rule_tags = set(self.compliance_tags)
        if context_tags.intersection(rule_tags):
            return True
        
        return False


@dataclass
class HumanGate:
    """A human gate instance."""
    gate_id: str
    gate_type: GateType
    rule: GateRule
    required_roles: List[str]
    timeout_minutes: int = 30
    escalation_path: Optional[List[str]] = None
    notification_channels: List[str] = field(default_factory=lambda: ["email", "slack"])
    
    def create_gate_instance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a gate instance for a specific context."""
        return {
            "gate_id": self.gate_id,
            "gate_type": self.gate_type.value,
            "context": context,
            "status": GateStatus.PENDING.value,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(minutes=self.timeout_minutes)).isoformat(),
            "required_roles": self.required_roles,
            "escalation_path": self.escalation_path,
            "actions_taken": [],
            "audit_trail": []
        }


class HumanGateSystem:
    """Enterprise human gate system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.gates: List[HumanGate] = []
        self.gate_instances: Dict[str, Dict] = {}
        self.audit_log: List[Dict] = []
        
        if config_path:
            self.load_config(config_path)
        
        # Load default gates
        self._load_default_gates()
    
    def _load_default_gates(self):
        """Load default enterprise gates."""
        # High-risk action gate
        high_risk_gate = HumanGate(
            gate_id="high_risk_approval",
            gate_type=GateType.APPROVAL_REQUIRED,
            rule=GateRule(
                risk_score_threshold=0.8,
                action_types=["retrain", "rollback"],
                cost_threshold=500.00,
                model_tier="tier1_critical",
                compliance_tags=["financial", "hipaa"]
            ),
            required_roles=["ml_ops", "manager"],
            timeout_minutes=60,
            escalation_path=["director", "vp_engineering"],
            notification_channels=["email", "slack", "pagerduty"]
        )
        
        # Compliance gate
        compliance_gate = HumanGate(
            gate_id="compliance_veto",
            gate_type=GateType.VETO_ONLY,
            rule=GateRule(
                compliance_tags=["gdpr", "hipaa", "pci"]
            ),
            required_roles=["compliance_officer"],
            timeout_minutes=120,
            notification_channels=["email"]
        )
        
        # Cost gate
        cost_gate = HumanGate(
            gate_id="cost_approval",
            gate_type=GateType.APPROVAL_REQUIRED,
            rule=GateRule(
                cost_threshold=1000.00
            ),
            required_roles=["manager", "director"],
            timeout_minutes=30,
            notification_channels=["email", "slack"]
        )
        
        # Acknowledgment gate for all retrains
        acknowledgment_gate = HumanGate(
            gate_id="retrain_acknowledgment",
            gate_type=GateType.ACKNOWLEDGMENT,
            rule=GateRule(
                action_types=["retrain"]
            ),
            required_roles=["ml_engineer"],
            timeout_minutes=15,
            notification_channels=["slack"]
        )
        
        self.gates = [high_risk_gate, compliance_gate, cost_gate, acknowledgment_gate]
    
    def load_config(self, config_path: str):
        """Load gate configuration from YAML."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for gate_config in config.get("gates", []):
            gate = HumanGate(
                gate_id=gate_config["id"],
                gate_type=GateType(gate_config["type"]),
                rule=GateRule(**gate_config["rule"]),
                required_roles=gate_config["required_roles"],
                timeout_minutes=gate_config.get("timeout_minutes", 30),
                escalation_path=gate_config.get("escalation_path"),
                notification_channels=gate_config.get("notification_channels", ["email"])
            )
            self.gates.append(gate)
    
    def evaluate_gates(self, context: Dict[str, Any]) -> List[Dict]:
        """Evaluate which gates should be triggered for a given context."""
        triggered_gates = []
        
        for gate in self.gates:
            if gate.rule.should_trigger(context):
                gate_instance = gate.create_gate_instance(context)
                self.gate_instances[gate_instance["gate_id"]] = gate_instance
                triggered_gates.append(gate_instance)
                
                # Log audit
                self._log_audit(
                    action="gate_triggered",
                    gate_id=gate.gate_id,
                    context=context,
                    message=f"Gate '{gate.gate_id}' triggered"
                )
        
        return triggered_gates
    
    def process_decision(self, gate_id: str, user: str, role: str, 
                        decision: str, comments: Optional[str] = None) -> bool:
        """Process a human decision on a gate."""
        if gate_id not in self.gate_instances:
            return False
        
        gate_instance = self.gate_instances[gate_id]
        
        # Check if user has required role
        if role not in gate_instance["required_roles"]:
            self._log_audit(
                action="gate_decision_rejected",
                gate_id=gate_id,
                user=user,
                role=role,
                decision=decision,
                message=f"User '{user}' with role '{role}' not authorized"
            )
            return False
        
        # Update gate status
        gate_instance["status"] = decision
        gate_instance["decided_by"] = user
        gate_instance["decided_at"] = datetime.utcnow().isoformat()
        gate_instance["comments"] = comments
        
        # Record action
        action = {
            "timestamp": datetime.utcnow().isoformat(),
            "user": user,
            "role": role,
            "decision": decision,
            "comments": comments
        }
        gate_instance["actions_taken"].append(action)
        
        # Log audit
        self._log_audit(
            action="gate_decision",
            gate_id=gate_id,
            user=user,
            role=role,
            decision=decision,
            comments=comments,
            message=f"Gate '{gate_id}' decided: {decision}"
        )
        
        return True
    
    def escalate_gate(self, gate_id: str, reason: str):
        """Escalate a gate to the next level."""
        if gate_id not in self.gate_instances:
            return False
        
        gate_instance = self.gate_instances[gate_id]
        
        # Get escalation path
        escalation_path = gate_instance.get("escalation_path", [])
        if not escalation_path:
            return False
        
        # Move to next escalation level
        current_level = gate_instance.get("escalation_level", 0)
        if current_level >= len(escalation_path):
            gate_instance["status"] = GateStatus.EXPIRED.value
            return False
        
        next_role = escalation_path[current_level]
        gate_instance["escalation_level"] = current_level + 1
        gate_instance["escalated_to"] = next_role
        gate_instance["status"] = GateStatus.ESCALATED.value
        gate_instance["escalation_reason"] = reason
        
        # Log audit
        self._log_audit(
            action="gate_escalated",
            gate_id=gate_id,
            escalated_to=next_role,
            reason=reason,
            message=f"Gate '{gate_id}' escalated to {next_role}"
        )
        
        return True
    
    def check_expired_gates(self):
        """Check and update status of expired gates."""
        now = datetime.utcnow()
        expired_count = 0
        
        for gate_id, gate_instance in list(self.gate_instances.items()):
            expires_at = datetime.fromisoformat(gate_instance["expires_at"])
            if now > expires_at and gate_instance["status"] == GateStatus.PENDING.value:
                gate_instance["status"] = GateStatus.EXPIRED.value
                expired_count += 1
                
                # Log audit
                self._log_audit(
                    action="gate_expired",
                    gate_id=gate_id,
                    message=f"Gate '{gate_id}' expired"
                )
        
        return expired_count
    
    def get_gate_status(self, gate_id: str) -> Optional[Dict]:
        """Get current status of a gate."""
        return self.gate_instances.get(gate_id)
    
    def can_proceed(self, gate_ids: List[str]) -> bool:
        """Check if all gates allow proceeding."""
        for gate_id in gate_ids:
            gate_instance = self.gate_instances.get(gate_id)
            if not gate_instance:
                return False
            
            status = gate_instance["status"]
            gate_type = gate_instance["gate_type"]
            
            if gate_type == GateType.VETO_ONLY.value:
                # Veto gates only block if explicitly rejected
                if status == GateStatus.REJECTED.value:
                    return False
                # Approved, expired, or pending all allow proceeding
                continue
            
            elif gate_type == GateType.APPROVAL_REQUIRED.value:
                # Approval gates require explicit approval
                if status != GateStatus.APPROVED.value:
                    return False
            
            elif gate_type == GateType.ACKNOWLEDGMENT.value:
                # Acknowledgment gates require non-pending status
                if status == GateStatus.PENDING.value:
                    return False
            
            elif gate_type == GateType.ESCALATION.value:
                # Escalation gates require final decision
                if status not in [GateStatus.APPROVED.value, GateStatus.REJECTED.value]:
                    return False
        
        return True
    
    def _log_audit(self, **kwargs):
        """Log an audit entry."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        self.audit_log.append(audit_entry)
    
    def save_audit_log(self, path: str):
        """Save audit log to file."""
        with open(path, 'w') as f:
            json.dump(self.audit_log, f, indent=2, default=str)
    
    def load_audit_log(self, path: str):
        """Load audit log from file."""
        if Path(path).exists():
            with open(path, 'r') as f:
                self.audit_log = json.load(f)


# Example usage
if __name__ == "__main__":
    # Initialize gate system
    gate_system = HumanGateSystem()
    
    # Example context that would trigger gates
    context = {
        "action": "retrain",
        "risk_score": 0.85,
        "estimated_cost": 1200.00,
        "model_tier": "tier1_critical",
        "compliance_tags": ["financial"],
        "pipeline_id": "pipeline-001",
        "model_id": "model-recommendation-v1"
    }
    
    # Evaluate which gates are triggered
    triggered_gates = gate_system.evaluate_gates(context)
    print(f"Triggered gates: {len(triggered_gates)}")
    for gate in triggered_gates:
        print(f"  - {gate['gate_id']} ({gate['gate_type']})")
    
    # Simulate human decisions
    if triggered_gates:
        # Manager approves high risk gate
        gate_system.process_decision(
            gate_id="high_risk_approval",
            user="john.doe",
            role="manager",
            decision="approved",
            comments="Cost justified for critical model"
        )
        
        # Compliance officer acknowledges
        gate_system.process_decision(
            gate_id="compliance_veto",
            user="jane.smith", 
            role="compliance_officer",
            decision="approved",
            comments="No compliance issues identified"
        )
        
        # Check if we can proceed
        gate_ids = [gate["gate_id"] for gate in triggered_gates]
        can_proceed = gate_system.can_proceed(gate_ids)
        print(f"\nCan proceed with action: {can_proceed}")
        
        # Save audit log
        gate_system.save_audit_log("gate_audit_log.json")

