#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-tenant ML Reliability Platform.
Manages multiple pipelines with isolated budgets, thresholds, and SLAs.
"""

import yaml
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import uuid


class TenantTier(Enum):
    """Tenant pricing/feature tiers."""
    ENTERPRISE = "enterprise"  # Full features, dedicated resources
    BUSINESS = "business"      # Advanced features, shared resources  
    STANDARD = "standard"      # Basic features, shared resources
    TRIAL = "trial"           # Limited features, evaluation only


class PipelineStatus(Enum):
    """Pipeline status."""
    ACTIVE = "active"
    PAUSED = "paused"
    MAINTENANCE = "maintenance"
    ARCHIVED = "archived"
    ERROR = "error"


@dataclass
class TenantConfig:
    """Configuration for a tenant."""
    tenant_id: str
    name: str
    tier: TenantTier
    max_pipelines: int
    max_models_per_pipeline: int
    monthly_cost_budget: float
    max_monthly_compute_hours: float
    sla_targets: Dict[str, float]  # e.g., {"availability": 0.999, "mttr": 2.0}
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    compliance_requirements: List[str] = field(default_factory=list)
    data_retention_days: int = 90
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    @property
    def daily_cost_budget(self) -> float:
        """Calculate daily cost budget."""
        return self.monthly_cost_budget / 30
    
    @property
    def daily_compute_hours(self) -> float:
        """Calculate daily compute hour limit."""
        return self.max_monthly_compute_hours / 30


@dataclass
class PipelineConfig:
    """Configuration for a pipeline within a tenant."""
    pipeline_id: str
    tenant_id: str
    name: str
    description: str
    model_type: str
    status: PipelineStatus = PipelineStatus.ACTIVE
    
    # Budget isolation
    pipeline_cost_budget: Optional[float] = None
    pipeline_compute_budget: Optional[float] = None
    
    # Threshold isolation
    drift_threshold: float = 0.20
    accuracy_drop_threshold: float = 0.15
    anomaly_rate_threshold: float = 0.05
    latency_increase_threshold: float = 0.20
    
    # SLA weights (customizable per pipeline)
    sla_weights: Dict[str, float] = field(default_factory=lambda: {
        "availability": 0.4,
        "accuracy": 0.3,
        "latency": 0.2,
        "cost": 0.1
    })
    
    # Governance
    approval_required: bool = False
    allowed_actions: List[str] = field(default_factory=lambda: ["fallback", "rollback", "retrain", "no_op"])
    notification_channels: List[str] = field(default_factory=lambda: ["email", "slack"])
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pipeline_id": self.pipeline_id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "description": self.description,
            "model_type": self.model_type,
            "status": self.status.value,
            "pipeline_cost_budget": self.pipeline_cost_budget,
            "pipeline_compute_budget": self.pipeline_compute_budget,
            "drift_threshold": self.drift_threshold,
            "accuracy_drop_threshold": self.accuracy_drop_threshold,
            "anomaly_rate_threshold": self.anomaly_rate_threshold,
            "latency_increase_threshold": self.latency_increase_threshold,
            "sla_weights": self.sla_weights,
            "approval_required": self.approval_required,
            "allowed_actions": self.allowed_actions,
            "notification_channels": self.notification_channels,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


@dataclass
class UsageMetrics:
    """Usage metrics for a tenant or pipeline."""
    current_cost: float = 0.0
    current_compute_hours: float = 0.0
    actions_taken: Dict[str, int] = field(default_factory=lambda: {
        "fallback": 0,
        "rollback": 0,
        "retrain": 0,
        "no_op": 0
    })
    sla_violations: int = 0
    incidents_resolved: int = 0
    mttr_minutes: List[float] = field(default_factory=list)
    
    @property
    def avg_mttr(self) -> float:
        """Calculate average MTTR."""
        if not self.mttr_minutes:
            return 0.0
        return sum(self.mttr_minutes) / len(self.mttr_minutes)
    
    def add_action(self, action: str, cost: float, compute_hours: float, mttr: Optional[float] = None):
        """Record an action."""
        self.current_cost += cost
        self.current_compute_hours += compute_hours
        
        if action in self.actions_taken:
            self.actions_taken[action] += 1
        
        if mttr is not None:
            self.mttr_minutes.append(mttr)
            self.incidents_resolved += 1


class MultiTenantPlatform:
    """Enterprise multi-tenant ML reliability platform."""
    
    def __init__(self, config_dir: str = "configs"):
        self.tenants: Dict[str, TenantConfig] = {}
        self.pipelines: Dict[str, PipelineConfig] = {}
        self.usage_metrics: Dict[str, UsageMetrics] = {}  # tenant_id -> metrics
        self.pipeline_metrics: Dict[str, UsageMetrics] = {}  # pipeline_id -> metrics
        self.config_dir = config_dir
        
        # Load existing configuration
        self._load_configuration()
    
    def _load_configuration(self):
        """Load tenant and pipeline configuration."""
        print(f"DEBUG _load_configuration: config_dir = {self.config_dir}")
        print(f"DEBUG _load_configuration: config_dir type = {type(self.config_dir)}")
        
        # Load tenants
        tenants_path = Path(self.config_dir) / "tenants.yaml"
        print(f"DEBUG: Looking for tenants at {tenants_path}")
        print(f"DEBUG: Path exists: {tenants_path.exists()}")
        
        if tenants_path.exists():
            try:
                with open(tenants_path, 'r', encoding='utf-8') as f:
                    tenants_data = yaml.safe_load(f)
                    print(f"DEBUG: Tenants data loaded: {tenants_data is not None}")
                    
                    if tenants_data and 'tenants' in tenants_data:
                        print(f"DEBUG: Found {len(tenants_data['tenants'])} tenants in YAML")
                        for tenant_data in tenants_data.get("tenants", []):
                            print(f"DEBUG: Loading tenant: {tenant_data.get('tenant_id', 'Unknown')}")
                            tenant = TenantConfig(
                                tenant_id=tenant_data["tenant_id"],
                                name=tenant_data["name"],
                                tier=TenantTier(tenant_data["tier"]),
                                max_pipelines=tenant_data["max_pipelines"],
                                max_models_per_pipeline=tenant_data["max_models_per_pipeline"],
                                monthly_cost_budget=tenant_data["monthly_cost_budget"],
                                max_monthly_compute_hours=tenant_data["max_monthly_compute_hours"],
                                sla_targets=tenant_data["sla_targets"],
                                feature_flags=tenant_data.get("feature_flags", {}),
                                compliance_requirements=tenant_data.get("compliance_requirements", []),
                                data_retention_days=tenant_data.get("data_retention_days", 90)
                            )
                            self.tenants[tenant.tenant_id] = tenant
                            self.usage_metrics[tenant.tenant_id] = UsageMetrics()
                            print(f"DEBUG: Tenant added: {tenant.tenant_id}")
                    else:
                        print("DEBUG: No tenants data found in YAML file")
            except Exception as e:
                print(f"ERROR loading tenants: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"WARNING: Tenants config not found at {tenants_path}")
        
        # Load pipelines
        pipelines_path = Path(self.config_dir) / "pipelines.yaml"
        print(f"DEBUG: Looking for pipelines at {pipelines_path}")
        print(f"DEBUG: Path exists: {pipelines_path.exists()}")
        
        if pipelines_path.exists():
            try:
                with open(pipelines_path, 'r', encoding='utf-8') as f:
                    pipelines_data = yaml.safe_load(f)
                    print(f"DEBUG: Pipelines data loaded: {pipelines_data is not None}")
                    
                    if pipelines_data and 'pipelines' in pipelines_data:
                        print(f"DEBUG: Found {len(pipelines_data['pipelines'])} pipelines in YAML")
                        for pipeline_data in pipelines_data.get("pipelines", []):
                            print(f"DEBUG: Loading pipeline: {pipeline_data.get('pipeline_id', 'Unknown')}")
                            pipeline = PipelineConfig(
                                pipeline_id=pipeline_data["pipeline_id"],
                                tenant_id=pipeline_data["tenant_id"],
                                name=pipeline_data["name"],
                                description=pipeline_data["description"],
                                model_type=pipeline_data["model_type"],
                                status=PipelineStatus(pipeline_data.get("status", "active")),
                                pipeline_cost_budget=pipeline_data.get("pipeline_cost_budget"),
                                pipeline_compute_budget=pipeline_data.get("pipeline_compute_budget"),
                                drift_threshold=pipeline_data.get("drift_threshold", 0.20),
                                accuracy_drop_threshold=pipeline_data.get("accuracy_drop_threshold", 0.15),
                                anomaly_rate_threshold=pipeline_data.get("anomaly_rate_threshold", 0.05),
                                latency_increase_threshold=pipeline_data.get("latency_increase_threshold", 0.20),
                                sla_weights=pipeline_data.get("sla_weights", {
                                    "availability": 0.4,
                                    "accuracy": 0.3,
                                    "latency": 0.2,
                                    "cost": 0.1
                                }),
                                approval_required=pipeline_data.get("approval_required", False),
                                allowed_actions=pipeline_data.get("allowed_actions", ["fallback", "rollback", "retrain", "no_op"]),
                                notification_channels=pipeline_data.get("notification_channels", ["email", "slack"]),
                                tags=pipeline_data.get("tags", [])
                            )
                            self.pipelines[pipeline.pipeline_id] = pipeline
                            self.pipeline_metrics[pipeline.pipeline_id] = UsageMetrics()
                            print(f"DEBUG: Pipeline added: {pipeline.pipeline_id}")
                    else:
                        print("DEBUG: No pipelines data found in YAML file")
            except Exception as e:
                print(f"ERROR loading pipelines: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"WARNING: Pipelines config not found at {pipelines_path}")
        
        print(f"DEBUG: Loaded {len(self.tenants)} tenants and {len(self.pipelines)} pipelines")
    
    def create_tenant(self, config: TenantConfig) -> bool:
        """Create a new tenant."""
        if config.tenant_id in self.tenants:
            return False
        
        self.tenants[config.tenant_id] = config
        self.usage_metrics[config.tenant_id] = UsageMetrics()
        
        # Save configuration
        self._save_tenants()
        return True
    
    def create_pipeline(self, config: PipelineConfig) -> bool:
        """Create a new pipeline for a tenant."""
        # Check if tenant exists
        if config.tenant_id not in self.tenants:
            return False
        
        tenant = self.tenants[config.tenant_id]
        
        # Check if tenant has capacity for more pipelines
        current_pipelines = sum(1 for p in self.pipelines.values() 
                              if p.tenant_id == config.tenant_id and p.status != PipelineStatus.ARCHIVED)
        if current_pipelines >= tenant.max_pipelines:
            return False
        
        # Set pipeline budgets if not specified
        if config.pipeline_cost_budget is None:
            config.pipeline_cost_budget = tenant.daily_cost_budget
        
        if config.pipeline_compute_budget is None:
            config.pipeline_compute_budget = tenant.daily_compute_hours
        
        # Store pipeline
        self.pipelines[config.pipeline_id] = config
        self.pipeline_metrics[config.pipeline_id] = UsageMetrics()
        
        # Save configuration
        self._save_pipelines()
        return True
    
    def check_action_allowed(self, pipeline_id: str, action: str, 
                           estimated_cost: float, estimated_compute_hours: float) -> Dict[str, Any]:
        """Check if an action is allowed for a pipeline."""
        if pipeline_id not in self.pipelines:
            return {"allowed": False, "reason": "Pipeline not found"}
        
        pipeline = self.pipelines[pipeline_id]
        tenant_id = pipeline.tenant_id
        
        if tenant_id not in self.tenants:
            return {"allowed": False, "reason": "Tenant not found"}
        
        tenant = self.tenants[tenant_id]
        pipeline_metrics = self.pipeline_metrics[pipeline_id]
        tenant_metrics = self.usage_metrics[tenant_id]
        
        # Check 1: Is action in allowed actions list?
        if action not in pipeline.allowed_actions:
            return {"allowed": False, "reason": f"Action '{action}' not in allowed actions list"}
        
        # Check 2: Pipeline budget check
        if pipeline.pipeline_cost_budget:
            if pipeline_metrics.current_cost + estimated_cost > pipeline.pipeline_cost_budget:
                return {"allowed": False, "reason": "Exceeds pipeline cost budget"}
        
        # Check 3: Pipeline compute budget check
        if pipeline.pipeline_compute_budget:
            if (pipeline_metrics.current_compute_hours + estimated_compute_hours > 
                pipeline.pipeline_compute_budget):
                return {"allowed": False, "reason": "Exceeds pipeline compute budget"}
        
        # Check 4: Tenant budget check
        if tenant_metrics.current_cost + estimated_cost > tenant.daily_cost_budget:
            return {"allowed": False, "reason": "Exceeds tenant daily cost budget"}
        
        # Check 5: Tenant compute budget check
        if (tenant_metrics.current_compute_hours + estimated_compute_hours > 
            tenant.daily_compute_hours):
            return {"allowed": False, "reason": "Exceeds tenant daily compute hours"}
        
        # Check 6: SLA impact analysis (simplified)
        # This would be more sophisticated in production
        
        return {
            "allowed": True,
            "pipeline_budget_remaining": pipeline.pipeline_cost_budget - pipeline_metrics.current_cost if pipeline.pipeline_cost_budget else None,
            "tenant_budget_remaining": tenant.daily_cost_budget - tenant_metrics.current_cost,
            "estimated_cost": estimated_cost,
            "estimated_compute": estimated_compute_hours
        }
    
    def record_action(self, pipeline_id: str, action: str, cost: float, 
                     compute_hours: float, mttr_minutes: Optional[float] = None):
        """Record an action taken by a pipeline."""
        if pipeline_id not in self.pipeline_metrics:
            return False
        
        pipeline = self.pipelines[pipeline_id]
        tenant_id = pipeline.tenant_id
        
        # Update pipeline metrics
        self.pipeline_metrics[pipeline_id].add_action(action, cost, compute_hours, mttr_minutes)
        
        # Update tenant metrics
        self.usage_metrics[tenant_id].add_action(action, cost, compute_hours, mttr_minutes)
        
        return True
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive status for a pipeline."""
        if pipeline_id not in self.pipelines:
            return None
        
        pipeline = self.pipelines[pipeline_id]
        metrics = self.pipeline_metrics.get(pipeline_id, UsageMetrics())
        tenant = self.tenants.get(pipeline.tenant_id)
        
        return {
            "pipeline": pipeline.to_dict(),
            "metrics": {
                "current_cost": metrics.current_cost,
                "current_compute_hours": metrics.current_compute_hours,
                "actions_taken": metrics.actions_taken,
                "sla_violations": metrics.sla_violations,
                "incidents_resolved": metrics.incidents_resolved,
                "avg_mttr_minutes": metrics.avg_mttr
            },
            "budgets": {
                "pipeline_cost_budget": pipeline.pipeline_cost_budget,
                "pipeline_cost_remaining": pipeline.pipeline_cost_budget - metrics.current_cost if pipeline.pipeline_cost_budget else None,
                "pipeline_compute_budget": pipeline.pipeline_compute_budget,
                "pipeline_compute_remaining": pipeline.pipeline_compute_budget - metrics.current_compute_hours if pipeline.pipeline_compute_budget else None,
                "tenant_daily_cost_budget": tenant.daily_cost_budget if tenant else None,
                "tenant_daily_cost_remaining": tenant.daily_cost_budget - self.usage_metrics[pipeline.tenant_id].current_cost if tenant else None
            },
            "thresholds": {
                "drift_threshold": pipeline.drift_threshold,
                "accuracy_drop_threshold": pipeline.accuracy_drop_threshold,
                "anomaly_rate_threshold": pipeline.anomaly_rate_threshold,
                "latency_increase_threshold": pipeline.latency_increase_threshold
            }
        }
    
    def get_tenant_status(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive status for a tenant."""
        if tenant_id not in self.tenants:
            return None
        
        tenant = self.tenants[tenant_id]
        metrics = self.usage_metrics[tenant_id]
        tenant_pipelines = [p for p in self.pipelines.values() if p.tenant_id == tenant_id]
        
        return {
            "tenant": {
                "tenant_id": tenant.tenant_id,
                "name": tenant.name,
                "tier": tenant.tier.value,
                "max_pipelines": tenant.max_pipelines,
                "current_pipelines": len(tenant_pipelines),
                "monthly_cost_budget": tenant.monthly_cost_budget,
                "max_monthly_compute_hours": tenant.max_monthly_compute_hours,
                "sla_targets": tenant.sla_targets,
                "compliance_requirements": tenant.compliance_requirements
            },
            "metrics": {
                "current_cost": metrics.current_cost,
                "current_compute_hours": metrics.current_compute_hours,
                "actions_taken": metrics.actions_taken,
                "sla_violations": metrics.sla_violations,
                "incidents_resolved": metrics.incidents_resolved,
                "avg_mttr_minutes": metrics.avg_mttr
            },
            "budgets": {
                "daily_cost_budget": tenant.daily_cost_budget,
                "daily_cost_remaining": tenant.daily_cost_budget - metrics.current_cost,
                "monthly_cost_budget": tenant.monthly_cost_budget,
                "daily_compute_hours": tenant.daily_compute_hours,
                "daily_compute_remaining": tenant.daily_compute_hours - metrics.current_compute_hours,
                "monthly_compute_hours": tenant.max_monthly_compute_hours
            },
            "pipelines": [
                {
                    "pipeline_id": p.pipeline_id,
                    "name": p.name,
                    "status": p.status.value,
                    "current_cost": self.pipeline_metrics.get(p.pipeline_id, UsageMetrics()).current_cost
                }
                for p in tenant_pipelines
            ]
        }
    
    def _save_tenants(self):
        """Save tenant configuration to file."""
        tenants_path = Path(self.config_dir) / "tenants.yaml"
        tenants_data = {
            "tenants": [
                {
                    "tenant_id": t.tenant_id,
                    "name": t.name,
                    "tier": t.tier.value,
                    "max_pipelines": t.max_pipelines,
                    "max_models_per_pipeline": t.max_models_per_pipeline,
                    "monthly_cost_budget": t.monthly_cost_budget,
                    "max_monthly_compute_hours": t.max_monthly_compute_hours,
                    "sla_targets": t.sla_targets,
                    "feature_flags": t.feature_flags,
                    "compliance_requirements": t.compliance_requirements,
                    "data_retention_days": t.data_retention_days
                }
                for t in self.tenants.values()
            ]
        }
        
        with open(tenants_path, 'w') as f:
            yaml.dump(tenants_data, f, default_flow_style=False)
    
    def _save_pipelines(self):
        """Save pipeline configuration to file."""
        pipelines_path = Path(self.config_dir) / "pipelines.yaml"
        pipelines_data = {
            "pipelines": [p.to_dict() for p in self.pipelines.values()]
        }
        
        with open(pipelines_path, 'w') as f:
            yaml.dump(pipelines_data, f, default_flow_style=False)


# Example usage
if __name__ == "__main__":
    # Initialize platform
    platform = MultiTenantPlatform("multi_tenant/configs")
    
    print(f"\nFinal count: {len(platform.tenants)} tenants, {len(platform.pipelines)} pipelines")
    
    # Create a tenant
    tenant_config = TenantConfig(
        tenant_id="acme-corp",
        name="ACME Corporation",
        tier=TenantTier.ENTERPRISE,
        max_pipelines=10,
        max_models_per_pipeline=5,
        monthly_cost_budget=10000.00,
        max_monthly_compute_hours=720,  # 24 hours/day * 30 days
        sla_targets={
            "availability": 0.999,
            "mttr": 2.0,
            "accuracy": 0.95
        },
        compliance_requirements=["soc2", "financial"],
        data_retention_days=365
    )
    
    platform.create_tenant(tenant_config)
    
    # Create a pipeline
    pipeline_config = PipelineConfig(
        pipeline_id="rec-pipeline-001",
        tenant_id="acme-corp",
        name="Recommendation Engine",
        description="Product recommendation model for e-commerce",
        model_type="classification",
        pipeline_cost_budget=1000.00,
        pipeline_compute_budget=24.0,
        drift_threshold=0.15,
        accuracy_drop_threshold=0.10,
        sla_weights={
            "availability": 0.5,
            "accuracy": 0.4,
            "latency": 0.1
        },
        approval_required=True,
        tags=["ecommerce", "recommendation", "production"]
    )
    
    platform.create_pipeline(pipeline_config)
    
    # Check if an action is allowed
    check_result = platform.check_action_allowed(
        pipeline_id="rec-pipeline-001",
        action="retrain",
        estimated_cost=200.00,
        estimated_compute_hours=2.0
    )
    
    print(f"Action allowed: {check_result['allowed']}")
    if check_result['allowed']:
        print(f"Remaining budget: ${check_result['pipeline_budget_remaining']:.2f}")
    
    # Record the action
    platform.record_action(
        pipeline_id="rec-pipeline-001",
        action="retrain",
        cost=180.50,
        compute_hours=1.8,
        mttr_minutes=2.5
    )
    
    # Get pipeline status
    status = platform.get_pipeline_status("rec-pipeline-001")
    print(f"\nPipeline cost: ${status['metrics']['current_cost']:.2f}")
    print(f"Average MTTR: {status['metrics']['avg_mttr_minutes']:.1f} minutes")
