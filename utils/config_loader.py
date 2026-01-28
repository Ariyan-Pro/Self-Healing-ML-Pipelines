"""
Configuration loader utility
"""
import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    """Loads and validates configuration files"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        self.config_cache = {}
    
    def load_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if filename in self.config_cache:
            return self.config_cache[filename]
        
        filepath = os.path.join(self.config_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.config_cache[filename] = config
        return config
    
    def load_pipeline_config(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        return self.load_config("pipeline.yaml")
    
    def load_healing_policies(self) -> Dict[str, Any]:
        """Load healing policies"""
        return self.load_config("healing_policies.yaml")
    
    def load_sla_config(self) -> Dict[str, Any]:
        """Load SLA configuration"""
        return self.load_config("sla_config.yaml")
    
    def load_cost_model(self) -> Dict[str, Any]:
        """Load cost model configuration"""
        return self.load_config("cost_model.yaml")
    
    def load_canary_config(self) -> Dict[str, Any]:
        """Load canary rollout configuration"""
        return self.load_config("canary_config.yaml")
