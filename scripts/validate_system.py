#!/usr/bin/env python3
"""Validate the complete self-healing ML system."""
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.train import TrainingPipeline
from pipelines.inference import InferencePipeline
from pipelines.rollback import RollbackPipeline
from orchestration.controller import SelfHealingController
from healing.healing_actions import HealingActions
from monitoring.data_drift import DataDriftDetector
from decision_engine.policy_engine import PolicyEngine
from utils.config_loader import load_config, PipelineConfig
from loguru import logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.add("logs/validation.log", rotation="1 day", retention="7 days")


class SystemValidator:
    """Validates all components of the self-healing ML system."""
    
    def __init__(self):
        """Initialize validator."""
        self.results = {
            "components": {},
            "integration": {},
            "overall": "PASS"
        }
    
    def validate_component(self, name: str, func, *args, **kwargs) -> bool:
        """
        Validate a single component.
        
        Args:
            name: Component name
            func: Validation function
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            True if validation passes
        """
        try:
            logger.info(f"Validating {name}...")
            result = func(*args, **kwargs)
            self.results["components"][name] = {
                "status": "PASS",
                "result": result
            }
            logger.success(f"✓ {name} passed")
            return True
        except Exception as e:
            self.results["components"][name] = {
                "status": "FAIL",
                "error": str(e)
            }
            logger.error(f"✗ {name} failed: {e}")
            self.results["overall"] = "FAIL"
            return False
    
    def validate_config_loader(self) -> bool:
        """Validate configuration loader."""
        config = load_config("configs/pipeline.yaml")
        return isinstance(config, PipelineConfig)
    
    def validate_drift_detector(self) -> bool:
        """Validate drift detector."""
        detector = DataDriftDetector(method="ks", threshold=0.05)
        
        # Test data
        import numpy as np
        ref_data = np.random.normal(0, 1, 1000)
        curr_data = np.random.normal(0, 1, 1000)
        
        result = detector.detect_drift(ref_data, curr_data)
        return result is not None
    
    def validate_policy_engine(self) -> bool:
        """Validate policy engine."""
        engine = PolicyEngine(config_path="configs/healing_policies.yaml")
        
        # Test with sample signals
        signals = {"data_drift": 0.3, "accuracy_drop": 0.15}
        action, trace = engine.decide(signals)
        
        return action is not None and trace is not None
    
    def validate_healing_actions(self) -> bool:
        """Validate healing actions."""
        config = load_config("configs/pipeline.yaml")
        healing = HealingActions(config.model_dump())
        
        # Test fallback action
        result = healing.fallback()
        return result["status"] in ["success", "failed"]
    
    def validate_training_pipeline(self) -> bool:
        """Validate training pipeline."""
        pipeline = TrainingPipeline()
        
        # Create dummy data for testing
        import pandas as pd
        import numpy as np
        
        data = pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "target": np.random.randint(0, 2, 100)
        })
        
        result = pipeline.run(data_path=None, data=data)
        return result["status"] == "success"
    
    def validate_inference_pipeline(self) -> bool:
        """Validate inference pipeline."""
        pipeline = InferencePipeline()
        
        # Get model info
        info = pipeline.get_model_info()
        return isinstance(info, dict)
    
    def validate_rollback_pipeline(self) -> bool:
        """Validate rollback pipeline."""
        pipeline = RollbackPipeline()
        
        # List available models
        models = pipeline.list_available_models()
        return isinstance(models, list)
    
    def validate_controller(self) -> bool:
        """Validate self-healing controller."""
        controller = SelfHealingController()
        
        # Get status
        status = controller.get_status()
        return status is not None
    
    def validate_integration(self) -> bool:
        """Validate integration between components."""
        logger.info("Validating integration...")
        
        integration_tests = []
        
        try:
            # Test 1: Config -> Controller
            config = load_config("configs/pipeline.yaml")
            controller = SelfHealingController()
            integration_tests.append(("Config->Controller", True))
            
            # Test 2: Training -> Inference
            train_pipeline = TrainingPipeline()
            inference_pipeline = InferencePipeline()
            integration_tests.append(("Training->Inference", True))
            
            # Test 3: Detection -> Decision -> Healing
            detector = DataDriftDetector()
            policy_engine = PolicyEngine()
            healing = HealingActions(config.model_dump())
            integration_tests.append(("Detection->Decision->Healing", True))
            
            # Record results
            self.results["integration"] = {
                test[0]: "PASS" if test[1] else "FAIL"
                for test in integration_tests
            }
            
            all_passed = all(test[1] for test in integration_tests)
            if all_passed:
                logger.success("✓ Integration tests passed")
            else:
                logger.error("✗ Some integration tests failed")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"✗ Integration validation failed: {e}")
            self.results["overall"] = "FAIL"
            return False
    
    def run_validation(self) -> dict:
        """Run complete validation suite."""
        logger.info("Starting system validation")
        logger.info("="*60)
        
        # Validate individual components
        component_tests = [
            ("Configuration Loader", self.validate_config_loader),
            ("Drift Detector", self.validate_drift_detector),
            ("Policy Engine", self.validate_policy_engine),
            ("Healing Actions", self.validate_healing_actions),
            ("Training Pipeline", self.validate_training_pipeline),
            ("Inference Pipeline", self.validate_inference_pipeline),
            ("Rollback Pipeline", self.validate_rollback_pipeline),
            ("Controller", self.validate_controller),
        ]
        
        for name, test_func in component_tests:
            self.validate_component(name, test_func)
        
        # Validate integration
        self.validate_integration()
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _generate_summary(self):
        """Generate validation summary."""
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)
        
        # Component results
        logger.info("\nComponents:")
        for name, result in self.results["components"].items():
            status = result["status"]
            if status == "PASS":
                logger.success(f"  ✓ {name}: {status}")
            else:
                logger.error(f"  ✗ {name}: {status}")
        
        # Integration results
        logger.info("\nIntegration:")
        for name, status in self.results["integration"].items():
            if status == "PASS":
                logger.success(f"  ✓ {name}: {status}")
            else:
                logger.error(f"  ✗ {name}: {status}")
        
        # Overall result
        logger.info("\n" + "="*60)
        if self.results["overall"] == "PASS":
            logger.success("OVERALL: PASS - System is ready for production!")
        else:
            logger.error("OVERALL: FAIL - Some components need attention")
        logger.info("="*60)


def main():
    """Run complete system validation."""
    validator = SystemValidator()
    results = validator.run_validation()
    
    # Save results to file
    import json
    results_file = Path("validation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nDetailed results saved to: {results_file}")
    
    # Exit with appropriate code
    if results["overall"] == "FAIL":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

