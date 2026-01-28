#!/usr/bin/env python3
"""Complete example of self-healing ML pipeline."""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.controller import SelfHealingController
from pipelines.train import TrainingPipeline
from pipelines.inference import InferencePipeline
from utils.config_loader import load_config
from loguru import logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.add("logs/run_pipeline.log", rotation="1 day", retention="7 days")


def generate_sample_data(n_samples: int = 100) -> pd.DataFrame:
    """Generate sample data for demonstration."""
    np.random.seed(42)
    
    # Generate features
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels with some noise
    coef = np.random.randn(n_features)
    y = (X @ coef + np.random.randn(n_samples) * 0.3) > 0
    y = y.astype(int)
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df


def main():
    """Run complete self-healing pipeline example."""
    logger.info("Starting Self-Healing ML Pipeline Demo")
    
    # 1. Load configuration
    logger.info("Step 1: Loading configuration")
    config = load_config("configs/pipeline.yaml")
    logger.info(f"Configuration loaded: model={config.model.name}")
    
    # 2. Generate sample data
    logger.info("Step 2: Generating sample data")
    data = generate_sample_data(100)
    
    # 3. Train initial model (we need to modify TrainingPipeline to accept DataFrame)
    logger.info("Step 3: Training initial model")
    
    # Save data temporarily to file for training
    temp_data_path = "data/raw/temp_training_data.csv"
    data.to_csv(temp_data_path, index=False)
    
    train_pipeline = TrainingPipeline()
    train_result = train_pipeline.run(
        data_path=temp_data_path,
        test_size=0.2
    )
    
    if train_result["status"] != "success":
        logger.error(f"Training failed: {train_result.get('error')}")
        return
    
    logger.info(f"Model trained: {train_result['model_info']['version']}")
    
    # 4. Initialize self-healing controller
    logger.info("Step 4: Initializing self-healing controller")
    controller = SelfHealingController()
    
    # 5. Run inference and monitoring cycles
    logger.info("Step 5: Running monitoring cycles")
    
    # Create some inference data with drift
    inference_data = {
        "features": {
            f"feature_{i}": np.random.randn(100) + (i * 0.1)  # Add some drift
            for i in range(5)
        },
        "reference_features": {
            f"feature_{i}": np.random.randn(100)
            for i in range(5)
        },
        "predictions": np.random.randint(0, 2, 100),
        "performance": {
            "accuracy": 0.85,
            "f1_score": 0.84
        }
    }
    
    # Run a few cycles
    for cycle in range(5):
        logger.info(f"Running cycle {cycle + 1}/5")
        
        # Simulate different conditions
        if cycle == 2:
            # Introduce significant drift
            inference_data["features"]["feature_0"] = np.random.randn(100) + 2.0
        
        if cycle == 3:
            # Introduce anomaly
            inference_data["predictions"] = np.concatenate([
                np.random.randint(0, 2, 90),
                np.full(10, 5)  # Anomalous values
            ])
        
        # Run healing cycle
        cycle_result = controller.run_healing_cycle(inference_data)
        
        logger.info(f"Cycle {cycle + 1} result: {cycle_result['action']}")
        
        if cycle_result["action"] != "no_action":
            logger.warning(f"Healing action triggered: {cycle_result['action']}")
            logger.info(f"Healing result: {cycle_result.get('healing_result', {})}")
    
    # 6. Get final status
    logger.info("Step 6: Getting system status")
    status = controller.get_status()
    
    logger.info("\n" + "="*50)
    logger.info("FINAL SYSTEM STATUS")
    logger.info("="*50)
    logger.info(f"Status: {status['state']}")
    logger.info(f"Cycles completed: {status['metrics']['cycles_completed']}")
    logger.info(f"Healing actions: {status['metrics']['healing_actions']}")
    logger.info(f"Drift detections: {status['metrics']['drift_detections']}")
    
    if status.get('healing_counts'):
        logger.info("Healing action counts:")
        for action, count in status['healing_counts'].items():
            logger.info(f"  {action}: {count}")
    
    # Clean up temp file
    import os
    if os.path.exists(temp_data_path):
        os.remove(temp_data_path)
    
    logger.info("Pipeline demo completed successfully!")


if __name__ == "__main__":
    main()
