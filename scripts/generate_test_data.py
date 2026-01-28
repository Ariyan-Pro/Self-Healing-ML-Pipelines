#!/usr/bin/env python3
"""Generate test data for self-healing ML system."""
import numpy as np
import pandas as pd
from pathlib import Path

def generate_test_data(n_samples: int = 1000, n_features: int = 10):
    """Generate synthetic test data."""
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels (binary classification)
    coef = np.random.randn(n_features)
    logits = X @ coef + np.random.randn(n_samples) * 0.5
    y = (logits > 0).astype(int)
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

if __name__ == "__main__":
    # Generate and save test data
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    train_data = generate_test_data(1000, 10)
    test_data = generate_test_data(200, 10)
    
    train_data.to_csv(data_dir / "train_data.csv", index=False)
    test_data.to_csv(data_dir / "test_data.csv", index=False)
    
    print(f"Generated test data: {len(train_data)} training, {len(test_data)} test samples")
