#!/bin/bash
# experiments/run.sh

echo "🧪 Self-Healing ML Pipeline - Experiment Runner"
echo "================================================"

# Create virtual environment if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r ../requirements.txt > /dev/null 2>&1

# Run experiments
echo ""
echo "Running comprehensive experiments..."
python run_all_experiments.py

echo ""
echo "Experiments completed! Check the experiment_results_* directory."