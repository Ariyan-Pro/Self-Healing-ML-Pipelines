import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

class TestExperiments(unittest.TestCase):
    def test_experiment_files_exist(self):
        files = [
            "synthetic_drift.py",
            "concept_shift_simulator.py", 
            "noise_injection.py",
            "run_all_experiments.py",
            "ablation_study.py"
        ]
        for file in files:
            path = os.path.join("experiments", file)
            self.assertTrue(os.path.exists(path), f"Missing: {path}")

if __name__ == "__main__":
    unittest.main()
