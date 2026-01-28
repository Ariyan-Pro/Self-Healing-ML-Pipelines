from setuptools import setup, find_packages

setup(
    name="self-healing-ml",
    version="1.0.0",
    description="Self-Healing ML Pipelines",
    packages=find_packages(),
    install_requires=[
        "numpy", "pandas", "scikit-learn", "scipy",
        "pyyaml", "pydantic", "loguru", "rich",
        "prometheus-client", "joblib"
    ],
)
