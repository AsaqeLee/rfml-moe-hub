from setuptools import setup, find_packages

setup(
    name="rfml-moe",
    version="0.1.0",
    description="Multi-modal MoE Drone RF Signal Detection Pipeline",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "click>=8.1.0",
        "rich>=13.7.0",
        "webdataset>=0.2.48",
        "huggingface-hub>=0.20.0",
    ],
    entry_points={
        "console_scripts": [
            "rfml=main:cli",
        ],
    },
)
