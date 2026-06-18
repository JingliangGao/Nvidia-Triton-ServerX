#!/usr/bin/env python3

import os
from setuptools import setup, find_packages

readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = ""

setup(
    name="model_analyzer",
    version="1.0.0",
    author="zhongpei",
    author_email="zhongpei@kylinos.cn",
    description="Model Analyzer - A tool for analyzing Kytensor model performance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["venv", "venv.*", "tests", "tests.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "psutil>=5.8.0",
        "tritonclient[all]>=2.20.0",
        "Pillow>=8.0.0",
        "transformers>=4.0.0",
    ],
    entry_points={
        "console_scripts": [
            "model-analyzer=model_analyzer.entrypoint:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)