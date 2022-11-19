#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DLC2Action Toolbox
© A. Mathis Lab
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dlc2action",
    version="0.2b1",
    author="A. Mathis Lab",
    author_email="alexander@deeplabcut.org",
    description="tba",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amathislab/DLC2Action",
    install_requires=[
        "tqdm>=4.62.3",
        "torch>=1.9",
        "numpy>=1.21.2",
        "scipy>=1.7.1",
        "pandas==1.4.3",
        "matplotlib>=3.4.3",
        "editdistance>=0.5.3",
        "optuna>=2.10.0",
        "openpyxl>=3.0.9",
        "plotly>=5.1.0",
        "ruamel.yaml==0.16.12",
        "p_tqdm<=1.2",
        "click>=8.0.3",
        "pytest>=7.1.2",
        "tables>=3.7.0",
        "torchvision>=0.13.1",
        "ftfy>=6.1.1",
        "regex>=2022.8.17",
        "scikit-learn>=1.1.2",
        "jupyter",  # added for demo
        "pims",
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)
