"""
DLC2Action Toolbox
© A. Mathis Lab
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dlc2action",
    version="1.0",
    author="A. Mathis Lab",
    author_email="alexander@deeplabcut.org",
    description="DLC2Action is a deep learning based action segmentation toolbox. It is designed to work directly on top of markerless pose estimation softwares such as DeepLabCut. It allows to train and evaluate various state-of-the-art action segmentation models on your own data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amathislab/DLC2Action",
    install_requires=[
        "tqdm",
        "torch",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "editdistance",
        "optuna",
        "openpyxl",
        "plotly",
        "ruamel.yaml",
        "p_tqdm",
        "click",
        "pyinquirer",
        "pytest",
        "tables",
        "torchvision",
        "tika",
        "pdfplumber",
        "ftfy",
        "regex",
        "scikit-learn",
        "einops",
        "ipykernel",
        "opencv-python",
        "seaborn",
        "pillow",
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": [
            "dlc2action_init = dlc2action.scripts.init:main",
        ]
    },
    python_requires=">=3.10",
)
