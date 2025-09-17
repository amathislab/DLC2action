<div align="center">

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
![Version](https://img.shields.io/badge/python_version-3.13-purple)
[![PyPI version](https://badge.fury.io/py/dlc2action.svg)](https://badge.fury.io/py/dlc2action)
[![Downloads](https://pepy.tech/badge/dlc2action)](https://pepy.tech/project/dlc2action)
[![Downloads](https://pepy.tech/badge/dlc2action/month)](https://pepy.tech/project/dlc2action/month)
[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

![logo](logos/title.png)

</div>

# 🌈 DLC2Action
DLC2Action is an action segmentation package that makes running and tracking experiments easy.

## 🛠️ Installation
You can simply install DLC2Action by typing:
```
pip install dlc2action
```

Or you can install DLC2Action for development by running this in your terminal.
```
git clone https://github.com/amathislab/DLC2Action
cd DLC2Action
conda create --name DLC2Action python=3.13
conda activate DLC2Action
python -m pip install .
```

## 📖 Features
The functionality of DLC2Action includes:
 - compiling and updating project-specific configuration files,
 - filling in the configuration dictionaries automatically whenever possible,
 - saving training parameters and results,
 - running predictions and hyperparameter searches,
 - creating active learning files,
 - loading hyperparameter search results in experiments and dumping them into configuration files,
 - comparing new experiment parameters with the project history and loading pre-computed features (to save time) and previously
   created splits (to enforce consistency) when there is a match,
 - filtering and displaying training, prediction and hyperparameter search history,
 - plotting training curve comparisons

and more.

## ⚡ A quick example
You can start a new project, run an experiment, visualize it and use the trained model to make a prediction
in a few lines of code.
```python
project = Project("project", data_type="dlc_track", annotation_type="csv")
project.update_parameters(...)
project.run_default_hyperparameter_search("search")
project.run_episode("episode", load_search="search")
project.evaluate(["episode"])
project.run_prediction("prediction", episode_names=["episode"], data_path="/path/to/new/data")
```

## 📊 Benchmarks

We provide standardized benchmarks on action segmentation to help you evaluate DLC2Action's performance. Check out the [benchmarks section](examples/benchmarks/README.md) for detailed results and comparisons.

## 📚 How to get more information ?

Check out the [examples](/examples) or [read the documentation](html_docs/dlc2action/index.html) for a taste of what else you can do.

## 🙏 Acknowledgments

DLC2Action is developed by members of the [A. Mathis Group](https://mathislab.org/) at EPFL. We are grateful to many people for feedback, alpha-testing, suggestions and contributions, in particular to Lucas Stoffl, Margaret Lane, Marouane Jaakik, Steffen Schneider and Mackenzie Mathis.

We are also grateful to the creators of the different benchmarks, as well as models that were adapted in DLC2action. In particular, the MS-TCN, the C2F-TCN, the ASFormer, the EDTCN and the MotionBERT models, and the CalMS21, the SIMBA CRIM13 and SIMBA-RAT, the OFT and EPM, the SHOT7m2 and hBABEL, and the Atari-HEAD datasets. Please refer to the [benchmarks section](examples/benchmarks/README.md) for detailed references and consider citing these works when using them.

## 📝 License

Note that the software is provided "as is", without warranty of any kind, express or implied. If you use the code or data, please cite us!

## 📑 Reference

Stay tuned for our first publication -- any feedback on this release candidate for version 1 is welcome. Thanks for using DLC2Action. Please reach out if you want to collaborate!
