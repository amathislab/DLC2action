[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

![](logos/title.png)

DLC2Action is an action segmentation package that makes running and tracking of machine learning experiments easy.

## Installation

### Via the Python Package Index

You can simply install DLC2Action by typing:

```
pip install "dlc2action==0.2b2"
```

### From Github

You can install DLC2Action for development by running this in your terminal.
```
git clone https://github.com/AlexEMG/DLC2Action
cd DLC2Action
conda create --name DLC2Action python=3.9
conda activate DLC2Action
python -m pip install .
```

## Features

The functionality of DLC2Action includes:
 - compiling and updating project-specific configuration files,
 - filling in configuration dictionaries automatically whenever possible,
 - saving training parameters and results,
 - running predictions and hyperparameter searches,
 - creating active learning files,
 - loading hyperparameter search results in experiments and dumping them into configuration files,
 - comparing new experiment parameters with the project history and loading pre-computed features (to save time) and previously
   created splits (to enforce consistency) when there is a match,
 - filtering and displaying training, prediction and hyperparameter search history,
 - plotting training curve comparisons

and more.

## A quick example

You can start a new project, run an experiment, visualize it and use the trained model to make a prediction
in a few lines of code.
```python
from dlc2action.project import Project

# create a new project
project = Project('project_name', data_type='data_type', annotation_type='annotation_type',
                  data_path='path/to/data/folder', annotation_path='path/to/annotation/folder')
# set important parameters, like the set labels you want to predict
project.update_parameters(...)
# run a training episode
project.run_episode('episode_1')
# plot the results
project.plot_episodes(['episode_1'], metrics=['recall'])
# use the model trained in episode_1 to make a prediction for new data
project.run_prediction('prediction_1', episode_names=['episode_1'], data_path='path/to/new_data/folder')
```

## How to get more information?

Check out the [examples](/examples) or [read the documentation](https://alexemg.github.io/DLC2action/html_docs/dlc2action.html) for a taste of what else you can do.


## Acknowledgments

[Liza Kozlova](https://github.com/elkoz) from the [A. Mathis Group at EPFL](https://www.mathislab.org/) is the main developer of DLC2Action.

We are grateful to many people for feedback, alpha-testing and suggestions, in particular to Andy Bonnetto, Lucas Stoffl, Margaret Lane, Marouane Jaakik, Steffen Schneider and Mackenzie Mathis.

## License:

Note that the software is provided "as is", without warranty of any kind, express or implied. If you use the code or data, please cite us!

## Reference:

Stay tuned for our first publication -- Any feedback on this beta release is welcome at this time. Thanks for using DLC2Action.
