#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
`dlc2action` is an action segmentation package that makes running and tracking experiments easy.

# Usage
`dlc2action` is designed to be modular.
You can use the high-level project interface for convenient experiment management or just import a metric
or an SSL module if you want more freedom. Here are some tutorials for using the package.

## Project
Project is the class that can create and maintain configuration files and keep track of your experiments.

### Creating
[creating]: #creating
To start a new project, you can create a new `dlc2action.project.project.Project` instance in python. 
Check `dlc2action.project.project.Project.print_data_types` and `dlc2action.project.project.Project.print_annotation_types` to see the implemented data 
and annotation types.
```python
from dlc2action.project import Project

project = Project(
    'project_name',
    data_type='data_type',
    annotation_type='annotation_type',
    data_path='path/to/data/folder',
    annotation_path='path/to/annotation/folder',
)
```

A new folder will be created at `projects_path/project_name` with all the necessary files. The default projects path is
a `DLC2Action` folder that will be created in your home directory.
The project structure looks like this.
```
.
project_name
├── config                                             # Settings files
├── meta                                               # Project meta files (experiment records)
├── saved_datasets                                     # Pre-computed dataset files
└── results
    ├── logs                                           # Training logs (human readable)
    │   └── episode.txt
    ├── models                                         # Model checkpoints
    │   └── episode
    │       ├── epoch25.pt
    │       └── epoch50.pt
    ├── searches                                       # Hyperparameter search results (graphs)
    │   └── search
    │       ├── search_param_importances.html_docs
    │       └── search_contour.html_docs
    ├── splits                                         # Split files
    │       ├── time_25.0%validation_10.0%test.txt
    │       └── random_20.0%validation_10.0%test.txt
    ├── suggestions                                    # Suggestion and active learning files
    │       └── active_learning
    │           ├── video1_suggestion.pickle
    │           ├── video2_suggestion.pickle
    │           └── al_points.pickle
    └── predictions                                    # Prediction files (pickled dictionaries)
            ├── episode_epoch25.pickle
            └── episode_epoch50_newdata.pickle
```

You can find a more detailed explanation of the structure at `dlc2action.project`.

After the project is created you can modify the parameters with the `dlc2action.project.project.Project.update_parameters` function.
Make sure to start by filling out the required parameters. You can get a list of those and a code hint with 
`dlc2action.project.project.Project.list_blanks`.

Run `project.help()` to find out more about other parameters you might want to modify.

### Training
[training]: #training
When you want to start your experiments, just create a `dlc2action.project.project.Project` instance again
(or use the one you created
to initialize the project). This time you don't have to set any parameters except the project name (and, if
you used it when creating the project, `projects_path`).

```python
from dlc2action.project import Project
project = Project('project_name')
```

The first thing you will want to do is train some models. There are three ways to run a *training episode*
in `dlc2action`.

1. **Run a single episode**

    ```python
    project.run_episode('episode_1')
    ```

    We have now run a training episode with the default project parameters (read from the configuration files)
    and saved it in the meta files under the name `episode_1`.

2. **Run multiple episodes in a row**

    ```python
    project.run_episodes(['episode_2', 'episode_3', 'episode_4'])
    ```

    That way the `dlc2action.task.universal_task.Task` instance will not be
    re-created every time, which might save you some time.

3. **Continue a previously run episode**

    ```python
    project.continue_episode('episode_2', num_epochs=500)
    ```

    In case you decide you want to continue an older episode, you can load all parameters and state dictionaries
    and set a new number of epochs. That way training will go on from where it has stopped and in the end the
    episode will be re-saved under the same name. Note that `num_epochs` denotes the **new total number of epochs**,
    so if `episode_2` has already been trained for 300 epochs, for example, it will now run for 200 epochs more,
    not 500.

Of course, changing the default parameters every time you want to run a new configuration is not very convenient.
And, luckily, you don't have to do that. Instead you can add a `parameters_update` parameter to
`dlc2action.project.project.Project.run_episode` (or `parameters_updates` to
`dlc2action.project.project.Project.run_episodes`; all other parameters generalize to multiple episodes
in a similar way). The third
function does not take many additional parameters since it aims to continue an experiment from exactly where
it ended.

```python
project.run_episode(
    'episode_5',
    parameters_update={
        'general': {'ssl': ['contrastive']},
        'ssl': {'contrastive': {'ssl_weight': 0.01}},
    },
)
```

In order to find the parameters you can modify, just open the *config* folder of your project and browse through
the files or call `dlc2action_fill` in your terminal (see [creating] section). The first-level keys
are the filenames without the extension (`'augmentations'`, `'data'`, `'general'`, `'losses'`,
`'metrics'`, `'ssl'`, `'training'`). Note that there are no
*model.yaml* or *features.yaml* files there for the `'model'` and `'features'` keys in the parameter
dictionaries. Those parameters are **read from the files in the *model* and *features* folders** that correspond
to the options you set in the `'general'` dictionary. For example, if at *general.yaml* `model_name` is set to
`'ms_tcn3'`, the `'model'` dictionary will be read from *model/ms_tcn3.yaml*.

If you want to create a new episode with modified parameters that loads a previously trained model, you can do that
by adding a `load_episode` parameter. By default we will load the last saved epoch, but you can also specify
which epoch you want with `load_epoch`. In that case the closest checkpoint will be chosen.

```python
project.run_episode(
    'episode_6',
    parameters_update={'training': {'batch_size': 64}},
    load_episode='episode_2',
    load_epoch=100,
)
```

### Optimizing
`dlc2action` also has tools for easy hyperparameter optimization. We use the `optuna` auto-ML package to perform
the searches and then save the best parameters and the search graphs (contour and parameter importance).
The best parameters
can then be loaded when you are run an episode or saved as the defaults in the configuration files.

To start a search, you need to run either the `project.project.Project.run_hyperparameter_search` or the 
`project.project.Project.run_default_hyperparameter_search` command. The first one allows you to optimize any hyperparameter you want
while the second provides you with a good default search space for a given model name. 

To run a default search, you just need to know the name of the model you want to experiment with.
```python
project.run_default_hyperparameter_search(
    "c2f_tcn_search",
    model_name="c2f_tcn",
    metric="f1",
    prune=True,
)
```

Defining your own search space is a bit more complicated. Let's
say we want to optimize for four parameters: overlap length, SSL task type, learning rate and
number of feature maps in the model. Here is the process
to figure out what we want to run.

1. **Find the parameter names**

    We check our parameter dictionary with `project.list_basic_parameters()` and find them in `"data"`, `"general"`, 
    `"training"` and `"model"` categories, respectively.
    That means that our parameter names are `'data/overlap'`, `'general/ssl'`, `'training/lr'` and
    `'model/num_f_maps'`.

2. **Define the search space**

    There are five types of search spaces in `dlc2action`:

    - `int`: integer values (uniform sampling),
    - `int_log`: integer values (logarithmic scale sampling),
    - `float`: float values (uniform sampling),
    - `float_log`: float values (logarithmic sampling),
    - `categorical`: choice between several values.

    The first four are defined with their minimum and maximum value while `categorical` requires a list of
    possible values. So the search spaces are described with tuples that look either like
    `(search_space_type, min, max)` or like `("categorical", list_of_values)`.

    We suspect that the optimal overlap is somewhere between 10 and 80 frames, SSL tasks should be either
    only contrastive or contrastive and masked_features together, sensible learning rate is between
    10<sup>-2</sup> and 10<sup>-4</sup> and the number of feature maps should be between 8 and 64.
    That makes our search spaces `("int", 10, 80)` for the overlap,
    `("categorical", [["contrastive"], ["contrastive", "masked_features"]])` for the SSL tasks,
    `("float_log", 1e-4, 1e-2)` for the learning rate and ("int_log", 8, 64) for the feature maps.

3. **Choose the search parameters**

    You need to decide which metric you are optimizing for and for how many trials. Note that it has to be one
    of the metrics you are computing: check `metric_functions` at *general.yaml* and add your
    metric if it's not there. The `direction` parameter determines whether this metric is minimized or maximized.
    The metric can also be averaged over a few of the most successful epochs (`average` parameter). If you want to
    use `optuna`'s pruning feature, set `prune` to `True`.

    You can also use parameter updates and load older experiments, as in the [training] section of this tutorial.

    Here we will maximize the recall averaged over 5 top epochs and run 50 trials with pruning.

Now we are ready to run!

```python
project.run_hyperparameter_search(
    search_space={
        "data/overlap": ("int", 10, 80),
        "general/ssl": (
            "categorical",
            [["contrastive"], ["contrastive", "masked_features"]]
        ),
        "training/lr": ("float_log": 1e-4, 1e-2),
        "model/num_f_maps": ("int_log", 8, 64),
    },
    metric="recall",
    n_trials=50,
    average=5,
    search_name="search_1",
    direction="maximize",
    prune=True,
)
```

After a search is finished, the best parameters are saved in the meta files and search graphs are saved at
*project_name/results/searches/search_1*. You can see the best parameters by running
`project.project.Project.list_best_parameters`:

```python
project.list_best_parameters('search_1')
```
Those results can also be loaded in a training episode or saved in the configuration files directly.

```python
project.update_parameters(
    load_search='search_1',
    load_parameters=['training/lr', 'model/num_f_maps'],
    round_to_binary=['model/num_f_maps'],
)
project.run_episode(
    'episode_best_params',
    load_search='search_1',
    load_parameters=['data/overlap', 'general/ssl'],
)
```
In this example we saved the learning rate and the number of feature maps in the configuration files and
loaded the other parameters to run
the `episode_best_params` training episode. Note how we used the `round_to_binary` parameter.
It will round the number of feature maps to the closest power of two (7 to 8, 35 to 32 and so on). This is useful
for parameters like the number of features or the batch size.

### Exploring results
[exploring]: #exploring-results
After you run a bunch of experiments you will likely want to get an overview.

#### Visualization
You can get a feeling for the predictions made by a model by running `project.project.Project.plot_predictions`.

```python
project.plot_predictions('episode_1', load_epoch=50)
```
This command will generate a prediction for a random sample and visualize it compared to the ground truth.
There is a lot of parameters you can customize, check them out in the documentation).

You can also analyze the results with `project.project.Project.plot_confusion_matrix`.

Another available visualization type is training curve comparison with `project.project.Project.plot_episodes`.
You can compare different metrics and modes across several episode or whithin one. For example, this command
will plot the validation accuracy curves for the two episodes.

```python
project.plot_episodes(['episode_1', 'episode_2'], metrics=['accuracy'])
```
And this will plot training and validation recall curves for `episode_3`.

```python
project.plot_episodes(['episode_3'], metrics=['recall'], modes=['train', 'val'])
```
You can also plot several episodes as one curve. That can be useful, for example, with episodes 2 and 6
in our tutorial, since `episode_6` loaded the model from `episode_2`.

```python
project.plot_episodes([['episode_2', 'episode_6'], 'episode_4'], metrics=['precision'])
```

#### Tables
Alternatively, you can start analyzing your experiments by putting them in a table. You can get a summary of
your training episodes with `project.project.Project.list_episodes`. It provides you with three ways to filter
the data.

1. **Episode names**

    You can directly say which episodes you want to look at and disregard all others. Let's say we want to see
    episodes 1, 2, 3 and 4.

2. **Parameters**

    In a similar fashion, you can choose to only display certain parameters. See all available parameters by
    running `project.list_episodes().columns`. Here we are only interested in the time of adding the record,
    the recall results and the learning rate.

3. **Value filter**

    The last option is to filter by parameter values. Filters are defined by strings that look like this:
    `'{parameter_name}::{sign}{value}'`. You can use as many filters as ypu want and separate them with a comma.
    The parameter names are the same as in point 2, the sign can be either
    `>`, `>=`, `<`, `<=` or `=` and you choose the value. Let's say we want to only get the episodes that took
    at least 30 minutes to train and got to a recall that is higher than 0.4. That translates to
    `'results/recall::>0.4,meta/training_time::>=00:30:00'`.

Putting it all together, we get this command.

```python
project.list_episodes(
    episodes=['episode_1', 'episode_2', 'episode_3', 'episode_4'],
    display_parameters=['meta/time', 'results/recall', 'training/lr'],
    episode_filter='results/recall::>0.4,meta/training_time::>=00:30:00',
)
```

There are similar functions for summarizing other history files: `project.project.Project.list_searches`,
`project.project.Project.list_predictions`, `project.project.Project.list_suggestions` and they all follow the
same pattern.

You can also get a summary of your experiment results with `project.project.Project.get_results_table` and
`project.project.Project.get_summary`. Those methods will aggregate the metrics over all runs of you experiment and allow 
for easy comparison between models or other parameters.

### Making predictions
When you feel good about one of the models, you can move on to making predictions. There are two ways to do that:
generate pickled prediction dictionaries with `project.project.Project.run_prediction` or active learning
and suggestion files for the [GUI] with `project.project.Project.run_suggestion`.

By default the predictions will be made for the entire dataset at the data path of the project. However, you
can also choose to only make them for the training, validation or test subset (set with `mode` parameter) or
even for entirely new data (set with `data_path` parameter). Note that if you set the `data_path`
value, **the split
parameters will be disregarded** and `mode` will be forced to `'all'`!

Here is an example of running these functions.

```python
project.run_prediction('prediction_1', episode_name='episode_3', load_epoch=150, mode='test')
```
This command will generate a prediction dictionary with the model from epoch 150 of `episode_3` for the test
subset of the project data (split according to the configuration files) and save it at
*project_name/results/predictions/prediction_1.pickle*. 

The history of prediction runs is recorded and at the [exploring] section
we have described how to access it.

[GUI]: https://github.com/amathislab/dlc2action_annotation
"""

from dlc2action.version import __version__, VERSION

__pdoc__ = {"options": False, "version": False, "scripts": False}
