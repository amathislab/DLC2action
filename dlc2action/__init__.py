#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
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

Alternatively, if you have installed the package with pip, you can run a command in your terminal.

```
$ dlc2action_init --name project_name -d data_type -a annotation_type -dp path/to/data_folder -ap path/to/annotation_folder
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

After the poprojectrject is created you can modify the
parameters manually in the `project_name/config` folder or with the `project.update_parameters()` function.
Make sure to fill in all the fields marked with `???`.

This can also be done through the terminal. With an `--all` flag the command will iterate through all parameters
and otherwise it will only ask you to fill in the `???` blanks.

```
$ dlc2action_fill --name project_name --all
```

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

To start a search, you need to run the `project.project.Project.run_hyperparameter_search` command. Let's
say we want to optimize for four parameters: overlap length, SSL task type, learning rate and
number of feature maps in the model. Here is the process
to figure out what we want to run.

1. **Find the parameter names**

    We look those parameters up in the config files and find them in *data.yaml*, *general.yaml*, *training.yaml*
    and *models/ms_tcn3.yaml*,
    respectively.
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
You can get a feeling for the predictions made by a model by running `project.project.Project.visualize_results`.

```python
project.visualize_results('episode_1', load_epoch=50)
```
This command will generate a prediction for a random sample and visualize it compared to the ground truth.
There is a lot of parameters you can customize, check them out in the documentation).

Another available visualization type is training curve comparison with `project.project.Project.plot_episodes`.
You can compare different metrics and modes across several episode or within one. For example, this command
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
project.run_suggestion(
    'suggestion_new_data',
    suggestion_episode='episode_4',
    suggestion_classes=['sleeping', 'eating'],
    suggestion_threshold=0.6,
    exclude_classes=['inactive'],
    data_path='/path/to/new_data_folder',
)
```
The first command will generate a prediction dictionary with the model from epoch 150 of `episode_3` for the test
subset of the project data (split according to the configuration files) and save it at
*project_name/results/predictions/prediction_1.pickle*. The second command will create suggestion and active
learning files for data at */path/to/new_data_folder* and save them at
*project_name/results/suggestions/suggestion_new_data*. There is a lot of parameters you can tune here to
get exactly what you need, so we really recommend reading the documentation of the
`project.project.Project.run_suggestion` function before using it.

The history of prediction and suggestion runs is recorded and at the [exploring] section
we have described how to access it.

[GUI]: https://github.com/amathislab/classic_annotation

# Contribution
Here you can learn how to add new elements to `dlc2action`.

## Models
[model]: #models
Models in `dlc2action` are formalized as `model.base_model.Model` instances. This class inherits from
`torch.nn.Module` but adds a more formalized structure and defines interaction with SSL modules.
Note that **SSL modules and base models are added separately** to allow for 'mixing and matching', so make sure
the model you want to add only predicts the target value.

The process to add a new model would be as follows.

1. **Separate your model into feature extraction and prediction**

    If we want to add an SSL module to your module, where should it go? Everything up to that point is
    your feature extractor and everything after is the prediction generator.

2. **Write the modules**

    Create a *{model_name}_modules.py* file at the *dlc2action/models* folder and write all necessary modules there.

3. **Formalize input and output**

    Make sure that your feature extraction and prediction generation modules accept a single input and return
    a single output.

4. **Write the model**

    Create a *{model_name}.py* file at the *dlc2action/models* folder and write the code for the
    `model.base_model.Model` child class. The `model.base_model.Model._feature_extractor` and
    `model.base_model.Model._predictor` functions need to return your feature extraction and prediction
    generation modules, respectively. See the `model.ms_tcn` code for reference.

5. **Add options**

    Add your model to the `models` dictionary at `dlc2action.options`. Add
    `{'{model_name}.{class_name}.dump_patches': False, '{model_name}.{class_name}.training': False}` to the
    `__pdoc__` dictionary at `dlc2action.model.__init__`.

6. **Add config parameters**

    Read the [config] explanation and add your parameters.

## SSL
SSL tasks are formalized as `ssl.base_ssl.SSLConstructor` instances. Read the documentation at `ssl` to
learn more about this and follow the process.

1. **Create a new file**

    Create a new file for your constructor at *dlc2action/ssl* and start defining a `ssl.base_ssl.SSLConstructor`
    child class.

2. **Determine the type of your task**

    You can find the descriptions of available SSL types at `ssl.base_ssl.SSLConstructor`. If what you want to do
    does not fit any of those types, you can add a new one. In that case you will have to modify the
    `model.base_model`
    code. More specifically, add the new name to the `available_ssl_types` variable and define the interaction
    at `model.base_model.Model.forward`. Set the `type` class variable to the type you decided on.

3. **Define the data transformation**

    You need to write a function that will take a *feature dictionary* (see `feature_extraction`) as input
    and return SSL input and SSL target as output. Read more about that at `ssl`. The `transformation` method
    of your class should perform that function.

4. **Define the network module**

    Take a look at `ssl.modules` and choose one of the existing modules or add your own. Remember that its
     `forward` method has
    to receive the output of a base model's feature extraction module (see the [model] tutorial for more details)
    as input and return a value that will go into the SSL loss function as output. Your class's `construct_module`
    method needs to return an instance of this module.

5. **Define the loss function**

    Choose your loss at `dlc2action.loss` or add a new one.
    If you do decide to add a new loss, see the [loss] tutorial.
    Set the `loss` method of your class to apply this loss.

6. **Add options**

    Add your model to the `ssl_constructors` dictionary at `dlc2action.options` and set a default loss weight
    in the `ssl_weights` dictionary at *training.yaml*.

7. **Add config parameters**

    Read the [config] explanation and add your parameters.

## Datasets
Datasets in `dlc2action` are defined through `data.base_store.InputStore` and `data.base_store.BehaviorStore`
classes. Before you start, please read the documentation at `data` to learn more about the assumptions we make
and the different classes involved in data processing. Then, follow these instructions.

1. **Figure out what you need to add**

    What are the data and annotation types of your dataset? Can you find either of them at `options.input_stores`
    and `options.annotation_stores`? If so, you are in luck. If not, move on to have a look at `data.annotation_store`
    and `data.input_store`. It's likely that the classes we already have already do most of what you need.
    For annotation, in most cases it should be enough to inherit from `data.annotation_store.ActionSegmentationStore`
    and only change the `data.annotation_store.ActionSegmentationStore._open_annotations` function. With input stores
    it can be more complicated, but should definitely use implementations at `data.input_store` as an example.

2. **Write the class**

    Create a new class at either `data.input_store` or `data.annotation_store`, inherit from whatever is the closest
    to what you need and implement all abstract functions.

3. **Add options**

    Add your store to `options.annotation_stores` or `options.input_stores`.

4. **Add config parameters**

    Read the [config] explanation and add your parameters.

## Losses
Adding a new loss is fairly straightforward since `dlc2action` uses `torch.nn.Module` instances as losses.

1. **Write the class**

    Add a new loss class inheriting from `torch.nn.Module` at `dlc2action.loss`. Add it to one of the existing
    submodules or create a new one if it isn't a good fit for anything. Generally, the `forward` method of your
    loss should take prediction and target tensors as input, in that order, and return a float value.

2. **Add options**

    If you want your new loss to be an option for the main loss function in training, you should add
    it to the `dlc2action.options.losses` dictionary. Note that in that case it has to have the input and output
    formats described above. If your loss expects input from a multi-stage model, like MS-TCN, add its name to
    `dlc2action.options.losses_multistage` as well.

3. **Add config parameters**

    Read the [config] explanation and add your parameters.

## Metrics
All metrics in `dlc2action` inherit from `metric.base_metric.Metric`. They store internal parameters that get updated
with each batch and are then used to compute the metric value. You need to write the functions for resetting the
parameters, updating them and calculating the value.

1. **Write the class**

    Add a new class at `metric.metrics`, inheriting from `metric.base_metric.Metric` and write the three
    abstract methods (read the documentation for details).

2. **Add options**

    Add your new metric to the `dlc2action.options.metrics` dictionary. Next, if it is supposed to decrease with
    increasing prediction quality, add its name to the `dlc2action.options.metrics_minimize` list and if it does not
    say anything about how good the predictions are, add it to `dlc2action.options.metrics_no_direction`.
    If it increases when predictions get better, you don't need to do anything else.

3. **Add config parameters**

    If you added your loss to the options at the previous step, you also need to add it to the config files.
    Read the [config] explanation and do that.

## Feature extractors
Feature extractors in `dlc2action` are defined fairly loosely. You need to make a new class that inherits from
one of `feature_extraction.FeatureExtractor` subclasses (i.e. `feature_extraction.PoseFeatureExtractor`). Each
of those subclasses are defined for a subclass of `data.base_store.InputStore` for a specific broad type of
data (i.e. `data.base_store.PoseInputStore` for pose estimation data).

When extracting the features, `data.base_store.InputStore` instances will pass a data dictionary to the feature
extractor and they will also contain the methods to extract information from that dictionary. You only
need to use that information (coordinates, list of body parts, number of frames and so on) to compute
some new features (distances, speeds, accelerations). Read the `feature_Extraction` documentation to find out more
about that. When you're done with reading you can start with those instructions.

1. **Write the class**

    Create a new class at `feature_extraction` that inherits from a `feature_extraction.FeatureExtractor` subclass.
    Look at the methods available for information extraction and write the `extract_features`
    function that will transform
    that information into some new features.

2. **Add a transformer**

    Since you've added some new features, `dlc2action` does not know how to handle them when adding augmentations.
    You need to either create a new `transformer.base_transformer.Transformer` or modify an existing one to fix
    that. Read the [augmentations]
    tutorial to find out how.

3. **Add options**

    Add your feature extractor to `dlc2action.options.feature_extractors` and set the corresponding transformer
    at `dlc2action.options.extractor_to_transformer`.

3. **Add config parameters**

    Read the [config] explanation and add your parameters.

## Augmentations
[augmentations]: #augmentations
Augmentations in `dlc2action` are carried out by `transformer.base_transformer.Transformer` instances.
Since they need to know how to handle different features, each transformer handles the output of specific
`feature_extraction.FeatureExtractor` subclasses (only one transformer corresponds to each feature extractor,
but each transformer can
work with multiple feature extractors). Hence the instructions.

1. **Figure out whether you need a new transformer**

    If the feature extractor you are interested in already has a corresponding transformer, you only need to
    add a new method to that transformer. You can find which transformer you need at
    `dlc2action.options.extractor_to_transformer`. If the feature extractor is new, it might still be enough to
    modify an existing transformer. Take a look at `transformer` and see if any of the existing classes are close
    enough to what you need. If they are not, you will have to create a new class.

2. **Add a new class** *(skip this step if you are not creating a new transformer)*

    If you did decide to create a new class, create a new *{transformer_name}.py* file at `dlc2action.transformer`
    and write it there, inheriting from `transformer.base_transformer.Transformer`.

3. **Add augmentation functions**

    Add new methods to your transformer class. Each method has to perform one augmentation, take
    `(main_input: dict, ssl_inputs: list, ssl_targets: list)`
    as input and return an output of the same format. Here `main_input` is a feature dictionary of the sample
    data, `ssl_inputs` is a list of SSL input feature dictionaries and `ssl_targets` is a list of SSL target
    feature dictionaries. The same augmentations are applied to all inputs if they are not `None` and
    have feature keys that
    are recognized by your transformer.

    When you are done writing the methods, add them with their names to the dictionary returned by
    `transformer.base_transformer.augmentations_dict`. If you think that your augmentation suits a very wide
    range of datasets, add its name to the list returned by `transformer.base_transformer.default_augmentations`
    as well.

4. **Add options** *(skip this step if you are not creating a new transformer)*

    Add your new transformer to `dlc2action.options.transformers`.

5. **Add config parameters**

    Read the [config] explanation and add your parameters.

## Config files
[config]: #config-files
It is important to modify the config files when you add something new to `dlc2action` since it is supposed to
mainly be used through the `project` interface that relies on the files being up-to-date.

To add configuration parameters for a new model, feature extractor,
input or annotation store, create a new file in the *model*, *features*,
*data* or *annotation* subfolder of *dlc2action/config*, respectively. If you are adding a new loss,
metric or SSL constructor, add a new dictionary to the corresponding file
at the same folder. **Make sure to use the same name as what you've added to the options file.** Then just
list all necessary parameters (use the same names as in the class signature), their default values
and an explanation in the commentary.

The commentary for every parameter has to start with its supposed type. Use one of `'bool'`, `'int'`, `'float'`,
`'str'`, `'list'` or `'set'` or define a set of possible values with square brackets (e.g. `['pose', 'image']`).
The type has to be followed with a semicolon and a short explanation. Please try to keep the explanations one line
long. Here is an example:

```
frame_limit: 15 # int; tracklets shorter than this number of frames will be discarded
```

Note that if a parameter that you need is already defined at *dlc2action/config/general.yaml*, **it will be pulled
automatically**, there is no need to add it to your new list of parameters. Just make sure that the name of the
parameter in your class signature and at *general.yaml* is the same.

In addition, there are also several *blanks* you can use to set the default parameters. Those *blanks*
are filled at runtime after the dataset is created. Read more about them at the `project` documentation.

When you are finished with defining the parameters, you should also describe them in the documentation (perhaps in
more detail) at the [options] section.

# Options
[options]: #options
Here you can find an overview of the parameters currently available in the settings.

## Data types

 - `'dlc_tracklet'`: DLC tracklet data,
 - `'dlc_track'`: DLC track data,

## Annotation types

 - `'dlc'`: the classic_annotation GUI format,
 - `'boris'`: BORIS format,

## Data

### Annotation

#### BORIS and DLC

- `behaviors: set` <br />
    the set of string names of the behaviors to predict
- `annotation_suffix: set` <br />
    the set of string suffixes such that annotation file names have the format of
    *{video_id}{annotation_suffix}*
- `correction: dict` <br />
    a dictionary of label corrections where keys are the names to be replaced and values are the replacements
    (e. g. `{'sleping': 'sleeping', 'calm locomotion': 'locomotion'}`)
- `error_class: str` <br />
    a class denoting errors in pose estimation (the intervals annotated with this class will be ignored)
- `min_frames_action: int` <br />
    the minimum number of frames for an action (shorter actions will be discarded during training)
- `filter_annotated: bool` <br />
    if true, `dlc2action` will discard long unannotated intervals during training
- `filter_background: bool` <br />
    if true, only label frames as background if a behavior is annotated somewhere close (in the same segment)
- `filter_visibility: bool` <br />
    if true, only label behaviors if there is suffucient visibility (a fraction of frames in a segment larger than
    `visibility_min_frac` has visibility score larger than `visibility_min_score`; the visibility score is defined by
    `data.base_store.InputStore` instances and should generally be from 0 to 1
- `visibility_min_score: float` <br />
    the minimum visibility score for visibility filtering (generally from 0 to 1)
- `visibility_min_frac: float` <br />
    the minimum fraction of visible frames for visibility filtering


#### CalMS21

No parameters

### Input

#### DLC track and tracklet

- `data_suffix: set` <br />
    the set of suffices such that the data files have the format of *{video_id}{data_suffix}*
- `data_prefix: set` <br />
    a set of prefixes used in file names for different views such that the data files have the format
    of *{data_prefix}{video_id}{data_suffix}*
- `feature_suffix: str` <br />
    the string suffix such that the precomputed feature files are named *{video_id}{feature_suffix}*;
    the files should be either stored at `data_path` or included in `file_paths`
- `contert_int_indices: bool` <br />
    if true, convert any integer key i in feature files to 'ind{i}'
- `canvas_shape: list` <br />
    the size of the canvas where the pose was defined
- `len_segment: int` <br />
    the length of segments (in frames) to cut the videos into
- `overlap: int` <br />
    the overlap (in frames) between neighboring segments
- `ignored_bodyparts: set` <br />
    the set of string names of bodyparts to ignore
- `default_agent_name: str` <br />
    the default agent name used in pose file generation
- `frame_limit: int` (only for tracklets) <br />
    tracklets shorter than this number of frames will be discarded

#### CalMS21

- `len_segment: int` <br />
    the length of segments (in frames) to cut the videos into
- `overlap: int` <br />
    the overlap (in frames) between neighboring segments
- `load_unlabeled: bool` <br />
    if true, unlabeled data will be loaded
- `file_prefix: str` <br />
    the prefix such that the data files are named {prefix}train.npy and {prefix}test.npy

### Features

#### Kinematic

- `interactive: bool` <br />
    if true, features are computed for pairs of clips (agents)
- `keys: set` <br />
    the keys to include
    (a subset of `{"coords", "intra_distance", "speed_joints", "angle_joints_radian", "acc_joints", "inter_distance"}`
    where
    - `'coords'` is the raw coordinates,
    - `'intra_distance'` is the distances between all pairs of keypoints,
    - `'speed_joints'` is the approximated vector speeds of all keypoints,
    - `'angle_joints_radian'` is the approximated angular speeds of all keypoints,
    - `'acc_joints'` is the approximated accelerations of all keypoints,
    - `'inter_distance'` is the distance between all pairs of keypoints where the points belong to different agents
        (used if `interactive` is true))

## Augmentations

### Kinematic

- `augmentations: set` <br />
    a list of augmentations (a subset of `{'rotate', 'mask', 'add_noise', 'shift', 'zoom', 'mirror'}`)
- `use_default_augmentations: bool` <br />
    if true, the default augmentation list will be used (`['mirror', 'shift', 'add_noise']`)
- `rotation_limits: list` <br />
    list of float rotation angle limits ([low, high]; default [-pi/2, pi/2])
- `mirror_dim: set` <br />
    set of integer dimension indices that can be mirrored
- `noise_std: float` <br />
    standard deviation of added noise
- `zoom_limits: list` <br />
    list of float zoom limits ([low, high]; default [0.5, 1.5])
- `masking_probability: float` <br />
    the probability of masking a joint

## Model

### MS-TCN3

- num_layers_PG: int` <br />
    number of layers in prediction generation stage
- num_layers_R: int` <br />
    number of layers in refinement stages
- `num_R: int` <br />
    number of refinement stages
- `num_f_maps: int` <br />
    number of feature maps
- `dims: int` <br />
    number of input features (set to 'dataset_features' to get the number from
    `dlc2action.data.dataset.BehaviorDataset.dataset_features()`)
- `dropout_rate: float` <br />
    dropout rate
- `skip_connections_refinement: bool` <br />
    if true, skip connections are added to the refinement stages
- `block_size_prediction: int` <br />
    if not null, skip connections are added to the prediction generation stage with this interval
- `direction_R: str` <br />
    causality of refinement stages; choose from:
    - `None`: no causality,
    - `bidirectional`: a combination of forward and backward networks,
    - `forward`: causal,
    - `backward`: anticausal
- `direction_PG: str` <br />
    causality of refinement stages (see `direction_R`)
- `shared_weights: bool` <br />
    if `True``, weights are shared across refinement stages
- `block_size_refinement: int` <br />
    if not 0, skip connections are added to the refinement stage with this interval
- `PG_in_FE: bool` <br />
    if `True`, the prediction generation stage is included in the feature extractor
- `rare_dilations: bool` <br />
    if `True`, dilation increases with layer less often
- `num_heads: int` <br />
    the number of parallel refinement stages


## General

- `model_name: str` <br />
    model name; choose from:
     - `'ms_tcn3'`: the original MS-TCN++ model with options to share weights across refinement stages or
        add skip connections.
- `num_classes: int` <br />
    number of classes (set to 'dataset_num_classes' to get the number from
    `dlc2action.data.dataset.BehaviorDataset.num_classes()`)
- `exclusive: bool` <br />
    if true, single-label classification is used; otherwise multi-label
- `ssl: set` <br />
    a set of SSL types to use; choose from:
     - `'contrastive'`: contrastive SSL with NT-Xent loss,
     - `'pairwise'`: pairwise comparison SSL with triplet or circle loss,
     - `'masked_features'`: prediction of randomly masked features with MSE loss,
     - `'masked_frames'`: prediction of randomly masked frames with MSE loss,
     - `'masked_joints'`: prediction of randomly masked joints with MSE loss,
     - `'masked_features_tcn'`: prediction of randomly masked features with MSE loss and a TCN module,
     - `'masked_frames_tcn'`: prediction of randomly masked frames with MSE loss and a TCN module,
     - `'masked_joints_tcn'`: prediction of randomly masked joints with MSE loss and a TCN module
- `metric_functions: set` <br />
    set of metric names; choose from:
     - `'recall'`: frame-wise recall,
     - `'segmental_recall'`: segmental recall,
     - `'precision'`: frame-wise precision,
     - `'segmental_precision'`: segmental precision,
     - `'f1'`: frame-wise F1 score,
     - `'segmental_f1'`: segmental F1 score,
     - `'edit_distance'`: edit distance (as a fraction of length of segment),
     - `'count'`: the fraction of predictions labeled with a specific behavior
- `loss_function: str` <br />
    name of loss function; choose from:
     - `'ms_tcn'`: a loss designed for the MS-TCN network; cross-entropy + MSE for consistency
- `feature_extraction: str` <br />
    the feature extraction method; choose from:
     - `'kinematic'`: extracts distances, speeds and accelerations for pose estimation data
- `only_load_annotated: bool` <br />
    if true, the input files that don't have a matching annotation file will be disregarded
- `ignored_clips: set` <br />
    a set of string clip ids (agent names) to be ignored

## Losses

### MS_TCN

- `weights: list` <br />
    list of weights for weighted cross-entropy
- `focal: bool` <br />
    if True, focal loss will be used
- `gamma: float` <br />
    the gamma parameter of focal loss
- `alpha: float` <br />
    the weight of consistency loss

## Metrics

### Recall, precision and F1 score (segmental and not)

- `main_class: int` <br />
    if not null, recall will only be calculated for main_class
- `average: str` <br />
    averaging method; choose from `'macro'`, `'micro'` or `'none'`
- `ignored_classes: set` <br />
    a set of class ids to ignore in calculation
- `iou_threshold: float` (only for segmental metrics) <br />
    intervals with IoU larger than this threshold are considered correct

### Count

- `classes: set` <br />
    a set of the class indices to count the occurrences of

## SSL

### Contrastive

- `ssl_features: int` <br />
    length of clip feature vectors
- `tau: float` <br />
    tau (NT-Xent loss parameter)
- `len_segment: int` <br />
    length of the segments that enter the SSL module
- `num_f_maps: list` <br />
    shape of the segments that enter the SSL module

### Pairwise

- `ssl_features: int` <br />
    length of clip feature vectors
- `margin: float` <br />
    margin (triplet loss parameter)
- `distance: str` <br />
    either 'cosine' or 'euclidean'
- `loss: str` <br />
    either 'triplet' or 'circle'
- `gamma: float` <br />
    gamma (triplet and circle loss parameter)
- `len_segment: int` <br />
    length of the segments that enter the SSL module
- `num_f_maps: list` <br />
    shape of the segments that enter the SSL module

### Masked joints, features and frames

- `frac_masked: float` <br />
    fraction of features to be masked
- `num_ssl_layers: int` <br />
    number of layers in the SSL module
- `num_ssl_f_maps: int` <br />
    number of feature maps in the SSL module
- `dims: int` <br />
    number of features per frame in original input data
- `num_f_maps: list` <br />
    shape of the segments that enter the SSL module

### Contrastive masked

- `ssl_features: int` <br />
    length of clip feature vectors
- `tau: float` <br />
    tau (NT-Xent loss parameter)
- `len_segment: int` <br />
    length of the segments that enter the SSL module
- `num_f_maps: list` <br />
    shape of the segments that enter the SSL module
- `num_masked: int` <br />
    number of frames to mask

## Training

- `lr: float` <br />
    learning rate
- `device: str` <br />
    device name (recognized by `torch`)
- `verbose: bool` <br />
    print training process
- `augment_train: int` <br />
    either 1 to use augmentations during training or 0 to not use
- `augment_val: int` <br />
    number of augmentations to average over at validation
- `ssl_weights: dict` <br />
    dictionary of SSL loss function weights (keys are SSL names, values are weights)
- `num_epochs: int` <br />
    number of epochs
- `to_ram: bool` <br />
    transfer the dataset to RAM for training (preferred if the dataset fits in working memory)
- `batch_size: int` <br />
    batch size
- `freeze_features: bool` <br />
    freeze the feature extractor parameters
- `ignore_tags: bool` <br />
    ignore meta tags (meta tags are generated by some datasets and used by some models and can contain
    information such as annotator id); when `True`, all meta tags are set to `None`
- `model_save_epochs: int` <br />
    interval for saving training checkpoints (the last epoch is always saved)
- `use_test: float` <br />
    the fraction of the test dataset to use in training without labels (for SSL tasks)
- `partition_method: str`  <br />
    the train/test/val partitioning method; choose from:
    - `'random'`: sort videos into subsets randomly,
    - `'random:test-from-name'` (or `'random:test-from-name:{name}'`): sort videos into training and validation
        subsets randomly and create
        the test subset from the video ids that start with a speific substring (`'test'` by default, or `name`
        if provided),
    - `'random:equalize:segments'` and `'random:equalize:videos'`: sort videos into subsets randomly but
        making sure that for the rarest classes at least `0.8 * val_frac` of the videos/segments that contain
        occurrences of the class get into the validation subset and `0.8 * test_frac` get into the test subset;
        this in ensured for all classes in order of increasing number of occurrences until the validation and
        test subsets are full
    - `'val-from-name:{val_name}:test-from-name:{test_name}'`: create the validation and test
        subsets from the video ids that start with specific substrings (`val_name` for validation
        and `test_name` for test) and sort all other videos into the training subset
    - `'folders'`: read videos from folders named *test*, *train* and *val* into corresponding subsets,
    - `'time'`: split each video into training, validation and test subsequences,
    - `'time:strict'`: split each video into validation, test and training subsequences
        and throw out the last segment in validation and test (to get rid of overlaps),
    - `'file'`: split according to a split file.
- `val_frac: float` <br />
    fraction of dataset to use as validation
- `test_frac: float` <br />
    fraction of dataset to use as test
- `split_path: str` <br />
    path to the split file (**only used when partition_method is `'file'`,
    otherwise disregarded and filled automatically**)
"""

from dlc2action.version import __version__, VERSION

__pdoc__ = {"options": False, "version": False, "scripts": False}
