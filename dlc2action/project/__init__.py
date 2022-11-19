#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
## Project interface

The most convenient way to use `dlc2action` is through the high-level project interface. It is defined in the
`project` module and its main functions are managing configuration files and keeping track of experiments.
When you create a `project.Project` instance with a previously unused name, it generates a new project folder with results,
history and configuration files.

```
.
project_name
├── config
├── meta
├── saved_datasets
└── results
    ├── logs
    │   └── episode.txt
    ├── models
    │   └── episode
    │       ├── epoch25.pt
    │       └── epoch50.pt
    ├── searches
    │   └── search
    │       ├── search_param_importances.html_docs
    │       └── search_contour.html_docs
    ├── splits
    │       ├── time_25.0%validation_10.0%test.txt
    │       └── random_20.0%validation_10.0%test.txt
    ├── suggestions
    │       └── active_learning
    │           ├── video1_suggestion.pickle
    │           ├── video2_suggestion.pickle
    │           └── al_points.pickle
    └── predictions
            ├── episode_epoch25.pickle
            └── episode_epoch50_newdata.pickle
```

Here is an explanation of this structure.

The **config** folder contains .yaml configuration files. Project instances can read them into a parameter dictionary
and update. Those readers understand several blanks for certain parameters that can be inferred from the data on
runtime:

* `'dataset_features'` will be replaced with the shape of features per frame in the data,
* `'dataset_classes'` will be replaced with the number of classes,
* `'dataset_inverse_weights'` at losses.yaml will be replaced with a list of float values that are inversely
* `'dataset_len_segment'` will be replaced with the length of segment in the data,
* `'model_features'` will be replaced with the shape of features per frame in the model feature extraction
    output (the input to SSL modules).
proportional to the number of frames labeled with the corresponding classes.

Pickled history files go in the **meta** folder. They are all pandas dataframes that store the relevant task
parameters, a summary of experiment results (where applicable) and some meta information, like additional
parameters or the time when the record was added. There are separate files for the history of training episodes,
hyperparameter searches, predictions, saved datasets and active learning file generations. The classes that handle
those files are defined at the `meta` module.

When a dataset is generated (the features are extracted and cut), it is saved in the **saved_datasets** folder. Every
time you create a new task, Project will check the saved dataset records and load pre-computed features if they
exist. You can always safely clean the datasets to save space with the remove_datasets() function.

Everything else is stored in the *results* folder. The text training log files go into the **logs** subfolder. Model
checkpoints (with `'model_state_dict'`, `'optimizer_state_dict'` and `'epoch'` keys) are saved in the **models**
subfolder. The main results of hyperparameter searches (best parameters and best values) are kept in the meta files
but they also generate html_docs plots that can be accessed in the **searches** subfolder. Split text files can be found
in the **splits** subfolder. They are also checked every time you create a task and if a split with the same
parameters already exists it will be loaded. Active learning files are saved in the **suggestions** subfolder.
Suggestions for each video are named *{video_id}_suggestion.pickle* and the active learning file is always
*al_points.pickle*. Finally, prediction files (pickled dictionaries) are stored in the **predictions** subfolder.
"""

from dlc2action.project.project import *
from dlc2action.project.meta import *
