#
# DLC2Action is not open-sourced yet.
# https://choosealicense.com/no-permission/
#
from dlc2action.project import Project


PROJECT_NAME = "test"  # the name of the project
PROJECTS_PATH = ...  # the path to the projects folder
DATA_TYPE = ...  # choose from Project.print_data_types()
ANNOTATION_TYPE = ...  # choose from Project.print_annotation_types()
DATA_PATH = ...  # path to data files
ANNOTATION_PATH = ...  # path to annotation files

FILE_PATHS = (
    None  # set to a list of file paths if you want to generate predictions for new data
)

project = Project(
    PROJECT_NAME,
    projects_path=PROJECTS_PATH,
    data_type=DATA_TYPE,
    annotation_type=ANNOTATION_TYPE,
    data_path=DATA_PATH,
    annotation_path=ANNOTATION_PATH,
)

# run project.help() for general information
# and project.help("data") to make sure that your files are organized correctly

project.update_parameters(
    ...
)  # run project.list_blanks() to see which parameters you need to update

MODELS = [
    "c2f_tcn",
    "transformer",
]  # see project.help("models") for a list of available models

for model in MODELS:  # run a hyperparameter search for your models
    project.run_default_hyperparameter_search(
        f"{model}_search",
        model_name=model,
        metric="f1",
    )

for model in MODELS:  # train models with the best hyperparameters
    project.run_episode(
        f"{model}_best",
        load_search=f"{model}_search",
        parameters_update={"general": {"model_name": model}},
        n_seeds=3,
        force=True,
    )

project.plot_episodes(  # compare training curves
    [f"{model}_best" for model in MODELS],
    metrics=["f1"],
    title="Best model training curves",
)

for model in MODELS:  # evaluate more metrics
    project.evaluate(
        [f"{model}_best"],
        parameters_update={
            "general": {"metric_functions": ["segmental_f1", "pr-auc", "f1"]},
            "metrics": {"f1": {"average": "none"}},
        },
    )

project.get_results_table(
    [f"{model}_best" for model in MODELS]
)  # get a table of the results

BEST_MODELS = [MODELS[0], MODELS[1]]  # choose the best model

project.run_prediction(
    [
        f"{model}_best" for model in BEST_MODELS
    ],  # if you choose multiple models, the predictions will be averaged
    force=True,
    file_paths=FILE_PATHS,  # make a prediction for new data
)

project.list_episodes(  # get the experiment history
    display_parameters=[
        "results/f1",
        "meta/time",
        "meta/training_time",
        "general/model_name",
    ],
    value_filter=f"results/f1::>0.3,general/model_name::{BEST_MODELS[0]}",
)
