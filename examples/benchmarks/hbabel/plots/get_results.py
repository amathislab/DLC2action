#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os
import numpy as np
from dlc2action.project import Project
import pandas as pd

DATA_PATH = f'path/to/converted_data'
DATA_TYPE = "dlc_track"
ANNOTATION_TYPE = "dlc"
PROJECTS_PATH = f"path/to/projects"


project_name = f"hBABEL_d2a_tops"
project = Project(
    project_name,
    data_path= DATA_PATH,
    annotation_path= DATA_PATH,
    projects_path=PROJECTS_PATH,
    data_type=DATA_TYPE,
    annotation_type= ANNOTATION_TYPE,
)


episode_names = project.list_episodes(print_results=False).index
episode_names = [e.split("::")[0] for e in episode_names if "top" in e]
episode_names = np.unique(episode_names)

df = {"models" : [], "f1_scores" : [], "trials" : [], "tops" : []}
for episode_name in episode_names:
    res,v = project.get_summary(episode_names = [episode_name], method = "best", return_values = True)
    top = int(episode_name.split("_")[-1])
    model = episode_name.split("_best_file")[0]
    f1_scores = v["f1"]

    df["models"] += [model]*len(f1_scores)
    df["f1_scores"] += f1_scores
    df["trials"] += list(np.arange(len(f1_scores)))
    df["tops"] += [top]*len(f1_scores)

df = pd.DataFrame(df)
print(df)
output_path = os.path.join("./results","hbabel.csv")
df.to_csv(output_path)
