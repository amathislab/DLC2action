#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
## Training and inference

`dlc2action.task.universal_task.Task` is the class that performs training and inference while
'dlc2action.task.task_dispatcher.TaskDispatcher` creates and updates `dlc2action.task.universal_task.Task` instances
in accordance with task parameter dictionaries.
"""

from dlc2action.task.task_dispatcher import TaskDispatcher
from dlc2action.task.universal_task import Task
