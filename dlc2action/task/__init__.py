#
# DLC2Action is not open-sourced yet.
# https://choosealicense.com/no-permission/
#
"""
## Training and inference

`dlc2action.task.universal_task.Task` is the class that performs training and inference while
'dlc2action.task.task_dispatcher.TaskDispatcher` creates and updates `dlc2action.task.universal_task.Task` instances
in accordance with task parameter dictionaries.
"""

from dlc2action.task.task_dispatcher import TaskDispatcher
from dlc2action.task.universal_task import Task
