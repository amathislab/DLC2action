<!--
Copyright 2025-present by A. Mathis Group and contributors. All rights reserved.

  This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.

-->
# 👁️‍🗨️ Atari-head-DLC2Action

<b>Segmentation of actions in the Atari-head dataset using gaze information.</b>

The dataset was [published](https://ojs.aaai.org/index.php/AAAI/article/view/6161) in 2020 on AAAI Conference for Artificial Intelligence and is available [Here](https://ojs.aaai.org/index.php/AAAI/article/view/6161).


For 20 Atari games, the dataset has recordings from more or less 20 participants in frames format. Every participant were playing between 10 and 15 min on each trial. The additional data consist in both cumulated score and instantaneous  score per frame, the average time for the player to make a decision in ms, the action from a set of 17 possible actions and a matrix of 2D gaze positions in pixels for each frames.


## 🛠️ Implementation

`/data` : code for extracting the data and generating DLC2Action compatible datasets for both poses and behaviors. <br/>
`/code` : contains all code related to DLC2Action model trainings. <br>
`/plots` : code for plotting information in relation to the dataset, the episode trainings or the predictions. Plots will be generated in the same folder. <br/>
`/utils` : contains basic function and tools, in `data_load.py` you can find a data loader class commonly used across all codes. <br/>


### 📘 Action dictionary

The actions are coded with integers:
>* NOOP : 0
>* FIRE : 1
>* UP : 2
>* RIGHT : 3
>* LEFT : 4
>* DOWN : 5
>* UP-RIGHT : 6
>* UP-LEFT : 7
>* DOWN-RIGHT : 8
>* DOWN-LEFT : 9
>* UP-FIRE : 10
>* RIGHT-FIRE : 11
>* LEFT-FIRE : 12
>* DOWN-FIRE : 13
>* UP-RIGHT-FIRE : 14
>* UP-LEFT-FIRE : 15
>* DOWN-RIGHT-FIRE : 16
>* DOWN-LEFT-FIRE : 17

### References

```
@inproceedings{zhang2020atari,
  title={Atari-head: Atari human eye-tracking and demonstration dataset},
  author={Zhang, Ruohan and Walshe, Calen and Liu, Zhuode and Guan, Lin and Muller, Karl and Whritner, Jake and Zhang, Luxin and Hayhoe, Mary and Ballard, Dana},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={34},
  number={04},
  pages={6811--6820},
  year={2020}
}
```