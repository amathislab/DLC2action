# SHOT7M2 - DLC2Action 🚀


## 🗂️ Data Preparation

SHOT7M2 made its debut in [Elucidating the Hierarchical Nature of Behavior with Masked Autoencoders](https://link.springer.com/chapter/10.1007/978-3-031-73039-9_7).
To get started, fetch the dataset by following the instructions on the [official BehaveMAE Github page](https://github.com/amathislab/BehaveMAE/blob/main/datasets/README.md).

**Convert the dataset into DLC2Action format:**
```
python data/SHOT2D2A.py
```

---

## 🏋️ Model Training & Evaluation

**Train your DLC2Action models:**
```
python code/train_dlc2action_SHOT.py
```

**Put your models to the test:**
```
python code/evaluate_SHOT.py
```

---

## 📊 Extract Results for Plotting

Turn your evaluation results into plot-ready files with a single command:
```
python plots/get_results.py
```

---

## 📚 Citation

If SHOT7M2 powers your research, please give credit where it's due:
```
@inproceedings{stoffl2024elucidating,
  title={Elucidating the hierarchical nature of behavior with masked autoencoders},
  author={Stoffl, Lucas and Bonnetto, Andy and d’Ascoli, St{\'e}phane and Mathis, Alexander},
  booktitle={European Conference on Computer Vision},
  pages={106--125},
  year={2024},
  organization={Springer}
}
```

---
