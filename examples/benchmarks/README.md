# 🏆 DLC2Action - Benchmarks

This directory contains the code and resources required to train and evaluate the proposed DLC2Action benchmarks.

## 🐾 2D Benchmarks

DLC2Action is evaluated on five 2D animal action segmentation benchmarks:

| Dataset         | Training Script                 | Download link |  Reference |
| --------------- | -----------------------------   | --------- | --------- |
| CalMS21         | [calms21.py](calms21.py)        |[data](https://data.caltech.edu/records/s0vdx-0k302)    | [[1]](#1-calms21)      |
| SimBA - CRIM13    | [simba_crim.py](simba_crim.py)  |[data](https://data.caltech.edu/records/4emt5-b0t10)   | [[2]](#2-crim13)       |
| SimBA - RAT     | [simba_rat.py](simba_rat.py)    |[data](https://osf.io/sr3ck/)    | [[3]](#3-simba-rat)       |
| Sturman - OFT   | [sturman_oft.py](sturman_oft.py)|[data](https://github.com/ETHZ-INS/DLCAnalyzer/tree/master/data/OFT), [videos](https://zenodo.org/records/3608658)    | [[4]](#4-oft-and-epm)     |
| Sturman - EPM   | [sturman_epm.py](sturman_epm.py)|[data](https://github.com/ETHZ-INS/DLCAnalyzer/tree/master/data/EPM)    | [[4]](#4-oft-and-epm)     |

For the OFT and EPM dataset, please run `examples/labels_processing_example.py` to convert the dataset into DLC2Action compatible files.


## 🏀 3D Benchmarks: hBABEL and SHOT7M2

DLC2Action is further evaluated on two 3D human action segmentation benchmarks: hBABEL and SHOT7M2.
Both datasets were introduced in [Elucidating the Hierarchical Nature of Behavior with Masked Autoencoders](https://link.springer.com/chapter/10.1007/978-3-031-73039-9_7) [[5]](#5-shot7m2-and-hbabel).

For detailed instructions on training and evaluation, refer to the respective documentation for [hBABEL](hbabel/README.md) and [SHOT7M2](shot7m2/README.md).

## 🎬 Using Video Features

Examples demonstrating the use of video features for training DLC2Action models are provided in the [video_features](video_features) directory.

## 🧪 Ablation Studies

Examples for training models using only coordinate data (without kinematic features) are available in the [no_kin_features](no_kin_features) directory.

## 📜 References

### [1] **CalMS21**
```
@article{sun2021multi,
  title={The multi-agent behavior dataset: Mouse dyadic social interactions},
  author={Sun, Jennifer J and Karigo, Tomomi and Chakraborty, Dipam and Mohanty, Sharada P and Wild, Benjamin and Sun, Quan and Chen, Chen and Anderson, David J and Perona, Pietro and Yue, Yisong and others},
  journal={arXiv preprint arXiv:2104.02710},
  year={2021}
}
```

### [2] **CRIM13**
```
@misc{crim13,
    title={CRIM13 (Caltech Resident-Intruder Mouse 13)}, DOI={10.22002/D1.1892}, publisher={CaltechDATA},
    author={Xavier P. Burgos-Artizzu and Piotr Dollar and Dayu Lin and David J. Anderson and Pietro Perona},
    year={2021}
}
```

### [3] **SimBA-RAT**
```
@article{goodwin2024simple,
  title={Simple Behavioral Analysis (SimBA) as a platform for explainable machine learning in behavioral neuroscience},
  author={Goodwin, Nastacia L and Choong, Jia J and Hwang, Sophia and Pitts, Kayla and Bloom, Liana and Islam, Aasiya and Zhang, Yizhe Y and Szelenyi, Eric R and Tong, Xiaoyu and Newman, Emily L and others},
  journal={Nature neuroscience},
  volume={27},
  number={7},
  pages={1411--1424},
  year={2024},
  publisher={Nature Publishing Group US New York}
}
```

### [4] **OFT and EPM**
```
@article{sturman2020deep,
  title={Deep learning-based behavioral analysis reaches human accuracy and is capable of outperforming commercial solutions},
  author={Sturman, Oliver and von Ziegler, Lukas and Schl{\"a}ppi, Christa and Akyol, Furkan and Privitera, Mattia and Slominski, Daria and Grimm, Christina and Thieren, Laetitia and Zerbi, Valerio and Grewe, Benjamin and others},
  journal={Neuropsychopharmacology},
  volume={45},
  number={11},
  pages={1942--1952},
  year={2020},
  publisher={Nature Publishing Group}
}
```

### [5] **Shot7M2 and hBABEL**
```
@inproceedings{stoffl2025elucidating,
  title={Elucidating the hierarchical nature of behavior with masked autoencoders},
  author={Stoffl, Lucas and Bonnetto, Andy and d’Ascoli, St{\'e}phane and Mathis, Alexander},
  booktitle={European Conference on Computer Vision},
  pages={106--125},
  year={2025},
  organization={Springer}
}
```
