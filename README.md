# awesome-sEMG-hand-gesture-recognition
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)

This repository provide some unofficial pytorch implementations of awesome works for surface electromyography (sEMG) based hand gesture recognition.

Please star this repo if you find our work is helpful for you.

* Network architectures to be reproduced
  * [ ] [Multi-stream CNN (WeiNet)](https://www.sciencedirect.com/science/article/abs/pii/S0167865517304439)
  * [ ] [XceptionTime](https://arxiv.org/abs/1911.03803)
  * [ ] [BiTCN](https://link.springer.com/chapter/10.1007/978-981-33-4932-2_30)

## Environment
The code is developed using python 3.7 on Ubuntu 20.04. NVIDIA GPU is needed.

## Data preparing
The experiment are taken on the [Ninapro dataset](http://ninaweb.hevs.ch/). The first sub-dataset DB1 and second sub-dataset DB2 are ultilized. 
1. Firstly download the [Ninapro DB1](http://ninaweb.hevs.ch/data1) and [Ninapro DB2](http://ninaweb.hevs.ch/data2) datasets.
Your directory tree should look like this: 

```
${ROOT}/data/ninapro
├── db1
|   |—— s1
|   |—— s2
|   |   ...
|   └── s27
|       |—— S27_A1_E1.mat
|       |—— S27_A1_E2.mat
|       └── S27_A1_E3.mat
└── db2
    |—— DB2_s1
    |—— DB2_s2
    |   ...
    └── DB2_s40
        |—— S40_E1_A1.mat
        |—— S40_E2_A1.mat
        └── S40_E3_A1.mat
```

2. Then run the following code to obtain the json file for experiment config:
```
python ./experiments/CinC17/build_dataset.py
```

## Quick Start
### Installation
1. Clone this repo
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
### Training
* Train net
  ```
  
  ```
### Evaluation
* validate net
  ```
  
  ```

## Results Demonstration

## Contact
If you have any questions, feel free to contact me through jia.zeng@sjtu.edu.cn or Github issues.