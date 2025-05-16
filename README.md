# Multi-scale Fused Graph Convolutional Network for Multi-site Photovoltaic Power Forecasting
This is the origin Pytorch implementation of the paper "Multi-scale Fused Graph Convolutional Network for Multi-site Photovoltaic Power Forecasting". Here, a novel and effective spatiotemporal model (*i.e., MSF-GCN*) is constructed for multi-site photovoltaic power forecasting. 
Please see our [paper](https://www.sciencedirect.com/science/article/pii/S0196890425002961) for more details.
<p align="center">
    <img src="./assets/method.png">
</p>

# Getting Started
To get started, ensure you have Conda installed on your system and follow these steps to set up the environment:
* python == 3.11.9
* matplotlib == 3.10.1
* numpy == 1.24.0
* pandas == 1.3.5
* scikit_learn == 1.4.2
* torch == 2.3.0

### Dataset
All the datasets can be found in ```./dataset/```, which are obtained from the following public links and cover different climate types.
* PVOD : http://dx.doi.org/10.11922/sciencedb.01094
* EODP : https://opendata.elia.be
* NREL : https://www.nrel.gov/grid/solar-power-data.html

### Examples
The experiment scripts for all datasets are provided under the folder ```./scripts/```. You can easily reproduce the results of the *MSF-GCN* using the following Python command:

```
# PVOD dataset
bash scripts/MSFGCN_PVOD.sh
# EODP dataset
bash scripts/MSFGCN_EODP.sh
# NREL dataset
bash scripts/MSFGCN_NERL.sh
```
# Acknowledgement
This work was supported by the National Natural Science Foundation of China (72242104), the China Postdoctoral Science Foundation (2024M761027), and the Interdisciplinary Research Program of Hust (2024JCYJ020).

The library is constructed based on the following repos:
* https://github.com/thuml/Time-Series-Library
* https://github.com/tsinghua-fib-lab/Traffic-Benchmark

# Citation
```
@article{Sima2025MSF,
  author    = {Qi, Sima and Xinze, Zhang and Siyue, Yang and Liang, Shen and Yukun, Bao},
  title     = {Multi-scale fused Graph Convolutional Network for multi-site photovoltaic power forecasting},
  journal   = {Energy Conversion and Management},
  volume    = {333},
  year      = {2025},
  pages     = {119773},
  issn      = {0196-8904},
  doi       = {10.1016/j.enconman.2025.119773},
}
```
# Contact
For any questions, you are welcome to contact us via qisima@hust.edu.cn.
