# Multi-Modal-Inverse-Constrained-Reinforcement-Learning

This is the code for the paper [Multi-Modal Inverse Constrained Reinforcement Learning from a Mixture of Demonstrations](https://neurips.cc/virtual/2023/poster/72837) published in NeurIPS 2023. Note that:
1. Our method relies on [MuJoCo](https://mujoco.org/) and [CommonRoad RL](https://commonroad.in.tum.de/commonroad-rl).
2. The implementation of the algorithm is based on the code from [ICRL-benchmark](https://github.com/Guiliang/ICRL-benchmarks-public/tree/main).

## Create Conda Environment
1. please install conda (cuda-11.0)
2. install [Pytorch (version==1.7.1)](https://pytorch.org) in the conda env.
```
conda env create -n cn-py37 python=3.7 -f python_environment.yml
conda activate cn-py37
```

## Set Continuous Environment
1. set MuJoCo (you can get help from [mujoco-tutorial](https://github.com/Guiliang/ICRL-benchmarks-public/blob/main/virtual_env_tutorial.md))
```
pip install -U 'mujoco-py<2.2,>=2.1'
pip install -e ./mujuco_environment
export MUJOCO_PY_MUJOCO_PATH=YOUR_MUJOCO_DIR/.mujoco/mujoco210
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:YOUR_MUJOCO_DIR/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```
2. set commonroad, you can follow the [commonroad tutorial](https://github.com/Guiliang/ICRL-benchmarks-public/blob/main/realisitic_env_tutorial.md)

## Generate Expert Dataset
For the discrete environment:
```
./local_run_pi.sh
```
For the continuous environment:
```
./local_run_ppo.sh
```
After training pi/ppo algorithm, intermediate results will be saved in the save_model folder. You should note that when running generate_data_for_constraint_inference.py, the parameter needs to specify the file name of where the intermediate results are stored.

## Training
```
./server_run_mmicrl_0.sh
```
I train the MMICRL model with 4 × RTX 3090 in continuous environments. For discrete environment simulations, the CPU is also capable of running code rapidly.

## Visualization
```
cd MMICRL
cd interface
python generate_running_plots.py
```
you can see the result in the plot_results folder.

## Welcome to Cite and Star
If you find this idea helpful, please cite it:
```
@inproceedings{
qiao2023MMICRL,
title={Multi-Modal Inverse Constrained Reinforcement Learning from a Mixture of Demonstrations},
author={Guanren Qiao and Guiliang Liu and Pascal Poupart and Zhiqiang Xu},
booktitle={Annual Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=4mPiqh4pLb}
}
```
Besides, you can also consider citing this paper:
```
@inproceedings{
liu2023benchmarking,
title={Benchmarking Constraint Inference in Inverse Reinforcement Learning},
author={Guiliang Liu and Yudong Luo and Ashish Gaurav and Kasra Rezaee and Pascal Poupart},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=vINj_Hv9szL}
}
```
