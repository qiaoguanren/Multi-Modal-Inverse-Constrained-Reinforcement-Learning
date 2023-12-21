#!/bin/bash
task_name="train-mujoco_HCWithPos_123"
launch_time=$(date +"%m-%d-%y-%H:%M:%S")
log_dir="log-server-${task_name}-${launch_time}.out"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/xxx/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
pip install -e ./mujuco_environment/
cd ./interface/
export CUDA_VISIBLE_DEVICES=3
python train_mmicrl.py ../config/mujuco_mixture_HCWithPos-v0/train_MMICRL_HCWithPos-v0.yaml -n 5 -s 123 -l "$log_dir"
python train_mmicrl.py ../config/mujoco_mixture_WGW-v0/train_MMICRL_WGW-v0.yaml -n 1 -s 123 -l "$log_dir"
cd ../