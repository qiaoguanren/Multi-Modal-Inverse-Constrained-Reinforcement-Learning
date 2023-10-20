task_name="train-pi"
launch_time=$(date +"%m-%d-%y-%H:%M:%S")
log_dir="log-local-${task_name}-${launch_time}.out"
source /scratch1/miniconda3/bin/activate
conda activate cn-py37
export MUJOCO_PY_MUJOCO_PATH=/scratch1/PycharmProjects/constraint-learning-mixture-experts/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch1/PycharmProjects/constraint-learning-mixture-experts/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
cd ./interface/
python train_pi.py ../config/mujoco_mixture_WGW-v0/train_me_c-0_pi_lag_WGW-v0.yaml -n 5 -s 123 -l "$log_dir"
python train_pi.py ../config/mujoco_mixture_WGW-v0/train_me_c-1_pi_lag_WGW-v0.yaml -n 5 -s 123 -l "$log_dir"
python generate_data_for_constraint_inference.py -nsy 0 -ct mixture -ld 2 -n 1 -tn PI-Lag-WallGrid -mn train_me_c-0_pi_lag_WGW-v0-Mar-20-2023-11_38-seed_123
python generate_data_for_constraint_inference.py -nsy 0 -ct mixture -ld 2 -n 1 -tn PI-Lag-WallGrid -mn train_me_c-1_pi_lag_WGW-v0-Mar-20-2023-12_38-seed_123
cd ../