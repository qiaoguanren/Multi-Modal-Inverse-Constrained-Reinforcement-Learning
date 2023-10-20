task_name="train-gridworld-meicrl"
launch_time=$(date +"%m-%d-%y-%H:%M:%S")
log_dir="log-local-${task_name}-${launch_time}.out"
export MUJOCO_PY_MUJOCO_PATH=/scratch1/PycharmProjects/constraint-learning-mixture-experts/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch1/PycharmProjects/constraint-learning-mixture-experts/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
source /scratch1/miniconda3/bin/activate
conda activate cn-py37
pip install -e ./mujuco_environment/
cd ./interface/
python train_mmicrl.py ../config/mujoco_mixture_WGW-v0/train_MEICRL_WGW-v0.yaml -s 123 -n 1 -l "$log_dir"
cd ../
