task_name="train-ppo"
launch_time=$(date +"%m-%d-%y-%H:%M:%S")
log_dir="log-local-${task_name}-${launch_time}.out"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/xxx/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
pip install -e ./mujuco_environment/
cd ./interface/
python train_ppo.py ../config/mujuco_mixture_Swimmer-v0/train_me_c-0_ppo_lag_HCWithPos-v0.yaml -n 5 -s 123
python train_ppo.py ../config/mujuco_mixture_Swimmer-v0/train_me_c-1_ppo_lag_HCWithPos-v0.yaml -n 5 -s 123
python generate_data_for_constraint_inference.py -n 5 -mn train_me_c-0_ppo_lag_HCWithPos-v0-multi_env-Apr-02-2023-22:10-seed_123 -tn PPO-Lag-HCWithPos -ld 2 -ct Mixture -nsy 0
python generate_data_for_constraint_inference.py -n 5 -mn train_me_c-1_ppo_lag_HCWithPos-v0-multi_env-Apr-03-2023-02:08-seed_123 -tn PPO-Lag-HCWithPos -ld 2 -ct Mixture -nsy 0
#python train_ppo.py ../config/highD_distance_constraint/train_me_c-0_ppo_lag_highD_distance_constraint.yaml -n 5 -s 123
#python train_ppo.py ../config/highD_distance_constraint/train_me_c-1_ppo_lag_highD_distance_constraint.yaml -n 5 -s 123
#python generate_data_for_constraint_inference.py -n 5 -mn train_me_c-0_ppo_lag_highD_distance_constraint-multi_env-Aug-06-2023-11:34-seed_123 -tn PPO-Lag-highD-distance -ld 2 -ct Mixture -nsy 0
#python generate_data_for_constraint_inference.py -n 5 -mn train_me_c-1_ppo_lag_highD_distance_constraint-multi_env-Aug-07-2023-08:11-seed_123 -tn PPO-Lag-highD-distance -ld 2 -ct Mixture -nsy 0
cd ../
