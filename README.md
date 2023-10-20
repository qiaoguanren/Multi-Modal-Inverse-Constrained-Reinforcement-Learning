# Inverse Constraint RL for Auto-Driving

## commonroad_rl
```
mkdir ./environment
```
This project applies the commonroad_rl environment, please clone their project into to folder ./environment. please refer to the environment setting at
''https://gitlab.lrz.de/tum-cps/commonroad-rl/-/tree/master/''

## stable_baseline3
This project utilizes some implementation in stable_baseline3, but no worry, I have included their code into this project


## Running
To run the code, 

### create env
```
mkdir ./save_model
mkdir ./evaluate_model
source /pkgs/anaconda3/bin/activate
conda env create -n cn-py37 -f python_environment.yml
conda activate cn-py37
```
###  Setup [mujoco](https://github.com/openai/mujoco-py)
```
pip install -U 'mujoco-py<2.2,>=2.1'
pip install -e ./mujuco_environment
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/h/galen/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```
###  Setup [commonroad-rl](https://gitlab.lrz.de/tum-cps/commonroad-rl)
```
mkdir ./commonroad_environment
```
###  prepare the data
```
```

### start the training
```
source your-conda-activate
conda activate your-env
cd ./interface
python train_commonroad_ppo.py ../config/train_ppo_highD.yaml -d 1  # for debug mode
python train_commonroad_ppo.py ../config/train_ppo_highD.yaml # for running
```
