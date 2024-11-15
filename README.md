# Closed-Loop Testing of Autonomous Driving Systems: A Safety-Critical, Effective, and Realistic Evaluation Using the Adversarial Cognitive Driver Model

![Platform](images/platform.png)
## Table of Contents
1. [Environment Setup](#environment-setup)
## Environment Setup
1. Clone the repository
```shell
git clone https://github.com/HCPS-ShanghaiTech/ACDM_open.git
cd ACDM_open
```
2. Create environment && I
3. nstall the dependencies
```shell
conda create -n acdm python=3.10 # Need Python version >= 3.10
conda activate acdm
pip install -r requirement.txt
```
### Quickly Started
```shell
# 切换地图
$ python load_world.py
# 运行系统
$ python main.py 
```
### 场景库输入格式
第一行输入车辆数n: int
第二行输入被测车辆的(init_vel, driving_style(没有则置0), 控制器("CDM", "ACDM", "OTHER"))
接下来的n-1行输入每辆车的(init_vel, lane_id, rel_pos, driving_style(没有则置0), 控制器("CDM", "ACDM", "OTHER"))
lane_id为0在不同车道, 1在相同车道
To be modified...