# Closed-Loop Testing of Autonomous Driving Systems: A Safety-Critical, Effective, and Realistic Evaluation Using the Adversarial Cognitive Driver Model

![Platform](images/platform.png)
## Table of Contents
1. [Environment Setup](#environment-setup)
## Environment Setup
1. Clone the repository
```bash
git clone https://github.com/HCPS-ShanghaiTech/ACDM_open.git
cd ACDM_open
```
2. Create environment && Install the dependencies
```bash
conda create -n acdm python=3.10 # Need Python version >= 3.10
conda activate acdm
pip install -r requirement.txt
```
## Quickly Started
#### Experiment Scripts
1. Import submodule (needed for all d2rl experiment)
```bash
git submodule update --init
```
2. Run scripts
- For Windows
```bash
winscript/short_experiment.bat
winscript/long_experiment.bat
```
- For Linux
```bash
sh linuxscripts/short_experiment.sh
sh linuxscripts/long_experiment.sh
```
#### Preset Scene Library
1. Load carla world
```python
python load_world.py
```
2. Preset scene example
```python
python main.py -t "short" -s "ttc" -n "acdm" -e "cdm"
```
### 场景库输入格式
第一行输入车辆数n: int
第二行输入被测车辆的(init_vel, driving_style(没有则置0), 控制器("CDM", "ACDM", "OTHER"))
接下来的n-1行输入每辆车的(init_vel, lane_id, rel_pos, driving_style(没有则置0), 控制器("CDM", "ACDM", "OTHER"))
lane_id为0在不同车道, 1在相同车道
To be modified...