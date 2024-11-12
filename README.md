# A-CDM 实验版本


### 运行顺序
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