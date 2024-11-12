"""
生成三辆车的场景
"""

import math
import csv
import random
import sys

sys.path.append("..")
import utils.globalvalues as gv
import utils.dyglobalvalues as dgv

import main as m

ttc_scene_path = "../scenelibrary/shortscenes/ttc_scene.csv"
ttc_cdm_path = "../scenelibrary/shortscenes/ttc_scene_cdm.csv"
ttc_idm_path = "../scenelibrary/shortscenes/ttc_scene_idm.csv"
ttc_na_cdm_path = "../scenelibrary/shortscenes/ttc_scene_na_cdm.csv"
ttc_na_idm_path = "../scenelibrary/shortscenes/ttc_scene_na_idm.csv"
risk_scene_path = "../scenelibrary/shortscenes/risk_scene.csv"
action_scene_path = "../scenelibrary/shortscenes/action_scene.csv"
min_dist_abs = 10
max_dist_abs = 30
min_ttc = 1
max_ttc = 10


def clear_csv(file_path):
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([])


def random_scenes_action(ego_mode, npc_mode, num_npcs, world, vehicle_bp):
    # 先随机筛选主车信息
    ego_speed = random.randint(20, 30)
    ego_v = random.randint(25, 40)
    data = [[num_npcs + 1]]
    data.append([str(ego_speed), str(ego_v), ego_mode])
    # 主车生成点
    carla_map = world.get_map()
    main_spawn_point = carla_map.get_spawn_points()[264]
    full_realvehicle_dict = {}
    init_vel, driving_style, control_mode = ego_speed, ego_v, ego_mode
    # 生成车辆
    vehicle_bp.set_attribute("color", "0, 0, 0")
    main_vehicle = world.spawn_actor(vehicle_bp, main_spawn_point)
    main_id = main_vehicle.id
    # 配置控制器
    full_realvehicle_dict[main_id] = m.set_cdm_vehicle(
        main_vehicle,
        driving_style,
        main_id,
        carla_map,
        main_spawn_point,
        init_vel,
    )
    for i in range(num_npcs):
        npc_speed = random.randint(20, 30)
        npc_lane = random.randint(0, 1)
        npc_dist = random.randint(-40, 40)
        npc_v = random.randint(30, 60)
        data.append(
            [
                str(npc_speed),
                str(npc_lane),
                str(npc_dist),
                str(npc_v),
                str(npc_mode),
            ]
        )
        ego_data = data[1]
        for i in range(2, len(data)):
            npc_i = data[i]
            if int(npc_i[1]) == 0:
                # 不同车道不能并排
                if int(npc_i[2]) == 0:
                    for carid in full_realvehicle_dict.keys():
                        full_realvehicle_dict[carid].vehicle.destroy()
                    return
            if int(npc_i[1]) == 1:
                # 与主车在一条车道
                if abs(int(npc_i[2])) <= min_dist_abs:
                    for carid in full_realvehicle_dict.keys():
                        full_realvehicle_dict[carid].vehicle.destroy()
                    return
                ttc = int(npc_i[2]) / max(int(ego_data[0]) - int(npc_i[0]), 1e-9)
                if ttc <= min_ttc:
                    for carid in full_realvehicle_dict.keys():
                        full_realvehicle_dict[carid].vehicle.destroy()
                    return
            for j in range(i + 1, len(data)):
                npc_j = data[j]
                if npc_i[1] == npc_j[1]:
                    # npc在同一条车道
                    if abs(int(npc_i[2]) - int(npc_j[2])) <= min_dist_abs:
                        for carid in full_realvehicle_dict.keys():
                            full_realvehicle_dict[carid].vehicle.destroy()
                        return
                    ttc = (int(npc_i[2]) - int(npc_j[2])) / max(
                        int(npc_j[0]) - int(npc_i[0]), 1e-9
                    )
                    if ttc <= min_ttc:
                        for carid in full_realvehicle_dict.keys():
                            full_realvehicle_dict[carid].vehicle.destroy()
                        return
        init_vel, lane_id, rel_pos, driving_style, control_mode = (
            npc_speed,
            npc_lane,
            npc_dist,
            npc_v,
            npc_mode,
        )
        vehicle_bp.set_attribute("color", gv.COLOR_DICT[gv.COLOR_LIST[i]])
        spawn_point = m.get_spawn_point(
            carla_map.get_waypoint(main_spawn_point.location), lane_id, rel_pos
        )
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        full_realvehicle_dict[vehicle.id] = m.set_cdm_vehicle(
            vehicle,
            driving_style,
            main_id,
            carla_map,
            spawn_point,
            init_vel,
        )

    step = -5
    world.tick()
    while step < 1:
        if step >= 0:
            # CDM决策
            for carid in full_realvehicle_dict.keys():
                car = full_realvehicle_dict.get(carid)
                if car.vehicle.id == main_id:
                    car.run_step(full_realvehicle_dict, network=None)
            # 终止运行
            for carid in full_realvehicle_dict.keys():
                full_realvehicle_dict[carid].vehicle.destroy()
        world.tick()
        step += 1
    reward_dict = full_realvehicle_dict.get(
        main_id
    ).controller.reward_calculator.reward_dict
    available_actions_count = 0
    for ak in reward_dict.keys():
        if reward_dict[ak] > 0:
            available_actions_count += 1
    if available_actions_count <= 1 and available_actions_count > 0:
        return data
    else:
        return


def random_scenes_risk(ego_mode, npc_mode, num_npcs):
    # 先随机筛选主车信息
    ego_speed = random.randint(20, 30)
    ego_v = random.randint(25, 40)
    data = [[num_npcs + 1]]
    data.append([str(ego_speed), str(ego_v), ego_mode])
    # 写入npc
    for _ in range(num_npcs):
        npc_speed = random.randint(20, 30)
        npc_lane = random.randint(0, 1)
        npc_dist = random.randint(-40, 40)
        npc_v = random.randint(30, 60)
        data.append(
            [
                str(npc_speed),
                str(npc_lane),
                str(npc_dist),
                str(npc_v),
                str(npc_mode),
            ]
        )
    ego_data = data[1]
    for i in range(2, len(data)):
        npc_i = data[i]
        if int(npc_i[1]) == 0:
            # 不同车道不能并排
            if int(npc_i[2]) == 0:
                return
            dist = math.sqrt(abs(int(npc_i[2])) ** 2 + gv.LANE_WIDTH**2)
            if dist > max_dist_abs:
                return
        if int(npc_i[1]) == 1:
            # 与主车在一条车道
            if abs(int(npc_i[2])) <= min_dist_abs:
                return
            ttc = int(npc_i[2]) / max(int(ego_data[0]) - int(npc_i[0]), 1e-9)
            dist = abs(int(npc_i[2]))
            if ttc <= min_ttc:
                return
            if dist > max_dist_abs:
                return
        for j in range(i + 1, len(data)):
            npc_j = data[j]
            if npc_i[1] == npc_j[1]:
                # npc在同一条车道
                if abs(int(npc_i[2]) - int(npc_j[2])) <= min_dist_abs:
                    return
                ttc = (int(npc_i[2]) - int(npc_j[2])) / max(
                    int(npc_j[0]) - int(npc_i[0]), 1e-9
                )
                if ttc <= min_ttc:
                    return
    print(data)
    return data


def random_scenes_ttc(ego_mode, npc_mode, num_npcs):
    # 先随机筛选主车信息
    ego_speed = random.randint(20, 30)
    ego_v = random.randint(25, 40)
    data = [[num_npcs + 1]]
    data.append([str(ego_speed), str(ego_v), ego_mode])
    # 写入npc
    for _ in range(num_npcs):
        npc_speed = random.randint(20, 30)
        npc_lane = random.randint(0, 1)
        npc_dist = random.randint(-40, 40)
        npc_v = random.randint(30, 60)
        data.append(
            [
                str(npc_speed),
                str(npc_lane),
                str(npc_dist),
                str(npc_v),
                str(npc_mode),
            ]
        )
    ego_data = data[1]
    for i in range(2, len(data)):
        npc_i = data[i]
        if int(npc_i[1]) == 0:
            # 不同车道不能并排
            if int(npc_i[2]) == 0:
                return
        if int(npc_i[1]) == 1:
            # 与主车在一条车道
            if abs(int(npc_i[2])) <= min_dist_abs:
                return
            ttc = int(npc_i[2]) / max(int(ego_data[0]) - int(npc_i[0]), 1e-9)
            if ttc > max_ttc or ttc <= min_ttc:
                return
        for j in range(i + 1, len(data)):
            npc_j = data[j]
            if npc_i[1] == npc_j[1]:
                # npc在同一条车道
                if abs(int(npc_i[2]) - int(npc_j[2])) <= min_dist_abs:
                    return
                ttc = (int(npc_i[2]) - int(npc_j[2])) / max(
                    int(npc_j[0]) - int(npc_i[0]), 1e-9
                )
                if ttc > max_ttc or ttc <= min_ttc:
                    return
    print(data)
    return data


if __name__ == "__main__":
    clear_csv(ttc_scene_path)
    random.seed(2024)
    with open(ttc_cdm_path, "w", newline="") as f:
        writer = csv.writer(f)
        scene_count = 0
        while scene_count < 100:
            data = random_scenes_ttc("CDM", "ACDM", 2)
            if data:
                writer.writerows(data)
                scene_count += 1
    random.seed(2024)
    with open(ttc_idm_path, "w", newline="") as f:
        writer = csv.writer(f)
        scene_count = 0
        while scene_count < 100:
            data = random_scenes_ttc("IDM", "ACDM", 2)
            if data:
                writer.writerows(data)
                scene_count += 1
    random.seed(2024)
    with open(ttc_na_cdm_path, "w", newline="") as f:
        writer = csv.writer(f)
        scene_count = 0
        while scene_count < 100:
            data = random_scenes_ttc("CDM", "IDM", 2)
            if data:
                writer.writerows(data)
                scene_count += 1
    random.seed(2024)
    with open(ttc_na_idm_path, "w", newline="") as f:
        writer = csv.writer(f)
        scene_count = 0
        while scene_count < 100:
            data = random_scenes_ttc("IDM", "IDM", 2)
            if data:
                writer.writerows(data)
                scene_count += 1
    # clear_csv(risk_scene_path)
    # random.seed(20240801)
    # with open(risk_scene_path, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     scene_count = 0
    #     while scene_count < 100:
    #         data = random_scenes_risk("CDM", "ACDM", 2)
    #         if data:
    #             writer.writerows(data)
    #             scene_count += 1
    # random.seed(202408)
    # (
    #     client,
    #     world,
    #     carla_map,
    #     vehicle_bp,
    #     spectator,
    #     pred_net,
    # ) = m.initialize()
    # with open(action_scene_path, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     scene_count = 0
    #     while scene_count < 100:
    #         data = random_scenes_action("CDM", "ACDM", 2, world, carla_map, vehicle_bp)
    #         if data:
    #             writer.writerows(data)
    #             scene_count += 1
# def select_one_scene(num_cars, main_s, main_v, )

# with open("score.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(header)
#     # write the data
#     writer.writerow(data)
