"""
一些需要的数学计算方法
"""

import carla
import math
import utils.dyglobalvalues as dgv
from utils.globalvalues import (
    ROADS_LENGTH,
    ROUND_LENGTH,
    STEP_DT,
    LON_ACC_DICT,
)


def cal_yaw_by_location(target_location, start_location):
    """
    将一个相对位置向量转换为朝向
    """
    direction_vector = target_location - start_location
    return math.degrees(math.atan2(direction_vector.y, direction_vector.x))


def cal_magnitude(location):
    """
    计算location中(x, y)的长度的平方
    """
    return carla.Vector2D(x=location.x, y=location.y).squared_length()


def cal_length(location):
    """
    计算location中(x, y)长度
    """
    return carla.Vector2D(x=location.x, y=location.y).length()


def cal_total_in_round_length(input_wp):
    """
    计算一圈中离起点的距离
    """
    if ROADS_LENGTH.get(input_wp.road_id):
        return input_wp.s + ROADS_LENGTH.get(input_wp.road_id)[1]
    else:
        return 0


def is_in_front(transform_a, transform_b):
    """
    几何方法判断a车是否在b车的前方
    """
    # 获取车辆A和B的位置坐标
    pos_a = transform_a.location
    pos_b = transform_b.location

    # 获取车辆B的朝向向量
    forward_vector_b = transform_b.get_forward_vector()

    # 计算车辆A相对于车辆B的位置向量
    relative_vector = pos_a - pos_b

    # 计算向量的点积
    dot_product = forward_vector_b.dot(relative_vector)

    # 如果点积大于0，则车辆A在车辆B的前方
    return dot_product > 0


def cal_distance_along_road(start_wp, target_wp):
    """
    判断两车沿着路面的相对位置, target相对start
    """
    start_s, target_s = cal_total_in_round_length(start_wp), cal_total_in_round_length(
        target_wp
    )

    # 假设两点处于同一圈
    distance_case_1 = target_s - start_s

    # 假设start waypoint在第一圈， target waypoint在第二圈
    distance_case_2 = target_s + ROUND_LENGTH - start_s

    # 假设start waypoint在第二圈， target waypoint在第一圈
    distance_case_3 = target_s - (start_s + ROUND_LENGTH)

    return min([distance_case_1, distance_case_2, distance_case_3], key=abs)


def get_real_vehicle_waypoint(real_vehicle):
    return dgv.get_map().get_waypoint(real_vehicle.vehicle.get_location())


def cal_rel_location_curve(target_location, start_location):
    """
    计算在弯曲道路上target相对于start的位置, 将弯曲道路拉直再计算
    """
    carla_map = dgv.get_map()
    start_wp, target_wp = carla_map.get_waypoint(
        start_location
    ), carla_map.get_waypoint(target_location)
    result_location = carla.Location(0, 0, 0)
    # 目前只考虑两车道, 所以没考虑左右关系
    result_location.y = abs(start_wp.lane_id - target_wp.lane_id) * start_wp.lane_width
    # z坐标其实用不上
    result_location.z = target_location.z - start_location.z
    # x坐标代表沿路线的距离
    result_location.x = cal_distance_along_road(start_wp, target_wp)

    return result_location


def d_distance_of_lon_action(virtualvehicle, control_action, dT):
    """计算当前action之下dT秒后车辆前进的纵向距离"""
    acc = LON_ACC_DICT.get(control_action)
    # 等差数列求和近似加速度计算公式（场景离散更新）
    return max(
        virtualvehicle.scalar_velocity * dT + 0.5 * acc * dT * (dT + STEP_DT), 1e-9
    )


def get_lon_closest_vehicle(virtual_vehicle_dict, ego_id):
    """
    筛选出前后方最近车辆（推演阶段）
    """
    ego_v_vehicle = virtual_vehicle_dict.get(ego_id)
    min_dist_front = 1e9
    min_dist_back = -1e9
    min_vid_front = None
    min_vid_back = None
    ego_lane_id = ego_v_vehicle.waypoint.lane_id
    # 对每辆车判断是否处于同一车道且前后距离最短
    for vid in virtual_vehicle_dict.keys():
        if vid != ego_id:
            v_vehicle = virtual_vehicle_dict.get(vid)
            rel_distance = cal_distance_along_road(
                ego_v_vehicle.waypoint, v_vehicle.waypoint
            )
            # 前方
            if v_vehicle.waypoint.lane_id == ego_lane_id:
                # 前方车辆
                if 0 < rel_distance < min_dist_front:
                    min_dist_front = rel_distance
                    min_vid_front = vid
                # 后方车辆
                if min_dist_back < rel_distance < 0:
                    min_dist_back = rel_distance
                    min_vid_back = vid
    # 返回前后方最近车辆与其相对距离
    return min_vid_front, min_dist_front, min_vid_back, min_dist_back


def get_front_closest_vehicle(virtual_vehicle_dict, ego_id):
    """
    筛选出前方最近车辆（推演阶段）
    """
    ego_v_vehicle = virtual_vehicle_dict.get(ego_id)
    min_dist_front = 1e9
    min_vid_front = None
    ego_lane_id = ego_v_vehicle.waypoint.lane_id
    # 对每辆车判断是否处于同一车道且在前方距离最短
    for vid in virtual_vehicle_dict.keys():
        if vid != ego_id:
            v_vehicle = virtual_vehicle_dict.get(vid)
            rel_distance = cal_distance_along_road(
                ego_v_vehicle.waypoint, v_vehicle.waypoint
            )
            if v_vehicle.waypoint.lane_id == ego_lane_id:
                # 前方车辆
                if 0 < rel_distance < min_dist_front:
                    min_dist_front = rel_distance
                    min_vid_front = vid
    # 返回前方最近车辆与其相对距离
    return min_vid_front, min_dist_front


def get_side_closest_vehicle(virtual_vehicle_dict, ego_id):
    """
    筛选出相邻车道最近车辆（推演阶段）
    """
    ego_v_vehicle = virtual_vehicle_dict.get(ego_id)
    min_dist_side = 1e9
    min_vid_side = None
    ego_lane_id = ego_v_vehicle.waypoint.lane_id
    # 对每辆车判断是否处于不同车道且绝对距离最短
    for vid in virtual_vehicle_dict.keys():
        if vid != ego_id:
            v_vehicle = virtual_vehicle_dict.get(vid)
            rel_distance = cal_distance_along_road(
                ego_v_vehicle.waypoint, v_vehicle.waypoint
            )
            if v_vehicle.waypoint.lane_id != ego_lane_id:
                if abs(rel_distance) < min_dist_side:
                    min_dist_side = abs(rel_distance)
                    min_vid_side = vid
    # 返回相邻车道最近车辆与其相对距离
    return min_vid_side, min_dist_side


def add_height_to_waypoint(waypoint_transform, vehicle_height):
    """
    沿道路法向量添加高度值
    """
    # 获取waypoint的位置和旋转
    location = waypoint_transform.location
    rotation = waypoint_transform.rotation

    # 计算法线向量
    pitch_rad = math.radians(rotation.pitch)
    normal_vector = carla.Location(x=-math.sin(pitch_rad), y=0, z=math.cos(pitch_rad))

    # 根据车辆高度沿法线方向进行偏移
    location.x += normal_vector.x * vehicle_height
    location.y += normal_vector.y * vehicle_height
    location.z += normal_vector.z * vehicle_height

    # 返回新的位置
    return location


def normalize(input_dict):
    """
    对字典的value(list)做归一化(输入字典的value必须是非负列表)
    """
    norm_dict = input_dict.copy()
    for key in input_dict.keys():
        value_sum = sum(input_dict[key])
        # 全为0时, 不做处理
        if value_sum != 0:
            for i in range(len(input_dict[key])):
                norm_dict[key][i] /= value_sum
    return norm_dict


def add_dicts(dict_list, weight_list):
    """
    输入一系列相同规格的字典(value为列表), 将其value逐个按比例相加, 比例必须全为正数
    """
    if not dict_list:
        raise RuntimeError("Dictionary is empty.")
    assert len(dict_list) == len(weight_list)
    res_dict = dict_list[0].copy()
    num_dict = len(weight_list)
    # 调整weight_list, 以第一个值为基准
    if weight_list[0] != 1:
        for i in range(num_dict):
            weight_list[i] /= weight_list[0]

    for key in res_dict.keys():
        for i in range(len(dict_list[0][key])):
            for j in range(1, num_dict):
                res_dict[key][i] += weight_list[j] * dict_list[j][key][i]

    return res_dict


def waypoints_are_same_location(waypoint1, waypoint2, tolerance=1e-4):
    """判断两个wp是不是在一个位置"""
    location1 = waypoint1.transform.location
    location2 = waypoint2.transform.location

    # 比较位置
    if (
        abs(location1.x - location2.x) < tolerance
        and abs(location1.y - location2.y) < tolerance
        and abs(location1.z - location2.z) < tolerance
    ):
        return True
    else:
        return False
