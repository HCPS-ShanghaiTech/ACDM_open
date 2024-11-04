from utils.globalvalues import OBSERVE_DISTANCE
from utils.globalvalues import FOV
from utils.extendmath import cal_distance_along_road
from envs.world import CarModule
import copy
import math


class Observer(CarModule):
    """
    感知器：用于获取已知环境信息。
    """

    def __init__(self, ego_id, main_id, map) -> None:
        super().__init__(ego_id, main_id, map)
        self._close_vehicle_id_list = []

    def judge_if_close_to_ego(
        self, full_vehicle_id_dict, other_vehicle_id, mode="real"
    ):
        """
        判断是否距离自身较近
        """
        if mode == "real":
            other_wp = self._map.get_waypoint(
                full_vehicle_id_dict[other_vehicle_id]._vehicle.get_location()
            )
            ego_wp = self._map.get_waypoint(
                full_vehicle_id_dict[self._ego_id]._vehicle.get_location()
            )
        elif mode == "virtual":
            other_wp = full_vehicle_id_dict[other_vehicle_id]._waypoint
            ego_wp = full_vehicle_id_dict[self._ego_id]._waypoint
        else:
            raise ValueError("The mode value is wrong!")
        lat_distance = cal_distance_along_road(ego_wp, other_wp)
        if abs(lat_distance) <= OBSERVE_DISTANCE:
            return True
        else:
            return False

    def get_close_vehicle_id_list(
        self, full_vehicle_id_dict, in_sight_vehicle_id_dict={}, mode="real"
    ):
        """
        筛选距离较近的车辆id
        """
        res = []
        if len(in_sight_vehicle_id_dict) == 0:
            for vehicle_id in full_vehicle_id_dict.keys():
                if self.judge_if_close_to_ego(full_vehicle_id_dict, vehicle_id, mode):
                    res.append(vehicle_id)
        elif len(in_sight_vehicle_id_dict) > 0:
            for vehicle_id in in_sight_vehicle_id_dict.keys():
                if self.judge_if_close_to_ego(full_vehicle_id_dict, vehicle_id, mode):
                    res.append(vehicle_id)
        return res

    def get_car_in_sight_id_list(self, full_vehicle_id_dict, mode="real"):
        """
        返回在车辆前方120度视角内的所有车辆的ID列表
        """
        fov_rad = math.radians(FOV)
        in_sight_dict = {}

        ego_vehicle = full_vehicle_id_dict[self._ego_id]._vehicle

        ego_location = ego_vehicle.get_location()
        ego_transform = ego_vehicle.get_transform()
        ego_yaw = ego_transform.rotation.yaw
        ego_forward = ego_transform.get_forward_vector()

        for vehicle_id, vehicle_data in full_vehicle_id_dict.items():
            if vehicle_id == self._ego_id:
                in_sight_dict[vehicle_id] = vehicle_data
                continue

            other_vehicle = vehicle_data._vehicle

            other_location = other_vehicle.get_location()
            direction_vector = other_location - ego_location
            direction_vector.z = 0  # 忽略Z轴，仅考虑X和Y

            # 计算角度
            dot = (
                ego_forward.x * direction_vector.x + ego_forward.y * direction_vector.y
            )
            det = (
                ego_forward.x * direction_vector.y - ego_forward.y * direction_vector.x
            )
            angle = math.atan2(det, dot)

            # 检查是否在 FOV 内
            if -fov_rad / 2 <= angle <= fov_rad / 2:
                in_sight_dict[vehicle_id] = vehicle_data

        return in_sight_dict


class FullObserver(Observer):
    """全观测"""

    def __init__(self, ego_id, main_id, map) -> None:
        super().__init__(ego_id, main_id, map)

    def observe(self, full_vehicle_id_dict):
        self._close_vehicle_id_list = self.get_close_vehicle_id_list(
            full_vehicle_id_dict
        )
        return self._close_vehicle_id_list


class PartialObserver(Observer):
    """部分观测"""

    def __init__(self, ego_id, main_id, map) -> None:
        super().__init__(ego_id, main_id, map)

    def observe(self, full_vehicle_id_dict):
        in_sight_vehicle_id_dict = self.get_car_in_sight_id_list(full_vehicle_id_dict)
        # print(in_sight_vehicle_id_dict.keys())
        self._close_vehicle_id_list = self.get_close_vehicle_id_list(
            full_vehicle_id_dict, in_sight_vehicle_id_dict
        )
        # print(self._close_vehicle_id_list)
        return self._close_vehicle_id_list


class CIPO_Observer(Observer):
    """有等级的观测"""

    def __init__(self, ego_id, main_id, map) -> None:
        super().__init__(ego_id, main_id, map)
        # 横向与纵向分别两种不同的规则
        self._lon_levels = {"Level1": [None], "Level2": [], "Level3": []}
        self._lat_levels = {"Level1": [None, None], "Level2": [], "Level3": []}

    def observe_partial(self, full_vehicle_id_dict):
        """返回: 较近车辆序列, 纵向CIPO序列, 横向CIPO序列"""
        in_sight_vehicle_id_dict = self.get_car_in_sight_id_list(full_vehicle_id_dict)
        # print(in_sight_vehicle_id_dict.keys())
        self.get_cipo_vehicle_id_dict(in_sight_vehicle_id_dict, "real")
        # print(self._ego_id, self._lon_levels, self._lat_levels)
        return self._close_vehicle_id_list, self._lon_levels, self._lat_levels

    def observe_full(self, full_vehicle_id_dict, mode="real"):
        """返回: 较近车辆序列, 纵向CIPO序列, 横向CIPO序列"""
        self.get_cipo_vehicle_id_dict(full_vehicle_id_dict, mode)
        # print(self._ego_id, self._lon_levels, self._lat_levels)
        return self._close_vehicle_id_list, self._lon_levels, self._lat_levels

    def get_cipo_vehicle_id_dict(self, full_vehicle_id_dict, mode):
        """筛选周围车辆的CIPO等级"""
        self._close_vehicle_id_list = []
        self._lon_levels = {"Level1": [None], "Level2": [], "Level3": []}
        self._lat_levels = {
            "Level1": [None, None],
            "Level2": [],
            "Level3": [],
        }  # 此处Level1分前后两种情况，提前分配空间
        # 首先筛选Level1
        min_dhw_lon, min_dhw_lat_pos = self.get_leve1_vehicle_id_list(
            full_vehicle_id_dict, mode
        )
        # 然后筛选Level2和Level3
        self.get_remain_levels_vehicle_id_list(
            min_dhw_lon, min_dhw_lat_pos, full_vehicle_id_dict, mode
        )

    def get_leve1_vehicle_id_list(self, full_vehicle_id_dict, mode):
        """首先筛选一轮level1的车辆, 方便后续等级的车辆筛选"""
        min_dhw_lat_pos = 1e9
        min_dhw_lat_neg = 1e9
        min_dhw_lon = 1e9  # 前方最近车辆的距离
        self._close_vehicle_id_list = self.get_close_vehicle_id_list(
            full_vehicle_id_dict, {}, mode
        )
        for vehicle_id in self._close_vehicle_id_list:
            if mode == "real":
                vehicle_wp = self._map.get_waypoint(
                    full_vehicle_id_dict[vehicle_id]._vehicle.get_location()
                )
                ego_wp = self._map.get_waypoint(
                    full_vehicle_id_dict[self._ego_id]._vehicle.get_location()
                )
            elif mode == "virtual":
                vehicle_wp = full_vehicle_id_dict[vehicle_id]._waypoint
                ego_wp = full_vehicle_id_dict[self._ego_id]._waypoint
            # vehicle相对ego的沿线距离
            lat_distance = cal_distance_along_road(ego_wp, vehicle_wp)
            # 对于即将由纵向动作生成的Node, 选取当前车道前方最近的车辆
            if (
                vehicle_wp.lane_id == ego_wp.lane_id
                and lat_distance > 0
                and lat_distance < min_dhw_lon
            ):
                self._lon_levels["Level1"][0] = vehicle_id
                min_dhw_lon = lat_distance
            # 对于即将由横向动作生成的Node, 选取相邻车道前后方的车辆
            if abs(vehicle_wp.lane_id - ego_wp.lane_id) == 1:
                # 选取相邻车道前方最近车辆
                if lat_distance >= 0 and lat_distance < min_dhw_lat_pos:
                    self._lat_levels["Level1"][0] = vehicle_id
                    min_dhw_lat_pos = lat_distance
                # 选取相邻车道后方最近车辆
                if lat_distance < 0 and -lat_distance < min_dhw_lat_neg:
                    self._lat_levels["Level1"][1] = vehicle_id
                    min_dhw_lat_neg = lat_distance
        return min_dhw_lon, min_dhw_lat_pos

    def get_remain_levels_vehicle_id_list(
        self, min_dhw_lon, min_dhw_lat_pos, full_vehicle_id_dict, mode="real"
    ):
        """筛选剩余level车辆"""
        for vehicle_id in self._close_vehicle_id_list:
            if mode == "real":
                vehicle_wp = self._map.get_waypoint(
                    full_vehicle_id_dict[vehicle_id]._vehicle.get_location()
                )
                ego_wp = self._map.get_waypoint(
                    full_vehicle_id_dict[self._ego_id]._vehicle.get_location()
                )
            elif mode == "virtual":
                vehicle_wp = full_vehicle_id_dict[vehicle_id]._waypoint
                ego_wp = full_vehicle_id_dict[self._ego_id]._waypoint
            # vehicle相对ego的沿线距离
            lat_distance = cal_distance_along_road(ego_wp, vehicle_wp)
            # 纵向Node的情况，选取相邻车道侧前方车辆
            if (
                vehicle_id not in self._lon_levels["Level1"]
                and vehicle_id != self._ego_id
            ):
                if (
                    abs(vehicle_wp.lane_id - ego_wp.lane_id) == 1
                    and lat_distance >= 0
                    and lat_distance <= min_dhw_lon
                ):
                    self._lon_levels["Level2"].append(vehicle_id)
                else:
                    self._lon_levels["Level3"].append(vehicle_id)
            # 横向Node的情况，选取同车道前方车辆
            if (
                vehicle_id not in self._lat_levels["Level1"]
                and vehicle_id != self._ego_id
            ):
                if (
                    vehicle_wp.lane_id == ego_wp.lane_id
                    and lat_distance >= 0
                    and lat_distance <= min_dhw_lat_pos
                ):
                    self._lat_levels["Level2"].append(vehicle_id)
                else:
                    self._lat_levels["Level3"].append(vehicle_id)
