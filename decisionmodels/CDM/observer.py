import math
import utils.globalvalues as gv
import utils.dyglobalvalues as dgv
from utils.extendmath import cal_distance_along_road
from envs import CarModule
from typing import Dict, Any
from vehicles import RealVehicle


class Observer(CarModule):
    """
    Observer: Used to acquire known environmental information.
    """

    def __init__(self, ego_id, main_id) -> None:
        super().__init__(ego_id, main_id)
        self.close_vehicle_id_list = []

    def judge_if_close_to_ego(
        self,
        other_vehicle_id,
        mode="real",
    ):
        """
        Determine if the vehicle is close to the ego vehicle.
        """
        if mode == "real":
            other_wp = dgv.get_map().get_waypoint(
                dgv.get_realvehicle(other_vehicle_id).vehicle.get_location()
            )
            ego_wp = dgv.get_map().get_waypoint(
                dgv.get_realvehicle(self.ego_id).vehicle.get_location()
            )
        elif mode == "virtual":
            other_wp = dgv.get_realvehicle(other_vehicle_id).clone_to_virtual().waypoint
            ego_wp = dgv.get_realvehicle(self.ego_id).clone_to_virtual().waypoint
        else:
            raise ValueError("The mode value is wrong!")
        lat_distance = cal_distance_along_road(ego_wp, other_wp)
        if abs(lat_distance) <= gv.OBSERVE_DISTANCE:
            return True
        else:
            return False

    def get_close_vehicle_id_list(
        self,
        realvehicle_id_list,
        in_sight_vehicle_id_dict: Dict[Any, RealVehicle] = {},
        mode="real",
    ):
        """
        Filter and return IDs of vehicles that are close.
        """
        res = []
        if len(in_sight_vehicle_id_dict) == 0:
            for vehicle_id in realvehicle_id_list:
                if self.judge_if_close_to_ego(vehicle_id, mode):
                    res.append(vehicle_id)
        elif len(in_sight_vehicle_id_dict) > 0:
            for vehicle_id in in_sight_vehicle_id_dict.keys():
                if self.judge_if_close_to_ego(vehicle_id, mode):
                    res.append(vehicle_id)
        return res

    def get_car_in_sight_id_list(self, realvehicle_id_list, mode="real"):
        """
        Return a list of vehicle IDs within a 120-degree forward field of view.
        """
        fov_rad = math.radians(gv.FOV)
        in_sight_dict = {}

        ego_vehicle = dgv.get_realvehicle(self.ego_id).vehicle

        ego_location = ego_vehicle.get_location()
        ego_transform = ego_vehicle.get_transform()
        ego_forward = ego_transform.get_forward_vector()

        for vehicle_id in realvehicle_id_list:
            vehicle_data = dgv.get_realvehicle(vehicle_id)
            if vehicle_id == self.ego_id:
                in_sight_dict[vehicle_id] = vehicle_data
                continue

            other_vehicle = vehicle_data.vehicle

            other_location = other_vehicle.get_location()
            direction_vector = other_location - ego_location
            direction_vector.z = 0  # Ignore Z-axis, consider X and Y only

            # Calculate the angle
            dot = (
                ego_forward.x * direction_vector.x + ego_forward.y * direction_vector.y
            )
            det = (
                ego_forward.x * direction_vector.y - ego_forward.y * direction_vector.x
            )
            angle = math.atan2(det, dot)

            # Check if within FOV
            if -fov_rad / 2 <= angle <= fov_rad / 2:
                in_sight_dict[vehicle_id] = vehicle_data

        return in_sight_dict


class FullObserver(Observer):
    """Full observation"""

    def __init__(self, ego_id, main_id) -> None:
        super().__init__(ego_id, main_id)

    def observe(self, realvehicle_id_list):
        self.close_vehicle_id_list = self.get_close_vehicle_id_list(realvehicle_id_list)
        return self.close_vehicle_id_list


class PartialObserver(Observer):
    """Partial observation"""

    def __init__(self, ego_id, main_id) -> None:
        super().__init__(ego_id, main_id)

    def observe(self, realvehicle_id_list):
        in_sight_vehicle_id_dict = self.get_car_in_sight_id_list()
        self.close_vehicle_id_list = self.get_close_vehicle_id_list(
            realvehicle_id_list, in_sight_vehicle_id_dict
        )
        return self.close_vehicle_id_list


class CIPO_Observer(Observer):
    """Level-based observation"""

    def __init__(self, ego_id, main_id) -> None:
        super().__init__(ego_id, main_id)
        # Two different sets of rules for longitudinal and lateral observations
        self.lon_levels = {"Level1": [None], "Level2": [], "Level3": []}
        self.lat_levels = {"Level1": [None, None], "Level2": [], "Level3": []}

    def observe_partial(self):
        """Return: Close vehicle sequence, longitudinal CIPO sequence, lateral CIPO sequence"""
        in_sight_vehicle_id_dict = self.get_car_in_sight_id_list()
        self.get_cipo_vehicle_id_dict(in_sight_vehicle_id_dict, "real")
        return self.close_vehicle_id_list, self.lon_levels, self.lat_levels

    def observe_full(self, realvehicle_id_list, mode="real"):
        """Return: Close vehicle sequence, longitudinal CIPO sequence, lateral CIPO sequence"""
        self.get_cipo_vehicle_id_dict(realvehicle_id_list, mode)
        return self.close_vehicle_id_list, self.lon_levels, self.lat_levels

    def get_cipo_vehicle_id_dict(self, realvehicle_id_list, mode):
        """Filter CIPO level of surrounding vehicles"""
        self.close_vehicle_id_list = []
        self.lon_levels = {"Level1": [None], "Level2": [], "Level3": []}
        self.lat_levels = {
            "Level1": [None, None],
            "Level2": [],
            "Level3": [],
        }  # Level1 here is divided into two cases, pre-allocate space
        # Firstly filter Level1
        min_dhw_lon, min_dhw_lat_pos = self.get_leve1_vehicle_id_list(
            realvehicle_id_list, mode
        )
        # Then filter Level2 and Level3
        self.get_remain_levels_vehicle_id_list(min_dhw_lon, min_dhw_lat_pos, mode)

    def get_leve1_vehicle_id_list(self, realvehicle_id_list, mode):
        """First filter level1 vehicles, facilitating subsequent level filtering"""
        min_dhw_lat_pos = 1e9
        min_dhw_lat_neg = 1e9
        min_dhw_lon = 1e9  # Distance to the closest vehicle ahead
        self.close_vehicle_id_list = self.get_close_vehicle_id_list(
            realvehicle_id_list, {}, mode
        )
        for vehicle_id in self.close_vehicle_id_list:
            if mode == "real":
                vehicle_wp = dgv.get_map().get_waypoint(
                    dgv.get_realvehicle(vehicle_id).vehicle.get_location()
                )
                ego_wp = dgv.get_map().get_waypoint(
                    dgv.get_realvehicle(self.ego_id).vehicle.get_location()
                )
            elif mode == "virtual":
                vehicle_wp = dgv.get_realvehicle(vehicle_id).clone_to_virtual().waypoint
                ego_wp = dgv.get_realvehicle(self.ego_id).clone_to_virtual().waypoint
            # Distance of vehicle relative to ego along the road
            lat_distance = cal_distance_along_road(ego_wp, vehicle_wp)
            # For Node generated by longitudinal actions, select the closest vehicle ahead in the current lane
            if (
                vehicle_wp.lane_id == ego_wp.lane_id
                and lat_distance > 0
                and lat_distance < min_dhw_lon
            ):
                self.lon_levels["Level1"][0] = vehicle_id
                min_dhw_lon = lat_distance
            # For Node generated by lateral actions, select vehicles ahead/behind in adjacent lanes
            if abs(vehicle_wp.lane_id - ego_wp.lane_id) == 1:
                # Select the closest vehicle ahead in the adjacent lane
                if lat_distance >= 0 and lat_distance < min_dhw_lat_pos:
                    self.lat_levels["Level1"][0] = vehicle_id
                    min_dhw_lat_pos = lat_distance
                # Select the closest vehicle behind in the adjacent lane
                if lat_distance < 0 and -lat_distance < min_dhw_lat_neg:
                    self.lat_levels["Level1"][1] = vehicle_id
                    min_dhw_lat_neg = lat_distance
        return min_dhw_lon, min_dhw_lat_pos

    def get_remain_levels_vehicle_id_list(
        self, min_dhw_lon, min_dhw_lat_pos, mode="real"
    ):
        """Filter remaining level vehicles"""
        for vehicle_id in self.close_vehicle_id_list:
            if mode == "real":
                vehicle_wp = dgv.get_map().get_waypoint(
                    dgv.get_realvehicle(vehicle_id).vehicle.get_location()
                )
                ego_wp = dgv.get_map().get_waypoint(
                    dgv.get_realvehicle(self.ego_id).vehicle.get_location()
                )
            elif mode == "virtual":
                vehicle_wp = dgv.get_realvehicle(vehicle_id).clone_to_virtual().waypoint
                ego_wp = dgv.get_realvehicle(self.ego_id).clone_to_virtual().waypoint
            # Distance of vehicle relative to ego along the road
            lat_distance = cal_distance_along_road(ego_wp, vehicle_wp)
            # In the case of longitudinal Node, select vehicles in the side-front of adjacent lanes
            if (
                vehicle_id not in self.lon_levels["Level1"]
                and vehicle_id != self.ego_id
            ):
                if (
                    abs(vehicle_wp.lane_id - ego_wp.lane_id) == 1
                    and lat_distance >= 0
                    and lat_distance <= min_dhw_lon
                ):
                    self.lon_levels["Level2"].append(vehicle_id)
                else:
                    self.lon_levels["Level3"].append(vehicle_id)
            # In the case of lateral Node, select vehicles ahead in the same lane
            if (
                vehicle_id not in self.lat_levels["Level1"]
                and vehicle_id != self.ego_id
            ):
                if (
                    vehicle_wp.lane_id == ego_wp.lane_id
                    and lat_distance >= 0
                    and lat_distance <= min_dhw_lat_pos
                ):
                    self.lat_levels["Level2"].append(vehicle_id)
                else:
                    self.lat_levels["Level3"].append(vehicle_id)
