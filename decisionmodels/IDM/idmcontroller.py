import utils.extendmath as emath
import utils.globalvalues as gv
import utils.dyglobalvalues as dgv
import math
import numpy as np
from decisionmodels.CDM.observer import Observer


class IntelligentDriverModel:
    """
    IDM(car following) + MOBIL(lane changing) Model
    """

    def __init__(
        self,
        observer: "IDMObserver",
        time_gap,
        desired_speed,
        max_acceleration,
        minimum_gap,
        acc_exp,
        comf_deceleration,
        politness,
    ):
        self.observer = observer
        self.time_gap = time_gap
        self.desired_speed = desired_speed
        self.max_acceleration = max_acceleration
        # The minimum acceptable distance from the preceding vehicle
        self.minimum_gap = minimum_gap
        self.acc_exp = acc_exp  # δ
        self.comf_deceleration = comf_deceleration
        self.control_acceleration = 0  # control action
        self.politness = politness  # Responsibility coefficient

    def get_control_acceleration(self, front_dist, ego_v, delta_v):
        """
        Calculate the acceleration for the next step (longitudinal)
        """
        MINI_speed = 70 / 3.6
        if delta_v:
            # Desired Gap
            front_dist = max(front_dist, 0.01)
            s_star = self.minimum_gap + max(
                0,
                (
                    ego_v * self.time_gap
                    + (ego_v * delta_v)
                    / (
                        2
                        * math.sqrt(abs(self.max_acceleration * self.comf_deceleration))
                    )
                ),
            )
            # Control Acceleration
            control_acceleration = self.max_acceleration * (
                1
                - math.pow(ego_v / self.desired_speed, self.acc_exp)
                - (math.pow(s_star, 2) / front_dist)
            )
        else:
            # Control Acceleration
            control_acceleration = self.max_acceleration * (
                1 - math.pow(ego_v / self.desired_speed, self.acc_exp)
            )

        if ego_v + control_acceleration <= MINI_speed:
            control_acceleration = (MINI_speed - ego_v) * gv.DECISION_DT

        return control_acceleration

    def run_forward(self, realvehicle_id_list, **kwargs):
        (
            front_id,
            front_dist,
            ego_v,
            delta_v,
            back_id,
            back_dist,
            sidef_id,
            sidef_dist,
            sideb_id,
            sideb_dist,
        ) = self.observer.observe(realvehicle_id_list)
        # Lane change
        LCthreshold = 2.5

        # Control Acceleration
        self.control_acceleration = np.clip(
            self.get_control_acceleration(front_dist, ego_v, delta_v),
            self.comf_deceleration,
            self.max_acceleration,
        )

        # Lane change judgment
        # Cannot change lanes when there are cars nearby
        if sidef_dist <= 0 or sideb_dist >= 0:
            return self.control_acceleration

        # Acceleration benefits after self lane change
        if sidef_id:
            ego_lc_delta_v = ego_v - dgv.get_realvehicle(sidef_id).scalar_velocity

        else:
            sidef_dist = ego_lc_delta_v = None
        benefit_ego_acc = (
            np.clip(
                self.get_control_acceleration(sidef_dist, ego_v, ego_lc_delta_v),
                self.comf_deceleration,
                self.max_acceleration,
            )
            - self.control_acceleration
        )

        # Rear acceleration benefit
        if back_id:
            back_v = dgv.get_realvehicle(back_id).scalar_velocity
            back_delta_v = back_v - ego_v
            back_acc = np.clip(
                self.get_control_acceleration(abs(back_dist), back_v, back_delta_v),
                self.comf_deceleration,
                self.max_acceleration,
            )
            if front_id:
                back_lc_delta_v = back_v - dgv.get_realvehicle(front_id).scalar_velocity
                back_lc_dist = abs(front_dist) + abs(back_dist) + gv.CAR_LENGTH
            else:
                back_lc_delta_v = back_lc_dist = None
            back_lc_acc = np.clip(
                self.get_control_acceleration(back_lc_dist, back_v, back_lc_delta_v),
                self.comf_deceleration,
                self.max_acceleration,
            )
            benefit_back_acc = back_lc_acc - back_acc
        else:
            benefit_back_acc = 0

        # Side rear acceleration benefit
        if sideb_id:
            sideb_v = dgv.get_realvehicle(sideb_id).scalar_velocity
            if sidef_id:
                sideb_delta_v = sideb_v - dgv.get_realvehicle(sidef_id).scalar_velocity
                sideb_oi_dist = abs(sidef_dist) + abs(sideb_dist) + gv.CAR_LENGTH
            else:
                sideb_delta_v = sideb_oi_dist = None
            sideb_acc = np.clip(
                self.get_control_acceleration(sideb_oi_dist, sideb_v, sideb_delta_v),
                self.comf_deceleration,
                self.max_acceleration,
            )
            sideb_lc_delta_v = sideb_v - ego_v
            sideb_lc_dist = abs(sideb_dist)
            sideb_lc_acc = np.clip(
                self.get_control_acceleration(sideb_lc_dist, sideb_v, sideb_lc_delta_v),
                self.comf_deceleration,
                self.max_acceleration,
            )
            benefit_sideb_acc = sideb_lc_acc - sideb_acc
        else:
            benefit_sideb_acc = 0

        # Calculate whether to change lanes or not
        if (
            benefit_ego_acc + self.politness * (benefit_back_acc + benefit_sideb_acc)
            > LCthreshold
        ):
            return "SLIDE"
        return self.control_acceleration


class IDMObserver(Observer):
    def __init__(self, ego_id, main_id) -> None:
        super().__init__(ego_id, main_id)

    def observe(self, realvehicle_id_list, mode="real"):
        """
        Controller parameters
        """
        (
            front_id,
            front_dist,
            back_id,
            back_dist,
            sidef_id,
            sidef_dist,
            sideb_id,
            sideb_dist,
        ) = self.get_closest_vehicles(realvehicle_id_list, mode)
        # The distance between the front car and the bumper
        front_dist -= gv.CAR_LENGTH
        back_dist += gv.CAR_LENGTH
        sidef_dist -= gv.CAR_LENGTH
        sideb_dist += gv.CAR_LENGTH
        # Speed gap
        ego_vehicle = dgv.get_realvehicle(self.ego_id)
        front_vehicle = dgv.get_realvehicle(front_id)

        if not front_vehicle:
            front_dist = delta_v = None
        else:
            delta_v = ego_vehicle.scalar_velocity - front_vehicle.scalar_velocity

        return (
            front_id,
            front_dist,
            ego_vehicle.scalar_velocity,
            delta_v,
            back_id,
            back_dist,
            sidef_id,
            sidef_dist,
            sideb_id,
            sideb_dist,
        )

    def get_closest_vehicles(self, realvehicle_id_list, mode="real"):
        """
        mode = real || virtual, Represents whether the input dictionary is a real vehicle or a virtual vehicle
        """
        self.close_vehicle_id_list = self.get_close_vehicle_id_list(realvehicle_id_list)
        ego_vehicle = dgv.get_realvehicle(self.ego_id)
        min_dist_front, min_dist_back, min_dist_sidef, min_dist_sideb = (
            1e9,
            -1e9,
            1e9,
            -1e9,
        )
        min_id_front, min_id_back, min_id_sidef, min_id_sideb = None, None, None, None
        if mode == "real":
            ego_wp = dgv.get_map().get_waypoint(ego_vehicle.vehicle.get_location())
        elif mode == "virtual":
            ego_wp = ego_vehicle.waypoint
        else:
            raise ValueError("The mode value is wrong!")
        for rvid in self.close_vehicle_id_list:
            if rvid != self.ego_id:
                vehicle = dgv.get_realvehicle(rvid)
                if mode == "real":
                    veh_wp = dgv.get_map().get_waypoint(vehicle.vehicle.get_location())
                elif mode == "virtual":
                    veh_wp = vehicle.waypoint
                else:
                    raise ValueError("The mode value is wrong!")
                rel_distance = emath.cal_distance_along_road(ego_wp, veh_wp)
                if veh_wp.lane_id == ego_wp.lane_id:
                    # Front
                    if 0 < rel_distance < min_dist_front:
                        min_dist_front = rel_distance
                        min_id_front = rvid
                    # Back
                    if min_dist_back < rel_distance < 0:
                        min_dist_back = rel_distance
                        min_id_back = rvid
                if veh_wp.lane_id != ego_wp.lane_id:
                    # Side Front
                    if 0 < rel_distance < min_dist_front:
                        min_dist_sidef = rel_distance
                        min_id_sidef = rvid
                    # Side Back
                    if min_dist_sideb < rel_distance < 0:
                        min_dist_sideb = rel_distance
                        min_id_sideb = rvid
        return (
            min_id_front,
            min_dist_front,
            min_id_back,
            min_dist_back,
            min_id_sidef,
            min_dist_sidef,
            min_id_sideb,
            min_dist_sideb,
        )
