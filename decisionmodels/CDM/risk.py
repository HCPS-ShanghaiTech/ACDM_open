from dstructures import EllipseGenerator
import utils.extendmath as emath
import utils.globalvalues as gv
import numpy as np
from envs import CarModule
from typing import List, Tuple
from dstructures import *


class Risk(CarModule):
    """
    Base class for calculating risk, various methods can inherit from this.
    """

    def __init__(self, ego_id, main_id) -> None:
        super().__init__(ego_id, main_id)
        self.risks = []  # Risks correspond one-to-one with leaves

    def get_preferred_leaves(
        self, leaves: List[Leaf], driving_style_risk
    ) -> Tuple[List[Leaf], List[Leaf]]:
        """
        Filter leaf nodes.
        """
        if len(self.risks) == 0:
            raise ValueError("You have not calculated risks.")

        preferred_leaves = []
        other_leaves = []  # Spare, now used for debugging
        for i in range(len(leaves)):
            if driving_style_risk >= self.risks[i]:
                preferred_leaves.append(leaves[i])
            else:
                other_leaves.append(leaves[i])
        # If all leaf nodes are removed, select the one with the lowest risk
        if len(preferred_leaves) == 0:
            preferred_leaves.append(leaves[np.argmin(self.risks)])

        return preferred_leaves, other_leaves


class CDMRisk(Risk):
    """
    Based on factors like elliptical social force, overspeeding, etc.
    """

    def __init__(self, ego_id, main_id) -> None:
        super().__init__(ego_id, main_id)

    def cal_risk_list(self, root: CIPORoot, leaves: List[Leaf]):
        """
        Calculate the risk list, aligned with leaves.
        """
        self.risks = [0 for _ in range(len(leaves))]
        for i in range(len(leaves)):
            social_force, _ = self.cal_social_force_max(root, leaves[i])
            penalty = self.cal_penalty_risk(root, leaves[i])
            ttc_risk = self.cal_ttc_risk(leaves[i])
            self.risks[i] = social_force + penalty + ttc_risk
            leaves[i].risk = self.risks[i]

    def cal_on_time_risk(self, root: CIPORoot, leaves: List[Leaf]):
        """
        Calculate real-time risk for result statistics.
        """
        # Since leaves are generated iteratively, the first leaf node must be all vehicles maintaining status, suitable for calculating current status
        leaf = leaves[0]
        total_sf_risk = self.cal_social_force_sum(root, leaf)
        ttc_risk = self.cal_ttc_risk(root)
        return total_sf_risk + ttc_risk

    def cal_ttc_risk(self, leaf: Leaf):
        """
        Calculate the time-to-collision risk for a single leaf node.
        """
        ego_v_vehicle = leaf.virtual_vehicle_dict.get(self.ego_id)
        min_vid_front, min_dist_front, min_vid_back, min_dist_back = (
            emath.get_lon_closest_vehicle(leaf.virtual_vehicle_dict, self.ego_id)
        )
        ttc_front = -1
        ttc_back = -1
        # Front TTC
        if min_vid_front is not None:
            ttc_front = max(min_dist_front, 0) / (
                -leaf.virtual_vehicle_dict.get(min_vid_front).scalar_velocity
                + ego_v_vehicle.scalar_velocity
                + 1e-9
            )
        # Rear TTC
        if min_vid_back is not None:
            ttc_back = -min(min_dist_back, 0) / (
                leaf.virtual_vehicle_dict.get(min_vid_back).scalar_velocity
                - ego_v_vehicle.scalar_velocity
                - 1e-9
            )
        return self.ttc_to_risk(ttc_front) + self.ttc_to_risk(ttc_back)

    def cal_social_force_sum(self, root: CIPORoot, leaf: Leaf):
        """
        Calculate the social force risk for a single leaf node (cumulative version).
        """
        total_risk = 0

        # Calculate ellipse parameters
        c1_location = root.virtual_vehicle_dict.get(self.ego_id).transform.location
        c2_location = leaf.virtual_vehicle_dict.get(self.ego_id).transform.location

        c = emath.cal_length(emath.cal_rel_location_curve(c2_location, c1_location))

        # Generate ellipse
        ellipse = EllipseGenerator(c1_location, c2_location, c)

        # Calculate risk
        for vid in leaf.virtual_vehicle_dict.keys():
            if vid != self.ego_id:
                car_location = leaf.virtual_vehicle_dict.get(vid).transform.location
                total_risk += ellipse.cal_risk_vector(car_location)
        return total_risk

    def cal_social_force_max(self, root: CIPORoot, leaf: Leaf):
        """
        Calculate the social force risk for a single leaf node (max version).
        Return the maximum risk and corresponding vehicle ID.
        """
        max_risk = 0
        max_vid = None
        # Calculate ellipse parameters
        c1_location = root.virtual_vehicle_dict.get(self.ego_id).transform.location
        c2_location = leaf.virtual_vehicle_dict.get(self.ego_id).transform.location

        c = emath.cal_length(emath.cal_rel_location_curve(c2_location, c1_location)) / 2

        # Generate ellipse
        ellipse = EllipseGenerator(c1_location, c2_location, c)

        # Calculate risk
        for vid in leaf.virtual_vehicle_dict.keys():
            if vid != self.ego_id:
                car_location = leaf.virtual_vehicle_dict.get(vid).transform.location
                elli_risk = ellipse.cal_risk_vector(car_location)
                if elli_risk >= max_risk:
                    max_risk = elli_risk
                    max_vid = vid
        return max_risk, max_vid

    def cal_penalty_risk(self, root: CIPORoot, leaf: Leaf):
        """
        Calculate penalties for overspeeding, lane changing, collision.
        """
        overspeed_penalty = 0
        collision_penalty = 0
        traj_accuracy = 10  # Trajectory accuracy, determining the number of continuous points for collision checking
        ego_vvehicle = leaf.virtual_vehicle_dict.get(self.ego_id)
        # Overspeeding
        speed = ego_vvehicle.scalar_velocity
        speed_threshold = 26.5
        overspeed_penalty_coeff = 3
        if speed > speed_threshold:
            overspeed_penalty = (
                overspeed_penalty_coeff * 10 * ((speed - speed_threshold) * 6 / 41) ** 2
            )

        # Collision
        for vid in leaf.virtual_vehicle_dict.keys():
            if collision_penalty >= 1000:
                break
            # Check for collision at intermediate trajectory points for each vehicle to avoid missing mid-process collisions

            if vid != self.ego_id:
                if (
                    -gv.CAR_LENGTH
                    <= emath.cal_distance_along_road(
                        root.virtual_vehicle_dict.get(self.ego_id).waypoint,
                        root.virtual_vehicle_dict.get(vid).waypoint,
                    )
                    <= gv.CAR_LENGTH
                ) and (
                    leaf.virtual_vehicle_dict.get(self.ego_id).control_action
                    in [
                        "SLIDE_RIGHT",
                        "SLIDE_LEFT",
                    ]
                ):
                    collision_penalty += 1000
                    continue
                assis_vego = root.virtual_vehicle_dict.get(self.ego_id).clone_self()
                assis_vother = root.virtual_vehicle_dict.get(vid).clone_self()
                for _ in range(traj_accuracy):
                    assis_vego.transform.location += (
                        leaf.virtual_vehicle_dict.get(self.ego_id).transform.location
                        - root.virtual_vehicle_dict.get(self.ego_id).transform.location
                    ) / traj_accuracy
                    assis_vother.transform.location += (
                        leaf.virtual_vehicle_dict.get(vid).transform.location
                        - root.virtual_vehicle_dict.get(vid).transform.location
                    ) / traj_accuracy

                    if assis_vego.judge_collision(assis_vother):
                        collision_penalty += 1000
                        break
                del assis_vego
                del assis_vother

        return overspeed_penalty + collision_penalty

    @staticmethod
    def ttc_to_risk(ttc):
        """
        Convert time-to-collision to risk.
        """
        threshold = 9
        if ttc < 0 or ttc > threshold:
            return 0
        return (threshold - ttc) ** 2
