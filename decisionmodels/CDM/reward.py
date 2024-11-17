from utils.globalvalues import ACTION_SCORE
from envs import CarModule
from dstructures import *
from typing import List
from dstructures import EllipseGenerator
import utils.extendmath as emath
import utils.globalvalues as gv
import torch
import math


class Reward(CarModule):
    """
    Base class for calculating rewards, other methods can inherit from this.
    """

    def __init__(self, ego_id, main_id) -> None:
        super().__init__(ego_id, main_id)
        self.reward_dict = {
            "MAINTAIN": 0,
            "ACCELERATE": 0,
            "DECELERATE": 0,
            "SLIDE_LEFT": 0,
            "SLIDE_RIGHT": 0,
        }


class CDMReward(Reward):
    """
    Original CDM calculation method.
    """

    def __init__(self, ego_id, main_id) -> None:
        super().__init__(ego_id, main_id)

    def cal_reward_dict(self, preferred_leaves: List[Leaf], other_leaves: List[Leaf]):
        """
        Scoring method.
        """
        self.reward_dict = {
            "MAINTAIN": [],
            "ACCELERATE": [],
            "DECELERATE": [],
            "SLIDE_LEFT": [],
            "SLIDE_RIGHT": [],
        }
        for leaf in preferred_leaves:
            self.reward_dict[
                leaf.virtual_vehicle_dict[self.ego_id].control_action
            ].append(
                ACTION_SCORE[leaf.virtual_vehicle_dict[self.ego_id].control_action]
            )
        for leaf in other_leaves:
            self.reward_dict[
                leaf.virtual_vehicle_dict[self.ego_id].control_action
            ].append(0)
        return self.reward_dict


class ACDMReward(Reward):
    """
    ACDM rewards.
    """

    def __init__(self, ego_id, main_id) -> None:
        super().__init__(ego_id, main_id)

    def cal_reward_dict(
        self, root, preferred_leaves, other_leaves, action_batch
    ) -> dict:
        """
        Scoring method.
        """
        # Clear previous rewards
        self.reward_dict = {
            "MAINTAIN": [],
            "ACCELERATE": [],
            "DECELERATE": [],
            "SLIDE_LEFT": [],
            "SLIDE_RIGHT": [],
        }
        # Calculate rewards
        ab_idx = 0
        for leaf in preferred_leaves:
            (
                self.reward_dict[
                    leaf.virtual_vehicle_dict[self.ego_id].control_action
                ].append(
                    self.cal_in_obs_reward(root, leaf)
                    * (
                        self.cal_mainrisk_reward(leaf)
                        + self.cal_traj_reward(
                            root, leaf, action_batch[ab_idx], action_batch[ab_idx + 1]
                        )
                    )  # Distribute output from batch
                    + ACTION_SCORE[
                        leaf.virtual_vehicle_dict[self.ego_id].control_action
                    ]
                    * 0.001
                )
            )
            ab_idx += 2
        # Add other nodes
        for leaf in other_leaves:
            self.reward_dict[
                leaf.virtual_vehicle_dict[self.ego_id].control_action
            ].append(0)
        return self.reward_dict

    def cal_mainrisk_reward(self, leaf):
        """
        Calculate reward based on ACDM's social force risk for the main vehicle at a single leaf node.
        """
        main_vvehicle = leaf.virtual_vehicle_dict.get(self.main_id)
        if main_vvehicle:
            # Calculate ellipse parameters
            c1_location = main_vvehicle.transform.location
            # Project forward 1 second
            c2_location = main_vvehicle.waypoint.next(main_vvehicle.scalar_velocity)[
                0
            ].transform.location

            c = (
                emath.cal_length(emath.cal_rel_location_curve(c2_location, c1_location))
                / 2
            )

            # Generate ellipse
            ellipse = EllipseGenerator(c1_location, c2_location, c)

            # Calculate risk for main vehicle
            car_location = leaf.virtual_vehicle_dict.get(self.ego_id).transform.location
            main_risk_reward = ellipse.cal_risk_vector(car_location)
        else:
            main_risk_reward = 0

        return main_risk_reward

    def cal_in_obs_reward(self, root, leaf):
        """
        Give a bonus if close to the main vehicle.
        """
        bonus = 1
        ego_vvehicle = leaf.virtual_vehicle_dict.get(self.ego_id)
        main_vvehicle = leaf.virtual_vehicle_dict.get(self.main_id)
        if main_vvehicle:
            if (
                gv.CAR_LENGTH
                + max(
                    0,
                    root.virtual_vehicle_dict.get(self.ego_id).scalar_velocity
                    - root.virtual_vehicle_dict.get(self.main_id).scalar_velocity,
                )
                <= emath.cal_distance_along_road(
                    main_vvehicle.waypoint, ego_vvehicle.waypoint
                )
                <= gv.OBSERVE_DISTANCE
            ):
                return bonus
            else:
                return 0
        else:
            return 0

    def cal_traj_reward(self, root, leaf, action, action_noego):
        """
        Predict changes in the trajectory of the test vehicle based on neural network output.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        action_dict = {0: 0, 1: 1, 2: -3, 3: 4}
        # Return 0 if not observed by main
        if (
            self.main_id not in leaf.virtual_vehicle_dict.keys()
            or self.main_id not in root.virtual_vehicle_dict.keys()
            or self.main_id == self.ego_id
        ):
            return 0
        # Neural network output
        action = torch.max(action, 0).indices.item()
        action_noego = torch.max(action_noego, 0).indices.item()
        coeff = 10
        if action != action_noego:
            if action == 3 or action_noego == 3:
                return (
                    math.sqrt(action_dict[action] ** 2 + action_dict[action_noego] ** 2)
                    * coeff
                )
            else:
                return abs(action_dict[action] - action_dict[action_noego]) * coeff
        else:
            return 0

    def upload_input_data(self, root: CIPORoot, leaf: Leaf):
        """
        Save the input tensor to root
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_tensor = self.from_perception_to_tensor(root, leaf).to(device)
        # Generate inputs with and without self
        noego_root, noego_leaf = root.clone_self(), leaf.clone_self()
        del noego_root.virtual_vehicle_dict[self.ego_id]
        del noego_leaf.virtual_vehicle_dict[self.ego_id]
        input_tensor_noego = self.from_perception_to_tensor(noego_root, noego_leaf).to(
            device
        )
        root.traj_cache.append(input_tensor, input_tensor_noego)

    @staticmethod
    def from_perception_to_tensor(root, leaf):
        """
        Generate corresponding neural network input from perception.
        """
        dhw_ub, dhw_lb, mainv_ub, mainv_lb, relx_ub, relx_lb = 150, 0, 50, 20, 100, 0
        seq_len = 25
        res_tensor = torch.ones(seq_len, 5) * -9999
        traj_vvehicle_dict = {}
        mean_velocity = (
            root.virtual_vehicle_dict.get(root.main_id).scalar_velocity
            + leaf.virtual_vehicle_dict.get(leaf.main_id).scalar_velocity
        ) / 2
        mv_tensor = (mean_velocity - mainv_lb) / (mainv_ub - mainv_lb)
        for i in range(seq_len):
            # Infer intermediate process based on start and end positions, do not update on the first step
            if i == 0:
                for vid in root.virtual_vehicle_dict.keys():
                    traj_vvehicle_dict[vid] = root.virtual_vehicle_dict.get(
                        vid
                    ).clone_self()
            else:
                # Handling curves is more troublesome, assume crossing the lane line at 0.5s
                for vid in traj_vvehicle_dict.keys():
                    update_times = int((1 / gv.STEP_DT) // seq_len)
                    for _ in range(update_times):
                        traj_vvehicle_dict.get(vid).scalar_velocity += (
                            gv.LON_ACC_DICT.get(
                                leaf.virtual_vehicle_dict.get(vid).control_action
                            )
                            * gv.STEP_DT
                        )
                        wp_gap = (
                            traj_vvehicle_dict.get(vid).scalar_velocity * gv.STEP_DT
                        )
                        if wp_gap == 0:
                            print(wp_gap)
                        if wp_gap > 0:
                            traj_vvehicle_dict.get(vid).waypoint = (
                                traj_vvehicle_dict.get(vid).waypoint.next(wp_gap)[0]
                            )
                        if wp_gap < 0:
                            traj_vvehicle_dict.get(vid).waypoint = (
                                traj_vvehicle_dict.get(vid).waypoint.previous(wp_gap)[0]
                            )
                    if i > seq_len // 2:
                        if (
                            leaf.virtual_vehicle_dict.get(vid).control_action
                            == "SLIDE_LEFT"
                            and traj_vvehicle_dict.get(vid).waypoint.lane_id
                            == gv.LANE_ID["Right"]
                        ):
                            traj_vvehicle_dict.get(vid).waypoint = (
                                traj_vvehicle_dict.get(vid).waypoint.get_left_lane()
                            )
                        if (
                            leaf.virtual_vehicle_dict.get(vid).control_action
                            == "SLIDE_RIGHT"
                            and traj_vvehicle_dict.get(vid).waypoint.lane_id
                            == gv.LANE_ID["Left"]
                        ):
                            traj_vvehicle_dict.get(vid).waypoint = (
                                traj_vvehicle_dict.get(vid).waypoint.get_right_lane()
                            )
            # Front and nearest side lane vehicle IDs and relative distances
            front_vid, dhw = emath.get_front_closest_vehicle(
                traj_vvehicle_dict, root.main_id
            )
            if front_vid is None:
                dhw_tensor, ttc_tensor = -9999, -9999
            else:
                dhw_tensor = (dhw_ub - abs(dhw - dhw_lb)) / (dhw_ub - dhw_lb)
                ttc_tensor = (
                    traj_vvehicle_dict.get(root.main_id).scalar_velocity
                    - traj_vvehicle_dict.get(front_vid).scalar_velocity
                ) / (dhw - gv.CAR_LENGTH + 1e-9)

            side_vid, relx = emath.get_side_closest_vehicle(
                traj_vvehicle_dict, root.main_id
            )
            if side_vid is None:
                relx_tensor, sidettc_tensor = -9999, -9999
            else:
                relx_tensor = (abs(relx - relx_lb)) / (relx_ub - relx_lb)
                sidettc_tensor = (
                    traj_vvehicle_dict.get(root.main_id).scalar_velocity
                    - traj_vvehicle_dict.get(side_vid).scalar_velocity
                    + 1e-9
                ) / abs(relx + 1e-9)
                sidettc_tensor = min(sidettc_tensor, 1)
            res_tensor[i] = torch.Tensor(
                [ttc_tensor, dhw_tensor, mv_tensor, relx_tensor, sidettc_tensor]
            )
        return res_tensor
