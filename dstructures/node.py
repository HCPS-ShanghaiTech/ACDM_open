import utils.globalvalues as gv
import utils.dyglobalvalues as dgv
import utils.extendmath as emath
import itertools
import torch
from envs import CarModule
from typing import Dict, Any
from vehicles.virtualvehicle import VirtualVehicle
from utils.globalvalues import LON_ACC_DICT, STEP_DT


def d_distance_of_lon_action(virtualvehicle, control_action, dT):
    """
    Calculate the longitudinal distance moved by the vehicle after dT seconds under the current action
    """
    acc = LON_ACC_DICT.get(control_action)
    return max(
        virtualvehicle.scalar_velocity * dT + 0.5 * acc * dT * (dT + STEP_DT), 1e-9
    )


class EnumerateTree(CarModule):
    """
    Contains root and leaf nodes.
    """

    def __init__(self, ego_id, main_id) -> None:
        super().__init__(ego_id, main_id)
        self.root = None
        self.leaves = []
        self.num_lon_leaves = 0
        self.num_lat_leaves = 0
        self.probability = 0
        self.is_valid = False

    def generate_root_from_cipo(self, close_vehicle_id_list, lon_levels, lat_levels):
        """
        Generate the Root node from the CIPO Observer.
        """
        virtual_vehicle_dict = {}
        for cid in close_vehicle_id_list:
            virtual_vehicle_dict[cid] = dgv.get_realvehicle(cid).clone_to_virtual()
        self.root = CIPORoot(
            self.ego_id,
            self.main_id,
            virtual_vehicle_dict,
            lon_levels,
            lat_levels,
        )
        self.probability = 1

    def generate_root_from_leaf(self, leaf, lon_levels, lat_levels):
        """
        Generate the Root node from a leaf.
        """
        self.root = CIPORoot(
            self.ego_id,
            self.main_id,
            leaf.virtual_vehicle_dict,
            lon_levels,
            lat_levels,
        )
        self.probability = leaf.risk

    def grow_tree(self):
        """
        Generate leaf nodes from the root node.
        Return the list of leaf nodes and the count of two types of leaf nodes.
        """
        if self.root == None:
            raise ValueError("You have not generated a root node.")

        lon_leaves, num_lon = self.root.generate_leaves("longitude")
        lat_leaves, num_lat = self.root.generate_leaves("lateral")
        self.leaves = lon_leaves + lat_leaves
        self.num_lon_leaves = num_lon
        self.num_lat_leaves = num_lat

        return self.leaves, num_lon, num_lat


class Node(CarModule):
    """
    Node: Represents the driving environment of a given scenario.
    """

    def __init__(self, ego_id, main_id, virtual_vehicle_dict: VirtualVehicle) -> None:
        super().__init__(ego_id, main_id)
        self.virtual_vehicle_dict = (
            virtual_vehicle_dict  # Typically considers only nearby vehicles
        )

    def judge_could_slide(self):
        _, min_dist_side = emath.get_side_closest_vehicle(
            self.virtual_vehicle_dict, self.ego_id
        )
        if abs(min_dist_side) <= (gv.CAR_LENGTH):
            return False
        return True


class Leaf(Node):
    """
    Leaf: Represents the driving environment at the next moment.
    """

    def __init__(
        self,
        leaf_id,
        ego_id,
        main_id,
        virtual_vehicle_dict: Dict[Any, VirtualVehicle],
    ) -> None:
        super().__init__(ego_id, main_id, virtual_vehicle_dict)
        self.risk = 0
        self.leaf_id = leaf_id

    def clone_self(self):
        """
        Addresses limitations of deepcopy with certain data types in CARLA.
        """
        new_vvehicle_dict = {}
        for vid in self.virtual_vehicle_dict.keys():
            new_vvehicle_dict[vid] = self.virtual_vehicle_dict[vid].clone_self()
        return Leaf(
            self.leaf_id,
            self.ego_id,
            self.main_id,
            new_vvehicle_dict,
        )

    def update_dict(self, dict):
        new_vvehicle_dict = {}
        for vid in dict.keys():
            new_vvehicle_dict[vid] = dict[vid].clone_self()
        return Leaf(
            self.leaf_id,
            self.ego_id,
            self.main_id,
            new_vvehicle_dict,
        )

    def upload_input_data(self, root: "CIPORoot"):
        """
        Save the input tensor to root
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_tensor = self.from_perception_to_tensor(self, root).to(device)
        # Generate inputs with and without self
        noego_root, noego_leaf = root.clone_self(), self.clone_self()
        del noego_root.virtual_vehicle_dict[self.ego_id]
        del noego_leaf.virtual_vehicle_dict[self.ego_id]
        input_tensor_noego = self.from_perception_to_tensor(noego_root, noego_leaf).to(
            device
        )
        root.traj_cache.append(input_tensor)
        root.traj_cache.append(input_tensor_noego)

    @staticmethod
    def from_perception_to_tensor(leaf: "Leaf", root: "CIPORoot"):
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


class CIPORoot(Node):
    """
    Root Node: Contains the driving environment of the current time.
    """

    def __init__(
        self,
        ego_id,
        main_id,
        virtual_vehicle_dict: Dict[Any, VirtualVehicle],
        lon_levels,
        lat_levels,
    ) -> None:
        super().__init__(ego_id, main_id, virtual_vehicle_dict)

        self.lon_levels = lon_levels
        self.lat_levels = lat_levels
        self.traj_cache = list()

    def clone_self(self):
        """
        Addresses limitations of deepcopy with certain data types in CARLA.
        """
        new_vvehicle_dict = {}
        for vid in self.virtual_vehicle_dict.keys():
            new_vvehicle_dict[vid] = self.virtual_vehicle_dict[vid].clone_self()
        return CIPORoot(
            self.ego_id,
            self.main_id,
            new_vvehicle_dict,
            self.lon_levels,
            self.lat_levels,
        )

    def generate_leaves(self, dir_type):
        """
        Generate leaf nodes.
        dir_type = "longitude" or "lateral"
        Return all leaf nodes and their count.
        """
        leaves = []
        iter_list = self.generate_iter_list(
            dir_type
        )  # dir_type determines if leaf nodes are longitudinal or lateral
        v_vehicle_id_list = list(self.virtual_vehicle_dict.keys())
        leaf_id = 0
        # Traverse each combination of actions
        for comb in iter_list:
            virtual_vehicle_dict_leaf = {}
            for i in range(len(comb)):
                # Ensure v_action and vid correspond one-to-one
                v_action = comb[i]
                vid = v_vehicle_id_list[i]
                v_vehicle = self.virtual_vehicle_dict.get(vid)
                virtual_vehicle_dict_leaf[vid] = (
                    self.generate_next_step_virtual_vehicle(v_vehicle, v_action)
                )
            leaves.append(
                Leaf(
                    leaf_id,
                    self.ego_id,
                    self.main_id,
                    virtual_vehicle_dict_leaf,
                )
            )
            leaf_id += 1
        return leaves, len(leaves)

    def generate_iter_list(self, dir_type):
        """
        Generates permutations of action space with variable length for each vehicle's action set.
        dir_type = "longitude" or "lateral"
        """
        res = []
        consider_actions = []
        if not self.virtual_vehicle_dict:
            return res
        # Longitudinal actions
        if dir_type == "longitude":
            for vid in self.virtual_vehicle_dict.keys():
                v_vehicle = self.virtual_vehicle_dict.get(vid)
                # For ego vehicle
                if vid == self.ego_id:
                    consider_actions = ["MAINTAIN", "ACCELERATE", "DECELERATE"]
                elif vid in self.lon_levels["Level1"]:
                    consider_actions = ["MAINTAIN", "DECELERATE"]
                elif vid in self.lon_levels["Level2"]:
                    # Two-lane restriction
                    if v_vehicle.waypoint.lane_id == gv.LANE_ID["Left"]:
                        consider_actions = ["MAINTAIN", "SLIDE_RIGHT"]
                    if v_vehicle.waypoint.lane_id == gv.LANE_ID["Right"]:
                        consider_actions = ["MAINTAIN", "SLIDE_LEFT"]
                elif vid in self.lon_levels["Level3"]:
                    consider_actions = ["MAINTAIN"]
                else:
                    consider_actions = ["MAINTAIN"]
                res.append(consider_actions)
        # Lateral actions
        if dir_type == "lateral":
            for vid in self.virtual_vehicle_dict.keys():
                v_vehicle = self.virtual_vehicle_dict.get(vid)
                if vid == self.ego_id:
                    # Two-lane restriction
                    if v_vehicle.waypoint.lane_id == gv.LANE_ID["Left"]:
                        consider_actions = ["SLIDE_RIGHT"]
                    if v_vehicle.waypoint.lane_id == gv.LANE_ID["Right"]:
                        consider_actions = ["SLIDE_LEFT"]
                elif vid == self.lat_levels["Level1"][0]:
                    # At the side-front of the main vehicle
                    consider_actions = ["MAINTAIN", "DECELERATE"]
                elif vid == self.lat_levels["Level1"][1]:
                    # At the side-rear of the main vehicle
                    consider_actions = ["MAINTAIN", "ACCELERATE"]
                elif vid in self.lat_levels["Level2"]:
                    # Two-lane restriction
                    if v_vehicle.waypoint.lane_id == gv.LANE_ID["Left"]:
                        consider_actions = ["MAINTAIN", "SLIDE_RIGHT"]
                    if v_vehicle.waypoint.lane_id == gv.LANE_ID["Right"]:
                        consider_actions = ["MAINTAIN", "SLIDE_LEFT"]
                elif vid in self.lat_levels["Level3"]:
                    consider_actions = ["MAINTAIN"]
                else:
                    consider_actions = ["MAINTAIN"]
                res.append(consider_actions)
        return list(itertools.product(*res))

    def generate_next_step_virtual_vehicle(self, virtual_vehicle, control_action):
        """
        Generate the next step virtual vehicle.
        """
        virtual_vehicle_next = virtual_vehicle.clone_self()
        virtual_vehicle_next.control_action = control_action
        if control_action in ["MAINTAIN", "ACCELERATE", "DECELERATE"]:
            # Calculate forward distance in discrete time
            d_distance = d_distance_of_lon_action(virtual_vehicle, control_action, 1)
            virtual_vehicle_next.waypoint = virtual_vehicle.waypoint.next(d_distance)[0]
        if control_action == "SLIDE_LEFT":
            d_distance = max(virtual_vehicle.scalar_velocity * 1, 1e-9)
            # Waypoint on the left road
            virtual_vehicle_next.waypoint = virtual_vehicle.waypoint.next(d_distance)[
                0
            ].get_left_lane()
            if virtual_vehicle_next.waypoint is None:
                print(virtual_vehicle.waypoint.lane_id, "should be -3")
        if control_action == "SLIDE_RIGHT":
            d_distance = max(virtual_vehicle.scalar_velocity * 1, 1e-9)
            # Waypoint on the right road
            virtual_vehicle_next.waypoint = virtual_vehicle.waypoint.next(d_distance)[
                0
            ].get_right_lane()
            if virtual_vehicle_next.waypoint is None:
                print(virtual_vehicle.waypoint.lane_id, "should be -2")
        virtual_vehicle_next.transform = virtual_vehicle_next.waypoint.transform
        virtual_vehicle_next.scalar_velocity = (
            virtual_vehicle.scalar_velocity + gv.LON_ACC_DICT.get(control_action)
        )
        return virtual_vehicle_next
