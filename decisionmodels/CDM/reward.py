from utils.globalvalues import ACTION_SCORE
from envs.world import CarModule
from dstructures.ellipse import EllipseGenerator
from collections import Counter
import utils.extendmath as emath
import utils.globalvalues as gv
import torch
import math


class Reward(CarModule):
    """
    计算reward的基类, 其他方法可继承于此
    """

    def __init__(self, ego_id, main_id, map) -> None:
        super().__init__(ego_id, main_id, map)
        self._reward_dict = {
            "MAINTAIN": 0,
            "ACCELERATE": 0,
            "DECELERATE": 0,
            "SLIDE_LEFT": 0,
            "SLIDE_RIGHT": 0,
        }


class CDMReward(Reward):
    """
    原始CDM的计算方法
    """

    def __init__(self, ego_id, main_id, map) -> None:
        super().__init__(ego_id, main_id, map)

    def cal_reward_dict(self, preferred_leaves, other_leaves):
        """
        赋分法
        """
        self._reward_dict = {
            "MAINTAIN": [],
            "ACCELERATE": [],
            "DECELERATE": [],
            "SLIDE_LEFT": [],
            "SLIDE_RIGHT": [],
        }
        for leaf in preferred_leaves:
            self._reward_dict[
                leaf._virtual_vehicle_dict[self._ego_id]._control_action
            ].append(
                ACTION_SCORE[leaf._virtual_vehicle_dict[self._ego_id]._control_action]
            )
        for leaf in other_leaves:
            self._reward_dict[
                leaf._virtual_vehicle_dict[self._ego_id]._control_action
            ].append(0)
        # print(self._ego_id, self._reward_dict)
        return self._reward_dict


class ACDMReward(Reward):
    """
    ACDM的reward
    """

    def __init__(self, ego_id, main_id, map) -> None:
        super().__init__(ego_id, main_id, map)

    def cal_reward_dict(self, root, preferred_leaves, other_leaves, nnet):
        """
        赋分法
        """
        # 清空
        self._reward_dict = {
            "MAINTAIN": [],
            "ACCELERATE": [],
            "DECELERATE": [],
            "SLIDE_LEFT": [],
            "SLIDE_RIGHT": [],
        }
        # 计算
        for leaf in preferred_leaves:
            (
                self._reward_dict[
                    leaf._virtual_vehicle_dict[self._ego_id]._control_action
                ].append(
                    self.cal_in_obs_reward(root, leaf)
                    * (
                        self.cal_mainrisk_reward(leaf) * 1
                        + self.cal_traj_reward(root, leaf, nnet) * 1
                    )
                    + ACTION_SCORE[
                        leaf._virtual_vehicle_dict[self._ego_id]._control_action
                    ]
                    * 0.001
                )
            )
        # 添加其他节点
        for leaf in other_leaves:
            self._reward_dict[
                leaf._virtual_vehicle_dict[self._ego_id]._control_action
            ].append(0)
        # print(self._ego_id, self._reward_dict)
        return self._reward_dict

    def cal_mainrisk_reward(self, leaf):
        """
        计算单个叶子结点下ACDM对主车社会力风险所导致的reward
        """
        main_vvehicle = leaf._virtual_vehicle_dict.get(self._main_id)
        if main_vvehicle:
            # 计算椭圆参数
            c1_location = main_vvehicle._transform.location
            # 往后再推演1s
            c2_location = main_vvehicle._waypoint.next(main_vvehicle._scalar_velocity)[
                0
            ].transform.location

            c = (
                emath.cal_length(
                    emath.cal_rel_location_curve(self._map, c2_location, c1_location)
                )
                / 2
            )

            # 生成椭圆
            ellipse = EllipseGenerator(self._map, c1_location, c2_location, c)

            # 计算对main vehicle的risk
            car_location = leaf._virtual_vehicle_dict.get(
                self._ego_id
            )._transform.location
            main_risk_reward = ellipse.cal_risk_vector(car_location)
        else:
            main_risk_reward = 0

        return main_risk_reward

    def cal_in_obs_reward(self, root, leaf):
        """
        若距离main vehicle较近, 给予bonus。
        """
        bonus = 1
        ego_vvehicle = leaf._virtual_vehicle_dict.get(self._ego_id)
        main_vvehicle = leaf._virtual_vehicle_dict.get(self._main_id)
        if main_vvehicle:
            if (
                gv.CAR_LENGTH
                + max(
                    0,
                    root._virtual_vehicle_dict.get(self._ego_id)._scalar_velocity
                    - root._virtual_vehicle_dict.get(self._main_id)._scalar_velocity,
                )
                <= emath.cal_distance_along_road(
                    main_vvehicle._waypoint, ego_vvehicle._waypoint
                )
                <= gv.OBSERVE_DISTANCE
            ):
                return bonus
            else:
                return 0
        else:
            return 0

    def cal_traj_reward(self, root, leaf, nnet):
        """
        根据神经网络的输出预测被测车辆的轨迹变化。
        """
        device = "cuda" if torch.cuda.is_available else "cpu"
        action_dict = {0: 0, 1: 1, 2: -3, 3: 4}
        # 自己不在main的观测中, 直接返回0
        if (
            self._main_id not in leaf._virtual_vehicle_dict.keys()
            or self._main_id not in root._virtual_vehicle_dict.keys()
            or self._main_id == self._ego_id
        ):
            return 0
        # 生成包含自己和不包含自己两种输入
        input_tensor = (
            self.from_perception_to_tensor(root, leaf).reshape(1, 25, 5).to(device)
        )
        noego_root, noego_leaf = root.clone_self(), leaf.clone_self()
        del noego_root._virtual_vehicle_dict[self._ego_id]
        del noego_leaf._virtual_vehicle_dict[self._ego_id]
        input_tensor_noego = (
            self.from_perception_to_tensor(noego_root, noego_leaf)
            .reshape(1, 25, 5)
            .to(device)
        )
        # 神经网络输出
        nnet.eval()
        with torch.no_grad():
            action = nnet(input_tensor, input_tensor == -9999)
            action_noego = nnet(input_tensor_noego, input_tensor_noego == -9999)
        # print(torch.max(action, 1).indices.item(), torch.max(action_noego, 1).indices)
        action = torch.max(action, 1).indices.item()
        action_noego = torch.max(action_noego, 1).indices.item()

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

    @staticmethod
    def from_perception_to_tensor(root, leaf):
        """
        根据perception生成对应的神经网络的输入。
        """
        dhw_ub, dhw_lb, mainv_ub, mainv_lb, relx_ub, relx_lb = 150, 0, 50, 20, 100, 0
        seq_len = 25
        res_tensor = torch.ones(seq_len, 5) * -9999
        traj_vvehicle_dict = {}
        mean_velocity = (
            root._virtual_vehicle_dict.get(root._main_id)._scalar_velocity
            + leaf._virtual_vehicle_dict.get(leaf._main_id)._scalar_velocity
        ) / 2
        mv_tensor = (mean_velocity - mainv_lb) / (mainv_ub - mainv_lb)
        for i in range(seq_len):
            # 根据首尾位置推断中间过程, 第一步不更新
            if i == 0:
                for vid in root._virtual_vehicle_dict.keys():
                    traj_vvehicle_dict[vid] = root._virtual_vehicle_dict.get(
                        vid
                    ).clone_self()
                    # traj_vvehicle_dict.get(vid)._waypoint = root._map.get_waypoint(
                    #     traj_vvehicle_dict.get(vid)._transform.location
                    # )
            else:
                # 弯道处理起来比较麻烦, 假设0.5s时跨过车道线
                for vid in traj_vvehicle_dict.keys():
                    update_times = int((1 / gv.STEP_DT) // seq_len)
                    for _ in range(update_times):
                        traj_vvehicle_dict.get(vid)._scalar_velocity += (
                            gv.LON_ACC_DICT.get(
                                leaf._virtual_vehicle_dict.get(vid)._control_action
                            )
                            * gv.STEP_DT
                        )
                        wp_gap = (
                            traj_vvehicle_dict.get(vid)._scalar_velocity * gv.STEP_DT
                        )
                        if wp_gap == 0:
                            print(wp_gap)
                        if wp_gap > 0:
                            traj_vvehicle_dict.get(vid)._waypoint = (
                                traj_vvehicle_dict.get(vid)._waypoint.next(wp_gap)[0]
                            )
                        if wp_gap < 0:
                            traj_vvehicle_dict.get(vid)._waypoint = (
                                traj_vvehicle_dict.get(vid)._waypoint.previous(wp_gap)[
                                    0
                                ]
                            )
                    if i > seq_len // 2:
                        if (
                            leaf._virtual_vehicle_dict.get(vid)._control_action
                            == "SLIDE_LEFT"
                            and traj_vvehicle_dict.get(vid)._waypoint.lane_id
                            == gv.LANE_ID["Right"]
                        ):
                            traj_vvehicle_dict.get(vid)._waypoint = (
                                traj_vvehicle_dict.get(vid)._waypoint.get_left_lane()
                            )
                        if (
                            leaf._virtual_vehicle_dict.get(vid)._control_action
                            == "SLIDE_RIGHT"
                            and traj_vvehicle_dict.get(vid)._waypoint.lane_id
                            == gv.LANE_ID["Left"]
                        ):
                            traj_vvehicle_dict.get(vid)._waypoint = (
                                traj_vvehicle_dict.get(vid)._waypoint.get_right_lane()
                            )
            # 前方和相邻车道最近的车辆id与相对距离
            front_vid, dhw = emath.get_front_closest_vehicle(
                traj_vvehicle_dict, root._main_id
            )
            if front_vid == None:
                dhw_tensor, ttc_tensor = -9999, -9999
            else:
                dhw_tensor = (dhw_ub - abs(dhw - dhw_lb)) / (dhw_ub - dhw_lb)
                ttc_tensor = (
                    traj_vvehicle_dict.get(root._main_id)._scalar_velocity
                    - traj_vvehicle_dict.get(front_vid)._scalar_velocity
                ) / (dhw - gv.CAR_LENGTH + 1e-9)

                # ttc_tensor = min(ttc_tensor, 1)
            side_vid, relx = emath.get_side_closest_vehicle(
                traj_vvehicle_dict, root._main_id
            )
            if side_vid == None:
                relx_tensor, sidettc_tensor = -9999, -9999
            else:
                relx_tensor = (abs(relx - relx_lb)) / (relx_ub - relx_lb)
                sidettc_tensor = (
                    traj_vvehicle_dict.get(root._main_id)._scalar_velocity
                    - traj_vvehicle_dict.get(side_vid)._scalar_velocity
                    + 1e-9
                ) / abs(relx + 1e-9)
                sidettc_tensor = min(sidettc_tensor, 1)
            res_tensor[i] = torch.Tensor(
                [ttc_tensor, dhw_tensor, mv_tensor, relx_tensor, sidettc_tensor]
            )
        return res_tensor
