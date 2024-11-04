import utils.globalvalues as NUM_OF_TREE
from utils.extendmath import waypoints_are_same_location
import itertools

"""
Cognitive Driver Model
"""


class CognitiveDriverModel:
    def __init__(
        self,
        driving_style,
        observer,
        enumeratetree,
        risk_calculator,
        reward_calculator,
        decision_mode,
    ) -> None:
        self._driving_style = driving_style
        self._observer = observer
        self._enumeratetree = enumeratetree
        self._risk_calculator = risk_calculator
        self._reward_calculator = reward_calculator
        self._decision_mode = decision_mode

    def update_from_observe(self, full_vehicle_id_dict, nnet=None):
        # 开始没有认知，给全观测
        if (not self._enumeratetrees[0]._leaves) or self.warm_up > 0:
            close_vehicle_id_list, lon_levels, lat_levels = self._observer.observe_full(
                full_vehicle_id_dict
            )
            self._enumeratetrees[0].generate_root_from_cipo(
                full_vehicle_id_dict, close_vehicle_id_list, lon_levels, lat_levels
            )
            self._enumeratetrees[0]._is_valid = True
            self.warm_up -= 1
        else:
            close_vehicle_id_list, lon_levels, lat_levels = (
                self._observer.observe_partial(full_vehicle_id_dict)
            )

            all_leaves = []
            for tid in range(len(self._enumeratetrees)):
                if self._enumeratetrees[tid]._is_valid == False:
                    continue
                true_list = {}
                guess_list = {}
                for cid in full_vehicle_id_dict.keys():
                    # 观测内的用真实观测
                    if cid in close_vehicle_id_list:
                        virtual_vehicle = full_vehicle_id_dict.get(
                            cid
                        ).clone_to_virtual()
                        true_list[cid] = virtual_vehicle
                    # 不在观测内，根据上次perception推测
                    else:
                        unique_leaf_id_list = self.select_unique_leaf(
                            cid, self._enumeratetrees[tid]._leaves
                        )
                        unique_virtual_vehicle_list = []
                        for id in unique_leaf_id_list:
                            unique_virtual_vehicle_list.append(
                                self._enumeratetrees[tid]
                                ._leaves[id]
                                ._virtual_vehicle_dict.get(cid)
                            )
                        guess_list[cid] = unique_virtual_vehicle_list

                guess_combinations = [
                    dict(zip(guess_list.keys(), combination))
                    for combination in itertools.product(*guess_list.values())
                ]
                result = [
                    {**true_list, **guess_combination}
                    for guess_combination in guess_combinations
                ]
                new_leaves = []
                for new_dict in result:
                    new_leaf = self._enumeratetrees[0]._leaves[0].update_dict(new_dict)
                    new_leaves.append(new_leaf)
                self._risk_calculator.cal_risk_list(
                    self._enumeratetrees[tid]._root, new_leaves
                )
                all_leaves.extend(new_leaves)
            sorted_leaves = sorted(
                all_leaves, key=lambda leaf: leaf._risk, reverse=True
            )
            if len(sorted_leaves) > 3:
                top_risk_leaves = sorted_leaves[:3]
            else:
                top_risk_leaves = sorted_leaves[:]
            for i in range(len(self._enumeratetrees)):
                if i < len(top_risk_leaves):
                    close_vehicle_id_list, lon_levels, lat_levels = (
                        self._observer.observe_full(
                            top_risk_leaves[i]._virtual_vehicle_dict, "virtual"
                        )
                    )
                    self._enumeratetrees[i].generate_root_from_leaf(
                        top_risk_leaves[i], lon_levels, lat_levels
                    )
                    self._enumeratetrees[i]._is_valid = True
                else:
                    self._enumeratetrees[i]._is_valid = False

    def select_unique_leaf(self, cid, leaves):
        consider_actions = []
        unique_leaf_id_list = []

        for i in range(len(leaves)):
            v_vehicle = leaves[i]._virtual_vehicle_dict.get(cid)
            if v_vehicle and v_vehicle._control_action not in consider_actions:
                consider_actions.append(v_vehicle._control_action)
                unique_leaf_id_list.append(i)

        return unique_leaf_id_list


class NormalCognitiveDriverModel(CognitiveDriverModel):
    def __init__(
        self,
        driving_style,
        observer,
        enumeratetree,
        risk_calculator,
        reward_calculator,
        decision_mode,
    ) -> None:
        super().__init__(
            driving_style,
            observer,
            enumeratetree,
            risk_calculator,
            reward_calculator,
            decision_mode,
        )
        # 跳过第一帧
        self.warm_up = 2

    def run_forward(self, full_vehicle_id_dict, nnet):
        """前向计算一轮"""
        # 观测器返回结果, ACDM采用CIPO观测器, 返回三个结果
        close_vehicle_id_list, lon_levels, lat_levels = self._observer.observe_full(
            full_vehicle_id_dict
        )

        # 树生成根节点
        self._enumeratetree.generate_root_from_cipo(
            full_vehicle_id_dict, close_vehicle_id_list, lon_levels, lat_levels
        )

        # 树生成叶子结点
        leaves, num_lon, num_lat = self._enumeratetree.grow_tree()

        # 计算风险
        self._risk_calculator.cal_risk_list(self._enumeratetree._root, leaves)

        # 筛选叶子结点
        preferred_leaves, other_leaves = self._risk_calculator.get_preferred_leaves(
            leaves, self._driving_style
        )

        # 计算收益
        reward_dict = self._reward_calculator.cal_reward_dict(
            preferred_leaves, other_leaves
        )

        # 决策
        return self._decision_mode.get_decision(reward_dict, num_lon, num_lat, True)

    # def update_from_observe(self, full_vehicle_id_dict, nnet=None):
    #     # 开始没有认知
    #     if not self._leaves:
    #         close_vehicle_id_list, lon_levels, lat_levels = self._observer.observe_full(
    #             full_vehicle_id_dict
    #         )
    #         self._enumeratetrees[0].generate_root_from_cipo(
    #             full_vehicle_id_dict, close_vehicle_id_list, lon_levels, lat_levels
    #         )
    #         while len(self._enumeratetrees) > 1:
    #             self._enumeratetrees.pop()
    #     else:
    #         close_vehicle_id_list, lon_levels, lat_levels = self._observer.observe_partial(
    #             full_vehicle_id_dict
    #         )
    #         # 根据观测筛选叶子节点
    #         for i in range(len(self._leaves)):
    #             for cid in close_vehicle_id_list:
    #                 virtual_vehicle = full_vehicle_id_dict.get(cid).clone_to_virtual()
    #                 if not waypoints_are_same_location(virtual_vehicle._waypoint, self._leaves[i]._virtual_vehicle_dict[cid]._waypoint):
    #                     self._leaves[i] = None
    #                     break
    #
    #         # 生成perception
    #         filtered_leaves = [x for x in self._leaves if x is not None]
    #
    #         # 没有符合观测的叶子节点
    #         if len(filtered_leaves) == 0:
    #             close_vehicle_id_list, lon_levels, lat_levels = self._observer.observe_full(
    #                 full_vehicle_id_dict
    #             )
    #             self._enumeratetrees[0].generate_root_from_cipo(
    #                 full_vehicle_id_dict, close_vehicle_id_list, lon_levels, lat_levels
    #             )
    #             while len(self._enumeratetrees) > 1:
    #                 self._enumeratetrees.pop()
    #
    #         # 根据风险排序，选出前三风险的叶子节点作为这一时刻的perception
    #         else:
    #             sorted_leaves = sorted(filtered_leaves, key=lambda leaf: leaf._risk, reverse=True)
    #             if len(sorted_leaves) > 3:
    #                 top_risk_leaves = sorted_leaves[:3]
    #             else:
    #                 top_risk_leaves = sorted_leaves[:]
    #             for i in range(len(top_risk_leaves)):
    #                 close_vehicle_id_list, lon_levels, lat_levels = self._observer.observe_full(
    #                     top_risk_leaves[i]._virtual_vehicle_dict, "virtual"
    #                 )
    #                 self._enumeratetrees[i].generate_root_from_leaf(
    #                     top_risk_leaves[i], lon_levels, lat_levels
    #                 )
    #             while len(self._enumeratetrees) > len(top_risk_leaves):
    #                 self._enumeratetrees.pop()

    # def run_forward(self, full_vehicle_id_dict, nnet=None):
    #     """前向计算一轮"""

    #     self.update_from_observe(full_vehicle_id_dict)
    #     lons = []
    #     lats = []
    #     reward_dicts = []
    #     risks = []
    #     for i in range(len(self._enumeratetrees)):
    #         if self._enumeratetrees[i]._is_valid == False:
    #             continue
    #         risks.append(self._enumeratetrees[i]._probability)
    #         # 树生成叶子结点
    #         leaves, num_lon, num_lat = self._enumeratetrees[i].grow_tree()
    #         lons.append(num_lon)
    #         lats.append(num_lat)

    #         # 计算风险
    #         self._risk_calculator.cal_risk_list(self._enumeratetrees[i]._root, leaves)

    #         # 筛选叶子结点
    #         preferred_leaves, other_leaves = self._risk_calculator.get_preferred_leaves(
    #             leaves, self._driving_style
    #         )

    #         # 计算收益
    #         reward_dict = self._reward_calculator.cal_reward_dict(
    #             preferred_leaves, other_leaves
    #         )
    #         reward_dicts.append(reward_dict)

    #         # # 全展开树
    #         # leaves = self._enumeratetrees[i].grow_tree_full()
    #         # self._risk_calculator.cal_risk_list(self._enumeratetrees[i]._root, leaves)

    #     # 决策
    #     return self._decision_mode.get_decisions(reward_dicts, lons, lats, risks)


class AdversarialCognitiveDriverModel(CognitiveDriverModel):
    def __init__(
        self,
        driving_style,
        observer,
        enumeratetree,
        risk_calculator,
        reward_calculator,
        decision_mode,
    ) -> None:
        super().__init__(
            driving_style,
            observer,
            enumeratetree,
            risk_calculator,
            reward_calculator,
            decision_mode,
        )
        # 跳过第一帧
        self.warm_up = 2

    # def run_forward(self, full_vehicle_id_dict, nnet=None):
    #     """前向计算一轮"""

    #     self.update_from_observe(full_vehicle_id_dict)
    #     lons = []
    #     lats = []
    #     reward_dicts = []
    #     risks = []
    #     for i in range(len(self._enumeratetrees)):
    #         if self._enumeratetrees[i]._is_valid == False:
    #             continue
    #         risks.append(self._enumeratetrees[i]._probability)
    #         # 树生成叶子结点
    #         leaves, num_lon, num_lat = self._enumeratetrees[i].grow_tree()
    #         lons.append(num_lon)
    #         lats.append(num_lat)

    #         # 计算风险
    #         self._risk_calculator.cal_risk_list(self._enumeratetrees[i]._root, leaves)

    #         # 筛选叶子结点
    #         preferred_leaves, other_leaves = self._risk_calculator.get_preferred_leaves(
    #             leaves, self._driving_style
    #         )

    #         # 计算收益
    #         reward_dict = self._reward_calculator.cal_reward_dict(
    #             self._enumeratetrees[i]._root, preferred_leaves, other_leaves, nnet
    #         )
    #         reward_dicts.append(reward_dict)

    #         # # 全展开树
    #         # leaves = self._enumeratetrees[i].grow_tree_full()
    #         # self._risk_calculator.cal_risk_list(self._enumeratetrees[i]._root, leaves)

    #     # 决策
    #     return self._decision_mode.get_decisions(reward_dicts, lons, lats, risks)

    def run_forward(self, full_vehicle_id_dict, nnet):
        """前向计算一轮"""
        # 观测器返回结果, ACDM采用CIPO观测器, 返回三个结果
        close_vehicle_id_list, lon_levels, lat_levels = self._observer.observe_full(
            full_vehicle_id_dict
        )

        # 树生成根节点
        self._enumeratetree.generate_root_from_cipo(
            full_vehicle_id_dict, close_vehicle_id_list, lon_levels, lat_levels
        )

        could_change_lane = self._enumeratetree._root.judge_could_slide()
        # 树生成叶子结点
        leaves, num_lon, num_lat = self._enumeratetree.grow_tree()

        # 计算风险
        self._risk_calculator.cal_risk_list(self._enumeratetree._root, leaves)

        # 筛选叶子结点
        preferred_leaves, other_leaves = self._risk_calculator.get_preferred_leaves(
            leaves, self._driving_style
        )

        # 计算收益
        reward_dict = self._reward_calculator.cal_reward_dict(
            self._enumeratetree._root, preferred_leaves, other_leaves, nnet
        )

        # 决策
        return self._decision_mode.get_decision(
            reward_dict, num_lon, num_lat, could_change_lane
        )
