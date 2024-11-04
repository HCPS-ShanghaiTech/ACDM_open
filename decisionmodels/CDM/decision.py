class Decision:
    """
    决策基类
    """

    def __init__(self) -> None:
        self._decision = "MAINTAIN"

    def decide(self):
        """
        返回控制结果
        """
        return self._decision


class LaplaceDecision(Decision):
    """
    CDM基于叶子结点的决策: 等可能决策法
    """

    def __init__(self) -> None:
        super().__init__()

    def get_decision(self, reward_dict, lon_num, lat_num, could_change_lane):
        """获得决策结果"""
        for key in reward_dict.keys():
            if key in ["ACCELERATE", "DECELERATE", "MAINTAIN"]:
                reward_dict[key] = sum(reward_dict[key]) / (
                    lon_num if lon_num != 0 else 1
                )
            else:
                reward_dict[key] = sum(reward_dict[key]) / (
                    3 * lat_num if lat_num != 0 else 1
                )
        if not could_change_lane:
            del reward_dict["SLIDE_LEFT"]
            del reward_dict["SLIDE_RIGHT"]
        self._decision = max(reward_dict, key=reward_dict.get)
        return self._decision

    def get_decisions(self, reward_dicts, lons, lats, risks):
        """获得决策结果"""
        total_risk = sum(risks)
        if total_risk != 0:
            normalized_risk = [x / total_risk for x in risks]
        else:
            normalized_risk = [1 / len(risks) for _ in range(len(risks))]
        reward_sum = {}
        for i in range(len(reward_dicts)):
            for key in reward_dicts[i].keys():
                if key in ["ACCELERATE", "DECELERATE", "MAINTAIN"]:
                    reward_dicts[i][key] = sum(reward_dicts[i][key]) / (
                        lons[i] if lons[i] != 0 else 1
                    )
                else:
                    reward_dicts[i][key] = sum(reward_dicts[i][key]) / (
                        3 * lats[i] if lats[i] != 0 else 1
                    )
                if key not in reward_sum:
                    reward_sum[key] = reward_dicts[i][key] * normalized_risk[i]
                else:
                    reward_sum[key] += reward_dicts[i][key] * normalized_risk[i]
            self._decision = max(reward_sum, key=reward_sum.get)
        return self._decision
