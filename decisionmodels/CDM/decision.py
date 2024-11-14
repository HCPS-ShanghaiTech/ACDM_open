class Decision:
    """
    Base class for decision-making.
    """

    def __init__(self) -> None:
        self.decision = "MAINTAIN"

    def decide(self):
        """
        Return the control result.
        """
        return self.decision


class LaplaceDecision(Decision):
    """
    CDM decision based on leaf nodes: Equally possible decision method.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_decision(
        self, reward_dict: dict, lon_num: int, lat_num: int, could_change_lane: bool
    ) -> str:
        """Get decision result"""
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
        self.decision = max(reward_dict, key=reward_dict.get)
        return self.decision
