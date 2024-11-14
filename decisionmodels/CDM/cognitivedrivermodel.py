"""
Cognitive Driver Model
"""

import utils.globalvalues as gv
from .observer import CIPO_Observer
from dstructures import EnumerateTree
from .risk import CDMRisk
from .reward import *
from .decision import LaplaceDecision


class NormalCognitiveDriverModel:
    def __init__(
        self,
        driving_style: int,
        observer: CIPO_Observer,
        enumeratetree: EnumerateTree,
        risk_calculator: CDMRisk,
        reward_calculator: CDMReward,
        decision_mode: LaplaceDecision,
    ) -> None:
        self.driving_style = driving_style
        self.observer = observer
        self.enumeratetree = enumeratetree
        self.risk_calculator = risk_calculator
        self.reward_calculator = reward_calculator
        self.decision_mode = decision_mode

    def run_forward(self, realvehicle_id_list, **kwargs):
        """Perform one forward calculation cycle"""
        # The observer returns results. ACDM uses the CIPO observer, which returns three results.
        close_vehicle_id_list, lon_levels, lat_levels = self.observer.observe_full(
            realvehicle_id_list
        )

        # The tree generates the root node
        self.enumeratetree.generate_root_from_cipo(
            close_vehicle_id_list, lon_levels, lat_levels
        )

        # The tree generates leaf nodes
        leaves, num_lon, num_lat = self.enumeratetree.grow_tree()

        # Calculate risk
        self.risk_calculator.cal_risk_list(self.enumeratetree.root, leaves)

        # Filter leaf nodes
        preferred_leaves, other_leaves = self.risk_calculator.get_preferred_leaves(
            leaves, self.driving_style
        )

        # Calculate reward
        reward_dict = self.reward_calculator.cal_reward_dict(
            preferred_leaves, other_leaves
        )

        # Make decision
        return self.decision_mode.get_decision(
            reward_dict, num_lon, num_lat, could_change_lane=True
        )

    # For multiprocessing, divide run_forward(args) to 3 part
    def run_forward_start(self, realvehicle_id_list, **kwargs):
        """Perform one forward calculation cycle"""
        # The observer returns results. ACDM uses the CIPO observer, which returns three results.
        close_vehicle_id_list, lon_levels, lat_levels = self.observer.observe_full(
            realvehicle_id_list
        )

        # The tree generates the root node
        self.enumeratetree.generate_root_from_cipo(
            close_vehicle_id_list, lon_levels, lat_levels
        )

        could_change_lane = self.enumeratetree.root.judge_could_slide()

        return could_change_lane

    @staticmethod
    def run_forward_middle(enumeratetree, risk_calculator, driving_style):
        try:
            # The tree generates leaf nodes
            leaves, num_lon, num_lat = enumeratetree.grow_tree()
            # Calculate risk
            risk_calculator.cal_risk_list(enumeratetree.root, leaves)

            # Filter leaf nodes
            preferred_leaves, other_leaves = risk_calculator.get_preferred_leaves(
                leaves, driving_style
            )

            return num_lon, num_lat, preferred_leaves, other_leaves
        except Exception as e:
            error_message = f"Exception: {str(e)}"
            return error_message

    def run_forward_end(
        self,
        num_lon,
        num_lat,
        preferred_leaves,
        other_leaves,
        could_change_lane,
        **kwargs,
    ):
        # Calculate reward
        reward_dict = self.reward_calculator.cal_reward_dict(
            preferred_leaves, other_leaves
        )

        # Make decision
        return self.decision_mode.get_decision(
            reward_dict, num_lon, num_lat, could_change_lane
        )


class AdversarialCognitiveDriverModel:
    def __init__(
        self,
        driving_style: int,
        observer: CIPO_Observer,
        enumeratetree: EnumerateTree,
        risk_calculator: CDMRisk,
        reward_calculator: ACDMReward,
        decision_mode: LaplaceDecision,
    ) -> None:
        self.driving_style = driving_style
        self.observer = observer
        self.enumeratetree = enumeratetree
        self.risk_calculator = risk_calculator
        self.reward_calculator = reward_calculator
        self.decision_mode = decision_mode

    def run_forward(self, realvehicle_id_list, **kwargs):
        """Perform one forward calculation cycle"""
        nnet = kwargs.get("network")
        # The observer returns results. ACDM uses the CIPO observer, which returns three results.
        close_vehicle_id_list, lon_levels, lat_levels = self.observer.observe_full(
            realvehicle_id_list
        )

        # The tree generates the root node
        self.enumeratetree.generate_root_from_cipo(
            close_vehicle_id_list, lon_levels, lat_levels
        )

        could_change_lane = self.enumeratetree.root.judge_could_slide()
        # The tree generates leaf nodes
        leaves, num_lon, num_lat = self.enumeratetree.grow_tree()

        # Calculate risk
        self.risk_calculator.cal_risk_list(self.enumeratetree.root, leaves)

        # Filter leaf nodes
        preferred_leaves, other_leaves = self.risk_calculator.get_preferred_leaves(
            leaves, self.driving_style
        )

        for leaf in preferred_leaves:
            leaf.upload_input_data(self.enumeratetree.root)
        # NN output
        traj_batch = torch.stack(self.enumeratetree.root.traj_cache)
        nnet.eval()
        with torch.no_grad():
            action_batch = nnet(traj_batch, traj_batch == -9999)
        # Calculate reward
        reward_dict = self.reward_calculator.cal_reward_dict(
            self.enumeratetree.root, preferred_leaves, other_leaves, action_batch
        )

        # Make decision
        return self.decision_mode.get_decision(
            reward_dict, num_lon, num_lat, could_change_lane
        )

    # For multiprocessing, divide run_forward(args) to 3 part
    def run_forward_start(self, realvehicle_id_list, **kwargs):
        """Perform one forward calculation cycle"""
        # The observer returns results. ACDM uses the CIPO observer, which returns three results.
        close_vehicle_id_list, lon_levels, lat_levels = self.observer.observe_full(
            realvehicle_id_list
        )

        # The tree generates the root node
        self.enumeratetree.generate_root_from_cipo(
            close_vehicle_id_list, lon_levels, lat_levels
        )

        could_change_lane = self.enumeratetree.root.judge_could_slide()

        return could_change_lane

    @staticmethod
    def run_forward_middle(enumeratetree, risk_calculator, driving_style):
        try:
            # The tree generates leaf nodes
            leaves, num_lon, num_lat = enumeratetree.grow_tree()
            # Calculate risk
            risk_calculator.cal_risk_list(enumeratetree.root, leaves)

            # Filter leaf nodes
            preferred_leaves, other_leaves = risk_calculator.get_preferred_leaves(
                leaves, driving_style
            )
            return num_lon, num_lat, preferred_leaves, other_leaves
        except Exception as e:
            error_message = f"Exception: {str(e)}"
            return error_message

    def run_forward_end(
        self,
        num_lon,
        num_lat,
        preferred_leaves,
        other_leaves,
        could_change_lane,
        **kwargs,
    ):
        nnet = kwargs.get("network")
        for leaf in preferred_leaves:
            leaf.upload_input_data(self.enumeratetree.root)
        # NN output
        traj_batch = torch.stack(self.enumeratetree.root.traj_cache)
        nnet.eval()
        with torch.no_grad():
            action_batch = nnet(traj_batch, traj_batch == -9999)
        # Calculate reward
        reward_dict = self.reward_calculator.cal_reward_dict(
            self.enumeratetree.root, preferred_leaves, other_leaves, action_batch
        )

        # Make decision
        return self.decision_mode.get_decision(
            reward_dict, num_lon, num_lat, could_change_lane
        )
