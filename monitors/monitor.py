import copy
import utils.extendmath as emath
import utils.globalvalues as gv
import utils.dyglobalvalues as dgv
from .posmatrix import PositionMatrix


class FiniteStateMachine:
    """
    Determines dangerous scenarios using the position matrix.
    """

    def __init__(self, step, pos_matirx: PositionMatrix, max_seq_len=4) -> None:
        self.state_list = [
            "Safe",
            "Overtake",
            "Overtaken",
            "Cutin",
            "RearEndRisk",
            "RearEndedRisk",
            "DriveInLine",
            "CarFollowingClose",
        ]
        self.current_step = step
        self.max_seq_len = max_seq_len
        self.pos_matrix = pos_matirx
        self.state_seq_to_main = {}
        self.last_seq_map = {}
        self.last_pos_repeat_num = {}

    def __del__(self):
        pass

    def update(self, step):
        """
        Iteratively update to determine whether the main vehicle is in a certain dangerous state,
        and identify the specific state (states do not strictly conflict, e.g., during a cut-in,
        there may be a risk of rear-end collision).
        """
        # Number of unsafe states
        num_unsafe = list()
        # Existing states require up to three states to judge
        assert self.max_seq_len >= 3
        # Update time
        self.current_step = step
        # Update Position Matrix
        self.last_seq_map = copy.deepcopy(self.state_seq_to_main)
        position_map, related_vehicles = self.pos_matrix.state_update()
        # Update one-to-one state sequences
        # state_seq_to_main format is as follows:
        # The key is the ID of the vehicle interacting with the test vehicle,
        # and the corresponding value is a list with a maximum length of max_seq_len.
        # Elements in the list are tuples: (key in posmatrix, main vehicle lane ID, current lane ID, step value of the last state)
        for rvid in related_vehicles:
            position_key = self.pos_matrix.get_key(position_map, rvid)
            if position_key == None:
                continue
            if rvid not in self.state_seq_to_main.keys():
                self.state_seq_to_main[rvid] = list()
                self.last_pos_repeat_num[rvid] = 1
            laneid = (
                dgv.get_map()
                .get_waypoint(dgv.get_realvehicle(rvid).vehicle.get_location())
                .lane_id
            )
            lane_id_main = (
                dgv.get_map()
                .get_waypoint(
                    dgv.get_realvehicle(self.pos_matrix.main_id).vehicle.get_location()
                )
                .lane_id
            )
            # Construct a tuple
            seq_elem_tuple = (position_key, lane_id_main, laneid, self.current_step)
            # If the sequence is empty
            if not self.state_seq_to_main[rvid]:
                self.state_seq_to_main[rvid].append(seq_elem_tuple)
            # If the last element of the sequence is the same as the current state, modify it in place
            elif position_key == self.state_seq_to_main[rvid][-1][0]:
                self.state_seq_to_main[rvid][-1] = seq_elem_tuple
                self.last_pos_repeat_num[rvid] += 1
            else:
                # If exceeding max_seq_len, control list length
                if len(self.state_seq_to_main[rvid]) > self.max_seq_len:
                    self.state_seq_to_main[rvid].pop(0)
                self.state_seq_to_main[rvid].append(seq_elem_tuple)
                self.last_pos_repeat_num[rvid] = 1
        # Determine if in a certain dangerous state
        for rvid in related_vehicles:
            # Check conditions for each state
            for state in self.state_list:
                if self.state_condition(rvid, state):
                    print(
                        f"Danger occurred at the {int(step * gv.STEP_DT)} second after the start of the scene",
                        "Interacting vehicles: ",
                        rvid,
                        " State: ",
                        state,
                    )
                    num_unsafe.append(state)
        return num_unsafe

    def state_condition(self, rvid, state="Safe"):
        """
        Determine the state based on the sequence of states.
        """
        ttc_threhold = 3
        seq = self.state_seq_to_main.get(rvid)
        last_seq = self.last_seq_map.get(rvid)
        if not seq:
            return False
        # Check if the current state is the latest
        latest_condition = seq[-1][-1] == self.current_step
        # Check if the sequence is repeated
        if seq and last_seq:
            repeat_condition = self.non_repeat_condition(seq, last_seq)
        else:
            repeat_condition = True
        if state == "Cutin":
            # Basic conditions that need to be met
            basic_condition = (
                repeat_condition
                and self.seq_lane_limit_condition(seq, 2)
                and latest_condition
                and self.keep_lane_condition(seq, 2, True)
            )
            # Specific conditions for determining a cut-in
            core_condition = (
                len(seq) >= 2
                and seq[-1][0] == "Front"
                and seq[-2][0]
                in [
                    "LeftFront",
                    "RightFront",
                    "LeftSide",
                    "RightSide",
                ]
            )
            return basic_condition and core_condition
        if state == "Overtaken":
            # Basic conditions that need to be met
            basic_condition = (
                repeat_condition
                and self.seq_lane_limit_condition(seq, 3)
                and latest_condition
            )
            # Specific conditions for determining overtaken
            core_condition = (
                len(seq) >= 4
                and seq[-1][0] in ["Front"]
                and seq[-4][0]
                in [
                    "LeftBack",
                    "RightBack",
                    "Back",
                ]
            )
            return basic_condition and core_condition
        if state == "Overtake":
            # Basic conditions that need to be met
            basic_condition = (
                repeat_condition
                and self.seq_lane_limit_condition(seq, 3)
                and latest_condition
            )
            # Specific conditions for determining overtake
            core_condition = (
                len(seq) >= 4
                and seq[-1][0]
                in [
                    "Back",
                ]
                and seq[-4][0] in ["Front", "LeftFront", "RightFront"]
            )
            return basic_condition and core_condition
        if state in ["RearEndRisk", "RearEndedRisk", "CarFollowingClose"]:
            min_id_front, min_dist_front, min_id_back, min_dist_back = (
                self.get_lon_closest_vehicle(
                    self.pos_matrix.related_vehicles,
                    self.pos_matrix.main_id,
                )
            )
            front_ttc = -1e9
            if min_id_front:
                front_ttc = min_dist_front / (
                    dgv.get_realvehicle(self.pos_matrix.main_id).scalar_velocity
                    - dgv.get_realvehicle(min_id_front).scalar_velocity
                    + 1e-9
                )
            back_ttc = -1e9
            if min_id_back:
                back_ttc = -min_dist_back / (
                    dgv.get_realvehicle(min_id_back).scalar_velocity
                    - dgv.get_realvehicle(self.pos_matrix.main_id).scalar_velocity
                    + 1e-9
                )
            if state == "RearEndRisk":
                return front_ttc > 0 and front_ttc <= ttc_threhold
            if state == "RearEndedRisk":
                return back_ttc > 0 and back_ttc <= ttc_threhold
            if state == "CarFollowingClose":
                return (
                    seq
                    and self.last_pos_repeat_num[rvid] % 5 == 0
                    and (
                        (abs(min_dist_front) <= 25 and abs(front_ttc) >= 10)
                        or (abs(min_dist_back) <= 25 and abs(back_ttc) >= 10)
                    )
                )
        if state == "DriveInLine":
            return (
                seq
                and seq[-1][0] in ["LeftSide", "RightSide"]
                and self.last_pos_repeat_num[rvid] % 5 == 0
            )

    @staticmethod
    def non_repeat_condition(seq, last_seq):
        """
        Determine if the state is repeated based on similarity to the previous sequence.
        """
        if len(seq) <= 1 or len(last_seq) <= 1:
            return True
        return seq[:-2] != last_seq[:-2]

    @staticmethod
    def seq_lane_limit_condition(seq, limit_len):
        """
        Check if the sequence length is sufficient.
        """
        return len(seq) >= limit_len

    @staticmethod
    def keep_lane_condition(seq, search_len, is_main):
        """
        Determine if the lane is maintained. Search_len represents the time length to search the state.
        Is_main indicates whether to check the main vehicle. False for checking the current vehicle.
        """
        assert search_len > 0
        seq_len = len(seq)
        if search_len <= 1:
            return True
        real_search_len = min(search_len, seq_len)
        if is_main:
            laneid_id = 1
        else:
            laneid_id = 2
        std_laneid = seq[-1][laneid_id]
        for i in range(seq_len - real_search_len, seq_len - 1):
            if seq[i][laneid_id] != std_laneid:
                return False
        return True

    @staticmethod
    def get_lon_closest_vehicle(rel_vehicle_list, ego_id):
        """
        Identify the closest vehicles in front of and behind the main vehicle.
        """
        carla_map = dgv.get_map()
        ego_vehicle = dgv.get_realvehicle(ego_id)
        min_dist_front = 1e9
        min_dist_back = -1e9
        min_id_front = None
        min_id_back = None
        ego_wp = carla_map.get_waypoint(ego_vehicle.vehicle.get_location())
        # Evaluate each vehicle to see if it is in the same lane and at the shortest front/back distance
        for rvid in rel_vehicle_list:
            if rvid != ego_id:
                vehicle = dgv.get_realvehicle(rvid)
                veh_wp = carla_map.get_waypoint(vehicle.vehicle.get_location())
                rel_distance = emath.cal_distance_along_road(ego_wp, veh_wp)
                # In front
                if veh_wp.lane_id == ego_wp.lane_id:
                    # Front vehicle
                    if 0 < rel_distance < min_dist_front:
                        min_dist_front = rel_distance
                        min_id_front = rvid
                    # Rear vehicle
                    if min_dist_back < rel_distance < 0:
                        min_dist_back = rel_distance
                        min_id_back = rvid
        # Return the closest vehicles in front and behind, along with their relative distances
        return min_id_front, min_dist_front, min_id_back, min_dist_back
