import carla
import csv
import json
import torch
import copy
import logging
import multiprocessing
import utils.globalvalues as gv
import utils.extendmath as emath
import utils.dyglobalvalues as dgv
import decisionmodels.CDM.cognitivedrivermodel as cdm
import decisionmodels.CDM.observer as obs
import decisionmodels.CDM.risk as rsk
import decisionmodels.CDM.reward as rwd
import decisionmodels.CDM.decision as dcs
import decisionmodels.IDM.idmcontroller as idm
import os
from typing import Dict, Any
from datetime import datetime
from dstructures import EnumerateTree
from vehicles import RealVehicle
from model.mask_lstm import NaiveLSTM
from monitors import FiniteStateMachine

LOG_PATH = os.path.join(
    "logs", "run_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)

RESULT_FOLDER = "results"
MAIN_CONTROL_MODES = ("CDM", "IDM")
NPC_CONTROL_MODES = ("CDM", "IDM", "ACDM")
DECISION_MATRIX = [[0] * 5 for _ in range(5)]
DECISION_MAP = {
    "ACCELERATE": 0,
    "MAINTAIN": 1,
    "DECELERATE": 2,
    "SLIDE_LEFT": 3,
    "SLIDE_RIGHT": 4,
}
D2RL_PATH = os.path.join("decisionmodels", "D2RL")
SCENE_COUNT = {
    "Collision": 0,
    "NPCsCollision": 0,
    "RearEndRisk": 0,
    "Cutin": 0,
    "DriveInLine": 0,
    "Overtake": 0,
    "CarFollowingClose": 0,
    "Overtaken": 0,
    "RearEndedRisk": 0,
}
COLLISION_COUNT = {
    "HeadCollision": 0,
    "BackCollision": 0,
    "SideAttack": 0,
    "SideVictim": 0,
}


def get_initialized_scene_count():
    """Experiment value"""
    return SCENE_COUNT.copy()


def get_initialized_collision_count():
    """Experiment value"""
    return COLLISION_COUNT.copy()


def get_result_path():
    """Result path"""
    return os.path.join(
        RESULT_FOLDER, "run_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".json"
    )


def get_total_result_path():
    """Total result path"""
    return os.path.join(RESULT_FOLDER, "result.json")


def get_scene_info_list(scene_path: str):
    """
    from csv file to scene information
    """
    logging.info("Converting the scene file")
    with open(scene_path, "r") as f:
        row_list = list(csv.reader(f))
    scene_info_list = list()
    row_idx = 0
    while row_idx < len(row_list):
        row = row_list[row_idx]
        if len(row) == 1:
            num_cars = int(row[0])
            scene_info_list.append(row_list[row_idx : row_idx + num_cars + 1])
            row_idx += num_cars + 1
        else:
            row_idx += 1
    return scene_info_list


def resolve_scene_info(scene_info: list):
    """
    from single scene info to full information
    """
    num_cars = int(scene_info[0][0])
    main_speed, main_driving_style, main_control_mode = scene_info[1]
    main_speed, main_driving_style = int(main_speed), int(main_driving_style)
    npc_info = list()
    for raw_info in scene_info[2:]:
        npc_info.append([int(ele) for ele in raw_info[:4]] + [raw_info[-1]])
    return num_cars, main_speed, main_driving_style, main_control_mode, npc_info


def collision_classify(ego_vehicle: RealVehicle, npc_vehicle: RealVehicle) -> str:
    """
    Using for judge the location of collision for ego vehicle.
    Return choose in ["HeadCollision", "BackCollision", "SideAttack", "SideVictim"]
    """
    ego_wp = emath.get_real_vehicle_waypoint(ego_vehicle)
    npc_wp = emath.get_real_vehicle_waypoint(npc_vehicle)
    lon_dist = emath.cal_distance_along_road(ego_wp, npc_wp)
    if lon_dist > gv.CAR_LENGTH * 0.8:
        return "HeadCollision"
    elif lon_dist < -gv.CAR_LENGTH * 0.8:
        return "BackCollision"
    else:
        if npc_vehicle.control_action in ("SLIDE_LEFT", "SLIDE_RIGHT"):
            return "SideVictim"
        else:
            return "SideAttack"


def decision_exp_set(real_action: str, sim_action: str) -> None:
    """
    Using for relative experiment, this function will recieve real action and sim
    action without someone npc vehicle, and set the DECISION_MATRIX which counts
    the number of real action -> sim action.
    """
    assert (
        real_action in gv.ACTION_SPACE and sim_action in gv.ACTION_SPACE
    ), 'Action name error, choose in ["MAINTAIN", "ACCELERATE", "DECELERATE", "SLIDE_LEFT", "SLIDE_RIGHT"]'
    global DECISION_MATRIX, DECISION_MAP
    DECISION_MATRIX[DECISION_MAP.get(real_action)][DECISION_MAP.get(sim_action)] += 1


def decision_exp_reset() -> None:
    """
    Using for relative experiment, reset DECISION_MATRIX.
    """
    global DECISION_MATRIX
    DECISION_MATRIX = [[0] * 5 for _ in range(5)]


def cont_action_map(action, realvehicle_id=None) -> str:
    """
    Map action from continuous actions to discrete actions.
    """
    carla_map = dgv.get_map()
    realvehicle = dgv.get_realvehicle(realvehicle_id)
    if type(action) == str:
        if action in ["SLIDE_LEFT", "SLIDE_RIGHT"]:
            return action
        elif realvehicle != None and carla_map != None:
            current_waypoint = carla_map.get_waypoint(
                realvehicle.vehicle.get_location()
            )
            if current_waypoint.lane_id == gv.LANE_ID["Left"]:
                return "SLIDE_RIGHT"
            if current_waypoint.lane_id == gv.LANE_ID["Right"]:
                return "SLIDE_LEFT"
    maintain_threshold = 0.3
    if abs(action) <= maintain_threshold:
        return "MAINTAIN"
    if action < 0:
        return "DECELERATE"
    if action > 0:
        return "ACCELERATE"


def single_scene_log(scene_idx, mean_speed, task_status, unsafe_num_dict, **kwargs):
    """
    Logs for single scene end
    """
    logging.info(f"Scene {scene_idx} end")
    logging.info("Task %s", task_status.lower())
    if kwargs.get("collision_type"):
        logging.info("Collision type is %s", kwargs["collision_type"].lower())
    logging.info(f"Mean Speed is {round(mean_speed,2)}")
    logging.info("Dangerous events: %s", unsafe_num_dict)


def save_result(
    complete_time,
    num_scenes,
    scene_path,
    npc_mode,
    mean_speed,
    duration,
    events_distribution,
    collision_distribution,
    relevancy,
    rel_matrix,
):
    result_data = {
        "CompleteTime": complete_time,
        "ScenePath": scene_path,
        "NpcMode": npc_mode,
        "NumScenes": num_scenes,
        "MeanSpeed": mean_speed,
        "Duration": duration,
        "EventDistribution": events_distribution,
        "CollisionDistribution": collision_distribution,
        "Relevance": relevancy,
        "RelMatrix": rel_matrix,
    }
    result_path = get_total_result_path()
    logging.info("Saving result data to %s", os.path.abspath(result_path))
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    if isinstance(data, list):
        data.append(result_data)
    else:
        logging.error("The root element in json file is not list, can't add new item.")
    with open(result_path, "w") as f:
        json.dump(data, f, indent=4)
    logging.info("Saving done.")


def initialize() -> None:
    """
    Global Initialization
    """
    logging.info("Initializing...")

    client = carla.Client("localhost", 2000)  # Connect to the Carla server
    client.set_timeout(10.0)  # Set timeout error

    world = client.get_world()  # Get the world
    settings = world.get_settings()  # Get world settings
    settings.synchronous_mode = True  # Set synchronous mode
    settings.fixed_delta_seconds = (
        gv.STEP_DT
    )  # Time interval for each frame in synchronous mode
    world.apply_settings(settings)  # Apply the settings

    carla_map = world.get_map()
    dgv.set_map(carla_map)  # Get the map

    blueprint_library = world.get_blueprint_library()  # Get the blueprint library

    device = "cuda" if torch.cuda.is_available else "cpu"
    pred_net = NaiveLSTM(
        gv.INPUT_SIZE, gv.HIDDEN_SIZE, gv.NUM_CLASSES, gv.NUM_LAYERS
    ).to(
        device
    )  # Load neural network parameters
    pred_net.load_state_dict(torch.load("./model/Traj_2lanes.pt"))
    ini_info = {
        "client": client,
        "world": world,
        "vehicle_bp": blueprint_library.filter("model3")[0],
        "spectator": world.get_spectator(),
        "pred_net": pred_net,
        "main_spawn_point": carla_map.get_spawn_points()[264],
        "exp_info": {
            "all_scene_count": get_initialized_scene_count(),
            "collision_num_dict": get_initialized_collision_count(),
        },
    }
    logging.info("Initialize done.")

    return ini_info


def set_cdm_vehicle(
    vehicle: carla.Vehicle,
    driving_style,
    main_id,
    spawn_point: carla.Transform,
    init_velocity,
    is_main=False,
):
    """
    Install CDM to RealVehicle
    """
    if is_main:
        id = "0"
    else:
        id = vehicle.id
    observer = obs.CIPO_Observer(id, main_id)
    enumtree = EnumerateTree(id, main_id)
    risk_cal = rsk.CDMRisk(id, main_id)
    reward_cal = rwd.CDMReward(id, main_id)
    decision = dcs.LaplaceDecision()
    controller = cdm.NormalCognitiveDriverModel(
        driving_style, observer, enumtree, risk_cal, reward_cal, decision
    )
    return RealVehicle(vehicle, spawn_point, controller, init_velocity)


def set_d2rl_vehicle(
    vehicle: carla.Vehicle, main_id, spawn_point: carla.Transform, init_velocity
):
    """
    Install D2RL to RealVehicle
    """
    if os.path.exists(D2RL_PATH) or (
        os.path.isdir(D2RL_PATH) and os.listdir(D2RL_PATH)
    ):
        import decisionmodels.D2RL.d2rldecisionmodel as d2rl

        id = vehicle.id
        controller = d2rl.D2RLDecisionModel(id, main_id)
        return RealVehicle(vehicle, spawn_point, controller, init_velocity)
    else:
        logging.error("D2RL file missing")


def set_acdm_vehicle(
    vehicle: carla.Vehicle,
    driving_style,
    main_id,
    spawn_point: carla.Transform,
    init_velocity,
):
    """
    Install ACDM to RealVehicle
    """
    id = vehicle.id
    observer = obs.CIPO_Observer(id, main_id)
    enumtree = EnumerateTree(id, main_id)
    risk_cal = rsk.CDMRisk(id, main_id)
    reward_cal = rwd.ACDMReward(id, main_id)
    decision = dcs.LaplaceDecision()
    controller = cdm.AdversarialCognitiveDriverModel(
        driving_style, observer, enumtree, risk_cal, reward_cal, decision
    )
    return RealVehicle(vehicle, spawn_point, controller, init_velocity)


def set_idm_vehicle(
    vehicle: carla.Vehicle,
    main_id,
    spawn_point: carla.Transform,
    init_velocity,
    is_main=False,
):
    """
    Install IDM to RealVehicle
    """
    if is_main:
        id = "0"
    else:
        id = vehicle.id
    observer = idm.IDMObserver(id, main_id)
    controller = idm.IntelligentDriverModel(
        observer,
        gv.TYPICAL_IDM_VALUE["TimeGap"],
        gv.TYPICAL_IDM_VALUE["DesiredSpeed"],
        gv.TYPICAL_IDM_VALUE["Acceleration"],
        gv.TYPICAL_IDM_VALUE["MinimumGap"],
        gv.TYPICAL_IDM_VALUE["AccelerationExp"],
        gv.TYPICAL_IDM_VALUE["ComfortDeceleration"],
        gv.TYPICAL_IDM_VALUE["Politness"],
    )
    return RealVehicle(vehicle, spawn_point, controller, init_velocity)


def get_spawn_point(main_waypoint: carla.Waypoint, lane, rel_distance):
    """
    以主车为中心获取车辆生成点
    Param:
    main_spawn_point -> carla.Waypoint() 主车生成点
    lane -> int 0表示左车道, 1表示右车道
    rel_distance -> float 相对距离
    Return: carla.Transform()
    """
    if lane != 0 and lane != 1:
        raise ValueError("Wrong lane id!")
    lane_id = "Left" if lane == 0 else "Right"
    spawn_wp = None
    # 生成在相同的车道上, 避免碰撞
    if gv.LANE_ID[lane_id] == main_waypoint.lane_id:
        if abs(rel_distance) <= gv.CAR_LENGTH:
            raise ValueError("The spawn point is too close!")
        spawn_wp = main_waypoint
    # 生成在不同的车道上
    else:
        if lane == 0:
            spawn_wp = main_waypoint.get_left_lane()
        if lane == 1:
            spawn_wp = main_waypoint.get_right_lane()
    # 获取相对位置的waypoint
    if rel_distance > 0:
        spawn_wp = spawn_wp.next(rel_distance)[0]
    else:
        spawn_wp = spawn_wp.previous(-rel_distance)[0]

    # 沿道路法向量增加高度
    res = spawn_wp.transform
    res.location = emath.add_height_to_waypoint(res, gv.CAR_HEIGHT / 2)
    return res


def task_dispatcher(task_function, *params, **kwargs):
    return task_function(*params, **kwargs)


def normal_loop(
    step,
    main_id,
    realvehicle_id_list: list,
    npc_realvehicle_list,
    main_control_mode,
    monitor: FiniteStateMachine,
    unsafe_num_dict,
    pred_net,
    rel_steps,
    total_steps,
    num_unsafe,
):
    """
    Decision of each car in single scene for Normal experiment
    """
    if step % (gv.DECISION_DT / gv.STEP_DT) == 0:
        tasks = list()
        for carid in dgv.get_realvehicle_id_list():
            car = dgv.get_realvehicle(carid)
            # Normal Decision
            if car.vehicle.id in npc_realvehicle_list:
                # If adversarial npc mode, run a partial decision
                car.run_step(realvehicle_id_list, network=pred_net)
                main_car = dgv.get_realvehicle(main_id)
                partial_realvehicle_id_list = realvehicle_id_list.copy()
                partial_realvehicle_id_list.remove(carid)
                tasks.append(
                    (main_car.controller.run_forward, partial_realvehicle_id_list, {})
                )
                partial_action = main_car.run_partial_step(realvehicle_id_list)
                # If main car's action changed, record relevance
                if main_control_mode == "CDM":
                    if partial_action != main_car.control_action:
                        rel_steps += 1
                    decision_exp_set(main_car.control_action, partial_action)
                if main_control_mode == "IDM":
                    real_action = cont_action_map(main_car.control_action, main_car)
                    partial_action = cont_action_map(partial_action, main_car)
                    if partial_action != real_action:
                        rel_steps += 1
                    decision_exp_set(real_action, partial_action)
                total_steps += 1
            elif car.vehicle.id == main_id:
                car.run_step(realvehicle_id_list, network=pred_net)
        # Dangerous scene experiment
        unsafe_list = monitor.update(step)
        for state in unsafe_list:
            if state not in unsafe_num_dict:
                unsafe_num_dict[state] = 1
            else:
                unsafe_num_dict[state] += 1
        num_unsafe += len(unsafe_list)
    return rel_steps, total_steps, num_unsafe


def d2rl_loop(
    step,
    main_id,
    realvehicle_id_list: list,
    npc_realvehicle_list,
    main_control_mode,
    bv_action_idx_dict,
    bv_pdf_dict,
    monitor: FiniteStateMachine,
    unsafe_num_dict,
    global_controller,
    rel_steps,
    total_steps,
    num_unsafe,
    lc_maintain_count_dict,
):
    """
    Decision of each car in single scene for D2RL experiment
    """
    # Main car decision
    if step % (gv.DECISION_DT / gv.STEP_DT) == 0:
        main_car = dgv.get_realvehicle(main_id)
        main_car.run_step(realvehicle_id_list)
        for cid in npc_realvehicle_list:
            # If adversarial npc mode, run a partial decision
            partial_realvehicle_id_list = realvehicle_id_list.copy()
            partial_realvehicle_id_list.remove(cid)
            partial_action = main_car.run_partial_step(partial_realvehicle_id_list)
            # If main car's action changed, record relevance
            if main_control_mode == "CDM":
                if partial_action != main_car.control_action:
                    rel_steps += 1
                decision_exp_set(main_car.control_action, partial_action)
            if main_control_mode == "IDM":
                real_action = cont_action_map(main_car.control_action, main_id)
                partial_action = cont_action_map(partial_action, main_id)
                if partial_action != real_action:
                    rel_steps += 1
                decision_exp_set(real_action, partial_action)
            total_steps += 1
        unsafe_list = monitor.update(step)
        # Dangerous scene experiment
        for state in unsafe_list:
            if state not in unsafe_num_dict:
                unsafe_num_dict[state] = 1
            else:
                unsafe_num_dict[state] += 1
        num_unsafe += len(unsafe_list)
    # D2RL decision
    if step % (gv.D2RL_DECISION_DT / gv.STEP_DT) == 0:
        main_car = dgv.get_realvehicle(main_id)
        close_vehicle_id_list = main_car.controller.observer.get_close_vehicle_id_list(
            realvehicle_id_list
        )
        last_bv_action_idx_dict = bv_action_idx_dict.copy()
        bv_pdf_dict, bv_action_idx_dict = global_controller.global_decision(
            realvehicle_id_list, close_vehicle_id_list, step
        )
        # Judge if lane change decide
        for key in last_bv_action_idx_dict.keys():
            if last_bv_action_idx_dict[key] in [0, 1] and lc_maintain_count_dict[
                key
            ] < int(gv.LANE_CHANGE_TIME / gv.D2RL_DECISION_DT):
                bv_action_idx_dict[key] = last_bv_action_idx_dict[key]
                lc_maintain_count_dict[key] += 1
            else:
                lc_maintain_count_dict[key] = 0
        for key in dgv.get_realvehicle_id_list():
            if key != main_id:
                car = dgv.get_realvehicle(key)
                car.control_action = car.controller.single_decision(
                    bv_pdf_dict, bv_action_idx_dict
                )
                # Min speed limit
                if car.scalar_velocity <= 60 / 3.6:
                    car.control_action = 60 / 3.6 - car.scalar_velocity
    return rel_steps, total_steps, num_unsafe
