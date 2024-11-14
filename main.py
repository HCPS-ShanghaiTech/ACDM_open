import carla
import os
import argparse
import logging
import multiprocessing
import utils.globalvalues as gv
import utils.extendmath as emath
import utils.dyglobalvalues as dgv
from collections import Counter
from typing import Dict, Any
from monitors import PositionMatrix, FiniteStateMachine
from manager import *
import time


def run_experiment(scene_path, scene_type, duration, exp_mode, pool=None):
    """
    Run ACDM or baseline experiment
    """
    if (
        exp_mode == "d2rl"
        and os.path.exists(D2RL_PATH)
        or (os.path.isdir(D2RL_PATH) and os.listdir(D2RL_PATH))
    ):
        import decisionmodels.D2RL.d2rldecisionmodel as d2rl
    logging.info("Experiment on scene file: %s", scene_path)
    # Initialize
    init_info = initialize()
    world = init_info["world"]
    vehicle_bp = init_info["vehicle_bp"]
    spectator = init_info["spectator"]
    pred_net = init_info["pred_net"]
    main_spawn_point = init_info["main_spawn_point"]
    all_scene_count = init_info["exp_info"].get("all_scene_count")
    collision_num_dict = init_info["exp_info"].get("collision_num_dict")

    all_scene_rel_steps = 0
    all_scene_steps = 0
    total_time = 0
    all_scene_mean_speed = 0
    control_mode_dict = dict()

    scene_info_list = get_scene_info_list(scene_path)
    num_scenes = len(scene_info_list)
    # multiprocess
    is_pickleable = True if pool else False

    start_time = time.time()
    for scene_idx in range(num_scenes):
        num_cars, main_speed, main_driving_style, main_control_mode, npc_info = (
            resolve_scene_info(scene_info_list[scene_idx])
        )
        logging.info("Scene: %d", scene_idx + 1)
        dgv.reset_realvehicle_dict()
        npc_realvehicle_id_list = list()  # id list
        cdm_realvehicle_id_list = list()  # only for multiprocessing
        # Main Car Setting
        vehicle_bp.set_attribute("color", "0, 0, 0")
        main_vehicle = world.spawn_actor(vehicle_bp, main_spawn_point)
        # For d2rl experiment, main car id is set "0"
        main_id = "0" if exp_mode == "d2rl" else main_vehicle.id
        is_d2rl_main = True if exp_mode == "d2rl" else False
        logging.info("The main car's id is {}, color is black".format(main_id))
        control_mode_dict[str(main_id)] = main_control_mode
        match main_control_mode:
            case "CDM":
                main_rv = set_cdm_vehicle(
                    main_vehicle,
                    main_driving_style,
                    main_id,
                    main_spawn_point,
                    main_speed,
                    is_d2rl_main,
                    is_pickleable,
                )
                cdm_realvehicle_id_list.append(main_id)
            case "IDM":
                main_rv = set_idm_vehicle(
                    main_vehicle,
                    main_id,
                    main_spawn_point,
                    main_speed,
                    is_d2rl_main,
                )
            case _:
                logging.error("Main control mode should in %s", MAIN_CONTROL_MODES)
                raise ValueError("Wrong main control mode")
        dgv.update_realvehicle_dict(main_id, main_rv)
        if exp_mode == "d2rl":
            # D2RL experiment setting
            global_controller = d2rl.D2RLGlobalDecisionModel(main_id)
        # NPC Car Setting
        for i in range(num_cars - 1):
            npc_speed, npc_lane, rel_pos, npc_driving_style, npc_control_mode = (
                npc_info[i]
            )
            # D2RL is special
            npc_control_mode = "D2RL" if exp_mode == "d2rl" else npc_control_mode
            spawn_point = get_spawn_point(
                dgv.get_map().get_waypoint(main_spawn_point.location),
                npc_lane,
                rel_pos,
            )
            vehicle_bp.set_attribute("color", gv.COLOR_DICT[gv.COLOR_LIST[i]])
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            logging.info(
                "The npc car(%d) id is %s, color is %s",
                i,
                vehicle.id,
                gv.COLOR_LIST[i],
            )
            control_mode_dict[str(vehicle.id)] = npc_control_mode
            match npc_control_mode:
                case "CDM":
                    npc_rv = set_cdm_vehicle(
                        vehicle,
                        npc_driving_style,
                        main_id,
                        spawn_point,
                        npc_speed,
                        is_pickleable=is_pickleable,
                    )
                    cdm_realvehicle_id_list.append(vehicle.id)
                case "ACDM":
                    npc_rv = set_acdm_vehicle(
                        vehicle,
                        npc_driving_style,
                        main_id,
                        spawn_point,
                        npc_speed,
                        is_pickleable=is_pickleable,
                    )
                    cdm_realvehicle_id_list.append(vehicle.id)
                case "IDM":
                    npc_rv = set_idm_vehicle(vehicle, main_id, spawn_point, npc_speed)
                case "D2RL":
                    npc_rv = set_d2rl_vehicle(vehicle, main_id, spawn_point, npc_speed)
                case _:
                    logging.error("NPC control mode should in %s", NPC_CONTROL_MODES)
                    raise ValueError("Wrong main control mode")
            dgv.update_realvehicle_dict(vehicle.id, npc_rv)
            npc_realvehicle_id_list.append(vehicle.id)

        # -----------------Single Scene Experiment Params------------------- #
        step = -5  # Warm up for 5 steps
        stop = False  # Stop flag
        stop_lifetime = 100 if scene_type == "long" else 50  # remain steps after stop
        # No additional vehicles will be generated during the scene
        realvehicle_id_list = dgv.get_realvehicle_id_list()
        # Monitor
        pos_matrix = PositionMatrix(main_id)
        monitor = FiniteStateMachine(0, pos_matrix, 4)
        # Relevance
        rel_steps = 0
        total_steps = 0
        # Driving Distance
        num_round = 0
        total_round = 2
        main_s = last_main_s = -1e9
        # Mean Speed
        cur_scene_mean_speed = 0
        unsafe_num_dict = get_initialized_scene_count()
        num_unsafe = 0
        # Status in ("Success", "Failed", "Running")
        task_status = "Running"
        # For D2RL experiment
        lc_maintain_count_dict = dict()
        bv_pdf_dict, bv_action_idx_dict = dict(), dict()
        if exp_mode == "d2rl":
            for vid in realvehicle_id_list:
                lc_maintain_count_dict[vid] = 0
                bv_pdf_dict[vid] = None
                bv_action_idx_dict[vid] = None
        # -----------------------Single Scene Loop-------------------------- #
        world.tick()
        while True:
            if step >= 0:
                if stop == False:
                    if exp_mode == "d2rl":
                        rel_steps, total_steps, num_unsafe = d2rl_loop(
                            step,
                            main_id,
                            realvehicle_id_list,
                            npc_realvehicle_id_list,
                            main_control_mode,
                            bv_action_idx_dict,
                            bv_pdf_dict,
                            monitor,
                            unsafe_num_dict,
                            global_controller,
                            rel_steps,
                            total_steps,
                            num_unsafe,
                            lc_maintain_count_dict,
                        )
                    else:
                        if pool:
                            # Multiprocessing
                            rel_steps, total_steps, num_unsafe = normal_loop_multips(
                                step,
                                main_id,
                                realvehicle_id_list,
                                npc_realvehicle_id_list,
                                cdm_realvehicle_id_list,
                                main_control_mode,
                                monitor,
                                unsafe_num_dict,
                                pred_net,
                                rel_steps,
                                total_steps,
                                num_unsafe,
                                pool,
                            )
                        else:
                            # Normal (Recommend)
                            rel_steps, total_steps, num_unsafe = normal_loop(
                                step,
                                main_id,
                                realvehicle_id_list,
                                npc_realvehicle_id_list,
                                main_control_mode,
                                monitor,
                                unsafe_num_dict,
                                pred_net,
                                rel_steps,
                                total_steps,
                                num_unsafe,
                            )

                    last_main_s = main_s
                    main_wp = dgv.get_map().get_waypoint(main_vehicle.get_location())
                    main_s = emath.cal_total_in_round_length(main_wp)
                    # Judge drive a round
                    if last_main_s > main_s:
                        num_round += 1
                    # Time or distance satisfy
                    if (duration == None and num_round >= total_round) or (
                        duration != None and step > duration // gv.STEP_DT
                    ):
                        stop = True
                        cur_step = step
                        task_status = "Success"
                        cur_scene_mean_speed = (
                            (num_round * gv.ROUND_LENGTH + main_s)
                            / ((step + 1) * gv.STEP_DT + 1e-9)
                            if step > 0
                            else dgv.get_realvehicle(main_id).scalar_velocity
                        )
                        single_scene_log(
                            scene_idx + 1,
                            cur_scene_mean_speed,
                            task_status,
                            unsafe_num_dict,
                        )
                    # Control
                    for car in dgv.get_realvehicles():
                        car.descrete_control()
                    # Spectator follow main car
                    spectator_tranform = carla.Transform(
                        main_vehicle.get_location() + carla.Location(z=120),
                        carla.Rotation(pitch=-90, yaw=180),
                    )
                    spectator.set_transform(spectator_tranform)
                    # Judge collision
                    for i in range(num_cars):
                        for j in range(i + 1, num_cars):
                            if dgv.get_realvehicle(
                                realvehicle_id_list[i]
                            ).collision_callback(
                                dgv.get_realvehicle(realvehicle_id_list[j]).vehicle
                            ):
                                if (
                                    realvehicle_id_list[i] != main_id
                                    and realvehicle_id_list[j] != main_id
                                ):
                                    unsafe_num_dict["NPCsCollision"] += 1
                                    collision_type = "NpcCollision"
                                else:
                                    unsafe_num_dict["Collision"] += 1
                                    if realvehicle_id_list[i] == main_id:
                                        main_real_vehicle = dgv.get_realvehicle(
                                            realvehicle_id_list[i]
                                        )
                                        npc_real_vehicle = dgv.get_realvehicle(
                                            realvehicle_id_list[j]
                                        )
                                    if realvehicle_id_list[j] == main_id:
                                        main_real_vehicle = dgv.get_realvehicle(
                                            realvehicle_id_list[j]
                                        )
                                        npc_real_vehicle = dgv.get_realvehicle(
                                            realvehicle_id_list[i]
                                        )
                                    collision_type = collision_classify(
                                        main_real_vehicle, npc_real_vehicle
                                    )
                                    collision_num_dict[collision_type] += 1
                                cur_scene_mean_speed = (
                                    (num_round * gv.ROUND_LENGTH + main_s)
                                    / ((step + 1) * gv.STEP_DT + 1e-9)
                                    if step > 0
                                    else dgv.get_realvehicle(main_id).scalar_velocity
                                )
                                single_scene_log(
                                    scene_idx + 1,
                                    cur_scene_mean_speed,
                                    task_status,
                                    unsafe_num_dict,
                                    collision_type=collision_type,
                                )
                                stop = True
                                cur_step = step
                                break
                # ------------------------Single Scene End-------------------------- #
                if stop == True:
                    if step - cur_step >= stop_lifetime:
                        total_time += int(cur_step * gv.STEP_DT)
                        all_scene_count = dict(
                            Counter(all_scene_count) + Counter(unsafe_num_dict)
                        )
                        all_scene_rel_steps += rel_steps
                        all_scene_steps += total_steps
                        all_scene_mean_speed += cur_scene_mean_speed
                        # Clean the environment
                        for car in dgv.get_realvehicles():
                            car.vehicle.destroy()
                        realvehicle_id_list = []
                        num_cars = 0
                        break
                # Update world
                world.tick()
                step += 1
            else:
                world.tick()
                step += 1
    print(time.time() - start_time)
    save_result(
        datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
        num_scenes,
        os.path.abspath(scene_path),
        npc_control_mode,
        all_scene_mean_speed / num_scenes,
        duration,
        {key: value / total_time for key, value in all_scene_count.items()},
        collision_num_dict,
        all_scene_rel_steps / all_scene_steps,
        DECISION_MATRIX,
    )
    decision_exp_reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args")
    parser.add_argument(
        "-n",
        "--npcmode",
        default="acdm",
        help="Npc car control mode, choice: ['d2rl', 'acdm', 'cdm']",
    )
    parser.add_argument(
        "-e",
        "--egomode",
        default="cdm",
        help="Scene library, choose in [cdm, idm, na_cdm, na_idm]",
    )
    parser.add_argument(
        "-p",
        "--path",
        default="",
        help="Path",
    )
    parser.add_argument(
        "-s",
        "--selection-type",
        default="ttc",
        help="Scene Selection Type, choose in [ttc, risk]",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=10,
        help="Testing time for each scene, set None for 3km experiment.",
    )
    parser.add_argument(
        "-t",
        "--scenetype",
        default="short",
        help="Scene Type, choose in [long, short]",
    )
    parser.add_argument(
        "-m",
        "--multiprocessing",
        action="store_true",
        help="If using multiprocess version for CDM/ACDM",
    )
    args = parser.parse_args()
    if args.scenetype == "short":
        scenefile_path = (
            "scenelibrary/"
            + args.scenetype
            + "scenes/"
            + args.selection_type
            + "_scene_"
            + args.egomode
            + ".csv"
        )
    else:
        scenefile_path = (
            "scenelibrary/"
            + args.scenetype
            + "scenes/"
            + "select_scene_exp_"
            + args.egomode
            + ".csv"
        )
    print("Loaded", scenefile_path)
    duration = None if args.duration == "None" else args.duration
    if args.path:
        scenefile_path = args.path
    if args.npcmode not in ["d2rl", "acdm", "cdm", "idm"]:
        raise ValueError("Wrong npc mode name!")
    if args.multiprocessing:
        with multiprocessing.Pool(processes=(os.cpu_count() // 2) - 2) as pool:
            run_experiment(scenefile_path, args.scenetype, duration, args.npcmode, pool)
    else:
        run_experiment(scenefile_path, args.scenetype, duration, args.npcmode)
