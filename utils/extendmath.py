"""
Extended math calculator
"""

import carla
import math
import utils.dyglobalvalues as dgv
from typing import Dict, Any
from utils import pickleable
from vehicles.virtualvehicle import VirtualVehicle
from utils.globalvalues import (
    ROADS_LENGTH,
    ROUND_LENGTH,
)


def cal_length(location):
    """
    Length doe (x,y) in location
    """
    return carla.Vector2D(x=location.x, y=location.y).length()


def cal_total_in_round_length(input_wp):
    """
    Calculate the distance from the starting point in a circle
    """
    if ROADS_LENGTH.get(input_wp.road_id):
        return input_wp.s + ROADS_LENGTH.get(input_wp.road_id)[1]
    else:
        return 0


def is_in_front(transform_a, transform_b):
    """
    Determine whether car A is in front of car B using geometric methods
    """

    pos_a = transform_a.location
    pos_b = transform_b.location

    forward_vector_b = transform_b.get_forward_vector()

    relative_vector = pos_a - pos_b

    dot_product = forward_vector_b.dot(relative_vector)

    return dot_product > 0


def cal_distance_along_road(start_wp, target_wp):
    """
    Determine the relative position of two vehicles along the road surface, target relative to start
    """
    start_s, target_s = cal_total_in_round_length(start_wp), cal_total_in_round_length(
        target_wp
    )

    # Same round
    distance_case_1 = target_s - start_s

    # Start waypoint at current round, target waypoint at next round
    distance_case_2 = target_s + ROUND_LENGTH - start_s

    # Start waypoint at next round, target waypoint at current round
    distance_case_3 = target_s - (start_s + ROUND_LENGTH)

    return min([distance_case_1, distance_case_2, distance_case_3], key=abs)


def cal_rel_location_curve(target_location, start_location):
    """
    Calculate the position of the target relative to the start on a curved road, straighten the curved road, and then calculate
    """
    carla_map = dgv.get_map()
    if type(start_location) == pickleable.Location:
        start_location = pickleable.from_location_to_carla(start_location)
    if type(target_location) == pickleable.Location:
        target_location = pickleable.from_location_to_carla(target_location)
    start_wp, target_wp = carla_map.get_waypoint(
        start_location
    ), carla_map.get_waypoint(target_location)
    result_location = carla.Location(0, 0, 0)
    # Now only complete 2-lane version
    result_location.y = abs(start_wp.lane_id - target_wp.lane_id) * start_wp.lane_width
    result_location.z = 0  # will not use
    # x means longitude distance
    result_location.x = cal_distance_along_road(start_wp, target_wp)

    return result_location


def get_front_closest_vehicle(virtual_vehicle_dict: Dict[Any, VirtualVehicle], ego_id):
    """
    Filter out the nearest virtual_vehicle ahead
    """
    ego_v_vehicle = virtual_vehicle_dict.get(ego_id)
    min_dist_front = 1e9
    min_vid_front = None
    ego_lane_id = ego_v_vehicle.waypoint.lane_id
    for vid in virtual_vehicle_dict.keys():
        if vid != ego_id:
            v_vehicle = virtual_vehicle_dict.get(vid)
            rel_distance = cal_distance_along_road(
                ego_v_vehicle.waypoint, v_vehicle.waypoint
            )
            if v_vehicle.waypoint.lane_id == ego_lane_id:
                if 0 < rel_distance < min_dist_front:
                    min_dist_front = rel_distance
                    min_vid_front = vid
    return min_vid_front, min_dist_front


def get_side_closest_vehicle(virtual_vehicle_dict: Dict[Any, VirtualVehicle], ego_id):
    """
    Filter out the nearest virtual_vehicle back
    """
    ego_v_vehicle = virtual_vehicle_dict.get(ego_id)
    min_dist_side = 1e9
    min_vid_side = None
    ego_lane_id = ego_v_vehicle.waypoint.lane_id
    for vid in virtual_vehicle_dict.keys():
        if vid != ego_id:
            v_vehicle = virtual_vehicle_dict.get(vid)
            rel_distance = cal_distance_along_road(
                ego_v_vehicle.waypoint, v_vehicle.waypoint
            )
            if v_vehicle.waypoint.lane_id != ego_lane_id:
                if abs(rel_distance) < min_dist_side:
                    min_dist_side = abs(rel_distance)
                    min_vid_side = vid
    return min_vid_side, min_dist_side


def add_height_to_waypoint(waypoint_transform, vehicle_height):
    """
    Add height values along the normal vector of the road
    """
    location = waypoint_transform.location
    rotation = waypoint_transform.rotation

    pitch_rad = math.radians(rotation.pitch)
    normal_vector = carla.Location(x=-math.sin(pitch_rad), y=0, z=math.cos(pitch_rad))

    location.x += normal_vector.x * vehicle_height
    location.y += normal_vector.y * vehicle_height
    location.z += normal_vector.z * vehicle_height

    return location
