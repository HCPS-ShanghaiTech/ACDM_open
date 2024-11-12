"""
Dynamic global dict
"""

import carla

from vehicles.realvehicle import RealVehicle
from typing import Dict, Any


def get_map():
    if carla_map == None:
        raise ValueError()
    return carla_map


def set_map(map):
    global carla_map
    carla_map = map


def get_realvehicle(key):
    return full_realvehicle_dict.get(key)


def reset_realvehicle_dict():
    global full_realvehicle_dict
    full_realvehicle_dict = dict()


def update_realvehicle_dict(key, value):
    full_realvehicle_dict[key] = value


def get_realvehicle_id_list():
    return list(full_realvehicle_dict.keys())


def get_realvehicles():
    return list(full_realvehicle_dict.values())


def pop_realvehicle(key):
    return full_realvehicle_dict.pop(key)


carla_map = None
full_realvehicle_dict: Dict[Any, RealVehicle] = dict()
