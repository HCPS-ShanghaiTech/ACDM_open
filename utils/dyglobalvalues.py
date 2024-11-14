"""
Dynamic global dict
"""

import carla

from vehicles.realvehicle import RealVehicle
from typing import Dict, Any

CARLA_MAP = carla.Client("localhost", 2000).get_world().get_map()
FULL_REALVEHICLE_DICT: Dict[Any, RealVehicle] = dict()


def get_map():
    if CARLA_MAP == None:
        raise ValueError()
    return CARLA_MAP


def set_map(map):
    global CARLA_MAP
    CARLA_MAP = map


def get_realvehicle(key):
    return FULL_REALVEHICLE_DICT.get(key)


def reset_realvehicle_dict():
    global FULL_REALVEHICLE_DICT
    FULL_REALVEHICLE_DICT = dict()


def update_realvehicle_dict(key, value):
    FULL_REALVEHICLE_DICT[key] = value


def get_realvehicle_id_list():
    return list(FULL_REALVEHICLE_DICT.keys())


def get_realvehicles():
    return list(FULL_REALVEHICLE_DICT.values())


def pop_realvehicle(key):
    return FULL_REALVEHICLE_DICT.pop(key)
