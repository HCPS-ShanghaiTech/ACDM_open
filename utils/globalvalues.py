"""
Static global values
"""

# Car info(tesla model3)
CAR_LENGTH = 4.791779518127441
CAR_WIDTH = 2.163450002670288
CAR_HEIGHT = 1.4876600503921509
COLOR_DICT = {"red": "255, 0, 0", "green": "0, 255, 0", "white": "255, 255, 255"}
COLOR_LIST = ["red", "green", "white"] * 10
# CDM/ACDM/IDM decision gap
DECISION_DT = 1
# D2RL decision gap
D2RL_DECISION_DT = 0.1
# Update gap
STEP_DT = 0.02
# Seconds for lane change
LANE_CHANGE_TIME = 1
# Road info
NUM_LANE = 2
LANE_ID = {"Left": -2, "Right": -3}
MIN_SPEED = 60 / 3.6
LANE_WIDTH = 4
# Metre
OBSERVE_DISTANCE = 100
# Observation angle range
FOV = 120
# Max decision tree number
NUM_OF_TREE = 3
# Road ID and Length
ROADS_LENGTH = {
    0: [0.00, 0.00],
    37: [814.84, 0.00],
    2344: [28.99, 814.84],
    38: [300.24, 843.83],
    12: [24.93, 1144.07],
    34: [276.20, 1169.00],
    35: [21.07, 1445.20],
    2035: [29.01, 1466.27],
    36: [12.21, 1495.28],
}  # [road length, total length from start point]
# Length for a round
ROUND_LENGTH = 1507.49
# Action space
ACTION_SPACE = ["MAINTAIN", "ACCELERATE", "DECELERATE", "SLIDE_LEFT", "SLIDE_RIGHT"]
LON_ACTIONS = ["MAINTAIN", "ACCELERATE", "DECELERATE"]
LAT_ACTIONS = ["SLIDE_LEFT", "SLIDE_RIGHT"]
# Acceleration map
LON_ACC_DICT = {
    "MAINTAIN": 0,
    "ACCELERATE": 1,
    "DECELERATE": -3,
    "SLIDE_LEFT": 0,
    "SLIDE_RIGHT": 0,
}
# CDM Reward
ACTION_SCORE = {
    "MAINTAIN": 3,
    "ACCELERATE": 4,
    "DECELERATE": 1,
    "SLIDE_LEFT": 3,
    "SLIDE_RIGHT": 3,
}
# NN info
INPUT_SIZE = 5
HIDDEN_SIZE = 64
NUM_CLASSES = 4
NUM_LAYERS = 2
# IDM params
TYPICAL_IDM_VALUE = {
    "DesiredSpeed": 108 / 3.6,
    "TimeGap": 1.0,
    "MinimumGap": 2,
    "AccelerationExp": 4,
    "Acceleration": 1.0,
    "ComfortDeceleration": -2,
    "Politness": 1,
}
