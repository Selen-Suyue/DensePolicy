import numpy as np

from utils.constants import *


TO_TENSOR_KEYS = ['input_coords_list', 'input_feats_list', 'action', 'action_normalized', 'action_obj_normalized', 'qpos', 'qpos_normalized']

# camera intrinsics
INTRINSICS = {
    "043322070878": np.array([[909.72656250, 0, 645.75042725, 0],
                              [0, 909.66497803, 349.66162109, 0],
                              [0, 0, 1, 0]]),
    "750612070851": np.array([[922.37457275, 0, 637.55419922, 0],
                              [0, 922.46069336, 368.37557983, 0],
                              [0, 0, 1, 0]]),
    "104122060902": np.array([[915.384521484375, 0, 633.3715209960938, 0],
                              [0, 914.9421997070312, 354.1505432128906, 0],
                              [0, 0, 1, 0]])
}

# For MBA and DSP
INHAND_CAM = ["104422070044"]

# transformation matrix from inhand camera (corresponds to INHAND_CAM[0]) to tcp
INHAND_CAM_TCP = np.array([
    [0, -1, 0, 0],
    [1, 0, 0, 0.077],
    [0, 0, 1, 0.1865],
    [0, 0, 0, 1]
])

# For RISE
RISE_INHAND_CAM = ["043322070878"]

# transformation matrix from inhand camera (corresponds to INHAND_CAM[0]) to tcp
RISE_INHAND_CAM_TCP = np.array([
    [0, -1, 0, 0],
    [1, 0, 0, 0.077],
    [0, 0, 1, 0.2665],
    [0, 0, 0, 1]
])