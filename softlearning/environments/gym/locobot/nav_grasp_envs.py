import gym
from gym import spaces
import numpy as np
import os
from collections import OrderedDict

from . import locobot_interface

from .base_env import LocobotBaseEnv
from .utils import *
from .rooms import initialize_room
from .nav_envs import *

from softlearning.environments.gym.spaces import DiscreteBox

class LocobotNavigationVacuumEnv(MixedLocobotNavigationEnv):
    def __init__(self, **params):
        defaults = dict()
        defaults["action_space"] = DiscreteBox(
            low=-1.0, high=1.0, 
            dimensions=OrderedDict((("move", 2), ("vacuum", 0)))
        )
        defaults.update(params)

        super().__init__(**defaults)
        print("LocobotNavigationVacuumEnv params:", self.params)

    def do_move(self, action):
        key, value = action
        if key == "move":
            super().do_move(value)
        else:
            super().do_move([0.0, 0.0])

    def do_grasp(self, action):
        key, value = action
        if key == "vacuum":
            return super().do_grasp(value)
        else:
            return 0

