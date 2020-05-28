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

        self.total_vacuum_actions = 0

    def do_move(self, action):
        key, value = action
        if key == "move":
            super().do_move(value)
        else:
            super().do_move([0.0, 0.0])

    def do_grasp(self, action):
        key, value = action
        if key == "vacuum":
            grasps =  super().do_grasp(value)
            # super().do_move([0.2, 0.2])
            return grasps
        else:
            return 0

    def reset(self):
        obs = super().reset()
        self.total_vacuum_actions = 0
        return obs

    def step(self, action):
        # init return values
        reward = 0.0
        infos = {}

        # do move
        self.do_move(action)

        # do grasping
        num_grasped = self.do_grasp(action)
        reward += num_grasped

        # if num_grasped == 0:
        #     reward -= 0.1
        # reward -= 0.01
        
        # infos loggin
        infos["success"] = num_grasped
        infos["total_grasped"] = self.total_grasped

        base_pos = self.interface.get_base_pos()
        infos["base_x"] = base_pos[0]
        infos["base_y"] = base_pos[1]

        if action[0] == "vacuum":
            self.total_vacuum_actions += 1

        infos["vacuum_action"] = int(action[0] == "vacuum")
        infos["total_success_to_vacuum_ratio"] = (0 if self.total_vacuum_actions == 0 
                                                    else self.total_grasped / self.total_vacuum_actions)

        # steps update
        self.num_steps += 1
        done = self.num_steps >= self.max_ep_len

        # get next observation
        obs = self.get_observation()

        return obs, reward, done, infos
