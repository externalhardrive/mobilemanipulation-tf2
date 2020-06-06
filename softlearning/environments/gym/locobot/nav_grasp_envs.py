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

class LocobotNavigationDQNGraspingEnv(RoomEnv):
    def __init__(self, **params):
        defaults = dict()
        
        room_name = "grasping"
        room_params = dict(
            min_objects=1,
            max_objects=10,
            object_name="greensquareball", 
            spawn_loc=[0.36, 0],
            spawn_radius=0.3,
        )

        defaults['room_name'] = room_name
        defaults['room_params'] = room_params
        defaults['use_aux_camera'] = True
        defaults['aux_camera_look_pos'] = [0.4, 0, 0.05]
        defaults['aux_camera_fov'] = 35
        defaults['aux_image_size'] = 100
        defaults['observation_space'] = spaces.Dict()
        defaults['action_space'] = spaces.Discrete(15 * 31)
        defaults['max_ep_len'] = 1

        defaults.update(params)

        super().__init__(**defaults)

        self.discretizer = Discretizer([15, 31], [0.3, -0.16], [0.4666666, 0.16])
        self.num_repeat = 10
        self.num_steps_this_env = self.num_repeat

    def do_grasp(self, loc):
        self.interface.execute_grasp_direct(loc, 0.0)
        reward = 0
        for i in range(self.room.num_objects):
            block_pos, _ = self.interface.get_object(self.room.objects_id[i])
            if block_pos[2] > 0.04:
                reward = 1
                self.interface.move_object(
                    self.room.objects_id[i], 
                    [self.room.extent + np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.01])
                break
        self.interface.move_arm_to_start(steps=90, max_velocity=8.0)
        return reward

    def are_blocks_graspable(self):
        for i in range(self.room.num_objects):
            object_pos, _ = self.interface.get_object(self.room.objects_id[i], relative=True)
            if is_in_rect(object_pos[0], object_pos[1], 0.3, -0.16, 0.466666666, 0.16):
                return True 
        return False

    def should_reset(self):
        return not self.are_blocks_graspable()

    def reset(self):
        if self.num_steps_this_env >= self.num_repeat or self.should_reset():
            self.interface.reset_robot([0, 0], 0, 0, 0)
            while True:
                self.room.reset()
                if self.are_blocks_graspable():
                    break
            self.num_steps_this_env = 0
        return self.get_observation()
    
    def render(self, *args, **kwargs):
        return self.interface.render_camera(use_aux=True)

    def get_observation(self):
        return self.render()

    def step(self, action):
        action_discrete = int(action)
        action_undiscretized = self.discretizer.undiscretize(self.discretizer.unflatten(action_discrete))
        
        reward = self.do_grasp(action_undiscretized)
        self.num_steps_this_env += 1

        obs = self.reset()

        return obs, reward, True, {}
