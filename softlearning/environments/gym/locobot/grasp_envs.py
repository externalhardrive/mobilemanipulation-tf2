import gym
from gym import spaces
import numpy as np
import os
from collections import OrderedDict

from . import locobot_interface

from .nav_envs import RoomEnv
from .utils import *
from .rooms import initialize_room

class LocobotDiscreteGraspingEnv(RoomEnv):
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
        defaults['observation_space'] = None
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
        
        reward = env.do_grasp(action_undiscretized)
        self.num_steps_this_env += 1

        obs = self.reset()

        return obs, reward, True, {}

