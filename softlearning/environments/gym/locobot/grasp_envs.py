import gym
from gym import spaces
import numpy as np
import os
from collections import OrderedDict
import time
from . import locobot_interface

from .nav_envs import RoomEnv
from .utils import *
from .rooms import initialize_room
import math

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
        defaults['aux_camera_look_pos'] = [0.4, 0, 0.05] #0.3821222484111786, -0.060383960604667664, 0.08656357228755951
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


class LocobotContinuousMultistepGraspingEnv(RoomEnv):
    def __init__(self, **params):
        defaults = dict()
        
        room_name = "grasping"
        room_params = dict(
            min_objects=1,
            max_objects=5,
            object_name="greensquareball", 
            spawn_loc=[0.36, 0],
            spawn_radius=0.3,
        )
        defaults['height_hack'] = True
        defaults['room_name'] = room_name
        defaults['room_params'] = room_params
        defaults['use_aux_camera'] = True
        defaults['aux_camera_look_pos'] = [0.4, 0, 0.05]
        defaults['aux_camera_fov'] = 35
        defaults['aux_image_size'] = 100
        defaults['observation_space'] = spaces.Dict()
        #defaults['action_space'] = spaces.Discrete(15 * 31)
        defaults['max_ep_len'] = 15
        defaults["observation_space"] = spaces.Dict({
            "current_ee": spaces.Box(low=-1.0, high=1.0, shape=(5,)),
            # "pixels": added by PixelObservationWrapper
        })
        defaults['height_hack'] = False

#         defaults["use_gripper_cam"] = True
        defaults.update(params)
        self.height_hack = defaults['height_hack']
        if self.height_hack:
            defaults["action_space"] = spaces.Box(low=-1.0, high=1.0, shape=(4,))

        else:
            defaults["action_space"] = spaces.Box(low=-1.0, high=1.0, shape=(5,))
            
        super().__init__(**defaults)
        self.local_ee_mins = np.array([0.09391462802886963, -0.0514984130859375, -0.09128464758396149, -math.pi, 0.])
        self.local_ee_maxes = np.array([0.5971627235412598, 0.13770374655723572, 0.6860376000404358, math.pi, 0.02])

        self.action_max = np.array([0.1, 0.1, 0.1, 0.5, 0.02])
        self.action_min = np.array([-0.1, -0.1, -0.1, -0.5, 0.001])

        #self.discretizer = Discretizer([15, 31], [0.3, -0.16], [0.4666666, 0.16])
        #self.num_repeat = 10
        #self.num_steps_this_env = self.num_repeat

#         self.use_gripper_cam = defaults['use_gripper_cam']

    def normalize_ee(self, local_ee):
        #print("localee", local_ee)
        ee = local_ee-(self.local_ee_maxes+self.local_ee_mins)/2
        ee = ee/(self.local_ee_maxes-self.local_ee_mins)
        return ee
    
    def denormalize_action(self, action):
        """ Action is between -1 and 1"""
        #print("action before", action)
        action = np.clip(action, -1, 1)
        action = ((action+1) / 2)  * (self.action_max-self.action_min)
        action = action + self.action_min
        #print("action after", action)
        return action
        
    def do_grasp(self, a):
        self.interface.apply_continuous_action(self.denormalize_action(a))
        #self.interface.execute_grasp_direct(loc, 0.0)
        reward = 0
        distances = []
        ee_global, _ = self.interface.get_ee_global()
        ee_global = np.array(ee_global)
        
#         if ee_global[2] <= 0.1:
#             self.interface.close_gripper()
#             self._attempted_grasp = True

        
        for i in range(self.room.num_objects):
            block_pos, _ = self.interface.get_object(self.room.objects_id[i])
            distances.append(np.linalg.norm(ee_global-np.array(block_pos)))
            if block_pos[2] > 0.04:
                reward = 1
                self.interface.move_object(
                    self.room.objects_id[i], 
                    [self.room.extent + np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.01])
                break
        #self.interface.move_arm_to_start(steps=90, max_velocity=8.0)
        if reward == 0:
            reward = -min(distances)
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
        t = time.time()
        print("reset", t)
        if True: #self.num_steps_this_env >= self.num_repeat or self.should_reset():
            self.interface.reset_robot([0, 0], 0, 0, 0)
            while True:
                self.room.reset()
                if self.are_blocks_graspable():
                    break
            self.num_steps_this_env = 0
            self.interface.move_arm_to_start(steps=90, max_velocity=8.0)
        return self.get_observation()
    
    def render(self, *args, **kwargs):
        main_img = self.interface.render_camera(use_aux=True)
#         if self.use_gripper_cam:
#             gripper_img = self.interface.render_camera(use_aux=True, link=18)
#             return (main_img, gripper_img)
        return main_img

    def get_observation(self):
        obs = OrderedDict()

        if self.interface.renders:
            # pixel observations are generated by PixelObservationWrapper, unless we want to manually check it
            obs["pixels"] = self.render()
        
        ee_local, _ = self.interface.get_ee_local()
        wrist_state = np.array(self.interface.get_wrist_state())
#         print("ee",ee_local)
#         print("wrist", wrist_state)
#         print("cat", np.concatenate([ee_local, wrist_state]))
        ee_local = self.normalize_ee(np.concatenate([ee_local, wrist_state]))
        obs["current_ee"] = ee_local        
        return obs

    def step(self, action):
#         action_discrete = int(action)
#         action_undiscretized = self.discretizer.undiscretize(self.discretizer.unflatten(action_discrete))
        if self.height_hack:
            new_action = np.zeros(5)
            new_action[:2] = action[:2]
            new_action[3:] = action[2:]
            new_action[2] = -1.
        reward = self.do_grasp(action)

        obs = self.get_observation()
        infos = {'grasp_success': 0}
        if reward == 1:
            infos['grasp_success'] = 1
        return obs, reward, reward > 0, infos

