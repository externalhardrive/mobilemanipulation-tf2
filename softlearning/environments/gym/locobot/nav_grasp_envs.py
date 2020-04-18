import gym
from gym import spaces
import numpy as np
import os

from . import locobot_interface
from softlearning.environments.helpers import random_point_in_circle

from .base_env import LocobotBaseEnv
from .utils import *

class ImageLocobotNavigationGraspingEnv(LocobotBaseEnv):
    """ A field with walls containing lots of balls, the robot grasps one and it disappears.
        Resets periodically. """
    def __init__(self, **params):
        defaults = dict(
            num_objects=100, object_name="greensquareball", wall_size=5.0, 
            max_ep_len=200, use_dist_reward=True, grasp_reward=200)

        defaults["observation_type"] = "image"
        defaults["action_dim"] = 2
        defaults["image_size"] = locobot_interface.IMAGE_SIZE
        # setup aux camera for grasping policy
        defaults["camera_fov"] = 55
        defaults["use_aux_camera"] = True
        defaults["aux_image_size"] = 84
        defaults["aux_camera_fov"] = 25
        defaults["aux_camera_look_pos"] = np.array([0.42, 0, 0.02])
        defaults.update(params)

        super().__init__(**defaults)

        print("params:", self.params)

        self.wall_size = self.params["wall_size"]
        self.interface.spawn_walls(self.wall_size)

        self.robot_yaw = 0.0
        self.robot_pos = np.array([0.0, 0.0])

        self.num_objects = self.params["num_objects"]
        self.objects_id = [self.interface.spawn_object(URDF[self.params["object_name"]], np.array([0.0, 0.0, 10 + i])) for i in range(self.num_objects)]

        self.use_dist_reward = self.params["use_dist_reward"]
        self.grasp_reward = self.params["grasp_reward"]
        self.max_ep_len = self.params["max_ep_len"]
        self.num_steps = 0
        self.num_successes = 0

        self.import_baselines_ppo_model()

    def import_baselines_ppo_model(self):
        from baselines.ppo2.model import Model
        from baselines.common.policies import build_policy
        
        class GraspingEnvPlaceHolder:
            observation_space = spaces.Box(low=0, high=1., shape=(84, 84, 3))
            action_space = spaces.Box(-np.ones(2), np.ones(2))

        policy = build_policy(GraspingEnvPlaceHolder, 'cnn')

        self.model = Model(policy=policy, 
                    ob_space=GraspingEnvPlaceHolder.observation_space, 
                    ac_space=GraspingEnvPlaceHolder.action_space,
                    nbatch_act=1, nbatch_train=32, nsteps=128, ent_coef=0.01, vf_coef=0.5, 
                    max_grad_norm=0.5, comm=None, mpi_rank_weight=1)

        # NOTE: This line must be called before any other tensorflow neural network is initialized
        # TODO: Fix this so that it doesn't use tf.GraphKeys.GLOBAL_VARIABLES in tf_utils.py
        self.model.load(os.path.join(CURR_PATH, "baselines_models/balls"))

    def reset(self):
        self.interface.set_base_pos_and_yaw(self.robot_pos, self.robot_yaw)

        for i in range(self.num_objects):
            while True:
                x, y = np.random.uniform(-self.wall_size * 0.5, self.wall_size * 0.5, size=(2,))
                if not is_in_circle(x, y, 0, 0, 0.2):
                    break
            self.interface.move_object(self.objects_id[i], [x, y, 0.015])

        self.interface.move_joints_to_start()

        self.num_steps = 0
        self.num_successes = 0

        self.reset_stacked_obs()
        obs = self.render()
        
        return obs

    def get_grasp_prob(self, aux_obs):
        return self.model.value(aux_obs)[0]

    def get_grasp_loc(self, aux_obs, noise=0.01):
        grasp_loc, _, _, _ = self.model.step(aux_obs)
        grasp_loc = grasp_loc[0] * np.array([0.04, 0.12])
        grasp_loc += np.random.normal(0, noise, (2,))
        return grasp_loc

    def do_grasp(self, grasp_loc):
        self.interface.execute_grasp(grasp_loc, 0.0)
        reward = 0
        for i in range(self.num_objects):
            block_pos, _ = self.interface.get_object(self.objects_id[i])
            if block_pos[2] > 0.08:
                reward = 1
                self.interface.move_object(self.objects_id[i], [self.wall_size * 3.0, 0, 1])
                break
        self.interface.move_joints_to_start()
        return reward

    def get_closest_object_dist_sq(self, return_id=False):
        min_dist_sq = float('inf')
        min_object_id = None
        for i in range(self.num_objects):
            block_pos, _ = self.interface.get_object(self.objects_id[i], relative=True)
            curr_dist_sq = (block_pos[0] - 0.42) ** 2 + block_pos[1] ** 2
            if curr_dist_sq < min_dist_sq:
                min_dist_sq = curr_dist_sq
                min_object_id = i
        if return_id:
            return min_dist_sq, min_object_id
        else:
            return min_dist_sq

    def step(self, a):
        a = np.array(a)

        self.interface.move_base(a[0] * 10.0, a[1] * 10.0)

        reward = 0.0
        if self.use_dist_reward:
            dist_sq = self.get_closest_object_dist_sq(return_id=False)
            reward = -np.sqrt(dist_sq)

        aux_obs = self.interface.render_camera(use_aux=True)
        v = self.get_grasp_prob(aux_obs)

        if v > 0.85:
            grasp_loc = self.get_grasp_loc(aux_obs)
            grasp_succ = self.do_grasp(grasp_loc)
            if grasp_succ:
                reward += self.grasp_reward
                self.num_successes += 1

        if self.interface.renders:
            obs = self.render()
        else:
            # pixel observations are automatically generated by PixelObservationWrapper
            obs = None

        self.num_steps += 1
        done = self.num_steps >= self.max_ep_len
        infos = {}

        if done:
            infos["num_successes"] = self.num_successes

        return obs, reward, done, infos

class ImageLocobotNavigationEnv(ImageLocobotNavigationGraspingEnv):
    """ No Grasping, just uses -distance squared """
    def step(self, a):
        a = np.array(a)

        self.interface.move_base(a[0] * 10.0, a[1] * 10.0)

        reward = 0.0

        dist_sq, object_id = self.get_closest_object_dist_sq(return_id=True)
        if self.use_dist_reward:
            reward = -np.sqrt(dist_sq)

        success = 0
        object_pos, _ = self.interface.get_object(self.objects_id[object_id], relative=True)
        if is_in_rect(object_pos[0], object_pos[1], 0.42 - 0.04, -0.12, 0.42 + 0.04, 0.12):
            self.interface.move_object(self.objects_id[object_id], [self.wall_size * 3.0, 0, 1])
            reward += self.grasp_reward
            success = 1
            self.num_successes += 1
            
        if self.interface.renders:
            obs = self.render()
        else:
            # pixel observations are automatically generated by PixelObservationWrapper
            obs = None

        self.num_steps += 1
        done = self.num_steps >= self.max_ep_len

        infos = {"success": success}
        # if done:
        #     infos["num_successes"] = self.num_successes
            # print("num success:", self.num_successes)

        return obs, reward, done, infos

    def import_baselines_ppo_model(self):
        return 