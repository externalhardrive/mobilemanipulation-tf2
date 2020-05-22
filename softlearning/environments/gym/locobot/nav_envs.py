import gym
from gym import spaces
import numpy as np
import os
from collections import OrderedDict

from . import locobot_interface

from .base_env import LocobotBaseEnv
from .utils import *
from .rooms import initialize_room

class RoomEnv(LocobotBaseEnv):
    """ A room with objects spread about. """
    def __init__(self, **params):
        defaults = dict(
            room_name="simple",
            room_params={} # use room defaults
        )
        defaults.update(params)

        super().__init__(**defaults)
        print("RoomEnv params:", self.params)

        self.robot_yaw = 0.0
        self.robot_pos = np.array([0.0, 0.0])

        self.room = initialize_room(self.interface, self.params["room_name"], self.params["room_params"])

    def get_observation(self):
        raise NotImplementedError

    def reset(self):
        self.robot_yaw = np.random.uniform(0, np.pi * 2)
        self.interface.reset_robot(self.robot_pos, self.robot_yaw, 0, 0)
        
        self.room.reset()

        self.num_steps = 0

        return self.get_observation()

class BaseNavigationEnv(RoomEnv):
    def __init__(self, **params):
        defaults = dict(trajectory_log_dir=None, trajectory_log_freq=0)

        defaults["max_ep_len"] = 200
        defaults.update(params)

        super().__init__(**defaults)
        print("BaseNavigationEnv params:", self.params)

        self.trajectory_log_dir = self.params["trajectory_log_dir"]
        self.trajectory_log_freq = self.params["trajectory_log_freq"]
        if self.trajectory_log_dir and self.trajectory_log_freq > 0:
            os.makedirs(self.trajectory_log_dir, exist_ok=True)
            uid = str(np.random.randint(1e6))
            self.trajectory_log_path = os.path.join(self.trajectory_log_dir, "trajectory_" + uid + "_")
            self.trajectory_num = 0
            self.trajectory_step = 0
            self.trajectory_base = np.zeros((self.trajectory_log_freq, 2))
            self.trajectory_objects = OrderedDict({})

        self.total_grasped = 0
    
    def reset(self):
        obs = super().reset()
        self.total_grasped = 0

        self.update_trajectory_objects()

        return obs

    def update_trajectory_objects(self):
        if self.trajectory_log_dir and self.trajectory_log_freq > 0:
            objects = np.zeros((self.room.num_objects, 2))
            for i in range(self.room.num_objects):
                object_pos, _ = self.interface.get_object(self.room.objects_id[i], relative=False)
                objects[i, 0] = object_pos[0]
                objects[i, 1] = object_pos[1]
            self.trajectory_objects[self.trajectory_step] = objects

    def do_move(self, action):
        self.interface.move_base(action[0] * 10.0, action[1] * 10.0)
    
    def do_grasp(self, action):
        """ returns number of object picked up """
        success = 0
        for i in range(self.room.num_objects):
            object_pos, _ = self.interface.get_object(self.room.objects_id[i], relative=True)
            if is_in_rect(object_pos[0], object_pos[1], 0.42 - 0.04, -0.12, 0.42 + 0.04, 0.12):
                success += 1
                self.total_grasped += 1
                self.interface.move_object(self.room.objects_id[i], [self.room.extent * 3.0, 0, 1])
                
                self.update_trajectory_objects()

                break

        return success

    def step(self, action):
        # init return values
        reward = 0.0
        infos = {}

        # do move
        self.do_move(action)

        # do grasping
        num_grasped = self.do_grasp(action)
        reward += num_grasped
        
        # infos loggin
        infos["success"] = num_grasped
        infos["total_grasped"] = self.total_grasped

        base_pos = self.interface.get_base_pos()
        infos["base_x"] = base_pos[0]
        infos["base_y"] = base_pos[1]

        # store trajectory information (usually for reset free)
        if self.trajectory_log_dir and self.trajectory_log_freq > 0:
            self.trajectory_base[self.trajectory_step, 0] = base_pos[0]
            self.trajectory_base[self.trajectory_step, 1] = base_pos[1]
            self.trajectory_step += 1
            
            if self.trajectory_step == self.trajectory_log_freq:
                self.trajectory_step -= 1 # for updating trajectory
                self.update_trajectory_objects()

                self.trajectory_step = 0
                self.trajectory_num += 1

                data = OrderedDict({
                    "base": self.trajectory_base,
                    "objects": self.trajectory_objects
                })

                np.save(self.trajectory_log_path + str(self.trajectory_num), data)
                self.trajectory_objects = OrderedDict({})

        # steps update
        self.num_steps += 1
        done = self.num_steps >= self.max_ep_len

        # get next observation
        obs = self.get_observation()

        return obs, reward, done, infos

class ImageLocobotNavigationEnv(BaseNavigationEnv):
    """ A room with the robot moves uses distance control and auto picks up. """
    def __init__(self, **params):
        defaults = dict()

        defaults["observation_type"] = "image"
        defaults["action_dim"] = 2
        defaults["image_size"] = 100
        defaults["camera_fov"] = 55
        defaults.update(params)

        super().__init__(**defaults)
        print("ImageLocobotNavigationEnv params:", self.params)

    def get_observation(self):
        obs = OrderedDict()
        if self.interface.renders:
            # pixel observations are generated by PixelObservationWrapper
            obs['pixels'] = self.render()
        
        return obs

class MixedLocobotNavigationEnv(BaseNavigationEnv):
    """ A room with the robot moves around using velocity control and auto picks up. """
    def __init__(self, **params):
        defaults = dict(steps_per_second=2, max_velocity=20.0, max_acceleration=4.0)

        defaults["observation_space"] = spaces.Dict({
            "current_velocity": spaces.Box(low=-1.0, high=1.0, shape=(2,)),
            # "target_velocity": spaces.Box(low=-1.0, high=1.0, shape=(2,)),
            # "pixels": added by PixelObservationWrapper
        })
        defaults["action_dim"] = 2
        defaults["image_size"] = 100
        defaults["camera_fov"] = 55
        defaults.update(params)

        super().__init__(**defaults)
        print("MixedLocobotNavigationEnv params:", self.params)

        self.num_sim_steps_per_env_step = int(60 / self.params["steps_per_second"])
        self.max_velocity = self.params["max_velocity"]
        # self.velocity_change_scale = self.params["max_acceleration"] / self.params["steps_per_second"]
        self.target_velocity = np.array([0.0, 0.0])

    def reset(self):
        _ = super().reset()
        
        self.target_velocity = np.array([self.max_velocity * 0.2] * 2)
        self.interface.set_wheels_velocity(self.target_velocity[0], self.target_velocity[1])
        for _ in range(60):
            self.interface.step()
        
        self.total_grasped = 0

        obs = self.get_observation()
        return obs

    def get_observation(self):
        obs = OrderedDict()

        if self.interface.renders:
            # pixel observations are generated by PixelObservationWrapper, unless we want to manually check it
            obs["pixels"] = self.render()
        
        velocity = self.interface.get_wheels_velocity()
        obs["current_velocity"] = np.clip(velocity / self.max_velocity, -1.0, 1.0)
        # obs["target_velocity"] = np.clip(self.target_velocity / self.max_velocity, -1.0, 1.0)
        
        return obs

    def do_move(self, action):
        self.target_velocity = np.array(action) * self.max_velocity
        new_left, new_right = self.target_velocity

        self.interface.set_wheels_velocity(new_left, new_right)
        self.interface.do_steps(self.num_sim_steps_per_env_step)

class MixedLocobotNavigationReachEnv(MixedLocobotNavigationEnv):
    """ Object wouldn't be picked up. The goal is to stop at a block. """

    def do_grasp(self, action):
        """ returns number of object picked up """
        success = 0
        for i in range(self.room.num_objects):
            object_pos, _ = self.interface.get_object(self.room.objects_id[i], relative=True)
            if is_in_rect(object_pos[0], object_pos[1], 0.42 - 0.04, -0.12, 0.42 + 0.04, 0.12):
                success += 1
                self.total_grasped += 1
                break

        return success 

# class ImageLocobotNavigationGraspingEnv(ImageLocobotNavigationEnv):
#     """ A field with walls containing lots of balls, the robot grasps one and it disappears.
#         Resets periodically. """
#     def __init__(self, **params):
#         defaults = dict()
#         defaults["observation_type"] = "image"
#         defaults["action_dim"] = 2
#         defaults["image_size"] = locobot_interface.IMAGE_SIZE
#         # setup aux camera for grasping policy
#         defaults["camera_fov"] = 55
#         defaults["use_aux_camera"] = True
#         defaults["aux_image_size"] = 84
#         defaults["aux_camera_fov"] = 25
#         defaults["aux_camera_look_pos"] = np.array([0.42, 0, 0.02])
#         defaults.update(params)

#         super().__init__(**defaults)

#         print("params:", self.params)

#         self.import_baselines_ppo_model()

#     def import_baselines_ppo_model(self):
#         from baselines.ppo2.model import Model
#         from baselines.common.policies import build_policy
        
#         class GraspingEnvPlaceHolder:
#             observation_space = spaces.Box(low=0, high=1., shape=(84, 84, 3))
#             action_space = spaces.Box(-np.ones(2), np.ones(2))

#         policy = build_policy(GraspingEnvPlaceHolder, 'cnn')

#         self.model = Model(policy=policy, 
#                     ob_space=GraspingEnvPlaceHolder.observation_space, 
#                     ac_space=GraspingEnvPlaceHolder.action_space,
#                     nbatch_act=1, nbatch_train=32, nsteps=128, ent_coef=0.01, vf_coef=0.5, 
#                     max_grad_norm=0.5, comm=None, mpi_rank_weight=1)

#         # NOTE: This line must be called before any other tensorflow neural network is initialized
#         # TODO: Fix this so that it doesn't use tf.GraphKeys.GLOBAL_VARIABLES in tf_utils.py
#         self.model.load(os.path.join(CURR_PATH, "baselines_models/balls"))

#     def get_grasp_prob(self, aux_obs):
#         return self.model.value(aux_obs)[0]

#     def get_grasp_loc(self, aux_obs, noise=0.01):
#         grasp_loc, _, _, _ = self.model.step(aux_obs)
#         grasp_loc = grasp_loc[0] * np.array([0.04, 0.12])
#         grasp_loc += np.random.normal(0, noise, (2,))
#         return grasp_loc

#     def do_grasp(self, grasp_loc):
#         self.interface.execute_grasp(grasp_loc, 0.0)
#         reward = 0
#         for i in range(self.num_objects):
#             block_pos, _ = self.interface.get_object(self.objects_id[i])
#             if block_pos[2] > 0.08:
#                 reward = 1
#                 self.interface.move_object(self.objects_id[i], [self.wall_size * 3.0, 0, 1])
#                 break
#         self.interface.move_joints_to_start()
#         return reward

#     def step(self, a):
#         a = np.array(a)

#         self.interface.move_base(a[0] * 10.0, a[1] * 10.0)

#         reward = 0.0
#         if self.use_dist_reward:
#             dist_sq = self.get_closest_object_dist_sq(return_id=False)
#             reward = -np.sqrt(dist_sq)

#         aux_obs = self.interface.render_camera(use_aux=True)
#         v = self.get_grasp_prob(aux_obs)

#         grasp_succ = False
#         if v > 0.85:
#             grasp_loc = self.get_grasp_loc(aux_obs)
#             grasp_succ = self.do_grasp(grasp_loc)
#             if grasp_succ:
#                 reward += self.grasp_reward

#         if self.interface.renders:
#             obs = self.render()
#         else:
#             # pixel observations are automatically generated by PixelObservationWrapper
#             obs = None

#         done = self.num_steps >= self.max_ep_len
#         infos = {"success": int(grasp_succ)}

#         return obs, reward, done, infos