import numpy as np

from softlearning.environments.gym.locobot import *
from softlearning.environments.gym.locobot.nav_envs import RoomEnv
from softlearning.environments.gym.locobot.utils import *

class GraspingEnv:
    def __init__(self, robot_pos=np.array([0.0, 0.0])):
        room_name = "grasping"
        self.robot_pos = robot_pos
        room_params = dict(
            min_objects=8, 
            max_objects=10,
            object_name="greensquareball", 
            spawn_loc=np.array([0.36, 0]) + self.robot_pos,
            spawn_radius=0.6,
            #no_spawn_radius=0.,
            
            wall_size=4,
        )
        env = RoomEnv(
            renders=False, grayscale=False, step_duration=1/60 * 0,
            room_name=room_name,
            room_params=room_params,
            robot_pos = robot_pos,
            
            # use_aux_camera=True,
            # aux_camera_look_pos=[0.4, 0, 0.05],
            # aux_camera_fov=35,
            # aux_image_size=100,
            observation_space=None,
            action_space=None,
            max_ep_len=None,
        )
        
        from softlearning.environments.gym.locobot.utils import URDF
#         env.interface.spawn_object(URDF["greensquareball"], pos=[0.3, -0.16, 0.015])
#         env.interface.spawn_object(URDF["greensquareball"], pos=[0.3, 0.16, 0.015])
#         env.interface.spawn_object(URDF["greensquareball"], pos=[0.466666, -0.16, 0.015])
#         env.interface.spawn_object(URDF["greensquareball"], pos=[0.466666, 0.16, 0.015])
#         env.interface.spawn_object(URDF["greensquareball"], pos=[0.3, 0, 0.015])
#         env.interface.spawn_object(URDF["greensquareball"], pos=[0.466666, 0, 0.015])
        obs = env.interface.render_camera(use_aux=False)

        # import matplotlib.pyplot as plt 
        # plt.imsave("./others/logs/obs.bmp", obs)

        # plt.imsave("./others/logs/cropped.bmp", obs[38:98, 20:80, :])
        
        self._env = env
        self.reset()
        while True:
            self._env.room.reset()
            if self.are_blocks_graspable():
                return

    def crop_obs(self, obs):
        return obs[..., 38:98, 20:80, :]

    @property
    def grasp_image_size(self):
        return 60

    def do_grasp(self, action):
        if len(action) == 3:
            self._env.interface.execute_grasp_direct(action[:2], action[2])
        else:
            self._env.interface.execute_grasp_direct(action, 0.0)
        reward = 0
        for i in range(self._env.room.num_objects):
            block_pos, _ = self._env.interface.get_object(self._env.room.objects_id[i])
            if block_pos[2] > 0.04:
                reward = 1
                self._env.interface.move_object(
                    self._env.room.objects_id[i], 
                    [self._env.room.extent + np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.01])
                break
#         if reward:
#             place = self.from_normalized_action(np.random.random(2) - 0.5)
#             self._env.interface.execute_place_direct(place, 0.0)
        self._env.interface.move_arm_to_start(steps=90, max_velocity=8.0)
        return reward

    def from_normalized_action(self, normalized_action):
        if len(normalized_action) == 2:
            action_min = np.array([0.3, -0.16])
            action_max = np.array([0.466666666, 0.16])
            action_mean = (action_max + action_min) * 0.5
            action_scale = (action_max - action_min) * 0.5
        else:
            action_min = np.array([0.3, -0.16, 0])
            action_max = np.array([0.466666666, 0.16, 3.14])
            action_mean = (action_max + action_min) * 0.5
            action_scale = (action_max - action_min) * 0.5

        return normalized_action * action_scale + action_mean

    def are_blocks_graspable(self):
        for i in range(self._env.room.num_objects):
            object_pos, _ = self._env.interface.get_object(self._env.room.objects_id[i], relative=True)
            if is_in_rect(object_pos[0], object_pos[1], 0.3+self.robot_pos[0], -0.16+self.robot_pos[1], 0.466666666+self.robot_pos[0], 0.16+self.robot_pos[1]):
                return True 
        return False

    def reset(self):
        self._env.interface.reset_robot(self.robot_pos, 0, 0, 0)
        while True:
            self._env.room.reset()
            if self.are_blocks_graspable():
                return

    def should_reset(self):
        return not self.are_blocks_graspable()
    
    def get_observation(self):
        return self.crop_obs(self._env.interface.render_camera(use_aux=False))


class FullyConvGraspingEnv:
    def __init__(self, robot_pos=np.array([0.0, 0.0])):
        room_name = "grasping"
        self.robot_pos = robot_pos
        room_params = dict(
            min_objects=8, 
            max_objects=10,
            object_name="greensquareball", 
            spawn_loc=np.array([0.36, 0]) + self.robot_pos,
            spawn_radius=0.6,
            #no_spawn_radius=0.,
            
            wall_size=4,
        )
        env = RoomEnv(
            renders=True, grayscale=False, step_duration=1/60 * 0,
            room_name=room_name,
            room_params=room_params,
            robot_pos = robot_pos,
            
            # use_aux_camera=True,
            # aux_camera_look_pos=[0.4, 0, 0.05],
            # aux_camera_fov=35,
            # aux_image_size=100,
            observation_space=None,
            action_space=None,
            max_ep_len=None,
        )
        self.crop_y = (38,98)
        self.crop_x = (20,80)
        self.a_min = np.array([0.3, -0.16])
        self.a_max = np.array([0.466666, 0.16])
        from softlearning.environments.gym.locobot.utils import URDF
#         env.interface.spawn_object(URDF["greensquareball"], pos=[0.3, -0.16, 0.015])
#         env.interface.spawn_object(URDF["greensquareball"], pos=[0.3, 0.16, 0.015])
#         env.interface.spawn_object(URDF["greensquareball"], pos=[0.466666, -0.16, 0.015])
#         env.interface.spawn_object(URDF["greensquareball"], pos=[0.466666, 0.16, 0.015])
#         env.interface.spawn_object(URDF["greensquareball"], pos=[0.3, 0, 0.015])
#         env.interface.spawn_object(URDF["greensquareball"], pos=[0.466666, 0, 0.015])
        obs = env.interface.render_camera(use_aux=False)

        # import matplotlib.pyplot as plt 
        # plt.imsave("./others/logs/obs.bmp", obs)

        # plt.imsave("./others/logs/cropped.bmp", obs[38:98, 20:80, :])
        
        self._env = env
        self.reset()
        obs = self.get_observation()
        self.obs_downsample = 4
        self.obs_dim = np.array(obs.shape[:2])
        while True:
            self._env.room.reset()
            if self.are_blocks_graspable():
                return

    def crop_obs(self, obs):
        
        return obs[..., self.crop_y[0]:self.crop_y[1], self.crop_x[0]:self.crop_x[1], :]

    @property
    def grasp_image_size(self):
        return 60

    def do_grasp(self, action):
        """ action is discretized raveled index"""
        #import pdb; pdb.set_trace()
        #print("action", action)
        #action *= 4
        pixel = np.array(np.unravel_index(action, shape=(self.obs_dim/self.obs_downsample).astype(np.int32))).flatten()
        #print("pixel", pixel)
        pixel *= self.obs_downsample
        y = pixel[0] +self.crop_y[0]
        x = pixel[1] +self.crop_x[0]
        
        pos_x_y = self._env.interface.get_world_from_pixel(np.array([x,y]))[:2]
        pos_x_y = np.clip(pos_x_y, a_min=self.a_min, a_max = self.a_max)
        if len(pos_x_y) == 3:
            assert(False)
            self._env.interface.execute_grasp_direct(pos_x_y, action[2])
        else:
            self._env.interface.execute_grasp_direct(pos_x_y, 0.0)
        reward = 0
        for i in range(self._env.room.num_objects):
            block_pos, _ = self._env.interface.get_object(self._env.room.objects_id[i])
            if block_pos[2] > 0.04:
                reward = 1
                self._env.interface.move_object(
                    self._env.room.objects_id[i], 
                    [self._env.room.extent + np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.01])
                break
        #if reward:
#             place = self.from_normalized_action(np.random.random(2) - 0.5)
#             self._env.interface.execute_place_direct(place, 0.0)
        self._env.interface.move_arm_to_start(steps=90, max_velocity=8.0)
        return reward

    def from_normalized_action(self, normalized_action):
        if len(normalized_action) == 2:
            action_min = np.array([0.3, -0.16])
            action_max = np.array([0.466666666, 0.16])
            action_mean = (action_max + action_min) * 0.5
            action_scale = (action_max - action_min) * 0.5
        else:
            action_min = np.array([0.3, -0.16, 0])
            action_max = np.array([0.466666666, 0.16, 3.14])
            action_mean = (action_max + action_min) * 0.5
            action_scale = (action_max - action_min) * 0.5

        return normalized_action * action_scale + action_mean

    def are_blocks_graspable(self):
        for i in range(self._env.room.num_objects):
            object_pos, _ = self._env.interface.get_object(self._env.room.objects_id[i], relative=True)
            if is_in_rect(object_pos[0], object_pos[1], 0.3+self.robot_pos[0], -0.16+self.robot_pos[1], 0.466666666+self.robot_pos[0], 0.16+self.robot_pos[1]):
                return True 
        return False

    def reset(self):
        self._env.interface.reset_robot(self.robot_pos, 0, 0, 0)
        while True:
            self._env.room.reset()
            if self.are_blocks_graspable():
                return

    def should_reset(self):
        return not self.are_blocks_graspable()
    
    def get_observation(self):
        return self.crop_obs(self._env.interface.render_camera(use_aux=False))


    
class FakeGraspingDiscreteEnv:
    """ 1D grasping discrete from 1D 'images' """
    def __init__(self, line_width=32, min_objects=1, max_objects=5):
        self.line_width = line_width
        self.min_objects = min_objects
        self.max_objects = max_objects

        self.line = -np.ones((self.line_width,))

    def reset(self):
        self.line = -np.ones((self.line_width,))
        num_objects = np.random.randint(self.min_objects, self.max_objects+1)
        for _ in range(num_objects):
            i = np.random.randint(0, self.line_width)
            self.line[i] = 1.0

    def should_reset(self):
        return np.all(self.line < 0.0)
    
    def get_observation(self):
        return np.copy(self.line)

    def do_grasp(self, action):
        a = int(action)

        reward = 0.0
        if self.line[a] > 0.0:
            reward = 1.0
            self.line[a] = -1.0
        
        return reward