import numpy as np

from softlearning.environments.gym.locobot import *
from softlearning.environments.gym.locobot.nav_envs import RoomEnv

class GraspingEnv:
    def __init__(self):
        room_name = "grasping"
        room_params = dict(
            min_objects=1, 
            max_objects=10,
            object_name="greensquareball", 
            spawn_loc=[0.36, 0],
            spawn_radius=0.3,
        )
        env = RoomEnv(
            renders=False, grayscale=False, step_duration=1/60 * 0,
            room_name=room_name,
            room_params=room_params,
            use_aux_camera=True,
            aux_camera_look_pos=[0.4, 0, 0.05],
            aux_camera_fov=35,
            aux_image_size=100,
            observation_space=None,
            action_space=None,
            max_ep_len=None,
        )

        # from softlearning.environments.gym.locobot.utils import URDF
        # env.interface.spawn_object(URDF["greensquareball"], pos=[0.3, -0.16, 0.015])
        # env.interface.spawn_object(URDF["greensquareball"], pos=[0.3, 0.16, 0.015])
        # env.interface.spawn_object(URDF["greensquareball"], pos=[0.466666, -0.16, 0.015])
        # env.interface.spawn_object(URDF["greensquareball"], pos=[0.466666, 0.16, 0.015])
        # env.interface.spawn_object(URDF["greensquareball"], pos=[0.3, 0, 0.015])
        # env.interface.spawn_object(URDF["greensquareball"], pos=[0.466666, 0, 0.015])
        # env.interface.render_camera(use_aux=True)

        self._env = env

    def do_grasp(self, action):
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
        self._env.interface.move_arm_to_start(steps=90, max_velocity=8.0)
        return reward

    def are_blocks_graspable(self):
        for i in range(self._env.room.num_objects):
            object_pos, _ = self._env.interface.get_object(self._env.room.objects_id[i], relative=True)
            if is_in_rect(object_pos[0], object_pos[1], 0.3, -0.16, 0.466666666, 0.16):
                return True 
        return False

    def reset(self):
        self._env.interface.reset_robot([0, 0], 0, 0, 0)
        while True:
            self._env.room.reset()
            if self.are_blocks_graspable():
                return

    def should_reset(self):
        return not self.are_blocks_graspable()
    
    def get_observation(self):
        return self._env.interface.render_camera(use_aux=True)