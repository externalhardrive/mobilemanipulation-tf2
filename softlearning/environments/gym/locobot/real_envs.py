import math
import gym
from gym import spaces
import numpy as np
from skimage.transform import resize
import time

from locobot_interface.client import LocobotClient

class RealLocobotBaseEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self._client = LocobotClient()

    def render(self, mode='rgb_array'):
        img = self._client.get_image()
        img = resize(img, (84,84))
        return img


class RealLocobotOdomNavEnv(RealLocobotBaseEnv):
    def __init__(self):
        super().__init__()
        self._action_dim = 2
        observation_dim = 84 * 84 * 3
        observation_high = np.ones(observation_dim)
        action_min = -np.ones(self._action_dim)
        action_max = np.ones(self._action_dim)
        self.action_space = spaces.Box(action_min, action_max)
        self.observation_space = spaces.Box(-observation_high, observation_high)

        self._action_scaling = np.array([.3, 1.])
        self._goal = np.array([.5, 0.])

        self.pre_arm_out_of_way = [-1.5, 0, 0, 0, 0]
        self.arm_out_of_way = [-1.53551483, -1.0154953 ,  1.25939822,  1.12287402, -0.06135923]
        
        self._client.set_pan_tilt(.0, .9)
        self._client.set_joint_angles([0, 0, 0, math.pi/2, 0])
        self._client.set_joint_angles(self.pre_arm_out_of_way)
        self._client.set_joint_angles(self.arm_out_of_way)
        
        self._max_steps = 15
        self._num_steps = 0

        time.sleep(.5)

    def reset(self):
        print('Resetting base position to [0,0]')
        self._client.set_base_pos(0, 0, 0, relative=False, close_loop=True)
        self._num_steps = 0
        return self.render()

    def step(self, a):
        a[0] = (a[0] + 1) / 2
        action = a * self._action_scaling
        fwd_speed, turn_speed = action

        print('Executing base command [{},{}]'.format(fwd_speed, turn_speed))
        self._client.set_base_vel(fwd_speed, turn_speed, .5)

        if self._client.get_bumper_state() == 1:
            self._client.set_base_vel(-.2, 1., 1)

        # time.sleep(.25)

        pos, _ = self._client.get_odom()
        pos = np.array(pos)

        self._num_steps += 1

        reward = np.linalg.norm(pos[:2] - self._goal)
        done = (reward < .03) or (self._num_steps >= self._max_steps)
        obs = self.render()

        return obs, reward, done, {}

