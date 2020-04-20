import numpy as np
import gym
from gym import spaces

from . import locobot_interface

class LocobotBaseEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, **params):
        self.interface = locobot_interface.PybulletInterface(**params)
        self.params = self.interface.params

        self.channels = 1 if self.interface.grayscale else 3

        self.stack_frames = self.params.get("stack_frames", 1)
        if self.stack_frames > 1:
            self.stacked_obs = np.zeros((params["image_size"], params["image_size"], self.channels * self.stack_frames), dtype=np.uint8)
            self.stacked_obs_initalized = False

        if "observation_type" in params:
            observation_type = params["observation_type"]
            if observation_type == "image" and params.get("baselines", False):
                self.observation_space = spaces.Box(low=0, high=1., 
                                        shape=(params["image_size"], params["image_size"], self.channels * self.stack_frames))
            else:
                if observation_type == "image":
                    observation_dim = params["image_size"] * params["image_size"] * self.channels * self.stack_frames
                elif observation_type == "state":
                    observation_dim = params["state_dim"]
                else:
                    raise ValueError("Unsupported observation_type: " + str(params["observation_type"]))
                observation_high = np.ones(observation_dim)
                self.observation_space = spaces.Box(-observation_high, observation_high)
        else:
            self.observation_space = params["observation_space"]

        if "action_dim" in params:
            self.action_dim = params["action_dim"]
            action_high = np.ones(self.action_dim)
            self.action_space = spaces.Box(-action_high, action_high)
        else:
            self.action_space = params["action_space"]

        self.max_ep_len = self.params["max_ep_len"]
        self.num_steps = 0

    def reset_stacked_obs(self):
        self.stacked_obs_initalized = False

    def render(self, mode='rgb_color', close=False):
        obs = self.interface.render_camera()
        if self.stack_frames > 1:
            if self.stacked_obs_initalized:
                self.stacked_obs[:, :, self.channels:] = self.stacked_obs[:, :, :-self.channels] 
                self.stacked_obs[:, :, :self.channels] = obs
            else:
                frames = []
                for _ in range(self.stack_frames):
                    frames.append(obs)
                self.stacked_obs = np.concatenate(frames, axis=2)
                self.stacked_obs_initalized = True
            return self.stacked_obs
        else:
            return obs 

    def get_pixels(self, *args, **kwargs):
        return self.render(*args, **kwargs)

    def enable_recording(self):
        self.interface.enable_recording()

    def get_frames(self):
        return self.interface.get_frames()