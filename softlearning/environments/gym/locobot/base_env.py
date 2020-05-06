import numpy as np
import gym
from gym import spaces

from . import locobot_interface

import pprint

class LocobotBaseEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, **params):
        self.interface = locobot_interface.PybulletInterface(**params)
        self.params = self.interface.params

        print()
        print("LocobotBaseEnv params:")
        pprint.pprint(dict(
            self=self,
            **self.params
        ))
        print()

        if "observation_space" in params:
            self.observation_space = params["observation_space"]
        else:
            observation_type = params["observation_type"]
            if observation_type == "image" and params.get("baselines", False):
                self.observation_space = spaces.Box(low=0, high=1., shape=(params["image_size"], params["image_size"], 3))
            else:
                s = {}
                if observation_type == "image":
                    # pixels taken care of by PixelObservationWrapper
                    pass
                elif observation_type == "state":
                    observation_high = np.ones(params["state_dim"])
                    s['state'] = spaces.Box(-observation_high, observation_high)
                else:
                    raise ValueError("Unsupported observation_type: " + str(params["observation_type"]))
                self.observation_space = spaces.Dict(s)

        if "action_space" in params:
            self.action_space = params["action_space"]
        else:
            self.action_dim = params["action_dim"]
            action_high = np.ones(self.action_dim)
            self.action_space = spaces.Box(-action_high, action_high)

        self.max_ep_len = self.params["max_ep_len"]
        self.num_steps = 0

    def render(self, mode='rgb_color', close=False):
        return self.interface.render_camera()

    def get_pixels(self, *args, **kwargs):
        return self.render(*args, **kwargs)

    def enable_recording(self):
        self.interface.enable_recording()

    def get_frames(self):
        return self.interface.get_frames()