import argparse
from collections import defaultdict

import time, os

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from softlearning.environments.gym.locobot import *
from softlearning.environments.gym.locobot.utils import *
from softlearning.environments.adapters.gym_adapter import GymAdapter

import matplotlib.image as mpimg
# mpimg.imsave("../bounding_box2.png", obs)

def save_obs(obs, path):
    if obs.shape[2] == 1:
        obs = np.concatenate([obs, obs, obs], axis=2)
    mpimg.imsave(path, obs)

def main(args):
    room_name = "simple_obstacles"
    room_params = dict(
        num_objects=100, 
        object_name="greensquareball", 
        wall_size=5.0,
        no_spawn_radius=0.8,
    )
    # inner_env = ImageLocobotNavigationEnv(
    #         renders=True, grayscale=False, step_duration=1/60,
    #         room_name=room_name,
    #         room_params=room_params,
    #         image_size=100,
    #         use_dist_reward=False, grasp_reward=1
    #     )
    inner_env = MixedLocobotNavigationEnv(
            renders=True, grayscale=False, step_duration=1/60,
            room_name="simple",
            room_params=room_params,
            image_size=100,
            steps_per_second=2,
            max_ep_len=200,
            max_velocity=20.0,
            max_acceleration=4.0,
        )

    env = GymAdapter(None, None,
        env=inner_env,
        pixel_wrapper_kwargs={
            'pixels_only': False,
        },
    )
    
    obs = env.reset()

    # inner_env.interface.set_wheels_velocity(20, 20)
    # for _ in range(1000):
    #     inner_env.interface.step()
    #     print(inner_env.interface.get_wheels_velocity())
    # for target in range(1, 20 + 1, 1):
    #     inner_env.interface.set_wheels_velocity(target, target)
    #     for _ in range(50):
    #         inner_env.interface.step()
    #         print(target, inner_env.interface.get_wheels_velocity())
    #     inner_env.render()
    #     input()

    i = 0
        
    while True:
        # save_obs(obs["pixels"], f"../images/obs{i}.png")
        print("velocity:", obs["velocity"])
        # print("target:", obs["target_velocity"])
        cmd = input().strip()
        try:
            if cmd == "exit":
                break
            if cmd == "r":
                obs = env.reset()
                i = 0
                continue
            else:
                move = [float(x) for x in cmd.split(" ")]
                move[1]
        except:
            print("cannot parse")
            continue

        a = np.array([move[0], move[1]])
        obs, rew, done, _ = env.step(a)
        i += 1

        print(rew)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)