import argparse
from collections import defaultdict

import time, os

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from softlearning.environments.gym.locobot import *
from softlearning.environments.gym.locobot.utils import *

RENDERS = True
NUM_OBJECTS = 80
WALL_SIZE = 5.0

import matplotlib.image as mpimg
# mpimg.imsave("../bounding_box2.png", obs)

def save_obs(obs, path):
    if obs.shape[2] == 1:
        obs = np.concatenate([obs, obs, obs], axis=2)
    mpimg.imsave(path, obs)

def main(args):
    # ImageLocobotNavigationGraspingEnv
    env = ImageLocobotNavigationEnv(
        renders=RENDERS, grayscale=True, step_duration=1/60 * 0.5,
        num_objects=NUM_OBJECTS, object_name="greensquareball", wall_size=WALL_SIZE, 
        image_size=84)
    
    obs = env.reset()

    i = 0
        
    while True:
        save_obs(obs, f"../images/obs{i}.png")
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