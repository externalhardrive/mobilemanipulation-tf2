import numpy as np 

from .envs import *

def create_grasping_env_discrete_samplers(
        env=None,
        discretizer=None,
        deterministic_model=None,
    ):
    total_dimensions = np.prod(discretizer.dimensions)

    def sample_random():
        obs = env.get_observation()
        action_discrete = np.random.randint(0, total_dimensions)
        action_undiscretized = discretizer.undiscretize(discretizer.unflatten(action_discrete)))
        reward = env.do_grasp(action_undiscretized)
    
        return obs, action_discrete, reward, {}

    def sample_deterministic():
        obs = env.get_observation()   
        action_discrete = deterministic_model(np.array([obs])).numpy()
        action_undiscretized = discretizer.undiscretize(discretizer.unflatten(action_discrete)))
        reward = env.do_grasp(action_undiscretized)

        return obs, action_discrete, reward, {}

    return sample_random, sample_deterministic

def create_grasping_env_autoregressive_discrete_samplers(
        env=None,
        discretizer=None,
        deterministic_model=None,
    ):
    
    def sample_random():
        obs = env.get_observation()
        action_discrete = np.random.randint(0, discretizer.dimensions)
        action_undiscretized = discretizer.undiscretize(action_discrete)
        reward = env.do_grasp(action_undiscretized)

        return obs, action_discrete, reward, {}

    def sample_deterministic():
        obs = env.get_observation()
        action_onehot = deterministic_model(np.array([obs]))
        action_discrete = np.array([tf.argmax(a, axis=-1).numpy().squeeze() for a in action_onehot])
        action_undiscretized = discretizer.undiscretize(action_discrete))
        reward = env.do_grasp(action_undiscretized)

        return obs, action_discrete, reward, {}

    return sample_random, sample_deterministic