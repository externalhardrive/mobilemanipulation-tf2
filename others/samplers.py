import numpy as np 

from envs import *

def create_grasping_env_discrete_sampler(
        env=None,
        discretizer=None,
        deterministic_model=None,
        min_samples_before_train=None,
        epsilon=None,
    ):
    total_dimensions = np.prod(discretizer.dimensions)

    def sample_random():
        obs = env.get_observation()
        action_discrete = np.random.randint(0, total_dimensions)
        action_undiscretized = discretizer.undiscretize(discretizer.unflatten(action_discrete))
        reward = env.do_grasp(action_undiscretized)
    
        return obs, action_discrete, reward, {'sample_random': 1}

    def sample_deterministic():
        obs = env.get_observation()   
        action_discrete = deterministic_model(np.array([obs])).numpy()
        action_undiscretized = discretizer.undiscretize(discretizer.unflatten(action_discrete))
        reward = env.do_grasp(action_undiscretized)

        return obs, action_discrete, reward, {'sample_deterministic': 1}

    def sampler(num_samples):
        rand = np.random.uniform()
        if rand < epsilon or num_samples < min_samples_before_train: # epsilon greedy or initial samples
            return sample_random()
        else:
            return sample_deterministic()

    return sampler

def create_grasping_env_autoregressive_discrete_sampler(
        env=None,
        discretizer=None,
        deterministic_model=None,
        min_samples_before_train=None,
        epsilon=None,
    ):
    
    def sample_random():
        obs = env.get_observation()
        action_discrete = np.random.randint(0, discretizer.dimensions)
        action_undiscretized = discretizer.undiscretize(action_discrete)
        reward = env.do_grasp(action_undiscretized)

        return obs, action_discrete, reward, {'sample_random': 1}

    def sample_deterministic():
        obs = env.get_observation()
        action_onehot = deterministic_model(np.array([obs]))
        action_discrete = np.array([tf.argmax(a, axis=-1).numpy().squeeze() for a in action_onehot])
        action_undiscretized = discretizer.undiscretize(action_discrete)
        reward = env.do_grasp(action_undiscretized)

        return obs, action_discrete, reward, {'sample_deterministic': 1}

    def sampler(num_samples):
        rand = np.random.uniform()
        if rand < epsilon or num_samples < min_samples_before_train: # epsilon greedy or initial samples
            return sample_random()
        else:
            return sample_deterministic()

    return sampler