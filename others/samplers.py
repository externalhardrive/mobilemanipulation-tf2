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

    def sampler(num_samples, force_deterministic=False):
        if force_deterministic:
            return sample_deterministic()
        rand = np.random.uniform()
        if rand < epsilon or num_samples < min_samples_before_train: # epsilon greedy or initial samples
            #print("sampling random rand", rand, "epsilon", epsilon, "num_samples", num_samples, "minsamples", min_samples_before_train)
            return sample_random()
        else:
            #print("deterministic")
            return sample_deterministic()

    return sampler


def create_fc_grasping_env_discrete_sampler(
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
        #action_undiscretized = discretizer.undiscretize(discretizer.unflatten(action_discrete))
        reward = env.do_grasp(action_discrete)
    
        return obs, action_discrete, reward, {'sample_random': 1}

    def sample_deterministic():
        obs = env.get_observation()   
        action_discrete = deterministic_model(np.array([obs])).numpy()
        #action_undiscretized = discretizer.undiscretize(discretizer.unflatten(action_discrete))
        reward = env.do_grasp(action_discrete)

        return obs, action_discrete, reward, {'sample_deterministic': 1}

    def sampler(num_samples, force_deterministic=False):
        if force_deterministic:
            return sample_deterministic()
        rand = np.random.uniform()
        if rand < epsilon or num_samples < min_samples_before_train: # epsilon greedy or initial samples
            #print("sampling random rand", rand, "epsilon", epsilon, "num_samples", num_samples, "minsamples", min_samples_before_train)
            return sample_random()
        else:
            #print("deterministic")
            return sample_deterministic()

    return sampler

def create_fake_grasping_discrete_sampler(
        env=None,
        discrete_dimension=None,
        deterministic_model=None,
        min_samples_before_train=None,
        epsilon=None,
    ):
    def sample_random():
        obs = env.get_observation()
        action = np.random.randint(0, discrete_dimension)
        reward = env.do_grasp(action)
    
        return obs, action, reward, {'sample_random': 1}

    def sample_deterministic():
        obs = env.get_observation()   
        action = deterministic_model(np.array([obs])).numpy()
        reward = env.do_grasp(action)

        return obs, action, reward, {'sample_deterministic': 1}

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

def create_grasping_env_ddpg_sampler(
        env=None,
        policy_model=None,
        unsquashed_model=None,
        action_dim=None,
        min_samples_before_train=1000,
        num_samples_at_end=50000,
        noise_std_start=0.5,
        noise_std_end=0.01,
    ):
    """ Linear annealing from start noise to end noise. """

    def sample_random():
        obs = env.get_observation()
        action = np.random.uniform(-1.0, 1.0, size=(action_dim,))
        reward = env.do_grasp(env.from_normalized_action(action))
    
        return obs, action, reward, {'action': action}

    def sample_with_noise(noise_std):
        obs = env.get_observation()

        noise = np.random.normal(size=(action_dim,)) * noise_std
        action_deterministic = policy_model(np.array([obs])).numpy()[0]
        action = np.clip(action_deterministic + noise, -1.0, 1.0)

        reward = env.do_grasp(env.from_normalized_action(action))
        
        infos = {
            'action': action, 
            'action_deterministic': action_deterministic, 
            'noise': noise, 
            'noise_std': noise_std,
        }

        if unsquashed_model:
            action_unsquashed = unsquashed_model(np.array([obs])).numpy()[0]
            infos['unsquashed_action'] = action_unsquashed 

        return obs, action, reward, infos

    def sampler(num_samples):
        if num_samples < min_samples_before_train: # epsilon greedy or initial samples
            return sample_random()
        
        noise_std = np.interp(num_samples, 
                              np.array([min_samples_before_train, num_samples_at_end]), 
                              np.array([noise_std_start, noise_std_end]))

        return sample_with_noise(noise_std)

    return sampler