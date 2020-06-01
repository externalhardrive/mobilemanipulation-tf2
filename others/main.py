import os 
import argparse

import numpy as np
import tensorflow as tf

from discretizer import *
from envs import *
from losses import *
from policies import *
from samplers import *
from replay_buffer import *
from train_functions import *
from training_loop import *

def autoregressive_discrete_dqn_grasping(args):
    # some hyper parameters
    image_size = 100
    discrete_dimensions = [15, 31]

    num_samples_per_env = 10
    num_samples_per_epoch = 100
    num_samples_total = int(1e5)
    min_samples_before_train = 1000

    train_frequency = 5
    train_batch_size = 200
    validation_prob = 0.1
    validation_batch_size = 100
    
    
    # create the policy
    logits_model, samples_model, deterministic_model = build_image_autoregressive_policy(
        image_size=image_size, 
        discrete_dimensions=discrete_dimensions,
        discrete_hidden_layers=[512, 512]
    )

    # create the optimizer
    optimizer = tf.optimizers.Adam(learning_rate=1e-5)
    
    # create the Discretizer
    discretizer = Discretizer(discrete_dimensions, [0.3, -0.16], [0.4666666, 0.16])

    # create the env
    env = GraspingEnv()

    # create the sampler
    sampler = create_grasping_env_autoregressive_discrete_sampler(
        env=env,
        discretizer=discretizer,
        deterministic_model=deterministic_model,
        min_samples_before_train=min_samples_before_train,
        epsilon=0.1,
    )

    # create the train and validation functions
    train_function = lambda data: train_autoregressive_discrete_sigmoid(logits_model, data, optimizer, discrete_dimensions)
    validation_function = lambda data: validation_autoregressive_discrete_sigmoid(logits_model, data, discrete_dimensions)

    # create the dataset
    train_buffer = ReplayBuffer(size=int(1e5), image_size=image_size, action_dim=len(discrete_dimensions))
    validation_buffer = ReplayBuffer(size=int(1e5), image_size=image_size, action_dim=len(discrete_dimensions))

    # testing time
    # logits_model.load_weights('./dataset/models/from_dataset_4/autoregressive')
    # epsilon = -1
    # min_samples_before_train = float('inf')
    # num_samples_total = 1

    all_diagnostics = training_loop(
        num_samples_per_env=num_samples_per_env,
        num_samples_per_epoch=num_samples_per_epoch,
        num_samples_total=num_samples_total,
        min_samples_before_train=min_samples_before_train,
        train_frequency=train_frequency,
        train_batch_size=train_batch_size,
        validation_prob=validation_prob,
        validation_batch_size=validation_batch_size,
        env=env,
        sampler=sampler,
        train_buffer=train_buffer, validation_buffer=validation_buffer,
        train_function=train_function, validation_function=validation_function,
    )

    save_folder = './others/logs/'
    name = 'autoregressive_10'

    if name:
        os.makedirs(save_folder, exist_ok=True)
        new_folder = os.path.join(save_folder, name)
        os.makedirs(new_folder, exist_ok=True)
        train_buffer.save(new_folder, "train_buffer")
        validation_buffer.save(new_folder, "validation_buffer")
        logits_model.save_weights(os.path.join(new_folder, "model"))
        np.save(os.path.join(new_folder, "diagnostics"), all_diagnostics)

def discrete_dqn_grasping(args):
    # some hyperparameters
    image_size = 100
    discrete_dimensions = [15, 31]
    discrete_dimension = 15 * 31

    num_samples_per_env = 10
    num_samples_per_epoch = 100
    num_samples_total = int(1e5)
    min_samples_before_train = 1000

    train_frequency = 5
    train_batch_size = 200
    validation_prob = 0.1
    validation_batch_size = 100
    
    # create the policy
    logits_model, samples_model, deterministic_model = build_image_discrete_policy(
        image_size=image_size, 
        discrete_dimension=discrete_dimension,
        discrete_hidden_layers=[512, 512]
    )

    # create the optimizer
    optimizer = tf.optimizers.Adam(learning_rate=1e-5)
    
    # create the Discretizer
    discretizer = Discretizer(discrete_dimensions, [0.3, -0.16], [0.4666666, 0.16])

    # create the env
    env = GraspingEnv()

    # create the sampler
    sampler = create_grasping_env_discrete_sampler(
        env=env,
        discretizer=discretizer,
        deterministic_model=deterministic_model,
        min_samples_before_train=min_samples_before_train,
        epsilon=0.1,
    )

    # create the train and validation functions
    train_function = lambda data: train_discrete_sigmoid(logits_model, data, optimizer, discrete_dimension)
    validation_function = lambda data: validation_discrete_sigmoid(logits_model, data, discrete_dimension)

    # create the dataset
    train_buffer = ReplayBuffer(size=num_samples_total, image_size=image_size, action_dim=discrete_dimension)
    validation_buffer = ReplayBuffer(size=num_samples_total, image_size=image_size, action_dim=discrete_dimension)

    # testing time
    # logits_model.load_weights('./dataset/models/from_dataset_4/autoregressive')
    # epsilon = -1
    # min_samples_before_train = float('inf')
    # num_samples_total = 1

    all_diagnostics = training_loop(
        num_samples_per_env=num_samples_per_env,
        num_samples_per_epoch=num_samples_per_epoch,
        num_samples_total=num_samples_total,
        min_samples_before_train=min_samples_before_train,
        train_frequency=train_frequency,
        train_batch_size=train_batch_size,
        validation_prob=validation_prob,
        validation_batch_size=validation_batch_size,
        env=env,
        sampler=sampler,
        train_buffer=train_buffer, validation_buffer=validation_buffer,
        train_function=train_function, validation_function=validation_function,
    )

    save_folder = './others/logs/'
    name = 'discrete_1'

    if name:
        os.makedirs(save_folder, exist_ok=True)
        new_folder = os.path.join(save_folder, name)
        os.makedirs(new_folder, exist_ok=True)
        train_buffer.save(new_folder, "train_buffer")
        validation_buffer.save(new_folder, "validation_buffer")
        logits_model.save_weights(os.path.join(new_folder, "model"))
        np.save(os.path.join(new_folder, "diagnostics"), all_diagnostics)

def ddpg_grasping(args):
    # some hyperparameters
    image_size = 100
    action_dim = 2

    num_samples_per_env = 10
    num_samples_per_epoch = 100
    num_samples_total = int(1e5)
    min_samples_before_train = 1000

    train_frequency = 5
    train_batch_size = 200
    
    # create the policy and Q function
    policy_model = build_image_deterministic_continuous_policy(
        image_size=image_size,
        action_dim=action_dim,
        feedforward_hidden_layers=(512, 512),
    )

    Q_model = build_image_continuous_Q_function(
        image_size=image_size,
        action_dim=action_dim,
        feedforward_hidden_layers=(512, 512),
    )

    # create the optimizer
    policy_optimizer = tf.optimizers.Adam(learning_rate=1e-5)
    Q_optimizer = tf.optimizers.Adam(learning_rate=1e-5)
    
    # create the env
    env = GraspingEnv()

    # create the sampler
    sampler = create_grasping_env_ddpg_sampler(
        env=env,
        policy_model=policy_model,
        action_dim=action_dim,
        min_samples_before_train=min_samples_before_train,
        num_samples_at_end=50000,
        noise_std_start=0.5,
        noise_std_end=0.01,
    )

    # create the train function
    def train_function(data): 
        Q_loss = train_ddpg_Q_function(Q_model, data, Q_optimizer)
        train_ddpg_policy(policy_model, Q_model, data['observations'], policy_optimizer)
        return Q_loss

    # create the dataset
    train_buffer = ReplayBuffer(size=num_samples_total, image_size=image_size, action_dim=action_dim)

    # run the training
    all_diagnostics = training_loop(
        num_samples_per_env=num_samples_per_env,
        num_samples_per_epoch=num_samples_per_epoch,
        num_samples_total=num_samples_total,
        min_samples_before_train=min_samples_before_train,
        train_frequency=train_frequency,
        train_batch_size=train_batch_size,
        validation_prob=None,
        validation_batch_size=None,
        env=env,
        sampler=sampler,
        train_buffer=train_buffer, validation_buffer=None,
        train_function=train_function, validation_function=None,
    )

    # save
    save_folder = './others/logs/'
    name = 'ddpg_1'

    if name:
        os.makedirs(save_folder, exist_ok=True)
        new_folder = os.path.join(save_folder, name)
        os.makedirs(new_folder, exist_ok=True)
        train_buffer.save(new_folder, "train_buffer")
        policy_model.save_weights(os.path.join(new_folder, "policy_model"))
        Q_model.save_weights(os.path.join(new_folder, "Q_model"))
        np.save(os.path.join(new_folder, "diagnostics"), all_diagnostics)

def main(args):
    # autoregressive_discrete_dqn_grasping(args)
    # discrete_dqn_grasping(args)
    ddpg_grasping(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)