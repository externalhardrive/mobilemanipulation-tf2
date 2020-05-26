import argparse
import os
import time

import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np 

from collections import OrderedDict
from pprint import pprint

from softlearning.environments.gym.locobot import *
from softlearning.environments.gym.locobot.nav_envs import RoomEnv

from softlearning.models.autoregressive_discrete import autoregressive_discrete_model
from softlearning.models.convnet import convnet_model

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors

def build_policy(image_size=100,
                 discrete_hidden_layers=(512, 512),
                 discrete_dimensions=(15, 31)):

    obs_in = tfk.Input((image_size, image_size, 3))
    conv_out = convnet_model(
        conv_filters=(64, 64, 64),
        conv_kernel_sizes=(3, 3, 3),
        conv_strides=(2, 2, 2),
        activation="relu"
    )(obs_in)
    
    discrete_logits_model, discrete_samples_model, discrete_deterministic_model = autoregressive_discrete_model(
        conv_out.shape[1],
        discrete_hidden_layers,
        discrete_dimensions,
        activation='relu',
        output_activation='linear',
        distribution_logits_activation='sigmoid'
    )
    actions_in = [tfk.Input(size) for size in discrete_dimensions]
    
    logits_out        = discrete_logits_model([conv_out] + actions_in)
    samples_out       = discrete_samples_model(conv_out)
    deterministic_out = discrete_deterministic_model(conv_out)

    logits_model        = tfk.Model([obs_in] + actions_in, logits_out)
    samples_model       = tfk.Model(obs_in, samples_out)
    deterministic_model = tfk.Model(obs_in, deterministic_out)

    return logits_model, samples_model, deterministic_model

def create_env():
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

    return env

def do_grasp(env, action):
    env.interface.execute_grasp_direct(action, 0.0)
    reward = 0
    for i in range(env.room.num_objects):
        block_pos, _ = env.interface.get_object(env.room.objects_id[i])
        if block_pos[2] > 0.04:
            reward = 1
            env.interface.move_object(env.room.objects_id[i], [env.room.extent, 0, 1])
            break
    env.interface.move_arm_to_start(steps=90, max_velocity=8.0)
    return reward

class Discretizer:
    def __init__(self, sizes, mins, maxs):
        self._sizes = np.array(sizes)
        self._mins = np.array(mins) 
        self._maxs = np.array(maxs) 

        self._step_sizes = (self._maxs - self._mins) / self._sizes

    def discretize(self, action):
        centered = action - self._mins
        indices = np.floor_divide(centered, self._step_sizes).astype(np.uint8)
        clipped = np.clip(indices, 0, self._sizes)
        return clipped

    def undiscretize(self, action):
        return action * self._step_sizes + self._mins + self._step_sizes * 0.5

class ReplayBuffer:
    def __init__(self, size, image_size, action_dim):
        self._size = size
        self._observations = np.zeros((size, image_size, image_size, 3), dtype=np.uint8)
        self._actions = np.zeros((size, action_dim), dtype=np.uint8)
        self._rewards = np.zeros((size, 1), dtype=np.uint8)
        self._num = 0

    def store_sample(self, observation, action, reward):
        self._observations[self._num] = observation
        self._actions[self._num] = action
        self._rewards[self._num] = reward
        self._num += 1

    def get_all_samples(self):
        data = {
            'observations': self._observations[:self._num],
            'actions': self._actions[:self._num],
            'rewards': self._rewards[:self._num],
        }
        return data

    def sample_batch(self, batch_size):
        inds = np.random.randint(0, self._num, size=(batch_size,))
        data = {
            'observations': self._observations[inds],
            'actions': self._actions[inds],
            'rewards': self._rewards[inds],
        }
        return data

    def save(self, folder_path, file_name='replaybuffer'):
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, file_name), self.get_all_samples())

    def load(self, path):
        data = np.load(path, allow_pickle=True)[()]
        self._num = data['observations'].shape[0]
        
        self._observations[:self._num] = data['observations']
        self._actions[:self._num] = data['actions']
        self._rewards[:self._num] = data['rewards']

@tf.function(experimental_relax_shapes=True)
def train(logits_model, data, optimizer, discrete_dimensions):
    observations = data['observations']
    rewards = tf.cast(data['rewards'], tf.float32)
    actions_discrete = data['actions']
    actions_onehot = [tf.one_hot(actions_discrete[:, i], depth=d) for i, d in enumerate(discrete_dimensions)]

    with tf.GradientTape() as tape:
        total_loss = tf.constant(0.)
        # get the logits for all the dimensions
        logits = logits_model([observations] + actions_onehot)
        for logits_per_dim, actions_onehot_per_dim in zip(logits, actions_onehot):
            # get only the logits for the actions taken
            taken_action_logits = tf.reduce_sum(logits_per_dim * actions_onehot_per_dim, axis=-1, keepdims=True)
            # calculate the sigmoid loss (because we know reward is 0 or 1)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=rewards, logits=taken_action_logits)
            loss = tf.nn.compute_average_loss(losses)
            total_loss += loss

    grads = tape.gradient(total_loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, logits_model.trainable_variables))

    return total_loss

def main(args):
    image_size = 100
    discrete_dimensions = [15, 31]

    # set up training loop
    num_samples_per_env = 10
    num_samples_per_epoch = 100
    num_samples_total = 100000
    min_samples_before_train = 500
    train_frequency = 1
    assert num_samples_per_epoch % num_samples_per_env == 0 and num_samples_total % num_samples_per_epoch == 0
    
    # create the policy
    logits_model, samples_model, deterministic_model = (
        build_policy(image_size=image_size, 
                     discrete_dimensions=discrete_dimensions,
                     discrete_hidden_layers=[512, 512]))
    optimizer = tf.optimizers.Adam(learning_rate=3e-4)
    
    # create the Discretizer
    discretizer = Discretizer(discrete_dimensions, [0.3, -0.16], [0.4666666, 0.16])

    # create the dataset
    buffer = ReplayBuffer(size=num_samples_total, image_size=image_size, action_dim=len(discrete_dimensions))

    # create the env
    env = create_env()

    # training loop
    num_samples = 0
    num_epoch = 0
    total_epochs = num_samples_total // num_samples_per_epoch
    training_start_time = time.time()
    discrete_dimensions_plus_one = [d + 1 for d in discrete_dimensions]

    while num_samples < num_samples_total:
        # diagnostics stuff
        diagnostics = OrderedDict((
            ('num_samples', 0),
            ('total_time', 0),
            ('time_this_epoch', 0),
            ('rewards_this_epoch', 0),
            ('average_loss_this_epoch', 0),
            ('num_random_this_epoch', 0),
            ('num_deterministic_this_epoch', 0),
        ))
        epoch_start_time = time.time()
        num_epoch += 1
        total_loss = 0.0
        num_train_steps = 0

        # run one epoch
        for i in range(num_samples_per_epoch):
            # reset the env (at the beginning as well)
            if i % num_samples_per_env == 0:
                env.interface.reset_robot([0, 0], 0, 0, 0)
                env.room.reset()

            # do sampling
            obs = env.interface.render_camera(use_aux=True)
            
            rand = np.random.uniform()
            if rand < 0.1: # epsilon greedy
                action_discrete = np.random.randint([0, 0], discrete_dimensions_plus_one)
                diagnostics['num_random_this_epoch'] += 1
            elif rand < -1: # epsilon half greedy?
                action_onehot = samples_model(np.array([obs]))
                action_discrete = np.array([tf.argmax(a, axis=-1).numpy().squeeze() for a in action_onehot])
            else:
                action_onehot = deterministic_model(np.array([obs]))
                action_discrete = np.array([tf.argmax(a, axis=-1).numpy().squeeze() for a in action_onehot])
                diagnostics['num_deterministic_this_epoch'] += 1
            action_undiscretized = discretizer.undiscretize(action_discrete)

            reward = do_grasp(env, action_undiscretized)
            diagnostics['rewards_this_epoch'] += reward

            buffer.store_sample(obs, action_discrete, reward)
            num_samples += 1

            # do training
            if num_samples >= min_samples_before_train and num_samples % train_frequency == 0:
                loss = train(logits_model, buffer.sample_batch(256), optimizer, discrete_dimensions)
                total_loss += loss.numpy()
                num_train_steps += 1

        # diagnostics stuff
        diagnostics['num_samples'] = num_samples
        diagnostics['total_time'] = time.time() - training_start_time
        diagnostics['time_this_epoch'] = time.time() - epoch_start_time
        diagnostics['average_loss_this_epoch'] = 'none' if num_train_steps == 0 else total_loss / num_train_steps

        print(f'Epoch {num_epoch}/{total_epochs}:')
        pprint(diagnostics)

    buffer.save("./dataset/data", "autoregressive_2_replay_buffer")
    logits_model.save_weights("./dataset/models/autoregressive_2_model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)