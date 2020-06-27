import gym 
import numpy as np 
import tensorflow as tf
import os

from gym import spaces

from collections import OrderedDict

from softlearning.rnd import *
from softlearning.utils.misc import RunningMeanVar

class PointGridExploration(gym.Env):
    def __init__(self, max_steps=20, is_training=False, trajectory_log_dir=None, trajectory_log_freq=0):
        self.max_steps = max_steps
        self.pos = np.array([0.0, 0.0])
        self.num_steps = 0

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        self.is_training = is_training
        if self.is_training:
            self.rnd_predictor, self.rnd_target = rnd_predictor_and_target(
                input_shapes={'observations': tf.TensorShape((2,))},
                output_shape=(512,),
                hidden_layer_sizes=(512, 512),
                activation='relu',
                output_activation='linear',
            )
            self.rnd_target.set_weights([np.random.normal(0, 0.1, size=weights.shape) for weights in self.rnd_target.get_weights()])
            self.rnd_running_mean_var = RunningMeanVar()
            self.rnd_predictor_optimizer = tf.optimizers.Adam(learning_rate=3e-4, name="rnd_predictor_optimizer")

        self.trajectory_log_dir = trajectory_log_dir
        self.trajectory_log_freq = trajectory_log_freq
        if self.trajectory_log_dir:
            os.makedirs(self.trajectory_log_dir, exist_ok=True)
            uid = str(np.random.randint(1e6))
            self.trajectory_log_path = os.path.join(self.trajectory_log_dir, "trajectory_" + uid + "_")
            self.trajectory_num = 0
            self.trajectory_pos = np.zeros((self.max_steps, 2))
            self.trajectories = OrderedDict()

            print(f"PointGridExploration {'training' if self.is_training else 'evaluation'} trajectory uid {uid}")

    def get_intrinsic_reward(self, obs):
        batch = {'observations': tree.map_structure(lambda x: x[np.newaxis, ...], obs)}
        return self.get_intrinsic_rewards(batch).squeeze()

    def get_intrinsic_rewards(self, batch, normalize=True):
        observations = batch['observations']

        predictor_values = self.rnd_predictor.values(observations)
        target_values = self.rnd_target.values(observations)

        intrinsic_rewards = tf.losses.MSE(y_true=target_values, y_pred=predictor_values).numpy().reshape(-1, 1)
        if normalize:
            intrinsic_rewards = intrinsic_rewards / self.rnd_running_mean_var.std

        return intrinsic_rewards

    @tf.function(experimental_relax_shapes=True)
    def update_rnd_predictor(self, batch):
        """Update the RND predictor network. """
        observations = batch['observations']
        target_values = self.rnd_target.values(observations)

        with tf.GradientTape() as tape:
            predictor_values = self.rnd_predictor.values(observations)

            predictor_losses = tf.losses.MSE(y_true=tf.stop_gradient(target_values), y_pred=predictor_values)
            predictor_loss = tf.nn.compute_average_loss(predictor_losses)

        predictor_gradients = tape.gradient(predictor_loss, self.rnd_predictor.trainable_variables)
        self.rnd_predictor_optimizer.apply_gradients(zip(predictor_gradients, self.rnd_predictor.trainable_variables))

        return predictor_losses

    def train_rnd(self, batch):
        # update predictor network
        predictor_losses = self.update_rnd_predictor(batch)
        
        # update running mean var
        unnormalized_intrinsic_rewards = self.get_intrinsic_rewards(batch, normalize=False)
        self.rnd_running_mean_var.update_batch(unnormalized_intrinsic_rewards)

        # diagnostics
        diagnostics = OrderedDict({
            "rnd_predictor_loss-mean": tf.reduce_mean(predictor_losses),
        })
        return diagnostics
    
    def process_batch(self, batch):
        intrinsic_rewards = self.get_intrinsic_rewards(batch)
        batch["rewards"] = intrinsic_rewards * 100.0
        train_diagnostics = self.train_rnd(batch)

        diagnostics = OrderedDict({
            **train_diagnostics,
            "intrinsic_running_std": self.rnd_running_mean_var.std,
            "intrinsic_reward-mean": np.mean(intrinsic_rewards),
            "intrinsic_reward-std": np.std(intrinsic_rewards),
            "intrinsic_reward-min": np.min(intrinsic_rewards),
            "intrinsic_reward-max": np.max(intrinsic_rewards),
        })
        return diagnostics

    def get_observation(self):
        return np.copy(self.pos) #/ self.max_steps

    def reset(self):
        self.pos = np.array([0.0, 0.0])
        self.num_steps = 0
        return self.get_observation()

    def step(self, action):
        if self.trajectory_log_dir:
            self.trajectory_pos[self.num_steps] = self.pos

        action = np.array(action)
        self.pos += action
        
        obs = self.get_observation()

        reward = 0.0

        self.num_steps += 1
        done = self.num_steps >= self.max_steps

        if done and self.trajectory_log_dir:
            self.trajectories[self.trajectory_num] = self.trajectory_pos
            self.trajectory_num += 1
            self.trajectory_pos = np.zeros((self.max_steps, 2))
            if self.trajectory_num % self.trajectory_log_freq == 0:
                np.save(self.trajectory_log_path + str(self.trajectory_num), self.trajectories)
                self.trajectories = OrderedDict()
                self.rnd_target.save_weights(os.path.join(self.trajectory_log_dir, "rnd_target"))
                self.rnd_predictor.save_weights(os.path.join(self.trajectory_log_dir, "rnd_predictor"))

        return obs, reward, done, {}
        
