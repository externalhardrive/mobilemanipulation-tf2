import gym
from gym import spaces
import numpy as np
import os
from collections import OrderedDict
import tensorflow as tf
from scipy.special import expit
import tree

from . import locobot_interface

from .base_env import LocobotBaseEnv
from .utils import *
from .rooms import initialize_room
from .nav_envs import *

from softlearning.environments.gym.spaces import DiscreteBox

from softlearning.utils.misc import RunningMeanVar

class LocobotNavigationVacuumEnv(MixedLocobotNavigationEnv):
    def __init__(self, **params):
        defaults = dict()
        defaults["action_space"] = DiscreteBox(
            low=-1.0, high=1.0, 
            dimensions=OrderedDict((("move", 2), ("vacuum", 0)))
        )
        defaults.update(params)

        super().__init__(**defaults)
        print("LocobotNavigationVacuumEnv params:", self.params)

        self.total_vacuum_actions = 0

    def do_move(self, action):
        key, value = action
        if key == "move":
            super().do_move(value)
        else:
            super().do_move([0.0, 0.0])

    def do_grasp(self, action, infos=None, return_grasped_object=False):
        key, value = action
        if key == "vacuum":
            grasps = super().do_grasp(value, return_grasped_object=return_grasped_object)
            # super().do_move([0.2, 0.2])
            return grasps
        else:
            return 0

    def reset(self):
        obs = super().reset()
        self.total_vacuum_actions = 0
        return obs

    def step(self, action):
        # init return values
        reward = 0.0
        infos = {}

        # do move
        self.do_move(action)

        # do grasping
        num_grasped = self.do_grasp(action, infos=infos)
        reward += num_grasped

        # if num_grasped == 0:
        #     reward -= 0.1
        # reward -= 0.01
        
        # infos loggin
        infos["success"] = num_grasped
        infos["total_grasped"] = self.total_grasped

        base_pos = self.interface.get_base_pos()
        infos["base_x"] = base_pos[0]
        infos["base_y"] = base_pos[1]

        if action[0] == "vacuum":
            self.total_vacuum_actions += 1

        infos["vacuum_action"] = int(action[0] == "vacuum")
        infos["total_success_to_vacuum_ratio"] = (0 if self.total_vacuum_actions == 0 
                                                    else self.total_grasped / self.total_vacuum_actions)

        # store trajectory information (usually for reset free)
        if self.trajectory_log_dir and self.trajectory_log_freq > 0:
            self.trajectory_base[self.trajectory_step, 0] = base_pos[0]
            self.trajectory_base[self.trajectory_step, 1] = base_pos[1]
            self.trajectory_step += 1
            
            if self.trajectory_step == self.trajectory_log_freq:
                self.trajectory_step -= 1 # for updating trajectory
                self.update_trajectory_objects()

                self.trajectory_step = 0
                self.trajectory_num += 1

                data = OrderedDict({
                    "base": self.trajectory_base,
                    "objects": self.trajectory_objects
                })

                np.save(self.trajectory_log_path + str(self.trajectory_num), data)
                self.trajectory_objects = OrderedDict({})

        # steps update
        self.num_steps += 1
        done = self.num_steps >= self.max_ep_len

        # get next observation
        obs = self.get_observation()

        return obs, reward, done, infos


class LocobotNavigationVacuumPerturbationEnv(LocobotNavigationVacuumEnv):
    """ Locobot Navigation with Perturbation """
    def __init__(self, **params):
        defaults = dict(
            is_training=False, 
            rnd_lr=3e-4,
            perturbation_batch_size=50,
            perturbation_train_frequency=5,
            num_perturbation_steps=20,
            perturbation_drop_step=9)
        
        defaults.update(params)

        super().__init__(**defaults)
        print("LocobotNavigationVacuumPerturbationEnv params:", self.params)

        self.perturbation_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        self.is_training = self.params["is_training"]
        if self.is_training:
            self.rnd_running_mean_var = RunningMeanVar()
            self.rnd_predictor_optimizer = tf.optimizers.Adam(learning_rate=self.params["rnd_lr"], name="rnd_predictor_optimizer")
            self.perturbation_training_iteration = 0
            self.perturbation_batch_size = self.params["perturbation_batch_size"]
            self.num_perturbation_steps = self.params["num_perturbation_steps"]
            self.perturbation_drop_step = self.params["perturbation_drop_step"]
            self.perturbation_train_frequency = self.params["perturbation_train_frequency"]

    def finish_init(self, replay_pool, perturbation_policy, perturbation_algorithm, rnd_predictor, rnd_target, **kwargs):
        self.replay_pool = replay_pool
        self.perturbation_policy = perturbation_policy
        self.perturbation_algorithm = perturbation_algorithm
        self.rnd_predictor = rnd_predictor
        self.rnd_target = rnd_target

    @tf.function(experimental_relax_shapes=True)
    def update_rnd_predictor(self, batch):
        """Update the RND predictor network. """
        observations = batch['observations']

        with tf.GradientTape() as tape:
            predictor_values = self.rnd_predictor.values(observations)
            target_values = self.rnd_target.values(observations)

            predictor_losses = tf.losses.MSE(y_true=target_values, y_pred=predictor_values)
            predictor_loss = tf.nn.compute_average_loss(predictor_losses)

        predictor_gradients = tape.gradient(predictor_loss, self.rnd_predictor.trainable_variables)
        
        self.rnd_predictor_optimizer.apply_gradients(zip(predictor_gradients, self.rnd_predictor.trainable_variables))

        return predictor_losses

    def train_perturbation_policy(self):
        self.perturbation_training_iteration += 1

        # get random batch
        batch = self.replay_pool.random_batch(self.perturbation_batch_size)

        # update RND predictor loss
        predictor_losses = self.update_rnd_predictor(batch)

        # compute intrinsic reward for this batch
        intrinsic_rewards = predictor_losses.numpy().reshape(-1, 1)
        self.rnd_running_mean_var.update_batch(intrinsic_rewards)
        intrinsic_rewards = intrinsic_rewards / self.rnd_running_mean_var.std

        batch["rewards"] = intrinsic_rewards

        # fix action to only include the continuous parts and is move, otherwise use 0
        actions = batch["actions"]
        gaussians = actions[:, 2:]
        is_move = actions[:, 0:1]
        batch["actions"] = gaussians * is_move

        sac_diagnostics = self.perturbation_algorithm._do_training(self.perturbation_training_iteration, batch)

        diagnostics = OrderedDict({
            # **sac_diagnostics,
            # "rnd_predictor_loss-mean": tf.reduce_mean(predictor_losses),
            "intrinsic_running_std": self.rnd_running_mean_var.std,
            "intrinsic_reward-mean": np.mean(intrinsic_rewards),
            "intrinsic_reward-std": np.std(intrinsic_rewards),
            "intrinsic_reward-min": np.min(intrinsic_rewards),
            "intrinsic_reward-max": np.max(intrinsic_rewards),
        })

        return diagnostics

    def do_perturbation_precedure(self, object_id, infos):
        # diagnostics = []
        for i in range(self.num_perturbation_steps):
            # move
            obs = self.get_observation(include_pixels=True)
            action = self.perturbation_policy.action(obs).numpy()
            self.do_move(("move", action))

            # drop the object
            if i == self.perturbation_drop_step:
                self.interface.move_object(self.room.objects_id[object_id], [0.4, 0.0, 0.015], relative=True)

        # if len(diagnostics) > 0:
        #     diagnostics = tree.map_structure(lambda *d: tf.reduce_mean(d).numpy(), *diagnostics)
        # if infos is not None:
        #     infos.update(diagnostics)

    def do_grasp(self, action, infos=None):
        grasps = super().do_grasp(action, return_grasped_object=True)
        if isinstance(grasps, (tuple, list)):
            reward, object_id = grasps
        else:
            reward = grasps
        if reward > 0.5 and self.is_training:
            self.do_perturbation_precedure(object_id, infos)
        return reward

    def step(self, action):
        infos = {}
        if self.is_training:
            infos["intrinsic_running_std"] = np.nan
            infos["intrinsic_reward-mean"] = np.nan
            infos["intrinsic_reward-std"] = np.nan
            infos["intrinsic_reward-min"] = np.nan
            infos["intrinsic_reward-max"] = np.nan
            if self.num_steps % self.perturbation_train_frequency == 0 and self.replay_pool.size >= self.perturbation_batch_size:
                diagnostics = self.train_perturbation_policy()
                infos.update(diagnostics)

        next_obs, reward, done, new_infos = super().step(action)

        infos.update(new_infos)

        return next_obs, reward, done, infos

class LocobotNavigationDQNGraspingEnv(RoomEnv):
    """ Combines navigation and grasping trained by DQN.
        Training cannot be parallelized.
    """

    grasp_deterministic_model = None

    def __init__(self, **params):
        defaults = dict(
            steps_per_second=2,
            max_velocity=20.0,
            num_grasp_repeat=1,
            is_training=True,
            grasp_training_params=dict(
                discrete_hidden_layers=[512, 512],
                lr=1e-5,
                batch_size=50,
                batch_size_successes_ratio=0.1,
                buffer_size=int(1e5),
                min_successes_before_train=5,
                min_fails_before_train=500,
                epsilon=0.1,
            ),
        )

        # movement camera
        defaults["image_size"] = 100
        defaults["camera_fov"] = 55

        # grasp camera
        # defaults['use_aux_camera'] = True
        # defaults['aux_camera_look_pos'] = [0.4, 0, 0.05]
        # defaults['aux_camera_fov'] = 35
        # defaults['aux_image_size'] = 100

        # observation space for base
        defaults['observation_space'] = spaces.Dict({
            "current_velocity": spaces.Box(low=-1.0, high=1.0, shape=(2,)),
            # "pixels": added by PixelObservationWrapper
        })
        
        # action space for base
        defaults['action_space'] = DiscreteBox(
            low=-1.0, high=1.0, 
            dimensions=OrderedDict((("move", 2), ("grasp", 0)))
        )

        defaults['max_ep_len'] = 200

        defaults.update(params)

        super().__init__(**defaults)

        # move stuff
        self.num_sim_steps_per_env_step = int(60 / self.params["steps_per_second"])
        self.max_velocity = self.params["max_velocity"]
        self.target_velocity = np.array([0.0, 0.0])

        # grasp stuff
        self.grasp_discretizer = Discretizer([15, 31], [0.3, -0.16], [0.4666666, 0.16])
        self.num_grasp_repeat = self.params["num_grasp_repeat"]
        self.total_grasp_actions = 0
        self.total_grasped = 0

        # grasp training setup
        self.is_training = self.params["is_training"]
        if self.is_training:
            if LocobotNavigationDQNGraspingEnv.grasp_deterministic_model is not None:
                raise ValueError("Cannot have two training environments at the same time")

            training_params = self.params["grasp_training_params"]

            logits_model, deterministic_model = build_image_discrete_policy(
                image_size=self.grasp_image_size,
                discrete_dimension=15*31,
                discrete_hidden_layers=training_params["discrete_hidden_layers"])

            LocobotNavigationDQNGraspingEnv.grasp_deterministic_model = deterministic_model
            self.grasp_deterministic_model = deterministic_model
            self.grasp_logits_model = logits_model

            self.grasp_optimizer = tf.optimizers.Adam(learning_rate=training_params["lr"])

            self.grasp_buffer_successes = ReplayBuffer(
                size=training_params["buffer_size"], 
                observation_shape=(self.grasp_image_size, self.grasp_image_size, 3), 
                action_dim=1, 
                observation_dtype=np.uint8, action_dtype=np.int32)
            self.grasp_buffer_fails = ReplayBuffer(
                size=training_params["buffer_size"], 
                observation_shape=(self.grasp_image_size, self.grasp_image_size, 3), 
                action_dim=1, 
                observation_dtype=np.uint8, action_dtype=np.int32)

            self.grasp_batch_size = training_params["batch_size"]
            self.grasp_batch_size_successes_ratio = training_params["batch_size_successes_ratio"]
            self.grasp_batch_size_successes = int(self.grasp_batch_size * self.grasp_batch_size_successes_ratio)
            self.grasp_batch_size_fails = self.grasp_batch_size - self.grasp_batch_size_successes
            self.grasp_min_successes_before_train = training_params["min_successes_before_train"]
            self.grasp_min_fails_before_train = training_params["min_fails_before_train"]
            self.grasp_epsilon = training_params["epsilon"]
        else:
            # TODO(externalhardrive): Add ability to load grasping model from file
            if LocobotNavigationDQNGraspingEnv.grasp_deterministic_model is None:
                raise ValueError("Training environment must be made first")
            self.grasp_deterministic_model = LocobotNavigationDQNGraspingEnv.grasp_deterministic_model

    def reset(self):
        _ = super().reset()

        self.total_grasped = 0
        self.total_grasp_actions = 0

        self.target_velocity = np.array([0, 0]) # np.array([self.max_velocity * 0.2] * 2)
        self.interface.set_wheels_velocity(self.target_velocity[0], self.target_velocity[1])
        for _ in range(60):
            self.interface.step()
        
        obs = self.get_observation()

        return obs

    def crop_obs(self, obs):
        return obs[..., 38:98, 20:80, :]

    @property
    def grasp_image_size(self):
        return 60
    
    def process_batch(self, batch):
        """ Modifies batch, the training batch data. """
        observations = self.crop_obs(batch["observations"]["pixels"])
        actions = batch["actions"]
        rewards = batch["rewards"]

        # actions goes: [is move, is grasp, move left, move right]
        is_grasp = actions[:, 1:2]

        # relabel if the action is a grasp and the reward is 0. If reward is 1 then no reason to relabel
        use_relabeled_value = is_grasp * (1.0 - rewards)

        max_Q_value = expit(np.max(self.grasp_logits_model(observations).numpy(), axis=-1, keepdims=True))
        batch["rewards"] = max_Q_value * use_relabeled_value + rewards * (1.0 - use_relabeled_value)

    def render(self, *args, **kwargs):
        return self.interface.render_camera(use_aux=False)

    def get_observation(self):
        obs = OrderedDict()

        if self.interface.renders:
            # pixel observations are generated by PixelObservationWrapper, unless we want to manually check it
            obs["pixels"] = self.render()
        
        velocity = self.interface.get_wheels_velocity()
        obs["current_velocity"] = np.clip(velocity / self.max_velocity, -1.0, 1.0)
        # obs["target_velocity"] = np.clip(self.target_velocity / self.max_velocity, -1.0, 1.0)
        
        return obs

    def do_move(self, action):
        self.target_velocity = np.array(action) * self.max_velocity
        new_left, new_right = self.target_velocity

        self.interface.set_wheels_velocity(new_left, new_right)
        self.interface.do_steps(self.num_sim_steps_per_env_step)

    def do_grasp(self, loc):
        self.interface.execute_grasp_direct(loc, 0.0)
        reward = 0
        for i in range(self.room.num_objects):
            block_pos, _ = self.interface.get_object(self.room.objects_id[i])
            if block_pos[2] > 0.04:
                reward = 1
                self.interface.move_object(
                    self.room.objects_id[i], 
                    [self.room.extent + np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.01])
                break
        self.interface.move_arm_to_start(steps=90, max_velocity=8.0)
        return reward

    def are_blocks_graspable(self):
        for i in range(self.room.num_objects):
            object_pos, _ = self.interface.get_object(self.room.objects_id[i], relative=True)
            if is_in_rect(object_pos[0], object_pos[1], 0.3, -0.16, 0.466666666, 0.16):
                return True 
        return False

    @tf.function(experimental_relax_shapes=True)
    def train_grasp(self, data):
        observations = data['observations']
        rewards = data['rewards']
        actions_discrete = data['actions']
        actions_onehot = tf.one_hot(actions_discrete[:, 0], depth=15*31)

        with tf.GradientTape() as tape:
            logits = self.grasp_logits_model(observations)
            taken_logits = tf.reduce_sum(logits * actions_onehot, axis=-1, keepdims=True)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=rewards, logits=taken_logits)
            loss = tf.nn.compute_average_loss(losses)

        grads = tape.gradient(loss, self.grasp_logits_model.trainable_variables)
        self.grasp_optimizer.apply_gradients(zip(grads, self.grasp_logits_model.trainable_variables))

        return loss

    def do_grasp_action(self, infos):
        num_grasps = 0
        reward = 0
        losses = []
        successes = []
        while num_grasps < self.num_grasp_repeat and reward < 1: #self.are_blocks_graspable():
            # get the grasping camera image
            obs = self.interface.render_camera(use_aux=False)
            obs = self.crop_obs(obs)

            # epsilon greedy or initial exploration
            if self.is_training:
                if (np.random.uniform() < self.grasp_epsilon or 
                        self.grasp_buffer_successes.num_samples < self.grasp_min_successes_before_train or
                        self.grasp_buffer_fails.num_samples < self.grasp_min_fails_before_train):
                    action_discrete = np.random.randint(0, 15*31)
                    infos["grasp_random"] = 1
                else:
                    action_discrete = self.grasp_deterministic_model(np.array([obs])).numpy()
                    infos["grasp_deterministic"] = 1
            else:
                action_discrete = self.grasp_deterministic_model(np.array([obs])).numpy()

            # convert to local grasp position and execute grasp
            action_undiscretized = self.grasp_discretizer.undiscretize(self.grasp_discretizer.unflatten(action_discrete))
            reward = self.do_grasp(action_undiscretized)

            successes.append(reward)

            # store in replay buffer
            if self.is_training:
                if reward > 0:
                    self.grasp_buffer_successes.store_sample(obs, action_discrete, reward)
                else:
                    self.grasp_buffer_fails.store_sample(obs, action_discrete, reward)
            
            num_grasps += 1

            # train once
            if self.is_training:
                if (self.grasp_buffer_successes.num_samples >= self.grasp_min_successes_before_train and 
                        self.grasp_buffer_fails.num_samples >= self.grasp_min_fails_before_train):
                    data_successes = self.grasp_buffer_successes.sample_batch(self.grasp_batch_size_successes)
                    data_fails = self.grasp_buffer_fails.sample_batch(self.grasp_batch_size_fails)
                    data = {
                        "observations": np.concatenate((data_successes["observations"], data_fails["observations"]), axis=0),
                        "actions": np.concatenate((data_successes["actions"], data_fails["actions"]), axis=0),
                        "rewards": np.concatenate((data_successes["rewards"], data_fails["rewards"]), axis=0)
                    }
                    loss = self.train_grasp(data)
                    losses.append(loss.numpy())

            # if success, stop grasping
            if reward > 0:
                self.total_grasped += 1
                break
        
        if len(losses) > 0:
            infos["grasp_training_loss"] = np.mean(losses)
        
        if len(successes) > 0:
            infos["average_success_per_taken_action"] = np.mean(successes)

        infos["num_grasps_per_action"] = num_grasps
        infos["success_per_action"] = int(reward > 0)

        # if num_grasps > 0:
        #     infos["num_grasps_per_taken_action"] = num_grasps
        #     infos["success_per_taken_action"] = int(reward > 0)

        return reward #* num_grasps

    def step(self, action):
        action_key, action_value = action

        reward = 0.0
        infos = {}

        infos["num_grasps_per_action"] = np.nan
        infos["success_per_action"] = np.nan
        # infos["num_grasps_per_taken_action"] = np.nan
        # infos["success_per_taken_action"] = np.nan
        infos["average_success_per_taken_action"] = np.nan

        if self.is_training:
            infos["grasp_training_loss"] = np.nan
            infos["grasp_random"] = np.nan
            infos["grasp_deterministic"] = np.nan
        
        if action_key == "move":
            self.do_move(action_value)
        elif action_key == "grasp":
            self.do_move([0, 0])
            reward = self.do_grasp_action(infos)
            self.total_grasp_actions += 1
        else:
            raise ValueError(f"action {action} is not in the action space")
        
        infos["total_grasp_actions"] = self.total_grasp_actions
        infos["total_grasped"] = self.total_grasped

        if self.is_training:
            infos["grasp_buffer_successes_samples"] = self.grasp_buffer_successes.num_samples
            infos["grasp_buffer_fails_samples"] = self.grasp_buffer_fails.num_samples

        self.num_steps += 1
        done = self.num_steps >= self.max_ep_len

        obs = self.get_observation()

        return obs, reward, done, infos
