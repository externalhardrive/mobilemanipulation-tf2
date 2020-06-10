import gym
from gym import spaces
import numpy as np
import os
from collections import OrderedDict
import tensorflow as tf
from scipy.special import expit

from . import locobot_interface

from .base_env import LocobotBaseEnv
from .utils import *
from .rooms import initialize_room
from .nav_envs import *

from softlearning.environments.gym.spaces import DiscreteBox

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

    def do_grasp(self, action):
        key, value = action
        if key == "vacuum":
            grasps =  super().do_grasp(value)
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
        num_grasped = self.do_grasp(action)
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

        # steps update
        self.num_steps += 1
        done = self.num_steps >= self.max_ep_len

        # get next observation
        obs = self.get_observation()

        return obs, reward, done, infos


class LocobotNavigationDQNGraspingEnv(RoomEnv):
    """ Combines navigation and grasping trained by DQN.
        Training cannot be parallelized.
    """

    grasp_deterministic_model = None

    def __init__(self, **params):
        defaults = dict(
            steps_per_second=2,
            max_velocity=20.0,
            num_grasp_repeat=10, 
            is_training=True,
            grasp_training_params=dict(
                discrete_hidden_layers=[512, 512],
                lr=1e-5,
                batch_size=50,
                buffer_size=int(1e5),
                min_samples_before_train=100,
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

            self.grasp_buffer = ReplayBuffer(
                size=training_params["buffer_size"], 
                observation_shape=(self.grasp_image_size, self.grasp_image_size, 3), 
                action_dim=1, 
                observation_dtype=np.uint8, action_dtype=np.int32)

            self.grasp_batch_size = training_params["batch_size"]
            self.grasp_min_samples_before_train = training_params["min_samples_before_train"]
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
        observations = self.crop_obs(batch["observations"])
        actions = batch["actions"]
        rewards = batch["rewards"]

        # actions goes: [is move, is grasp, move left, move right]
        is_grasp = actions[:, 1:2]
        print(batch)

        max_Q_value = expit(np.max(self.grasp_logits_model(observations).numpy(), axis=-1, keepdims=True))
        batch["rewards"] = max_Q_value * is_grasp + rewards * (1.0 - is_grasp)

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
        while num_grasps < self.num_grasp_repeat and self.are_blocks_graspable():
            # get the grasping camera image
            obs = self.interface.render_camera(use_aux=False)
            obs = self.crop_obs(obs)

            # epsilon greedy or initial exploration
            if self.is_training:
                if np.random.uniform() < self.grasp_epsilon or self.grasp_buffer.num_samples < self.grasp_min_samples_before_train:
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
                self.grasp_buffer.store_sample(obs, action_discrete, reward)
            
            num_grasps += 1

            # train once
            if self.is_training and self.grasp_buffer.num_samples >= self.grasp_min_samples_before_train:
                data = self.grasp_buffer.sample_batch(self.grasp_batch_size)
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

        if num_grasps > 0:
            infos["num_grasps_per_taken_action"] = num_grasps
            infos["success_per_taken_action"] = int(reward > 0)

        return reward #* num_grasps

    def step(self, action):
        action_key, action_value = action

        reward = 0.0
        infos = {}

        infos["num_grasps_per_action"] = np.nan
        infos["success_per_action"] = np.nan
        infos["num_grasps_per_taken_action"] = np.nan
        infos["success_per_taken_action"] = np.nan
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
            infos["grasp_buffer_samples"] = self.grasp_buffer.num_samples

        self.num_steps += 1
        done = self.num_steps >= self.max_ep_len

        obs = self.get_observation()

        return obs, reward, done, infos
