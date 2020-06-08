import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors

from softlearning.models.convnet import convnet_model
from softlearning.models.feedforward import feedforward_model

CURR_PATH = os.path.dirname(os.path.abspath(__file__))

URDF = {
    "locobot": os.path.join(CURR_PATH, 'urdf/locobot_description.urdf'),
    "miniblock": os.path.join(CURR_PATH, 'urdf/miniblock.urdf'),
    "greenbox": os.path.join(CURR_PATH, 'urdf/greenbox.urdf'),
    "redbox": os.path.join(CURR_PATH, 'urdf/redbox.urdf'),
    "largerminiblock": os.path.join(CURR_PATH, 'urdf/largerminiblock.urdf'),
    "greenball": os.path.join(CURR_PATH, 'urdf/greenball.urdf'),
    "greensquareball": os.path.join(CURR_PATH, 'urdf/greensquareball_v2.urdf'),
    "greensquareball_large": os.path.join(CURR_PATH, 'urdf/greensquareball_large.urdf'),
    "walls": os.path.join(CURR_PATH, 'urdf/walls.urdf'),
    "plane": os.path.join(CURR_PATH, 'urdf/plane.urdf'),
    "rectangular_pillar": os.path.join(CURR_PATH, 'urdf/rectangular_pillar.urdf'),
    "solid_box": os.path.join(CURR_PATH, 'urdf/solid_box.urdf'),
    "walls_2": os.path.join(CURR_PATH, 'urdf/medium_room/walls.urdf'),
    "textured_box": os.path.join(CURR_PATH, 'urdf/medium_room/box.urdf'),
}

TEXTURE = {
    "wood": os.path.join(CURR_PATH, 'urdf/medium_room/wood2.png'),
    "wall": os.path.join(CURR_PATH, 'urdf/medium_room/wall1.png'),
    "marble": os.path.join(CURR_PATH, 'urdf/medium_room/marble.png'),
    "crate": os.path.join(CURR_PATH, 'urdf/medium_room/crate.png'),
    "navy": os.path.join(CURR_PATH, 'urdf/medium_room/navy_cloth.png'),
    "red": os.path.join(CURR_PATH, 'urdf/medium_room/red_cloth.png'),
}


def is_in_rect(x, y, min_x, min_y, max_x, max_y):
    return min_x < x < max_x and min_y < y < max_y

def is_in_circle(x, y, center_x, center_y, radius):
    return (x - center_x) ** 2 + (y - center_y) ** 2 < radius ** 2


class Discretizer:
    def __init__(self, sizes, mins, maxs):
        self._sizes = np.array(sizes)
        self._mins = np.array(mins) 
        self._maxs = np.array(maxs) 

        self._step_sizes = (self._maxs - self._mins) / self._sizes

    @property
    def dimensions(self):
        return self._sizes

    def discretize(self, action):
        centered = action - self._mins
        indices = np.floor_divide(centered, self._step_sizes)
        clipped = np.clip(indices, 0, self._sizes)
        return clipped

    def undiscretize(self, action):
        return action * self._step_sizes + self._mins + self._step_sizes * 0.5

    def flatten(self, action):
        return np.ravel_multi_index(action, self._sizes, order='C')

    def unflatten(self, index):
        return np.array(np.unravel_index(index, self._sizes, order='C')).squeeze()


def build_image_discrete_policy(
        image_size=100,
        discrete_hidden_layers=(512, 512),
        discrete_dimension=15 * 31
    ):
    obs_in = tfk.Input((image_size, image_size, 3))
    conv_out = convnet_model(
        conv_filters=(64, 64, 64),
        conv_kernel_sizes=(3, 3, 3),
        conv_strides=(2, 2, 2),
        activation="relu",
    )(obs_in)
    
    logits_out = feedforward_model(
        discrete_hidden_layers,
        [discrete_dimension],
        activation='relu',
        output_activation='linear',
    )(conv_out)
    
    logits_model = tfk.Model(obs_in, logits_out)

    def deterministic_model(obs):
        logits = logits_model(obs)
        inds = tf.argmax(logits, axis=-1)
        return inds

    return logits_model, deterministic_model


class ReplayBuffer:
    """ Poor man's replay buffer. """
    def __init__(self, size, observation_shape, action_dim, observation_dtype=np.uint8, action_dtype=np.int32):
        self._size = size
        self._observations = np.zeros((size,) + observation_shape, dtype=observation_dtype)
        self._actions = np.zeros((size, action_dim), dtype=action_dtype)
        self._rewards = np.zeros((size, 1), dtype=np.float32)
        self._num = 0
        self._index = 0

    @property
    def num_samples(self):
        return self._num

    def store_sample(self, observation, action, reward):
        self._observations[self._index] = observation
        self._actions[self._index] = action
        self._rewards[self._index] = reward
        self._num = min(self._num + 1, self._size)
        self._index = (self._index + 1) % self._size

    def get_all_samples(self):
        data = {
            'observations': self._observations[:self._num],
            'actions': self._actions[:self._num],
            'rewards': self._rewards[:self._num],
        }
        return data

    def get_all_samples_in_batch(self, batch_size):
        datas = []
        for i in range(0, (self._num // batch_size) * batch_size, batch_size):
            data = {
                'observations': self._observations[i:i+batch_size],
                'actions': self._actions[i:i+batch_size],
                'rewards': self._rewards[i:i+batch_size],
            }
            datas.append(data)
        if self._num % batch_size != 0:
            datas.append(self.sample_batch(batch_size))
        return datas
    
    def get_all_samples_in_batch_random(self, batch_size):
        inds = np.concatenate([np.arange(self._num), np.arange((batch_size - self._num % batch_size) % batch_size)])
        np.random.shuffle(inds)

        observations = self._observations[inds]
        actions = self._actions[inds]
        rewards = self._rewards[inds]

        datas = []
        for i in range(0, self._num, batch_size):
            data = {
                'observations': observations[i:i+batch_size],
                'actions': actions[i:i+batch_size],
                'rewards': rewards[i:i+batch_size],
            }
            datas.append(data)
        return datas

    def get_all_success_in_batch_random(self, batch_size):
        successes = (self._rewards == 1)[:, 0]
        observations = self._observations[successes]
        actions = self._actions[successes]
        rewards = self._rewards[successes]
        num_success = len(observations)

        inds = np.concatenate([np.arange(num_success), np.arange((batch_size - num_success % batch_size) % batch_size)])
        np.random.shuffle(inds)

        observations = observations[inds]
        actions = actions[inds]
        rewards = rewards[inds]

        datas = []
        for i in range(0, num_success, batch_size):
            data = {
                'observations': observations[i:i+batch_size],
                'actions': actions[i:i+batch_size],
                'rewards': rewards[i:i+batch_size],
            }
            datas.append(data)
        return datas

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