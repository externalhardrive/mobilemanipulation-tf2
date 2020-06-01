import numpy as np

import os

class ReplayBuffer:
    def __init__(self, size, image_size, action_dim, action_dtype=np.int32):
        self._size = size
        self._observations = np.zeros((size, image_size, image_size, 3), dtype=np.uint8)
        self._actions = np.zeros((size, action_dim), dtype=action_dtype)
        self._rewards = np.zeros((size, 1), dtype=np.float32)
        self._num = 0

    @property
    def num_samples(self):
        return self._num

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