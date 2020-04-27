from collections import defaultdict

import numpy as np
import tree

from .base_sampler import BaseSampler

import sys, os
import time

class SimpleSampler(BaseSampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._total_samples = 0

        self._is_first_step = True

    def reset(self):
        if self.policy is not None:
            self.policy.reset()

        self._path_length = 0
        self._path_return = 0
        self._current_path = []
        self._current_observation = self.environment.reset()

    @property
    def _policy_input(self):
        return self._current_observation

    def _process_sample(self,
                        observation,
                        action,
                        reward,
                        terminal,
                        next_observation,
                        info):
        processed_observation = {
            'observations': observation,
            'actions': action,
            'rewards': np.atleast_1d(reward),
            'terminals': np.atleast_1d(terminal),
            'next_observations': next_observation,
            'infos': info,
        }

        return processed_observation

    def sample(self):
        if self._is_first_step:
            self.reset()

        action = self.policy.action(self._policy_input).numpy()
        
        if np.any(np.abs(action) >= 1.0):
            print("WARNING:")
            print("ACTION:", action)
            sys.stdout.flush()

        if np.any(np.isnan(action)):
            print("WARNING:")
            print("observation:", self._policy_input)
            print("action:", action)
            save_path = f"/home/externalhardrive/RAIL/mobilemanipulation-tf2/nohup_output/error_2/error_policy_{self._total_samples}/"
            os.makedirs(save_path, exist_ok=True)
            self.policy.save(save_path + "policy")
            np.save(save_path + "observation", self._policy_input)
            np.save(save_path + "curr_path", np.array(self._current_path))
            
            from pprint import pprint
            all_observations = self.pool.last_n_batch(self.pool.size)['observations']
            print("min:")
            pprint(tree.map_structure(lambda x: np.min(x, axis=0), all_observations))
            print()

            print("max:")
            pprint(tree.map_structure(lambda x: np.max(x, axis=0), all_observations))
            print()

            print("mean:")
            pprint(tree.map_structure(lambda x: np.mean(x, axis=0), all_observations))
            print()

            print("std:")
            pprint(tree.map_structure(lambda x: np.std(x, axis=0), all_observations))
            print()

            breakpoint()
            pass

            sys.stdout.flush()
            time.sleep(10)
            # temporary
            raise RuntimeError("NAN IN OUTPUT")
            action = np.zeros(action.shape)

        next_observation, reward, terminal, info = self.environment.step(
            action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        processed_sample = self._process_sample(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            info=info,
        )

        self._current_path.append(processed_sample)

        if terminal or self._path_length >= self._max_path_length:
            last_path = tree.map_structure(
                lambda *x: np.stack(x, axis=0), *self._current_path)

            self.pool.add_path({
                key: value
                for key, value in last_path.items()
                if key != 'infos'
            })

            self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return
            self._n_episodes += 1

            self.pool.terminate_episode()

            self._is_first_step = True
            # Reset is done in the beginning of next episode, see above.

        else:
            self._current_observation = next_observation
            self._is_first_step = False

        return next_observation, reward, terminal, info

    def get_diagnostics(self):
        diagnostics = super(SimpleSampler, self).get_diagnostics()
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
        })

        return diagnostics
