import argparse
import json
import os
from pathlib import Path
import pickle

import pandas as pd
import numpy as np

from collections import OrderedDict

from softlearning.environments.utils import get_environment_from_params
from softlearning import policies
from softlearning.samplers import rollouts
from softlearning.utils.tensorflow import set_gpu_memory_growth
from softlearning.utils.video import save_video
from examples.development.main import ExperimentRunner

def load_environment(variant, env_kwargs):
    environment_params = (
        variant['environment_params']['training']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])
    # environment_params["kwargs"]["renders"] = True
    # environment_params["kwargs"]["step_duration"] = 1/60
    environment_params["kwargs"].update(env_kwargs)

    environment = get_environment_from_params(environment_params)
    return environment


def load_policy(checkpoint_dir, variant, environment):
    policy_params = variant['policy_params'].copy()
    policy_params['config'] = {
        **policy_params['config'],
        'action_range': (environment.action_space.low,
                         environment.action_space.high),
        'input_shapes': environment.observation_shape,
        'output_shape': environment.action_shape,
    }

    policy = policies.get(policy_params)

    status = policy.load_weights(checkpoint_dir)
    status.assert_consumed().run_restore_ops()

    return policy

variant_path = 'nohup_output/error_policy/params.pkl'
with open(variant_path, 'rb') as f:
    variant = pickle.load(f)

env = load_environment(variant, {})
policy = load_policy('nohup_output/error_policy/error_policy/', variant, env)

print(variant)

observation = np.load('nohup_output/error_policy/error_policy/observation.npy', allow_pickle=True)[()]
