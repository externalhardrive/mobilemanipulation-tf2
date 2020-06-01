import os 
import numpy as np
import time

from collections import OrderedDict, defaultdict

import tree

import pprint

def training_loop(
        num_samples_per_env=10,
        num_samples_per_epoch=100,
        num_samples_total=100000,
        min_samples_before_train=1000,
        train_frequency=5,
        train_batch_size=256,
        validation_prob=0.1,
        validation_batch_size=100,
        env=None,
        sampler=None,
        train_buffer=None, validation_buffer=None,
        train_function=None, validation_function=None,
    ):

    assert num_samples_per_epoch % num_samples_per_env == 0 and num_samples_total % num_samples_per_epoch == 0

    print()
    print("Training Loop params:")
    pprint.pprint(dict(
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
    ))
    print()

    # training loop
    num_samples = 0
    num_epoch = 0
    total_epochs = num_samples_total // num_samples_per_epoch
    training_start_time = time.time()

    all_diagnostics = []

    while num_samples < num_samples_total:
        # diagnostics stuff
        diagnostics = OrderedDict((
            ('num_samples_total', 0),
            ('num_training_samples', 0),
            ('num_validation_samples', 0),
            ('total_time', 0),
            ('time', 0),
            ('average_training_loss', 0),
            ('validation_loss', 'none'),
            ('num_envs', 0),
            ('num_success', 0),
            ('average_success_ratio_per_env', 0),
            ('average_tries_per_env', 0),
            ('envs_with_success_ratio', 0),
            ('sampler_infos', None),
        ))
        epoch_start_time = time.time()
        num_epoch += 1
        total_training_loss = 0.0
        num_train_steps = 0

        # run one epoch
        sampler_infos = defaultdict(list)
        num_samples_this_env = 0
        successes_this_env = 0
        total_success_ratio = 0
        num_envs_with_success = 0
        for i in range(num_samples_per_epoch):
            # reset the env (at the beginning as well)
            if i == 0 or num_samples_this_env >= num_samples_per_env or env.should_reset():
                if i > 0:
                    success_ratio = successes_this_env / num_samples_this_env
                    total_success_ratio += success_ratio
                if successes_this_env > 0:
                    num_envs_with_success += 1
                env.reset()
                num_samples_this_env = 0
                successes_this_env = 0
                diagnostics['num_envs'] += 1

            # do sampling
            obs, action, reward, infos = sampler(num_samples)
            for k in infos:
                sampler_infos[k].append(infos[k])

            diagnostics['num_success'] += reward
            successes_this_env += reward

            if np.random.uniform() < validation_prob:
                validation_buffer.store_sample(obs, action, reward)
            else:
                train_buffer.store_sample(obs, action, reward)
            
            num_samples += 1
            num_samples_this_env += 1

            # do training
            if num_samples >= min_samples_before_train and num_samples % train_frequency == 0:
                data = train_buffer.sample_batch(train_batch_size)
                loss = train_function(data)
                total_training_loss += loss.numpy()
                num_train_steps += 1

        # diagnostics stuff
        diagnostics['num_samples_total'] = num_samples
        diagnostics['num_training_samples'] = train_buffer.num_samples
        diagnostics['num_validation_samples'] = validation_buffer.num_samples
        diagnostics['total_time'] = time.time() - training_start_time
        diagnostics['time'] = time.time() - epoch_start_time
        
        diagnostics['average_training_loss'] = 'none' if num_train_steps == 0 else total_training_loss / num_train_steps

        if validation_buffer.num_samples >= validation_batch_size:
            datas = validation_buffer.get_all_samples_in_batch(validation_batch_size)
            total_validation_loss = 0.0
            for data in datas:
                total_validation_loss += validation_function(data).numpy()
            diagnostics['validation_loss'] = total_validation_loss / len(datas)

        success_ratio = successes_this_env / num_samples_this_env
        total_success_ratio += success_ratio
        diagnostics['average_success_ratio_per_env'] = total_success_ratio / diagnostics['num_envs']
        diagnostics['average_tries_per_env'] = num_samples_per_epoch / diagnostics['num_envs']
        if successes_this_env > 0:
            num_envs_with_success += 1
        diagnostics['envs_with_success_ratio'] = num_envs_with_success / diagnostics['num_envs']

        condensed_infos = OrderedDict()
        for k, v in sampler_infos.items():
            condensed_infos[k + '-mean'] = np.mean(v)
            condensed_infos[k + '-sum'] = np.sum(v)
            condensed_infos[k + '-count'] = len(v)
        diagnostics['sampler_infos'] = condensed_infos

        print(f'Epoch {num_epoch}/{total_epochs}:')
        pprint.pprint(diagnostics)
        all_diagnostics.append(diagnostics)

    return all_diagnostics
