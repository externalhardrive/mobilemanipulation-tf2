import os 
import numpy as np
import time

def dqn_train(
    num_samples_per_env=10,
    num_samples_per_epoch=100,
    num_samples_total=100000,
    min_samples_before_train=1000,
    train_frequency=5,
    epsilon=0.1,
    train_batch_size=256,
    validation_prob=0.1,
    validation_batch_size=100,
    env=None,
    buffer=None,
    validation_buffer=None,
    train_model=None,
    random_sampler=None, deterministic_sampler=None,
    train_function=None, validation_function=None,
    discretizer=None, 
    optimizer=None,
    discrete_dimensions=None,
    name=None,
    save_folder='./others/logs/'
    ):

    print(dict(
        num_samples_per_env=num_samples_per_env,
        num_samples_per_epoch=num_samples_per_epoch,
        num_samples_total=num_samples_total,
        min_samples_before_train=min_samples_before_train,
        train_frequency=train_frequency,
        epsilon=epsilon,
        train_batch_size=train_batch_size,
        validation_prob=validation_prob,
        validation_batch_size=validation_batch_size,
        env=env,
        train_buffer=train_buffer,
        validation_buffer=validation_buffer,
        train_model=train_model,
        random_sampler=random_sampler, deterministic_sampler=deterministic_sampler,
        train_function=train_function, validation_function=train_function,
        discretizer=discretizer, 
        discrete_dimensions=discrete_dimensions,
        optimizer=optimizer,
        name=name,
        save_folder=save_folder
    ))

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
            ('num_random', 0),
            ('num_softmax', 0),
            ('num_deterministic', 0),
            ('num_envs', 0),
            ('num_success', 0),
            ('average_success_ratio_per_env', 0),
            ('average_tries_per_env', 0),
            ('envs_with_success_ratio', 0)
        ))
        epoch_start_time = time.time()
        num_epoch += 1
        total_training_loss = 0.0
        num_train_steps = 0

        # run one epoch
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
            rand = np.random.uniform()
            if rand < epsilon or num_samples < min_samples_before_train: # epsilon greedy or initial samples
                obs, action, reward, _ = random_sampler()
            else:
                obs, action, reward, _ = deterministic_sampler()

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
                loss = train_function(train_model, data, optimizer)
                total_training_loss += loss.numpy()
                num_train_steps += 1

        # diagnostics stuff
        diagnostics['num_samples_total'] = num_samples
        diagnostics['num_training_samples'] = buffer.num_samples
        diagnostics['num_validation_samples'] = validation_buffer.num_samples
        diagnostics['total_time'] = time.time() - training_start_time
        diagnostics['time'] = time.time() - epoch_start_time
        
        diagnostics['average_training_loss'] = 'none' if num_train_steps == 0 else total_training_loss / num_train_steps

        if validation_buffer.num_samples >= validation_batch_size:
            datas = validation_buffer.get_all_samples_in_batch(validation_batch_size)
            total_validation_loss = 0.0
            for data in datas:
                total_validation_loss += validation_function(logits_model, data).numpy()
            diagnostics['validation_loss'] = total_validation_loss / len(datas)

        success_ratio = successes_this_env / num_samples_this_env
        total_success_ratio += success_ratio
        diagnostics['average_success_ratio_per_env'] = total_success_ratio / diagnostics['num_envs']
        diagnostics['average_tries_per_env'] = num_samples_per_epoch / diagnostics['num_envs']
        if successes_this_env > 0:
            num_envs_with_success += 1
        diagnostics['envs_with_success_ratio'] = num_envs_with_success / diagnostics['num_envs']

        print(f'Epoch {num_epoch}/{total_epochs}:')
        pprint(diagnostics)
        all_diagnostics.append(diagnostics)

    if name:
        os.makedirs(save_folder, exist_ok=True)
        new_folder = os.path.join(save_folder, name)
        os.makedirs(new_folder, exist_ok=True)
        train_buffer.save(new_folder, f"train_buffer")
        validation_buffer.save(new_folder, "validation_buffer")
        train_model.save_weights(os.path.join(new_folder, "model"))
        np.save(os.path.join(new_folder, "diagnostics"), all_diagnostics)