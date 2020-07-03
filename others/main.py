import os 
import argparse

import numpy as np
import tensorflow as tf

from discretizer import *
from envs import *
from losses import *
from policies import *
from samplers import *
from replay_buffer import *
from train_functions import *
from training_loop import *
from datetime import datetime


# def autoregressive_discrete_dqn_grasping(args):
#     # some hyper parameters
#     image_size = 100
#     discrete_dimensions = [15, 31]

#     num_samples_per_env = 10
#     num_samples_per_epoch = 100
#     num_samples_total = int(1e5)
#     min_samples_before_train = 1000

#     train_frequency = 5
#     train_batch_size = 200
#     validation_prob = 0.1
#     validation_batch_size = 100
    
    
#     # create the policy
#     logits_model, samples_model, deterministic_model = build_image_autoregressive_policy(
#         image_size=image_size, 
#         discrete_dimensions=discrete_dimensions,
#         discrete_hidden_layers=[512, 512]
#     )

#     # create the optimizer
#     optimizer = tf.optimizers.Adam(learning_rate=1e-5)
    
#     # create the Discretizer
#     discretizer = Discretizer(discrete_dimensions, [0.3, -0.16], [0.4666666, 0.16])

#     # create the env
#     env = GraspingEnv()

#     # create the sampler
#     sampler = create_grasping_env_autoregressive_discrete_sampler(
#         env=env,
#         discretizer=discretizer,
#         deterministic_model=deterministic_model,
#         min_samples_before_train=min_samples_before_train,
#         epsilon=0.1,
#     )

#     # create the train and validation functions
#     train_function = lambda data: train_autoregressive_discrete_sigmoid(logits_model, data, optimizer, discrete_dimensions)
#     validation_function = lambda data: validation_autoregressive_discrete_sigmoid(logits_model, data, discrete_dimensions)

#     # create the dataset
#     train_buffer = ReplayBuffer(size=int(1e5), observation_shape=(image_size, image_size, 3), action_dim=len(discrete_dimensions))
#     validation_buffer = ReplayBuffer(size=int(1e5), observation_shape=(image_size, image_size, 3), action_dim=len(discrete_dimensions))

#     # testing time
#     # logits_model.load_weights('./dataset/models/from_dataset_4/autoregressive')
#     # epsilon = -1
#     # min_samples_before_train = float('inf')
#     # num_samples_total = 1

#     all_diagnostics = training_loop(
#         num_samples_per_env=num_samples_per_env,
#         num_samples_per_epoch=num_samples_per_epoch,
#         num_samples_total=num_samples_total,
#         min_samples_before_train=min_samples_before_train,
#         train_frequency=train_frequency,
#         train_batch_size=train_batch_size,
#         validation_prob=validation_prob,
#         validation_batch_size=validation_batch_size,
#         env=env,
#         sampler=sampler,
#         train_buffer=train_buffer, validation_buffer=validation_buffer,
#         train_function=train_function, validation_function=validation_function,
#     )

#     save_folder = './others/logs/'
#     name = 'autoregressive_10'

#     if name:
#         os.makedirs(save_folder, exist_ok=True)
#         new_folder = os.path.join(save_folder, name)
#         os.makedirs(new_folder, exist_ok=True)
#         train_buffer.save(new_folder, "train_buffer")
#         validation_buffer.save(new_folder, "validation_buffer")
#         logits_model.save_weights(os.path.join(new_folder, "model"))
#         np.save(os.path.join(new_folder, "diagnostics"), all_diagnostics)

def discrete_dqn_grasping(args):
    name = 'disc_dqn_'+ args.name
    
    # some hyperparameters
    image_size = 60
    
    if args.use_theta:
        discrete_dimensions = [15, 31, 4]
        discrete_dimension = 15 * 31*4
        # create the Discretizer
        discretizer = Discretizer(discrete_dimensions, [0.3, -0.16, 0], [0.4666666, 0.16, 3.14])
    else:
        discrete_dimensions = [15, 31]
        discrete_dimension = 15 * 31
        # create the Discretizer
        discretizer = Discretizer(discrete_dimensions, [0.3, -0.16], [0.4666666, 0.16])

    num_samples_per_env = 10
    num_samples_per_epoch = 200
    num_samples_total = int(1e4)
    min_samples_before_train = 1000
    num_eval_samples_per_epoch = 20
    train_frequency = 1
    train_batch_size = 128
    validation_prob = 0.1
    validation_batch_size = 16
    
    if True: #args.policy == 'discrete':
        # create the policy
        logits_model, deterministic_model = build_image_discrete_policy(
            image_size=image_size, 
            discrete_dimension=discrete_dimension,
            discrete_hidden_layers=[512, 512]
        )
#     elif args.policy == 'auto':
#         # create the policy
#         logits_model, samples_model, deterministic_model = build_image_autoregressive_policy(
#             image_size=image_size, 
#             discrete_dimensions=discrete_dimensions,
#             discrete_hidden_layers=[512, 512]
#         )

    # create the optimizer
    optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    


    # create the env
    env = GraspingEnv()
    #env = GraspingEnv(robot_pos=np.array([1., 1.]))
    env.reset()
    # create the sampler
    sampler = create_grasping_env_discrete_sampler(
        env=env,
        discretizer=discretizer,
        deterministic_model=deterministic_model,
        min_samples_before_train=min_samples_before_train,
        epsilon=0.1,
    )
    # create the sampler
    eval_sampler = create_grasping_env_discrete_sampler(
        env=env,
        discretizer=discretizer,
        deterministic_model=deterministic_model,
        min_samples_before_train=0,
        epsilon=0,
    )


    # create the train and validation functions
    train_function = lambda data: train_discrete_sigmoid(logits_model, data, optimizer, discrete_dimension)
    validation_function = lambda data: validation_discrete_sigmoid(logits_model, data, discrete_dimension)

    # create the dataset
    train_buffer = ReplayBuffer(size=num_samples_total+5000, observation_shape=(image_size, image_size, 3), action_dim=1)
    validation_buffer = ReplayBuffer(size=num_samples_total, observation_shape=(image_size, image_size, 3), action_dim=1)

    if args.load_data:
        data = np.load('2k_positives.npy',allow_pickle=True).item()
        for i in range(len(data['observations'])):
            train_buffer.store_sample(data['observations'][i], data['actions'][i], data['rewards'][i])
        data = np.load('2k_negatives.npy',allow_pickle=True).item()
        for i in range(len(data['observations'])):
            train_buffer.store_sample(data['observations'][i], data['actions'][i], data['rewards'][i])
    # testing time
    # logits_model.load_weights('./dataset/models/from_dataset_4/autoregressive')
    # epsilon = -1
    # min_samples_before_train = float('inf')
    # num_samples_total = 1
    now = datetime.now()
    savedir='./others/logs/'+name+now.strftime("%m%d%Y-%H-%M-%S")
    print("saving to ", savedir)
    all_diagnostics, train_buffer, validation_buffer = training_loop(
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
        eval_sampler=eval_sampler,
        num_eval_samples_per_epoch=num_eval_samples_per_epoch,
        train_buffer=train_buffer, validation_buffer=validation_buffer,
        train_function=train_function, validation_function=validation_function,
        savedir=savedir,
        pretrain=args.pretrain
    )

    save_folder = './others/logs/'


    if name:
        os.makedirs(save_folder, exist_ok=True)
        new_folder = os.path.join(save_folder, name)
        os.makedirs(new_folder, exist_ok=True)
        train_buffer.save(new_folder, "train_buffer")
        validation_buffer.save(new_folder, "validation_buffer")
        logits_model.save_weights(os.path.join(new_folder, "model"))
        np.save(os.path.join(new_folder, "diagnostics"), all_diagnostics)

def discrete_FCdqn_grasping(args):
    name = 'disc_FCdqn_'+ args.name
    
    # some hyperparameters
    image_size = 60

    num_samples_per_env = 10
    num_samples_per_epoch = 200
    num_samples_total = int(1e4)
    min_samples_before_train = 1000
    num_eval_samples_per_epoch = 20
    train_frequency = 1
    train_batch_size = 128
    validation_prob = 0.1
    validation_batch_size = 16
    
    if True: #args.policy == 'discrete':
        # create the policy
        logits_model, deterministic_model = build_fc_image_discrete_policy(
            image_size=image_size, 
            #discrete_dimension=discrete_dimension,
            #discrete_hidden_layers=[512, 512]
        )
#     elif args.policy == 'auto':
#         # create the policy
#         logits_model, samples_model, deterministic_model = build_image_autoregressive_policy(
#             image_size=image_size, 
#             discrete_dimensions=discrete_dimensions,
#             discrete_hidden_layers=[512, 512]
#         )

    # create the optimizer
    optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    
    if args.use_theta:
        discrete_dimensions = [image_size, image_size, 4]
        discrete_dimension = image_size *image_size31*4
        # create the Discretizer
        discretizer = Discretizer(discrete_dimensions, [0.3, -0.16, 0], [0.4666666, 0.16, 3.14])
    else:
        discrete_dimensions = [image_size//2, image_size//2]
        discrete_dimension = image_size * image_size // 4
        # create the Discretizer
        discretizer = Discretizer(discrete_dimensions, [0.3, -0.16], [0.4666666, 0.16])


    # create the env
    env = FullyConvGraspingEnv()
    #env = GraspingEnv(robot_pos=np.array([1., 1.]))
    env.reset()
    # create the sampler
    sampler = create_fc_grasping_env_discrete_sampler(
        env=env,
        discretizer=discretizer,
        deterministic_model=deterministic_model,
        min_samples_before_train=min_samples_before_train,
        epsilon=0.1,
    )
    # create the sampler
    eval_sampler = create_fc_grasping_env_discrete_sampler(
        env=env,
        discretizer=discretizer,
        deterministic_model=deterministic_model,
        min_samples_before_train=0,
        epsilon=0,
    )


    # create the train and validation functions
    train_function = lambda data: train_discrete_sigmoid(logits_model, data, optimizer, discrete_dimension)
    validation_function = lambda data: validation_discrete_sigmoid(logits_model, data, discrete_dimension)

    # create the dataset
    train_buffer = ReplayBuffer(size=num_samples_total+5000, observation_shape=(image_size, image_size, 3), action_dim=1)
    validation_buffer = ReplayBuffer(size=num_samples_total, observation_shape=(image_size, image_size, 3), action_dim=1)

    if args.load_data:
        data = np.load('2k_positives.npy',allow_pickle=True).item()
        for i in range(len(data['observations'])):
            train_buffer.store_sample(data['observations'][i], data['actions'][i], data['rewards'][i])
        data = np.load('2k_negatives.npy',allow_pickle=True).item()
        for i in range(len(data['observations'])):
            train_buffer.store_sample(data['observations'][i], data['actions'][i], data['rewards'][i])
    # testing time
    # logits_model.load_weights('./dataset/models/from_dataset_4/autoregressive')
    # epsilon = -1
    # min_samples_before_train = float('inf')
    # num_samples_total = 1
    now = datetime.now()
    savedir='./others/logs/'+name+now.strftime("%m%d%Y-%H-%M-%S")
    print("saving to ", savedir)
    all_diagnostics, train_buffer, validation_buffer = training_loop(
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
        eval_sampler=eval_sampler,
        num_eval_samples_per_epoch=num_eval_samples_per_epoch,
        train_buffer=train_buffer, validation_buffer=validation_buffer,
        train_function=train_function, validation_function=validation_function,
        savedir=savedir,
        pretrain=args.pretrain
    )

    save_folder = './others/logs/'


    if name:
        os.makedirs(save_folder, exist_ok=True)
        new_folder = os.path.join(save_folder, name)
        os.makedirs(new_folder, exist_ok=True)
        train_buffer.save(new_folder, "train_buffer")
        validation_buffer.save(new_folder, "validation_buffer")
        logits_model.save_weights(os.path.join(new_folder, "model"))
        np.save(os.path.join(new_folder, "diagnostics"), all_diagnostics)

        
# def ddpg_grasping(args):
#     # some hyperparameters
#     image_size = 100
#     action_dim = 2

#     num_samples_per_env = 10
#     num_samples_per_epoch = 100
#     num_samples_total = int(1e5)
#     min_samples_before_train = 200

#     train_frequency = 1
#     train_batch_size = 200
    
#     # create the policy and Q function
#     policy_model, unsquashed_model = build_image_deterministic_continuous_policy(
#         image_size=image_size,
#         action_dim=action_dim,
#         feedforward_hidden_layers=(256, 256),
#     )

#     Q_model = build_image_continuous_Q_function(
#         image_size=image_size,
#         action_dim=action_dim,
#         feedforward_hidden_layers=(256, 256),
#     )

#     # create the optimizer
#     policy_optimizer = tf.optimizers.Adam(learning_rate=1e-5)
#     Q_optimizer = tf.optimizers.Adam(learning_rate=1e-5)
    
#     # create the env
#     env = GraspingEnv()

#     # create the sampler
#     sampler = create_grasping_env_ddpg_sampler(
#         env=env,
#         policy_model=policy_model,
#         unsquashed_model=unsquashed_model,
#         action_dim=action_dim,
#         min_samples_before_train=min_samples_before_train + 800,
#         num_samples_at_end=50000,
#         noise_std_start=0.5,
#         noise_std_end=0.005,
#     )

#     # create the train function
#     num_train_steps = 0
#     def train_function(data): 
#         nonlocal num_train_steps
#         num_train_steps += 1
#         Q_loss = train_ddpg_Q_function(Q_model, data, Q_optimizer)
#         if num_train_steps >= 800:
#             policy_loss = train_ddpg_policy(policy_model, Q_model, data['observations'], policy_optimizer)
#         else:
#             policy_loss = tf.constant(0.0)
#         return Q_loss, policy_loss

#     # create the dataset
#     train_buffer = ReplayBuffer(size=num_samples_total, observation_shape=(image_size, image_size, 3), action_dim=action_dim, action_dtype=np.float32)

#     # run the training
#     all_diagnostics, train_buffer, validation_buffer = training_loop(
#         num_samples_per_env=num_samples_per_env,
#         num_samples_per_epoch=num_samples_per_epoch,
#         num_samples_total=num_samples_total,
#         min_samples_before_train=min_samples_before_train,
#         train_frequency=train_frequency,
#         train_batch_size=train_batch_size,
#         validation_prob=None,
#         validation_batch_size=None,
#         env=env,
#         sampler=sampler,
#         train_buffer=train_buffer, validation_buffer=None,
#         train_function=train_function, validation_function=None,
#     )

#     # save
#     save_folder = './others/logs/'
#     name = 'ddpg_1'

#     if name:
#         os.makedirs(save_folder, exist_ok=True)
#         new_folder = os.path.join(save_folder, name)
#         os.makedirs(new_folder, exist_ok=True)
#         train_buffer.save(new_folder, "train_buffer")
#         policy_model.save_weights(os.path.join(new_folder, "policy_model"))
#         Q_model.save_weights(os.path.join(new_folder, "Q_model"))
#         np.save(os.path.join(new_folder, "diagnostics"), all_diagnostics)

# def discrete_fake_grasping(args):
#     # some hyperparameters
#     line_width = 32
#     discrete_dimension = 32

#     num_samples_per_env = 10
#     num_samples_per_epoch = 1000
#     num_samples_total = int(1e5)
#     min_samples_before_train = 1000

#     train_frequency = 1
#     train_batch_size = 200
#     validation_prob = 0.1
#     validation_batch_size = 100
    
#     # create the policy
#     logits_model, _, deterministic_model = build_discrete_policy(
#         input_size=line_width, 
#         discrete_dimension=discrete_dimension,
#         discrete_hidden_layers=[512, 512]
#     )

#     # create the optimizer
#     optimizer = tf.optimizers.Adam(learning_rate=3e-4)
    
#     # create the env
#     env = FakeGraspingDiscreteEnv()

#     # create the sampler
#     sampler = create_fake_grasping_discrete_sampler(
#         env=env,
#         discrete_dimension=discrete_dimension,
#         deterministic_model=deterministic_model,
#         min_samples_before_train=min_samples_before_train,
#         epsilon=0.1,
#     )

#     # create the train and validation functions
#     train_function = lambda data: train_discrete_sigmoid(logits_model, data, optimizer, discrete_dimension)
#     validation_function = lambda data: validation_discrete_sigmoid(logits_model, data, discrete_dimension)

#     # create the dataset
#     train_buffer = ReplayBuffer(size=num_samples_total, observation_shape=(line_width,), action_dim=1, observation_dtype=np.float32)
#     validation_buffer = ReplayBuffer(size=num_samples_total, observation_shape=(line_width,), action_dim=1, observation_dtype=np.float32)

#     # testing time
#     # logits_model.load_weights('./dataset/models/from_dataset_4/autoregressive')
#     # epsilon = -1
#     # min_samples_before_train = float('inf')
#     # num_samples_total = 1

#     all_diagnostics, train_buffer, validation_buffer = training_loop(
#         num_samples_per_env=num_samples_per_env,
#         num_samples_per_epoch=num_samples_per_epoch,
#         num_samples_total=num_samples_total,
#         min_samples_before_train=min_samples_before_train,
#         train_frequency=train_frequency,
#         train_batch_size=train_batch_size,
#         validation_prob=validation_prob,
#         validation_batch_size=validation_batch_size,
#         env=env,
#         sampler=sampler,
#         train_buffer=train_buffer, validation_buffer=validation_buffer,
#         train_function=train_function, validation_function=validation_function,
#     )

#     save_folder = './others/logs/'
#     name = 'fake_grasping_1'

#     if name:
#         os.makedirs(save_folder, exist_ok=True)
#         new_folder = os.path.join(save_folder, name)
#         os.makedirs(new_folder, exist_ok=True)
#         train_buffer.save(new_folder, "train_buffer")
#         validation_buffer.save(new_folder, "validation_buffer")
#         logits_model.save_weights(os.path.join(new_folder, "model"))
#         np.save(os.path.join(new_folder, "diagnostics"), all_diagnostics)

def main(args):
    # autoregressive_discrete_dqn_grasping(args)
    #
    if args.policy == 'discrete':
        discrete_dqn_grasping(args)
    elif args.policy == 'fc':
        discrete_FCdqn_grasping(args)
    # ddpg_grasping(args)
    # discrete_fake_grasping(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="name of experiment", type=str, default='')
    parser.add_argument("--load_data", help="whether to preload data", default=False,action="store_true")
    parser.add_argument("--use_theta", help="whether to use a grasp angle", default=False,action="store_true")
    parser.add_argument("--policy", help="name of policy", type=str, default='discrete')

    parser.add_argument("--pretrain", help="number of steps to pretrain for", type=int, default=0)

    args = parser.parse_args()
    main(args)