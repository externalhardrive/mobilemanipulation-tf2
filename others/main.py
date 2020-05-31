from discretizer import *
from dqn import *
from envs import *
from loss import *
from policies import *
from replay_buffer import *
from train import *

def autoregressive_discrete_dqn_grasping(args):
    image_size = 100
    discrete_dimensions = [15, 31]
    
    epsilon = 0.1
    train_batch_size = 200
    validation_prob = 0.1
    validation_batch_size = 100

    num_samples_per_env = 10
    num_samples_per_epoch = 100
    num_samples_total = 100000
    min_samples_before_train = 10
    train_frequency = 5
    assert num_samples_per_epoch % num_samples_per_env == 0 and num_samples_total % num_samples_per_epoch == 0
    
    # create the policy
    logits_model, samples_model, deterministic_model = (
        build_policy(image_size=image_size, 
                     discrete_dimensions=discrete_dimensions,
                     discrete_hidden_layers=[512, 512]))
    optimizer = tf.optimizers.Adam(learning_rate=1e-5)

    # testing time
    # logits_model.load_weights('./dataset/models/from_dataset_4/autoregressive')
    # epsilon = -1
    # min_samples_before_train = float('inf')
    # num_samples_total = 1
    
    # create the Discretizer
    discretizer = Discretizer(discrete_dimensions, [0.3, -0.16], [0.4666666, 0.16])

    # create the samplers


    # create the train and validation functions


    # create the dataset
    train_buffer = ReplayBuffer(size=num_samples_total, image_size=image_size, action_dim=len(discrete_dimensions))
    validation_buffer = ReplayBuffer(size=num_samples_total, image_size=image_size, action_dim=len(discrete_dimensions))

    # buffer.load('./dataset/data/autoregressive_4_replay_buffer.npy')

    # create the env
    env = GraspingEnv()

    training_loop(
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
        buffer=buffer,
        validation_buffer=validation_buffer,
        train_model=logits_model,
        discretizer=discretizer, 
        discrete_dimensions=discrete_dimensions,
        optimizer=optimizer,
        name='autoregressive_10'
    )

def discrete_dqn_grasping(args):
    image_size = 100
    # discrete_dimensions = [15, 31]
    discrete_dimensions = [15 * 31]
    
    epsilon = 0.1
    train_batch_size = 200
    validation_prob = 0.1
    validation_batch_size = 100

    num_samples_per_env = 10
    num_samples_per_epoch = 100
    num_samples_total = 100000
    min_samples_before_train = 10
    train_frequency = 5
    assert num_samples_per_epoch % num_samples_per_env == 0 and num_samples_total % num_samples_per_epoch == 0
    
    # create the policy
    # logits_model, samples_model, deterministic_model = (
    #     build_policy(image_size=image_size, 
    #                  discrete_dimensions=discrete_dimensions,
    #                  discrete_hidden_layers=[512, 512]))
    logits_model, samples_model, deterministic_model = (
        build_discrete_policy(image_size=image_size, 
                              discrete_dimension=discrete_dimensions[0],
                              discrete_hidden_layers=[512, 512]))
    optimizer = tf.optimizers.Adam(learning_rate=1e-5)

    # testing time
    # logits_model.load_weights('./dataset/models/from_dataset_4/autoregressive')
    # epsilon = -1
    # min_samples_before_train = float('inf')
    # num_samples_total = 1
    
    # create the Discretizer
    # discretizer = Discretizer(discrete_dimensions, [0.3, -0.16], [0.4666666, 0.16])
    discretizer = Discretizer([15, 31], [0.3, -0.16], [0.4666666, 0.16])

    # create the dataset
    buffer = ReplayBuffer(size=num_samples_total, image_size=image_size, action_dim=len(discrete_dimensions))
    validation_buffer = ReplayBuffer(size=num_samples_total, image_size=image_size, action_dim=len(discrete_dimensions))

    # buffer.load('./dataset/data/autoregressive_4_replay_buffer.npy')

    # create the env
    env = create_env()

    training_loop(
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
        buffer=buffer,
        validation_buffer=validation_buffer,
        logits_model=logits_model, samples_model=samples_model, deterministic_model=deterministic_model,
        discretizer=discretizer, 
        discrete_dimensions=discrete_dimensions,
        optimizer=optimizer,
        name='autoregressive_10'
    )

def main(args):
    # autoregressive_discrete_dqn_grasping(args)
    discrete_dqn_grasping(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)