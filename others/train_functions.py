import tensorflow as tf 

from losses import *

@tf.function(experimental_relax_shapes=True)
def train_autoregressive_discrete_sigmoid(logits_model, data, optimizer, discrete_dimensions):
    observations = data['observations']
    rewards = tf.cast(data['rewards'], tf.float32)
    actions_discrete = data['actions']
    actions_onehot = [tf.one_hot(actions_discrete[:, i], depth=d) for i, d in enumerate(discrete_dimensions)]

    with tf.GradientTape() as tape:
        # get the logits for all the dimensions
        logits = logits_model([observations] + actions_onehot)
        loss = autoregressive_binary_cross_entropy_loss(logits, actions_onehot, rewards)

    grads = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, logits_model.trainable_variables))

    return loss

@tf.function(experimental_relax_shapes=True)
def validation_autoregressive_discrete_sigmoid(logits_model, data, discrete_dimension):
    observations = data['observations']
    rewards = tf.cast(data['rewards'], tf.float32)
    actions_discrete = data['actions']
    actions_onehot = [tf.one_hot(actions_discrete[:, i], depth=d) for i, d in enumerate(discrete_dimensions)]
    logits = logits_model([observations] + actions_onehot)
    loss = autoregressive_binary_cross_entropy_loss(logits, actions_onehot, rewards)
    return loss



@tf.function(experimental_relax_shapes=True)
def train_discrete_sigmoid(logits_model, data, optimizer, discrete_dimension):
    observations = data['observations']
    rewards = tf.cast(data['rewards'], tf.float32)
    actions_discrete = data['actions']
    actions_onehot = tf.one_hot(actions_discrete[:, 0], depth=discrete_dimension)

    with tf.GradientTape() as tape:
        logits = logits_model(observations)
        taken_logits = tf.reduce_sum(logits * actions_onehot, axis=-1, keepdims=True)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=rewards, logits=taken_logits)
        loss = tf.nn.compute_average_loss(losses)

    grads = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, logits_model.trainable_variables))

    return loss

@tf.function(experimental_relax_shapes=True)
def validation_discrete_sigmoid(logits_model, data, discrete_dimension):
    observations = data['observations']
    rewards = tf.cast(data['rewards'], tf.float32)
    actions_discrete = data['actions']
    actions_onehot = tf.one_hot(actions_discrete[:, 0], depth=discrete_dimension)
    logits = logits_model(observations)
    taken_logits = tf.reduce_sum(logits * actions_onehot, axis=-1, keepdims=True)
    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=rewards, logits=taken_logits)
    loss = tf.nn.compute_average_loss(losses)
    return loss



@tf.function(experimental_relax_shapes=True)
def train_autoregressive_discrete_softmax(logits_model, data, optimizer, discrete_dimensions):
    observations = data['observations']
    rewards = tf.cast(data['rewards'], tf.float32)
    actions_discrete = data['actions']
    actions_onehot = [tf.one_hot(actions_discrete[:, i], depth=d) for i, d in enumerate(discrete_dimensions)]
    actions_labeled = [rewards * a + (1. - rewards) * (1. - a) / (d - 1.) for a, d in zip(actions_onehot, discrete_dimensions)]

    with tf.GradientTape() as tape:
        # get the logits for all the dimensions
        logits = logits_model([observations] + actions_onehot)
        loss = autoregressive_softmax_cross_entropy_loss(logits, actions_labeled)

    grads = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, logits_model.trainable_variables))

    return loss

@tf.function(experimental_relax_shapes=True)
def validation_autoregressive_discrete_softmax(logits_model, data, discrete_dimensions):
    observations = data['observations']
    rewards = tf.cast(data['rewards'], tf.float32)
    actions_discrete = data['actions']
    actions_onehot = [tf.one_hot(actions_discrete[:, i], depth=d) for i, d in enumerate(discrete_dimensions)]
    actions_labeled = [rewards * a + (1. - rewards) * (1. - a) / (d - 1.) for a, d in zip(actions_onehot, discrete_dimensions)]
    logits = logits_model([observations] + actions_onehot)
    loss = autoregressive_softmax_cross_entropy_loss(logits, actions_labeled)
    return loss



@tf.function(experimental_relax_shapes=True)
def train_ddpg_Q_function(Q_model, data, optimizer):
    observations = data['observations']
    rewards = data['rewards']
    actions = data['actions']

    with tf.GradientTape() as tape:
        Q_values_logits = Q_model((observations, actions))
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=rewards, logits=Q_values_logits)
        loss = tf.nn.compute_average_loss(losses)

    grads = tape.gradient(loss, Q_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, Q_model.trainable_variables))

    return loss

@tf.function(experimental_relax_shapes=True)
def train_ddpg_policy(policy_model, Q_model, observations, optimizer):
    with tf.GradientTape() as tape:
        actions = policy_model(observations)
        Q_values_logits = Q_model((observations, actions))
        losses = -tf.math.sigmoid(Q_values_logits)
        loss = tf.nn.compute_average_loss(losses)

    grads = tape.gradient(loss, policy_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy_model.trainable_variables))

    return loss