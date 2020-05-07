from copy import deepcopy
from collections import OrderedDict

import pprint

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .rl_algorithm import RLAlgorithm

class R3L(RLAlgorithm):
    """ Loosely based off of R3L 
        https://openreview.net/pdf?id=rJe2syrtvS
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            forward_sac,
            perturbation_sac,
            rnd_predictor,
            rnd_target,
            plotter=None,

            rnd_lr=3-4,
            intrinsic_scale=1.0,
            extrinsic_scale=1.0,

            save_full_state=False,
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
        """

        super().__init__(**kwargs)

        print()
        print("R3L params:")
        pprint.pprint(dict(
            self=self,
            training_environment=training_environment,
            evaluation_environment=evaluation_environment,
            forward_sac=forward_sac,
            perturbation_sac=perturbation_sac,
            rnd_predictor=rnd_predictor,
            rnd_target=rnd_target,
            plotter=plotter,

            rnd_lr=rnd_lr,
            intrinsic_scale=intrinsic_scale,
            extrinsic_scale=extrinsic_scale,

            save_full_state=save_full_state,
            kwargs=kwargs,
        ))
        print()

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment

        self._forward_sac = forward_sac
        self._perturbation_sac = perturbation_sac

        self._rnd_predictor = rnd_predictor
        self._rnd_target = rnd_target

        self._plotter = plotter

        self._rnd_lr = rnd_lr
        self._intrinsic_scale = intrinsic_scale
        self._extrinsic_scale = extrinsic_scale

        self._save_full_state = save_full_state

        self._rnd_predictor_optimizer = tf.optimizers.Adam(
            learning_rate=self._rnd_lr,
            name="rnd_predictor_optimizer")

        self._rnd_running_mean = np.zeros((), np.float64)
        self._rnd_running_var = np.ones((), np.float64)
        self._rnd_running_count = 1e-4

        # start with perturbation policy because the start of first epoch will switch policy
        self._current_forward_policy = False
        self._policy = self._perturbation_sac._policy

    def _epoch_before_hook(self):
        super()._epoch_before_hook()

        # switch between forward and perturbation policy
        if self._current_forward_policy:
            self.sampler.switch_policy(self._perturbation_sac._policy)
            self._current_forward_policy = False 
        else:
            self.sampler.switch_policy(self._forward_sac._policy)
            self._current_forward_policy = True 

    @property
    def _evaluation_policy(self):
        return self._forward_sac._policy

    @tf.function(experimental_relax_shapes=True)
    def _update_rnd_predictor(self, batch):
        """Update the RND predictor network. """
        observations = batch['observations']

        with tf.GradientTape() as tape:
            predictor_values = self._rnd_predictor.values(observations)
            target_values = self._rnd_target.values(observations)

            predictor_losses = tf.losses.MSE(y_true=target_values, y_pred=predictor_values)
            predictor_loss = tf.nn.compute_average_loss(predictor_losses)

        predictor_gradients = tape.gradient(predictor_loss, self._rnd_predictor.trainable_variables)
        
        self._rnd_predictor_optimizer.apply_gradients(zip(predictor_gradients, self._rnd_predictor.trainable_variables))

        return predictor_losses

    def _update_running_mean_and_var(self, batch_rewards):
        """ Modified Welford's algorithm from the original RND paper
            https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/mpi_util.py#L200-L214
        """
        batch_mean = np.mean(batch_rewards)
        batch_var = np.var(batch_rewards)
        batch_count = batch_rewards.shape[0]
        
        delta = batch_mean - self._rnd_running_mean
        total_count = self._rnd_running_count + batch_count

        new_mean = self._rnd_running_mean + delta * batch_count / total_count
        m_a = self._rnd_running_var * self._rnd_running_count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self._rnd_running_count * batch_count / total_count
        new_var = m_2 / total_count

        self._rnd_running_mean = new_mean
        self._rnd_running_var = new_var
        self._rnd_running_count = total_count

    def _do_training(self, iteration, batch):
        # update RND predictor loss
        predictor_losses = self._update_rnd_predictor(batch)

        # compute intrinsic reward for this batch
        intrinsic_rewards = predictor_losses.numpy().reshape(-1, 1)
        self._update_running_mean_and_var(intrinsic_rewards)
        intrinsic_rewards = intrinsic_rewards / np.sqrt(self._rnd_running_var)

        # update the current policy we are using
        if self._current_forward_policy:
            batch['rewards'] = self._extrinsic_scale * batch['rewards'] + self._intrinsic_scale * intrinsic_rewards
            sac_diagnostics = self._forward_sac._do_training(iteration, batch)
        else:
            batch['rewards'] = intrinsic_rewards
            sac_diagnostics = self._perturbation_sac._do_training(iteration, batch)

        diagnostics = OrderedDict({
            **sac_diagnostics,
            'rnd_predictor_loss-mean': tf.reduce_mean(predictor_losses),
            'rnd_running_var': self._rnd_running_var,
            'rnd_intrinsic_reward-mean': np.mean(intrinsic_rewards),
            'rnd_intrinsic_reward-min': np.min(intrinsic_rewards),
            'rnd_intrinsic_reward-max': np.max(intrinsic_rewards),
        })

        return diagnostics

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as an ordered dictionary.

        Also calls the `draw` method of the plotter, if plotter defined.
        """
        # TODO(externalhardrive): better diagnostics
        diagnostics = OrderedDict((
            ('forward_sac', self._forward_sac.get_diagnostics(iteration, batch, training_paths, evaluation_paths)),
            ('perturbation_sac', self._perturbation_sac.get_diagnostics(iteration, batch, training_paths, evaluation_paths)),
            ('current_policy', 'forward' if self._current_forward_policy else 'perturbation'),
        ))

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_rnd_predictor_optimizer': self._rnd_predictor_optimizer,
        }

        return saveables