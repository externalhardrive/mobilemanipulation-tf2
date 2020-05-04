from copy import deepcopy
from collections import OrderedDict

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
            learning_rate=self._policy_lr,
            name="rnd_predictor_optimizer")

        self._rnd_running_mean = 0.0
        self._rnd_running_variance = 1.0

        self._current_forward_policy = False

    def _epoch_before_hook(self):
        super()._epoch_before_hook()

        # switch between forward and perturbation policy
        if self._current_forward_policy:
            self.sampler.switch_policy(self._perturbation_sac.policy)
            self._current_forward_policy = False 
        else:
            self.sampler.switch_policy(self._forward_sac.policy)
            self._current_forward_policy = True 

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

    def _do_training(self, iteration, batch):
        # update RND predictor loss
        predictor_losses = self._update_rnd_predictor(batch)

        # compute intrinsic loss for this batch
        intrinsic_rewards = ...

        # update the current policy we are using
        if self._current_forward_policy: #something
            batch['rewards'] = self._extrinsic_scale * batch['rewards'] + self._intrinsic_scale * intrinsic_rewards
            sac_diagnostics = self._forward_sac._do_training(iteration, batch)
        else:
            batch['rewards'] = intrinsic_rewards
            sac_diagnostics = self._perturbation_sac._do_training(iteration, batch)

        diagnostics = OrderedDict({
            **sac_diagnostics,
            'rnd_predictor_loss-mean': tf.reduce_mean(predictor_losses)
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