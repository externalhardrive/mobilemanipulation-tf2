import tensorflow as tf
import tree

from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import create_inputs
from softlearning.utils.tensorflow import apply_preprocessors
from softlearning import preprocessors as preprocessors_lib
from softlearning.utils.tensorflow import cast_and_concat

from .base_value_function import StateActionValueFunction


def create_double_value_function(value_fn, *args, **kwargs):
    # TODO(hartikainen): The double Q-function should support the same
    # interface as the regular ones. Implement the double min-thing
    # as a Keras layer.
    value_fns = tuple(value_fn(*args, **kwargs) for i in range(2))
    return value_fns


def double_feedforward_Q_function(*args, **kwargs):
    return create_double_value_function(
        feedforward_Q_function, *args, **kwargs)


def feedforward_Q_function(input_shapes,
                           *args,
                           preprocessors=None,
                           observation_keys=None,
                           name='feedforward_Q',
                           **kwargs):
    inputs = create_inputs(input_shapes)

    # TODO(externalhardrive): Need to find a better way of handling unspecified preprocessors
    empty_preprocessors = tree.map_structure(lambda x: None, inputs)
    if preprocessors is not None:
        if isinstance(preprocessors, (tuple, list)) and len(preprocessors) == 2:
            observation_preprocessor = empty_preprocessors[0]
            if preprocessors[0] is not None:
                if isinstance(preprocessors[0], dict):
                    observation_preprocessor.update(preprocessors[0])
                else:
                    raise NotImplementedError("observation preprocessors can only be of type dict")
            action_preprocessors = empty_preprocessors[1]
            if preprocessors[1] is not None:
                action_preprocessors = preprocessors[1]
            preprocessors = (observation_preprocessor, action_preprocessors)
        else:
            raise NotImplementedError("preprocessors can only be of shape (observation preprocessors, action preprocessors)")
    else:
        preprocessors = empty_preprocessors

    preprocessors = tree.map_structure_up_to(
        inputs, preprocessors_lib.deserialize, preprocessors)

    preprocessed_inputs = apply_preprocessors(preprocessors, inputs)

    # NOTE(hartikainen): `feedforward_model` would do the `cast_and_concat`
    # step for us, but tf2.2 broke the sequential multi-input handling: See:
    # https://github.com/tensorflow/tensorflow/issues/37061.
    out = tf.keras.layers.Lambda(cast_and_concat)(preprocessed_inputs)
    Q_model_body = feedforward_model(
        *args,
        output_shape=[1],
        name=name,
        **kwargs
    )

    Q_model = tf.keras.Model(inputs, Q_model_body(out), name=name)

    Q_function = StateActionValueFunction(
        model=Q_model, observation_keys=observation_keys, name=name)

    return Q_function
