import os
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# from softlearning.utils.keras import PicklableModel
from softlearning.models.convnet import convnet_model
from softlearning.models.feedforward import feedforward_model

CURR_PATH = os.path.dirname(os.path.abspath(__file__))

URDF = {
    "locobot": os.path.join(CURR_PATH, 'urdf/locobot_description.urdf'),
    "miniblock": os.path.join(CURR_PATH, 'urdf/miniblock.urdf'),
    "greenbox": os.path.join(CURR_PATH, 'urdf/greenbox.urdf'),
    "redbox": os.path.join(CURR_PATH, 'urdf/redbox.urdf'),
    "largerminiblock": os.path.join(CURR_PATH, 'urdf/largerminiblock.urdf'),
    "greenball": os.path.join(CURR_PATH, 'urdf/greenball.urdf'),
    "greensquareball": os.path.join(CURR_PATH, 'urdf/greensquareball_v2.urdf'),
    "walls": os.path.join(CURR_PATH, 'urdf/walls.urdf'),
    "plane": os.path.join(CURR_PATH, 'urdf/plane.urdf'),
    "rectangular_pillar": os.path.join(CURR_PATH, 'urdf/rectangular_pillar.urdf'),
    "solid_box": os.path.join(CURR_PATH, 'urdf/solid_box.urdf')
}

def load_convnet(path, image_size, **params):
    """ Load a RBG convnet model (conv layers + dense layers) from file.
        Args:
            path: absolute or relative path to the model
            image_size: size of the input image
            params: conv_filters, conv_kernel_sizes, conv_strides (from softlearning/models/convnet.py)
                    ff_layers, output_size, output_activation
                    relative_path: if True, then path is relative to the current locobot folder.
        Returns:
            a model
    """
    conv_filters = params.get("conv_filters", (64,64,64))
    conv_kernel_sizes = params.get("conv_kernel_sizes", (3,3,3))
    conv_strides = params.get("conv_strides", (2,2,2))
    ff_layers = params.get("ff_layers", (256,256))
    output_size = params.get("output_size", 1)
    output_activation = params.get("output_activation", 'sigmoid')

    input_pl = Input(shape=(image_size, image_size, 3), dtype=tf.uint8)
    conv = convnet_model(
            conv_filters=conv_filters,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides)(input_pl)
    ff = feedforward_model(ff_layers, output_size, output_activation=output_activation)(conv)
    model = Model(inputs=input_pl, outputs=ff)

    if params.get("relative_path", False):
        path = os.path.join(CURR_PATH, path)

    model.load_weights(path)
    return model

def is_in_rect(x, y, min_x, min_y, max_x, max_y):
    return min_x < x < max_x and min_y < y < max_y

def is_in_circle(x, y, center_x, center_y, radius):
    return (x - center_x) ** 2 + (y - center_y) ** 2 < radius ** 2