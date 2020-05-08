#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Surgery code to freeze, slim by dropping layers, etc.

@author: __author__
@status: __status__
@license: __license__
'''

from tfkerassurgeon import Surgeon
import tensorflow as tf
import conf as model_conf


class TransferModel:

    def __init__(self, base_model_name):
        self.base_model_name = base_model_name
        if base_model_name not in model_conf.MODEL_DICT.keys():
            raise ('{} not in {}'.format(base_model_name, model_conf.MODEL_DICT.keys()))
        return

    def build(self, class_size, l2_weight_decay_alpha=0.):
        """
        Build base model from the pre-trained model
        :param class_size: number of classes in Dense layer
        :param l2_weight_decay_alpha:
        :return: a Keras network model
        """
        cfg = eval("model_conf.MODEL_DICT['{}']".format(self.base_model_name))
        image_size = cfg['image_size']
        IMG_SHAPE = (image_size, image_size, 3)
        model_instance = cfg['model_instance']

        base_model = eval("{}(input_shape={},include_top=False,weights='imagenet')".format(model_instance, IMG_SHAPE))

        if l2_weight_decay_alpha > 0.:
            if cfg['has_depthwise_layers']:
                for layer in base_model.layers:
                    if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                        layer.add_loss(tf.keras.regularizers.l2(l2_weight_decay_alpha)(l.depthwise_kernel))
                    elif isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                        layer.add_loss(tf.keras.regularizers.l2(l2_weight_decay_alpha)(layer.kernel))
                    if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                        layer.add_loss(tf.keras.regularizers.l2(l2_weight_decay_alpha)(layer.bias))
            else:
                for layer in base_model.layers:
                    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                        layer.add_loss(tf.keras.regularizers.l2(l2_weight_decay_alpha)(layer.kernel))
                    if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                        layer.add_loss(tf.keras.regularizers.l2(l2_weight_decay_alpha)(layer.bias))

        # Freezing (or setting layer.trainable = False) prevents weights in these layers
        # from being updated during training.
        base_model.trainable = False

        fine_tune_at = cfg['fine_tune_at']

        # if fine tune at defined, freeze all the layers before the `fine_tune_at` layer
        if fine_tune_at > 0:
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False

        # if slimming the model, remove every other filter from conv2d layers - not this does not work
        # for all models.
        if cfg['slim']:
            surgeon = Surgeon(base_model)
            for layer in base_model.layers:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    weight = layer.get_weights()[0]
                    num_filters = len(weight[0, 0, 0, :])
                    surgeon.add_job('delete_channels', layer, channels=range(0, num_filters, 2))
            base_model = surgeon.operate()

        model_sequential = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(class_size, activation='softmax')
        ])

        return model_sequential, image_size, fine_tune_at


if __name__ == '__main__':
    mmaker = TransferModel()
    # build the basic model
    model = mmaker.build()
    model.summary()
