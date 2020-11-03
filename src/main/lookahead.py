#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Lookahead optimizer.

@author: __author__
@status: __status__
@license: __license__
'''

from tensorflow.python.keras import backend as K

class Lookahead(object):
    """Add the [Lookahead Optimizer](https://arxiv.org/abs/1907.08610) functionality for [keras](https://keras.io/).
    """

    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = 0

    def inject(self, model):
        """Inject the Lookahead algorithm for the given model.
        The following code is modified from the Keras _make_train_function method.
        See: https://github.com/keras-team/keras/blob/master/keras/engine/training.py#L497
        """
        if not hasattr(model, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')

        model._check_trainable_weights_consistency()
        if model.train_function is None:
            inputs = (model._feed_inputs +
                      model._feed_targets +
                      model._feed_sample_weights)
            inputs += [K.symbolic_learning_phase()]

            with K.get_graph().as_default():
                with K.name_scope('training'):
                    with K.name_scope(model.optimizer.__class__.__name__):
                        # training updates
                        fast_updates = model.optimizer.get_updates(
                            params=model._collected_trainable_weights, loss=model.total_loss)
                        fast_params = model._collected_trainable_weights
                        slow_params = fast_params

            # Unconditional updates
            fast_updates += model.get_updates_for(None)
            # Conditional updates relevant to this model
            fast_updates += model.get_updates_for(model.inputs)
            fast_updates += model.metrics[0].updates

            with K.name_scope('training'):
                slow_updates, copy_updates = [], []
                for p, q in zip(fast_params, slow_params):
                    slow_updates.append(K.update(q, q + self.alpha * (p - q)))
                    copy_updates.append(K.update(p, q))
                # Gets loss and metrics. Updates weights at each call.
                fast_train_function = K.function(
                    inputs, [model.total_loss] + model._compile_metrics_tensors['categorial_accuracy'],
                    updates=fast_updates,
                    name='fast_train_function',
                    **model._function_kwargs)
                def F(inputs):
                    self.count += 1
                    R = fast_train_function(inputs)
                    if self.count % self.k == 0:
                        K.batch_get_value(slow_updates)
                        K.batch_get_value(copy_updates)
                    return R
                model.train_function = F
