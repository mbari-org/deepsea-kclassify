import tensorflow as tf

def focal_loss(gamma=2., alpha=4.):
    """Focal loss for multi-classification.
        # Arguments
        # References
            - [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
        """
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        '''
        Focal loss for multi-classification
        y_true: tensor of ground truth labels, shape [batch size, number of classes]
        y_pred: tensor of model output, shape [batch size, number of classes]
        gamma: float, 0 < gamma < 1.
        alpha: float, 0 < alpha < 1
        :return: loss
        '''
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed
