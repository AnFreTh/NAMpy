import tensorflow as tf


class MinMaxEncodingLayer(tf.keras.layers.Layer):
    """
    Custom Keras layer for min-max scaling of input data.

    This layer scales input values to the range [-1, 1] using min-max scaling.

    Args:
        min_value (float): Minimum value for scaling.
        max_value (float): Maximum value for scaling.

    Returns:
        tf.Tensor: Scaled tensor in the range [-1, 1].

    Example:
        min_max_layer = MinMaxEncodingLayer(min_value=0, max_value=1)
        scaled_data = min_max_layer(inputs)
    """

    def __init__(self, min_value=-1, max_value=2, **kwargs):
        super(MinMaxEncodingLayer, self).__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        # Apply min-max scaling to the range [-1, 1]
        encoded = 2 * (inputs - self.min_value) / (self.max_value - self.min_value) - 1
        return encoded

    def get_config(self):
        config = super(MinMaxEncodingLayer, self).get_config()
        config.update({"min_value": self.min_value, "max_value": self.max_value})
        return config
