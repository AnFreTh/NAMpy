import tensorflow as tf
import numpy as np
from keras.utils import to_categorical


class NoPreprocessingLayer(tf.keras.layers.Layer):
    def __init__(self, type=None, **kwargs):
        super(NoPreprocessingLayer, self).__init__(**kwargs)

        self.type = type

    def build(self, input_shape):
        super(NoPreprocessingLayer, self).build(input_shape)

    def call(self, feature):
        if self.type is None:
            encoded_feature = np.expand_dims(feature, 1)
        elif self.type == "int":
            encoded_feature = np.expand_dims(feature, 1) + 1
        elif self.type == "one_hot":
            encoded_feature = to_categorical(feature)

        return encoded_feature

    def adapt(self, feature):
        pass
