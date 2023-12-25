import keras
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf


class NoPreprocessingCatLayer(keras.layers.Layer):
    def __init__(self, type=None, **kwargs):
        super(NoPreprocessingCatLayer, self).__init__(**kwargs)

        self.type = type

    def build(self, input_shape):
        super(NoPreprocessingCatLayer, self).build(input_shape)

    def call(self, feature):
        feature = tf.cast(feature, dtype=self.feature_dtype)
        # feature = feature.map(self.value_mapping)
        if self.type is None:
            encoded_feature = np.expand_dims(feature, 1)
        elif self.type == "int":
            try:
                feature.shape[1]
                encoded_feature = feature + 1
            except IndexError:
                encoded_feature = np.expand_dims(feature, 1)

        elif self.type == "one_hot":
            feature = tf.squeeze(feature)
            print(feature.shape, feature.dtype, feature)
            encoded_feature = to_categorical(feature, num_classes=self.num_classes)

        return encoded_feature

    def adapt(self, feature):
        self.feature_dtype = feature.dtype
        self.num_classes = len(np.unique(feature))
        pass


class NoPreprocessingLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoPreprocessingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(NoPreprocessingLayer, self).build(input_shape)

    def call(self, feature):
        return np.array(feature)

    def adapt(self, feature):
        pass
