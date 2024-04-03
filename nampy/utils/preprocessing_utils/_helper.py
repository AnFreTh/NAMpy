import tensorflow as tf
import numpy as np
from keras.utils import to_categorical


class NoPreprocessingCatLayer(tf.keras.layers.Layer):
    def __init__(self, type=None, **kwargs):
        super(NoPreprocessingCatLayer, self).__init__(**kwargs)

        self.type = type

    def build(self, input_shape):
        super(NoPreprocessingCatLayer, self).build(input_shape)

    def call(self, feature):
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
            encoded_feature = to_categorical(feature)

        return encoded_feature

    def adapt(self, feature):
        pass


class NoPreprocessingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoPreprocessingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(NoPreprocessingLayer, self).build(input_shape)

    def call(self, feature):
        return np.array(feature)

    def adapt(self, feature):
        pass


class OneHotConstantBins(tf.keras.layers.Layer):
    def __init__(self, num_bins, **kwargs):
        super(OneHotConstantBins, self).__init__(**kwargs)
        self.num_bins = num_bins

    def build(self, input_shape):
        super(OneHotConstantBins, self).build(input_shape)

    def call(self, feature):
        layer = tf.keras.layers.Discretization(
            bin_boundaries=self.bin_edges,
            output_mode="one_hot",
        )

        return layer(feature)

    def adapt(self, feature):
        min = np.min(feature)

        max = np.max(feature)

        self.bin_edges = np.linspace(min, max, self.num_bins)
