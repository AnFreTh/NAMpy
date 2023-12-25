from keras.layers import Layer
import keras
import tensorflow as tf
from xDL.shapefuncs.helper_nets.helper_funcs import (
    get_random_features_initializer,
    get_default_scale,
)


class AddWeightsLayer(keras.layers.Layer):
    def __init__(self, weight_value=1e-25, **kwargs):
        """
        Initialize the AddWeightsLayer. -> Layer that is added as mask to spline inputs

        Args:
            weight_value (float): The initial value of trainable weights (default is 1e-16).
            **kwargs: Additional keyword arguments for the base class constructor.
        """
        super(AddWeightsLayer, self).__init__(**kwargs)
        self.weight_value = weight_value

    def build(self, input_shape):
        """
        Build the layer by creating trainable weights.

        Args:
            input_shape (tuple): The shape of the input tensor.

        Returns:
            None
        """
        # Create trainable weights with the same shape as the input
        self.pseudo_knots_locations = self.add_weight(
            shape=(input_shape[-1],),
            initializer=tf.constant_initializer(self.weight_value),
            trainable=True,
            name="pseudo_knots_locations",
        )
        super(AddWeightsLayer, self).build(input_shape)

    def call(self, inputs):
        """
        Add the trainable weights to each row except the first and last columns of the input.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The modified input tensor with weights added.
        """
        input_shape = tf.shape(inputs)
        weights = (
            tf.ones((input_shape[0], input_shape[1])) * self.pseudo_knots_locations
        )

        return inputs + weights


class InterceptLayer(Layer):
    """
    Custom Keras layer to add an intercept (bias) term to the input tensor.

    Args:
        **kwargs: Additional keyword arguments to pass to the Layer constructor.

    Example:
        # Create a model with an InterceptLayer
        model = keras.Sequential([
            InterceptLayer(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
    """

    def __init__(self, **kwargs):
        super(InterceptLayer, self).__init__(**kwargs)
        self.intercept = self.add_weight(
            name="bias", initializer="zeros", trainable=True
        )

    def call(self, inputs):
        return inputs + self.intercept


class IdentityLayer(Layer):
    """
    Custom Keras layer to apply various activation functions to the input tensor.

    Args:
        activation (str): Name of the activation function to apply.
        **kwargs: Additional keyword arguments to pass to the Layer constructor.

    Raises:
        ValueError: If an unsupported activation function name is provided.

    Example:
        # Create a model with an IdentityLayer using a specific activation
        model = keras.Sequential([
            IdentityLayer(activation='relu'),
            keras.layers.Dense(64),
            IdentityLayer(activation='softmax')
        ])
    """

    def __init__(self, activation, **kwargs):
        super(IdentityLayer, self).__init__(**kwargs)
        self.activation = activation

    def call(self, inputs):
        if self.activation == "linear":
            return inputs
        elif self.activation == "relu":
            return keras.activations.relu(inputs)
        elif self.activation == "sigmoid":
            return keras.activations.sigmoid(inputs)
        elif self.activation == "softmax":
            return keras.activations.softmax(inputs)
        elif self.activation == "softplus":
            return keras.activations.softplus(inputs)
        elif self.activation == "softsign":
            return keras.activations.softsign(inputs)
        elif self.activation == "tanh":
            return keras.activations.tanh(inputs)
        elif self.activation == "elu":
            return keras.activations.elu(inputs)
        elif self.activation == "selu":
            return keras.activations.selu(inputs)
        elif self.activation == "exponential":
            return keras.activations.exponential(inputs)
        else:
            raise ValueError("Please use an activation function supported by Keras")


class GLULayer(keras.layers.Layer):
    def __init__(
        self,
        units: int = 16,
        fc_layer=None,
        virtual_batch_size=None,
        momentum: float = 0.98,
        **kwargs,
    ):
        """
        Creates a layer with a fully-connected linear layer, followed by batch
        normalization, and a gated linear unit (GLU) as the activation function.

        Parameters:
        -----------
        units: int
            Number of units in layer. Default (16).
        fc_layer:keras.layers.Dense
            This is useful when you want to create a GLU layer with shared parameters. This
            is necessary because batch normalization should still be uniquely initialized
            due to the masked inputs in TabNet steps being in a different scale than the
            original input. Default (None) creates a new FC layer.
        virtual_batch_size: int
            Batch size for Ghost Batch Normalization (GBN). Value should be much smaller
            than and a factor of the overall batch size. Default (None) runs regular batch
            normalization. If an integer value is specified, GBN is run with that virtual
            batch size.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values
            correspond to larger impact of batch on the rolling statistics computed in
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        """
        super(GLULayer, self).__init__(**kwargs)
        self.units = units
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum

        if fc_layer:
            self.fc = fc_layer
        else:
            self.fc = keras.layers.Dense(self.units * 2, use_bias=False)

        self.bn = keras.layers.BatchNormalization(
            virtual_batch_size=self.virtual_batch_size, momentum=self.momentum
        )

    def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        x = tf.math.multiply(x[:, : self.units], tf.nn.sigmoid(x[:, self.units :]))
        return x


import numpy as np
from keras import initializers


class RandomFourierFeatures(keras.Layer):
    def __init__(
        self,
        output_dim,
        kernel_initializer="gaussian",
        scale=None,
        trainable=False,
        name=None,
        **kwargs,
    ):
        _SUPPORTED_RBF_KERNEL_TYPES = ["gaussian", "laplacian"]
        if output_dim <= 0:
            raise ValueError(
                "`output_dim` should be a positive integer. " f"Received: {output_dim}"
            )
        if isinstance(kernel_initializer, str):
            if kernel_initializer.lower() not in _SUPPORTED_RBF_KERNEL_TYPES:
                raise ValueError(
                    f"Unsupported `kernel_initializer`: {kernel_initializer} "
                    f"Expected one of: {_SUPPORTED_RBF_KERNEL_TYPES}"
                )
        if scale is not None and scale <= 0.0:
            raise ValueError(
                "When provided, `scale` should be a positive float. "
                f"Received: {scale}"
            )
        super().__init__(trainable=trainable, name=name, **kwargs)
        self.output_dim = output_dim
        self.kernel_initializer = kernel_initializer
        self.scale = scale

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        # TODO(pmol): Allow higher dimension inputs. Currently the input is
        # expected to have shape [batch_size, dimension].
        if input_shape.rank != 2:
            raise ValueError(
                "The rank of the input tensor should be 2. "
                f"Received input with rank {input_shape.ndims} instead. "
                f"Full input shape received: {input_shape}"
            )
        if input_shape.dims[1].value is None:
            raise ValueError(
                "The last dimension of the input tensor should be defined. "
                f"Found `None`. Full input shape received: {input_shape}"
            )

        input_dim = input_shape.dims[1].value

        kernel_initializer = get_random_features_initializer(
            self.kernel_initializer, shape=(input_dim, self.output_dim)
        )

        self.unscaled_kernel = self.add_weight(
            name="unscaled_kernel",
            shape=(input_dim, self.output_dim),
            dtype=tf.float32,
            initializer=kernel_initializer,
            trainable=False,
        )

        self.bias = self.add_weight(
            name="bias",
            shape=(self.output_dim,),
            dtype=tf.float32,
            initializer=initializers.RandomUniform(minval=0.0, maxval=2 * np.pi),
            trainable=False,
        )

        if self.scale is None:
            self.scale = get_default_scale(self.kernel_initializer, input_dim)
        self.kernel_scale = self.add_weight(
            name="kernel_scale",
            shape=(1,),
            dtype=tf.float32,
            initializer=tf.compat.v1.constant_initializer(self.scale),
            trainable=True,
            constraint="NonNeg",
        )
        super().build(input_shape)

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        inputs = tf.cast(inputs, tf.float32)
        kernel = (1.0 / self.kernel_scale) * self.unscaled_kernel
        outputs = tf.matmul(a=inputs, b=kernel)
        outputs = tf.nn.bias_add(outputs, self.bias)
        return tf.cos(outputs)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank(2)
        if input_shape.dims[-1].value is None:
            raise ValueError(
                "The last dimension of the input tensor should be defined. "
                f"Found `None`. Full input shape received: {input_shape}"
            )
        return input_shape[:-1].concatenate(self.output_dim)

    def get_config(self):
        kernel_initializer = self.kernel_initializer
        if not isinstance(kernel_initializer, str):
            kernel_initializer = initializers.serialize(kernel_initializer)
        config = {
            "output_dim": self.output_dim,
            "kernel_initializer": kernel_initializer,
            "scale": self.scale,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
