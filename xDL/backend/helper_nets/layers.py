from tensorflow.keras.layers import Layer
import tensorflow as tf


class InterceptLayer(Layer):
    """
    Custom Keras layer to add an intercept (bias) term to the input tensor.

    Args:
        **kwargs: Additional keyword arguments to pass to the Layer constructor.

    Example:
        # Create a model with an InterceptLayer
        model = tf.keras.Sequential([
            InterceptLayer(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
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
        model = tf.keras.Sequential([
            IdentityLayer(activation='relu'),
            tf.keras.layers.Dense(64),
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
            return tf.keras.activations.relu(inputs)
        elif self.activation == "sigmoid":
            return tf.keras.activations.sigmoid(inputs)
        elif self.activation == "softmax":
            return tf.keras.activations.softmax(inputs)
        elif self.activation == "softplus":
            return tf.keras.activations.softplus(inputs)
        elif self.activation == "softsign":
            return tf.keras.activations.softsign(inputs)
        elif self.activation == "tanh":
            return tf.keras.activations.tanh(inputs)
        elif self.activation == "elu":
            return tf.keras.activations.elu(inputs)
        elif self.activation == "selu":
            return tf.keras.activations.selu(inputs)
        elif self.activation == "exponential":
            return tf.keras.activations.exponential(inputs)
        else:
            raise ValueError("Please use an activation function supported by Keras")
