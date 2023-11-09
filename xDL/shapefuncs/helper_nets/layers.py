from tensorflow.keras.layers import Layer
import tensorflow as tf


class AddWeightsLayer(tf.keras.layers.Layer):
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


class GLULayer(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int = 16,
        fc_layer=None,
        virtual_batch_size=None,
        momentum: float = 0.98,
        **kwargs
    ):
        """
        Creates a layer with a fully-connected linear layer, followed by batch
        normalization, and a gated linear unit (GLU) as the activation function.

        Parameters:
        -----------
        units: int
            Number of units in layer. Default (16).
        fc_layer:tf.keras.layers.Dense
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
            self.fc = tf.keras.layers.Dense(self.units * 2, use_bias=False)

        self.bn = tf.keras.layers.BatchNormalization(
            virtual_batch_size=self.virtual_batch_size, momentum=self.momentum
        )

    def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        x = tf.math.multiply(x[:, : self.units], tf.nn.sigmoid(x[:, self.units :]))
        return x
