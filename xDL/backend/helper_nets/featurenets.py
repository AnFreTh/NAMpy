import tensorflow as tf
from tensorflow.keras import regularizers
from xDL.backend.helper_nets.layers import *


def MLP(
    inputs, sizes, name, bias=True, activation="relu", dropout=0.5, output_dimension=1
):
    """
    Build a single feature neural network.

    This function constructs a neural network model for processing a single feature.
    -> returns a tf.keras.Model!

    Args:
        input (tf.Tensor): Input tensor representing the single feature.
        sizes (list): A list of integers specifying the number of units for each hidden layer.
        bias (bool, optional): Whether to include bias terms in the Dense layers. Defaults to True.
        activation (str, optional): Activation function to use for hidden layers. Defaults to "relu".
        dropout (float, optional): Dropout rate for regularization. Defaults to 0.5.
        output_dimension (int, optional): Number of units in the output layer. Defaults to 1.

    Returns:
        tf.keras.Model: A Keras model representing the single feature neural network.
    """
    dropout = tf.keras.layers.Dropout(dropout)
    x = tf.keras.layers.Dense(sizes[0], activation=activation, use_bias=bias)(inputs)
    x = dropout(x)
    for size in sizes[1:]:
        x = tf.keras.layers.Dense(size, activation=activation, use_bias=bias)(x)
    x = tf.keras.layers.Dense(output_dimension, activation="linear", use_bias=False)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
    model.reset_states()
    return model


def Interaction_MLP(
    inputs, sizes, name, bias=True, activation="relu", dropout=0.5, output_dimension=1
):
    """
    Build an interaction feature neural network.

    This function constructs a neural network model for interaction feature processing.
    -> returns a tf.keras.Model!

    Args:
        inputs (list): A list of input tensors. The input tensors to be concatenated.
        sizes (list): A list of integers specifying the number of units for each hidden layer.
        bias (bool, optional): Whether to include bias terms in the Dense layers. Defaults to True.
        activation (str, optional): Activation function to use for hidden layers. Defaults to "relu".
        dropout (float, optional): Dropout rate for regularization. Defaults to 0.5.
        output_dimension (int, optional): Number of units in the output layer. Defaults to 1.

    Returns:
        tf.keras.Model: A Keras model representing the interaction feature neural network.
    """
    dropout = tf.keras.layers.Dropout(dropout)
    x = tf.keras.layers.Concatenate()(inputs)
    x = tf.keras.layers.Dense(sizes[0], activation=activation, use_bias=bias)(x)
    x = dropout(x)
    for size in sizes[1:]:
        x = tf.keras.layers.Dense(size, activation=activation, use_bias=bias)(x)
    x = tf.keras.layers.Dense(output_dimension, activation="linear", use_bias=False)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
    model.reset_states()
    return model


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


def CubicSplineNet(
    inputs,
    sizes,
    name,
    bias=True,
    activation=None,
    dropout=0.5,
    output_dimension=1,
):
    """
    Create a Cubic Spline Neural Network.
    -> this takes a cubic basis expansion as an input and adds weights to each column, thus pseudo-adjusting the knot locations.

    Args:
        inputs (tf.Tensor): Input tensor to the network.
        sizes (list): List of sizes for each layer. -> unused
        name (str): Name of the model.
        bias (bool): Whether to use bias in Dense layers (default is True).
        activation (str): Activation function for Dense layers (default is None). -> unused
        dropout (float): Dropout rate (default is 0.5). -> unused
        output_dimension (int): Dimension of the output (default is 1).

    Returns:
        tf.keras.Model: The Cubic Spline Neural Network model.
    """
    # Add small weights to each row except the first and last columns

    weight_shift = AddWeightsLayer()
    x = weight_shift(inputs)
    x = tf.keras.layers.Dense(
        output_dimension,
        activation="linear",
        use_bias=bias,
        kernel_regularizer=regularizers.L1L2(l1=0.05, l2=0.05),
        activity_regularizer=regularizers.L2(0.05),
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
    model.reset_states()
    return model


def PolySplineNet(
    inputs,
    sizes,
    name,
    bias=True,
    activation=None,
    dropout=0.5,
    output_dimension=1,
):
    """
    Create a Polynomial Spline Neural Network.
    -> this takes a polynomial basis expansion as an input and adds weights to each column, thus pseudo-adjusting the knot locations.

    Args:
        inputs (tf.Tensor): Input tensor to the network.
        sizes (list): List of sizes for each layer. -> unused
        name (str): Name of the model.
        bias (bool): Whether to use bias in Dense layers (default is True).
        activation (str): Activation function for Dense layers (default is None). -> unused
        dropout (float): Dropout rate (default is 0.5). -> unused
        output_dimension (int): Dimension of the output (default is 1).

    Returns:
        tf.keras.Model: The polynomial Spline Neural Network model.
    """

    weight_shift = AddWeightsLayer()
    x = weight_shift(inputs)
    x = tf.keras.layers.Dense(
        output_dimension,
        activation="linear",
        use_bias=bias,
        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-5),
        activity_regularizer=regularizers.L2(1e-5),
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
    model.reset_states()
    return model


# Helper func for building MLP on transformer output ([cls] token)
def build_cls_mlp(input_dim, factors, dropout):
    """
    Build a multi-layer perceptron (MLP) for a transformer output.

    Args:
        input_dim (int): Dimension of the input.
        factors (list): List of factors for hidden layer sizes.
        dropout (float): Dropout rate.

    Returns:
        tf.keras.Sequential: The MLP model.
    """
    hidden_units = [input_dim // f for f in factors]
    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(tf.keras.layers.BatchNormalization()),
        mlp_layers.append(tf.keras.layers.Dense(units, activation="relu"))
        mlp_layers.append(tf.keras.layers.Dropout(dropout))

    return tf.keras.Sequential(mlp_layers)


def build_shape_funcs(dropout=0.1):
    """
    Build shape functions using a multi-layer perceptron (MLP).

    Args:
        dropout (float): Dropout rate (default is 0.1).

    Returns:
        tf.keras.Sequential: The MLP model for shape functions.
    """
    hidden_units = [128, 128, 64]

    mlp_layers = []
    for units in hidden_units:
        # mlp_layers.append(BatchNormalization()),
        mlp_layers.append(tf.keras.layers.Dense(units, activation="relu"))
        mlp_layers.append(tf.keras.layers.Dropout(dropout))

    mlp_layers.append(tf.keras.layers.Dense(1, activation="linear", use_bias=False))

    return tf.keras.Sequential(mlp_layers)
