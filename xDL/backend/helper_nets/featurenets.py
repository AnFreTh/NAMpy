import tensorflow as tf
from tensorflow.keras import regularizers
from xDL.backend.helper_nets.layers import *
from xDL.backend.transformerblock import TransformerBlock
import pandas as pd


def MLP(inputs, param_dict, output_dimension=1, name=None):
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

    assert (
        param_dict["Network"] == "MLP"
    ), "Network-Name error. The Network name passed to the MLP is not correct. Expected 'MLP'"

    sizes = param_dict["sizes"]
    dropout = param_dict["dropout"]
    activation = param_dict["activation"]
    if len(inputs) > 1:
        x = tf.keras.layers.Concatenate()(inputs)
    else:
        x = inputs[0]
    dropout = tf.keras.layers.Dropout(dropout)
    x = tf.keras.layers.Dense(sizes[0], activation=activation)(x)
    x = dropout(x)
    for size in sizes[1:]:
        x = tf.keras.layers.Dense(size, activation=activation)(x)
    x = tf.keras.layers.Dense(output_dimension, activation="linear", use_bias=False)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
    model.reset_states()
    return model


def Transformer(inputs, param_dict, output_dimension=1, name=None):
    assert (
        param_dict["Network"] == "Transformer"
    ), "Network-Name error. The Network name passed to the Transformer is not correct. Expected 'Transformer'"

    if len(inputs) == 1:
        print(
            "It is not recommended to use a Transformer for a single feature input. Consider using a MLP instead."
        )

    inputs = inputs[0]
    embedding_dim = param_dict["embedding_dim"]

    depth = param_dict["depth"]
    heads = param_dict["heads"]
    ff_dropout = param_dict["ff_dropout"]
    attn_dropout = param_dict["attn_dropout"]

    if "n_bins" in param_dict.keys():
        num_categories = param_dict["n_bins"] + 1

    if param_dict["encoding"] == "PLE":
        embeddings = tf.keras.layers.Dense(embedding_dim, activation="relu")(inputs)
        embeddings = tf.expand_dims(embeddings, axis=1, name="dimension_expansion")
        print(embeddings.shape)

    else:
        embeddings = tf.keras.layers.Embedding(
            input_dim=num_categories, output_dim=embedding_dim
        )(inputs)

        print(embeddings.shape)

    for _ in range(depth):
        embeddings = TransformerBlock(
            embedding_dim,
            heads,
            embedding_dim,
            att_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            explainable=False,
        )(embeddings)

    embeddings = tf.keras.layers.Flatten()(embeddings)
    embeddings = tf.keras.layers.Dense(128, activation="relu")(embeddings)
    x = tf.keras.layers.Dense(output_dimension, use_bias=False, activation="linear")(
        embeddings
    )
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


def CubicSplineNet(inputs, param_dict, output_dimension=1, name=None):
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

    assert (
        param_dict["Network"] == "CubicSplineNet"
    ), "Network-Name error. The Network name passed to the CubicSplineNet is not correct. Expected 'CubicSplineNet'"

    inputs = inputs[0]
    activation = param_dict["activation"]
    l1 = param_dict["l1_regularizer"]
    l2 = param_dict["l2_regularizer"]
    l2_activity = param_dict["l2_activity_regularizer"]

    weight_shift = AddWeightsLayer()
    x = weight_shift(inputs)

    x = tf.keras.layers.Dense(
        output_dimension,
        activation=activation,
        kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
        activity_regularizer=regularizers.L2(l2_activity),
        use_bias=False,
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
    model.reset_states()
    return model


def PolySplineNet(inputs, param_dict, output_dimension=1, name=None):
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

    assert (
        param_dict["Network"] == "PolySplineNet"
    ), "Network-Name error. The Network name passed to the PolySplineNet is not correct. Expected 'PolySplineNet'"

    inputs = inputs[0]
    activation = param_dict["activation"]
    l1 = param_dict["l1_regularizer"]
    l2 = param_dict["l2_regularizer"]
    l2_activity = param_dict["l2_activity_regularizer"]

    weight_shift = AddWeightsLayer()
    x = weight_shift(inputs)

    x = tf.keras.layers.Dense(
        output_dimension,
        activation=activation,
        use_bias=False,
        kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
        activity_regularizer=regularizers.L2(l2_activity),
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
        # mlp_layers.append(tf.keras.layers.BatchNormalization()),
        mlp_layers.append(
            tf.keras.layers.Dense(
                units,
                activation="relu",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            )
        )
        mlp_layers.append(tf.keras.layers.Dropout(dropout))

    return tf.keras.Sequential(mlp_layers)


def build_shape_funcs(
    dropout=0.1, output_dim=1, hidden_units=[128, 128, 64], activation="relu"
):
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
        mlp_layers.append(tf.keras.layers.BatchNormalization()),
        mlp_layers.append(tf.keras.layers.Dense(units, activation=activation))
        mlp_layers.append(tf.keras.layers.Dropout(dropout))

    mlp_layers.append(
        tf.keras.layers.Dense(output_dim, activation="linear", use_bias=False)
    )

    return tf.keras.Sequential(mlp_layers)


def helper_normalization_net(input_list):
    layer1 = tf.keras.layers.Concatenate()
    layer2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    x = layer1(input_list)
    x = layer2(x)

    return tf.keras.Model(inputs=input_list, outputs=x, name="helper_net")
