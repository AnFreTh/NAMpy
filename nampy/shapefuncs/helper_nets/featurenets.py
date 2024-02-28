import tensorflow as tf
from keras import regularizers
from nampy.shapefuncs.helper_nets.layers import (
    AddWeightsLayer,
    InterceptLayer,
    IdentityLayer,
    GLULayer,
)
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from keras.layers import Add
from nampy.shapefuncs.baseshapefunction import ShapeFunction


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, units, activation="relu", dropout=0.5):
        """
        Constructor for the ResidualBlock class.

        Parameters:
        - units (int): The number of units/neurons in the Dense layers.
        - activation (str): The activation function to use in the Dense layers (default is "relu").
        - dropout (float): The dropout rate to apply between the two Dense layers (default is 0.5).
        """
        super(ResidualBlock, self).__init__()
        super(ResidualBlock, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.hidden2 = tf.keras.layers.Dense(units, activation=activation)

    def call(self, inputs):
        """
        Defines the forward pass for the ResidualBlock.

        Parameters:
        - inputs: The input tensor to the residual block.

        Returns:
        - tf.Tensor: The output tensor after processing through the residual block.
        """
        x = self.hidden1(inputs)
        x = self.dropout(x)
        x = self.hidden2(x)
        return Add()([inputs, x])


class ResNet(ShapeFunction):
    def __init__(self, inputs, *args, **kwargs):
        """
        Constructor for the ResNet class.

        Parameters:
        - inputs: The input data to the ResNet.
        - *args: Additional positional arguments (not used in this constructor).
        - **kwargs: Additional keyword arguments (not used in this constructor).
        """
        super(ResNet, self).__init__(*args, **kwargs)

        assert (
            self.Network == "ResNet"
        ), "Network-Name error. The Network name passed to the ResNet is not correct. Expected 'ResNet'"

        if not hasattr(self, "activation"):
            self.activation = "relu"
        if not hasattr(self, "hidden_dims"):
            self.hidden_dims = [128, 128]
        if not hasattr(self, "dropout"):
            self.dropout = 0.5
        self.output_layer = tf.keras.layers.Dense(
            self.output_dimension, "linear", use_bias=False
        )
        self.concatenate = False
        if len(inputs) > 1:
            self.concatenate = True

        self.arch = []
        for dim in self.hidden_dims:
            self.arch.append(
                ResidualBlock(dim, activation=self.activation, dropout=self.dropout)
            )

    def forward(self, inputs):
        """
        Defines the forward pass for the ResNet.

        Parameters:
        - inputs: The input data to the ResNet.

        Returns:
        - tf.Tensor: The output tensor after processing through the ResNet.
        """

        if self.concatenate:
            inputs = tf.keras.layers.Concatenate()(inputs)
        x = inputs
        for block in self.arch:
            x = block(x)

        x = self.output_layer(x)
        return x


class MLP(ShapeFunction):
    def __init__(self, inputs, *args, **kwargs):
        """
        Constructor for the MLP class.

        Parameters:
        - inputs: The input data to the MLP.
        - *args: Additional positional arguments (not used in this constructor).
        - **kwargs: Additional keyword arguments (not used in this constructor).
        """
        super().__init__(*args, **kwargs)

        assert (
            self.Network == "MLP"
        ), "Network-Name error. The Network name passed to the MLP is not correct. Expected 'MLP'"

        if not hasattr(self, "activation"):
            self.activation = "relu"
        if not hasattr(self, "hidden_dims"):
            self.hidden_dims = [128, 64, 64]
        if not hasattr(self, "dropout"):
            self.dropout = 0.5
        self.output_layer = tf.keras.layers.Dense(
            self.output_dimension, "linear", use_bias=False
        )
        self.concatenate = False
        if len(inputs) > 1:
            self.concatenate = True

        self.arch = []
        for dim in self.hidden_dims:
            self.arch.append(tf.keras.layers.Dense(dim, activation=self.activation))
            self.arch.append(tf.keras.layers.Dropout(self.dropout))

    def forward(self, inputs):
        """
        Defines the forward pass for the MLP.

        Parameters:
        - inputs: The input data to the MLP.

        Returns:
        - tf.Tensor: The output tensor after processing through the MLP.
        """

        if self.concatenate:
            inputs = tf.keras.layers.Concatenate()(inputs)
        x = inputs
        for layer in self.arch:
            x = layer(x)

        x = self.output_layer(x)
        return x


class CubicSplineNet(ShapeFunction):
    def __init__(self, inputs, *args, **kwargs):
        """
        Constructor for the CubicSplineNet class.

        Parameters:
        - inputs: The input data to the CubicSplineNet.
        - *args: Additional positional arguments (not used in this constructor).
        - **kwargs: Additional keyword arguments (not used in this constructor).
        """
        super().__init__(*args, **kwargs)

        assert (
            self.Network == "CubicSplineNet"
        ), "Network-Name error. The Network name passed to the CubicSplineNet is not correct. Expected 'CubicSplineNet'"

        if not hasattr(self, "n_knots"):
            self.n_knots = 15
        if not hasattr(self, "activation"):
            self.activation = "relu"
        if not hasattr(self, "l1_regularizer"):
            self.l1_regularizer = 0.0005
        if not hasattr(self, "l2_regularizer"):
            self.l2_regularizer = 0.0005
        if not hasattr(self, "l2_activity_regularizer"):
            self.l2_activity_regularizer = 0.0005
        if not hasattr(self, "hidden_dims"):
            self.hidden_dims = [64]

        self.output_layer = tf.keras.layers.Dense(
            self.output_dimension,
            "linear",
            use_bias=False,
            kernel_regularizer=regularizers.L1L2(
                l1=self.l1_regularizer, l2=self.l2_regularizer
            ),
            activity_regularizer=regularizers.L2(self.l2_activity_regularizer),
        )

        self.arch = [AddWeightsLayer()]
        for dim in self.hidden_dims:
            self.arch.append(tf.keras.layers.Dense(dim, activation=self.activation))

    def forward(self, inputs):
        """
        Defines the forward pass for the CubicSplineNet.

        Parameters:
        - inputs: The input data to the CubicSplineNet.

        Returns:
        - tf.Tensor: The output tensor after processing through the CubicSplineNet.
        """
        x = inputs
        for layer in self.arch:
            x = layer(x)

        x = self.output_layer(x)
        return x


class PolynomialSplineNet(ShapeFunction):
    def __init__(self, inputs, *args, **kwargs):
        """
        Constructor for the PolynomialSplineNet class.

        Parameters:
        - inputs: The input data to the PolynomialSplineNet.
        - *args: Additional positional arguments (not used in this constructor).
        - **kwargs: Additional keyword arguments (not used in this constructor).
        """
        super().__init__(*args, **kwargs)

        assert (
            self.Network == "PolynomialSplineNet"
        ), "Network-Name error. The Network name passed to the CubicSplineNet is not correct. Expected 'CubicSplineNet'"

        if not hasattr(self, "n_knots"):
            self.n_knots = 15
        if not hasattr(self, "activation"):
            self.activation = "relu"
        if not hasattr(self, "l1_regularizer"):
            self.l1_regularizer = 0.0005
        if not hasattr(self, "l2_regularizer"):
            self.l2_regularizer = 0.0005
        if not hasattr(self, "l2_activity_regularizer"):
            self.l2_activity_regularizer = 0.0005
        if not hasattr(self, "hidden_dims"):
            self.hidden_dims = [64]

        self.output_layer = tf.keras.layers.Dense(
            self.output_dimension,
            "linear",
            use_bias=False,
            kernel_regularizer=regularizers.L1L2(
                l1=self.l1_regularizer, l2=self.l2_regularizer
            ),
            activity_regularizer=regularizers.L2(self.l2_activity_regularizer),
        )

        self.arch = [AddWeightsLayer()]
        for dim in self.hidden_dims:
            self.arch.append(tf.keras.layers.Dense(dim, activation=self.activation))

    def forward(self, inputs):
        """
        Defines the forward pass for the PolynomialSplineNet.

        Parameters:
        - inputs: The input data to the PolynomialSplineNet.

        Returns:
        - tf.Tensor: The output tensor after processing through the PolynomialSplineNet.
        """
        x = inputs
        for layer in self.arch:
            x = layer(x)

        x = self.output_layer(x)
        return x


class LinearPredictor(ShapeFunction):
    def __init__(self, inputs, *args, **kwargs):
        """
        Constructor for the LinearPredictor class.

        Parameters:
        - inputs: The input data to the LinearPredictor.
        - *args: Additional positional arguments (not used in this constructor).
        - **kwargs: Additional keyword arguments (not used in this constructor).
        """
        super().__init__(*args, **kwargs)

        assert (
            self.Network == "LinearPredictor"
        ), f"Network-Name error. The Network name passed to the LinearPredictor is not correct. Expected 'LinearPredictor', got {self.Network}"

        self.concatenate = False
        if len(inputs) > 1:
            self.concatenate = True

        self.arch = tf.keras.layers.Dense(
            self.output_dimension, "linear", use_bias=False
        )

    def forward(self, inputs):
        """
        Defines the forward pass for the LinearPredictor.

        Parameters:
        - inputs: The input data to the LinearPredictor.

        Returns:
        - tf.Tensor: The output tensor after processing through the LinearPredictor.
        """
        if self.concatenate:
            inputs = tf.keras.layers.Concatenate()(inputs)
        return self.arch(inputs)


class RandomFourierNet(ShapeFunction):
    def __init__(self, inputs, *args, **kwargs):
        """
        Constructor for the RandomFourierNet class.

        Parameters:
        - inputs: The input data to the RandomFourierNet.
        - *args: Additional positional arguments (not used in this constructor).
        - **kwargs: Additional keyword arguments (not used in this constructor).
        """
        super().__init__(*args, **kwargs)

        assert (
            self.Network == "RandomFourierNet"
        ), f"Network-Name error. The Network name passed to the RandomFourierNet is not correct. Expected 'RandomFourierNet', got {self.Network}"

        if not hasattr(self, "hidden_dims"):
            self.hidden_dims = [4096, 64]
        if not hasattr(self, "activation"):
            self.activation = "relu"

        self.concatenate = False
        if len(inputs) > 1:
            self.concatenate = True

        self.arch = [
            RandomFourierFeatures(
                output_dim=self.hidden_dims[0],
                trainable=True,
                kernel_initializer="gaussian",
            )
        ]
        for dim in self.hidden_dims[1:]:
            self.arch.append(tf.keras.layers.Dense(dim, activation=self.activation))

        self.output_layer = tf.keras.layers.Dense(
            self.output_dimension, "linear", use_bias=False
        )

    def forward(self, inputs):
        """
        Defines the forward pass for the RandomFourierNet.

        Parameters:
        - inputs: The input data to the RandomFourierNet.

        Returns:
        - tf.Tensor: The output tensor after processing through the RandomFourierNet.
        """
        if self.concatenate:
            inputs = tf.keras.layers.Concatenate()(inputs)
        x = inputs
        for layer in self.arch:
            x = layer(x)

        x = self.output_layer(x)
        return x


class ConstantWeightNet(ShapeFunction):
    def __init__(self, inputs, *args, **kwargs):
        """
        Constructor for the LinearPredictor class.

        Parameters:
        - inputs: The input data to the LinearPredictor.
        - *args: Additional positional arguments (not used in this constructor).
        - **kwargs: Additional keyword arguments (not used in this constructor).
        """
        super().__init__(*args, **kwargs)

        assert (
            self.Network == "ConstantWeightNet"
        ), f"Network-Name error. The Network name passed to the ConstantWeightNet is not correct. Expected 'ConstantWeightNet', got {self.Network}"

        self.concatenate = False
        if len(inputs) > 1:
            self.concatenate = True

        self.arch = InterceptLayer()

    def forward(self, inputs):
        """
        Defines the forward pass for the ConstantWeightNet.

        Parameters:
        - inputs: The input data to the ConstantWeightNet.

        Returns:
        - tf.Tensor: The output tensor after processing through the ConstantWeightNet.
        """
        if self.concatenate:
            inputs = tf.keras.layers.Concatenate()(inputs)
        return self.arch(inputs)


class Transformer(ShapeFunction):
    def __init__(self, inputs, *args, **kwargs):
        ### TODO!
        super().__init__(*args, **kwargs)

        assert (
            self.Network == "Transformer"
        ), "Network-Name error. The Network name passed to the Transformer is not correct. Expected 'Transformer'"

        if len(inputs) == 1:
            print(
                "It is not recommended to use a Transformer for a single feature input. Consider using a MLP instead."
            )

        if not hasattr(self, "embedding_dim"):
            self.embedding_dim = 32
        if not hasattr(self, "ff_dropout"):
            self.ff_dropout = 0.1
        if not hasattr(self, "dropout"):
            self.dropout = 0.5
        if not hasattr(self, "attn_dropout"):
            self.attn_dropout = 0.1
        if not hasattr(self, "heads"):
            self.heads = 8
        if not hasattr(self, "depth"):
            self.depth = 4

        if hasattr(self, "n_bins"):
            self.num_categories = self.n_bins + 1

        if not hasattr(self, "encoding"):
            self.encoding == "int"

    def forward(self, inputs):
        if self.encoding == "PLE":
            embeddings = tf.keras.layers.Dense(self.embedding_dim, activation="relu")(
                inputs
            )
            embeddings = tf.expand_dims(embeddings, axis=1, name="dimension_expansion")

        else:
            embeddings = tf.keras.layers.Embedding(
                input_dim=self.num_categories, output_dim=self.embedding_dim
            )(inputs)


# def Transformer(inputs, param_dict, output_dimension=1, name=None):
#    assert (
#        param_dict["Network"] == "Transformer"
#    ), "Network-Name error. The Network name passed to the Transformer is not correct. Expected 'Transformer'"
#
#    if len(inputs) == 1:
#        print(
#            "It is not recommended to use a Transformer for a single feature input. Consider using a MLP instead."
#        )
#
#    inputs = inputs[0]
#    embedding_dim = param_dict["embedding_dim"]
#
#    depth = param_dict["depth"]
#    heads = param_dict["heads"]
#    ff_dropout = param_dict["ff_dropout"]
#    attn_dropout = param_dict["attn_dropout"]
#
#    if "n_bins" in param_dict.keys():
#        num_categories = param_dict["n_bins"] + 1
#
#    if param_dict["encoding"] == "PLE":
#        embeddings = tf.keras.layers.Dense(embedding_dim, activation="relu")(inputs)
#        embeddings = tf.expand_dims(embeddings, axis=1, name="dimension_expansion")
#        print(embeddings.shape)
#
#    else:
#        embeddings = tf.keras.layers.Embedding(
#            input_dim=num_categories, output_dim=embedding_dim
#        )(inputs)
#
#        print(embeddings.shape)
#
#    for _ in range(depth):
#        embeddings = TransformerBlock(
#            embedding_dim,
#            heads,
#            embedding_dim,
#            att_dropout=attn_dropout,
#            ff_dropout=ff_dropout,
#            explainable=False,
#        )(embeddings)
#
#    embeddings = tf.keras.layers.Flatten()(embeddings)
#    embeddings = tf.keras.layers.Dense(128, activation="relu")(embeddings)
#    x = tf.keras.layers.Dense(output_dimension, use_bias=False, activation="linear")(
#        embeddings
#    )
#    model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
#    model.reset_states()
#    return model
