import tensorflow as tf
from keras.callbacks import *
from sklearn.model_selection import KFold
from xDL.backend.basemodel import BaseModel
from xDL.shapefuncs.transformer_encoder import TabTransformerEncoder
from xDL.shapefuncs.helper_nets.helper_funcs import build_cls_mlp
from xDL.backend.families import *

import warnings

# Filter out the specific warning by category
warnings.filterwarnings("ignore", category=UserWarning)


class TabTransformerLSS(BaseModel):
    def __init__(
        self,
        data,
        y,
        family,
        dropout=0.5,
        val_split=0.2,
        val_data=None,
        activation="relu",
        classification=False,
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        mlp_hidden_factors: list = [2, 4],
        hidden_units=[128, 128, 64],
        encoder=None,
        output_activation=tf.math.sigmoid,
        num_classes=1,
        binning_task="regression",
    ):
        """
        Initialize the TabTransformer model as described in https://arxiv.org/pdf/2012.06678.pdf.
        Args:
            data (pd.DataFrame): Input data as a Pandas DataFrame.
            y (str): target variable. must be included in data
            dropout (float): Dropout rate (default is 0.1).
            val_split (float): Validation data split ratio (default is 0.2).
            val_data (pd.DataFrame, optional): Validation data as a Pandas DataFrame (default is None).
            activation (str): Activation function for the model (default is "relu").
            classification (bool): Whether the problem is a classification task (default is False).
            embedding_dim (int): Dimension of embeddings (default is 32).
            depth (int): Depth of the transformer encoder (default is 4).
            heads (int): Number of attention heads (default is 8).
            attn_dropout (float): Attention dropout rate (default is 0.1).
            ff_dropout (float): Feedforward dropout rate (default is 0.1).
            mlp_hidden_factors (list): List of factors for hidden layer sizes (default is [2, 4]).
            encoder (object): Custom encoder for the model (default is None).
            output_activation (callable): Output layer activation function (default is tf.math.sigmoid).

        Attributes:
            data: The input data.
            dropout (float): The dropout rate for model layers.
            val_data: Validation data to use.
            val_split (float): The validation data split ratio.
            family (str): The distribution family for the target variable.
            activation (str): The activation function for model layers.
            classification (bool): True if a classification task, False for regression.
            TRANSFORMER_FEATURES (list): List of transformer features.
            encoder: The transformer encoder.
            transformer_mlp: The transformer MLP layer.
            final_mlp: The final MLP layer.
            output_activation (callable): The output activation function.
            training_dataset (tf.data.Dataset): training dataset containing the transformed inputs
            validation_dataset (tf.data.Dataset): validation dataset containing the transformed inputs
            plotting_dataset (tf.data.Dataset): dataset containing the transformed inputs adapted for creating the plots
            inputs (dict): dictionary with all tf.keras.Inputs -> mapping from feature name to feature
            input_dict (dict): dictionary containg all the model specification -> mapping from feature to network type, network size, name, input
            NUM_FEATURES (list): Convenience list with all numerical features
            CAT_FEATURES (list): Convenience list with all categorical features
            y (str): Name of the target variable.
            feature_names (list): List of feature names.
        """

        super(TabTransformerLSS, self).__init__(
            data=data,
            y=y,
            activation=activation,
            dropout=dropout,
            binning_task=binning_task,
        )

        if family == "Normal":
            self.family = Normal()
        elif family == "Logistic":
            self.family = Logistic()
        elif family == "InverseGamma":
            self.family = InverseGamma()
        elif family == "Poisson":
            self.family = Poisson()
        elif family == "JohnsonSU":
            self.family = JohnsonSU()
        elif family == "Gamma":
            self.family = Gamma()
        else:
            raise ValueError(
                "The family must be in ['Normal', 'Logistic', 'InverseGamma', 'Poisson', 'JohnsonSU', 'Gamma']. If you wish further distributions to be implemented please raise an Issue"
            )

        self.val_data = val_data
        self.val_split = val_split
        self.activation = activation
        self.dropout = dropout
        self.classification = classification
        self.explainable = False
        self.output_activation = output_activation
        self.hidden_units = hidden_units

        num_classes = num_classes

        # Initialise encoder
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = TabTransformerEncoder(
                categorical_features=self.CAT_FEATURES,
                numerical_features=self.NUM_FEATURES,
                embedding_dim=embedding_dim,
                depth=depth,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                explainable=False,
                data=self.data,
            )

        mlp_input_dim = embedding_dim * len(self.encoder.categorical)
        hidden_units = [mlp_input_dim // f for f in mlp_hidden_factors]

        self.final_mlp = build_cls_mlp(mlp_input_dim, mlp_hidden_factors, ff_dropout)

        self.output_layer = tf.keras.layers.Dense(
            self.family.dimension, activation="linear", use_bias=False
        )

    def NegativeLogLikelihood(self, y_true, y_hat):
        """Negative LogLIkelihood Loss function

        Args:
            y_true (_type_): True Labels
            y_hat (_type_): Predicted Distribution

        Returns:
            _type_: negative Log likelihood of respective input distribution
        """
        return -y_hat.log_prob(tf.cast(y_true, dtype=tf.float32))

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.final_mlp(x)
        output = self.output_layer(x)
        p_y = tfp.layers.DistributionLambda(lambda x: self.family.forward(x))(output)

        return p_y
