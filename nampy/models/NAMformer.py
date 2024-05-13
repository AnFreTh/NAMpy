import tensorflow as tf
from keras.callbacks import *
import numpy as np
from nampy.backend.black_box_basemodel import BaseModel
from nampy.shapefuncs.transformer_encoder import NAMformerEncoder
from nampy.shapefuncs.helper_nets.helper_funcs import build_cls_mlp
from nampy.visuals.plot_importances import (
    visualize_importances,
)
from keras.layers import Add
from nampy.visuals.analytics_plot import visual_analysis
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import warnings
from nampy.visuals.plot_interactive import plot_NAMformer
from nampy.visuals.plot_predictions import (
    plot_multi_output,
)

# Filter out the specific warning by category
warnings.filterwarnings("ignore", category=UserWarning)


class NAMformer(BaseModel):
    def __init__(
        self,
        data,
        y,
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
        output_activation="linear",
        binning_task="regression",
        explainable=True,
        batch_size=1024,
        num_encoding="PLE",
        n_bins=20,
        feature_dropout=0.1,
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

        task = "classification" if classification else "regression"

        super(NAMformer, self).__init__(
            data=data,
            y=y,
            activation=activation,
            dropout=dropout,
            binning_task=binning_task,
            batch_size=batch_size,
            num_encoding=num_encoding,
            n_bins=n_bins,
            task=task,
            val_split=val_split,
            val_data=val_data,
        )

        self.val_data = val_data
        self.val_split = val_split
        self.activation = activation
        self.dropout = dropout
        self.classification = classification
        self.explainable = explainable
        self.output_activation = output_activation
        self.hidden_units = hidden_units
        self.TRANSFORMER_FEATURES = self.CAT_FEATURES + self.NUM_FEATURES
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.mlp_hidden_factors = mlp_hidden_factors
        self.encoder = encoder
        self.num_encoding = num_encoding
        self.model_built = False
        self.n_bins = n_bins
        self.feature_dropout = feature_dropout

    def build(self, input_shape):
        """
        Build the model. This method should be called before training the model.
        """
        if self.model_built:
            return

        num_categories = [len(self.data[cat].unique()) + 1 for cat in self.CAT_FEATURES]
        numeric_categories = []
        for key in self.NUM_FEATURES:
            if self.num_encoding == "PLE":
                n_categories = self.n_bins
            else:
                n_categories = self.n_bins

            numeric_categories.append(n_categories + 1)

        num_categories += numeric_categories

        self._initialize_transformer(num_categories)
        self._initialize_transformer_mlp()
        self._initialize_output_layer(self.n_classes)
        self._initialize_shapefuncs(self.n_classes)

        self.model_built = True

    def _initialize_transformer(self, num_categories):
        # Initialise
        if self.encoder:
            pass
        else:
            self.encoder = NAMformerEncoder(
                categorical_features=self.CAT_FEATURES,
                numerical_features=self.NUM_FEATURES,
                num_categories=num_categories,
                embedding_dim=self.embedding_dim,
                depth=self.depth,
                heads=self.heads,
                attn_dropout=self.attn_dropout,
                ff_dropout=self.ff_dropout,
                explainable=self.explainable,
                data=self.data,
            )

        self.ln = tf.keras.layers.LayerNormalization()

    def _initialize_transformer_mlp(self):
        self.mlp_input_dim = self.embedding_dim * len(self.encoder.features)

        self.transformer_mlp = build_cls_mlp(
            self.mlp_input_dim, self.mlp_hidden_factors, self.ff_dropout
        )

    def _initialize_output_layer(self, num_classes):
        self.output_layer = tf.keras.layers.Dense(
            num_classes,
            self.output_activation,
        )

        self.FeatureDropoutLayer = tf.keras.layers.Dropout(self.feature_dropout)

    def _initialize_shapefuncs(self, num_classes):
        self.shapefuncs = []
        all_features = self.CAT_FEATURES + self.NUM_FEATURES

        for feature_name in all_features:
            model = Sequential(name=feature_name)
            model.add(
                Flatten(input_shape=(1, self.embedding_dim))
            )  # Adjusting input shape to account for the sequence length
            # model.add(Dense(units=64, activation="relu"))
            # model.add(Dropout(0.5))
            model.add(
                Dense(units=num_classes, activation=self.output_activation)
            )  # Output layer

            self.shapefuncs.append(model)

    def get_raw_embeddings(self, inputs):
        x, expl, uncontextualized_embeddings = self.encoder(inputs)
        return uncontextualized_embeddings

    def call(self, inputs, training=True):
        if self.encoder.explainable:
            x, expl, uncontextualized_embeddings = self.encoder(inputs)
            raw_embeddings = tf.identity(uncontextualized_embeddings)
            x = self.ln(x[:, 0, :])
            x = self.transformer_mlp(x)
            output = self.output_layer(x)  # FT-Transformer output
            feature_preds = [
                shapefunc(uncontextualized_embeddings[i])
                for i, shapefunc in enumerate(self.shapefuncs)
            ]

            # Apply dropout if in training mode
            if training:
                feature_preds_dropout = [
                    self.FeatureDropoutLayer(output) for output in feature_preds
                ]
                summed_outputs = Add()(feature_preds_dropout)
            else:
                summed_outputs = Add()(feature_preds)

            att_testing_weights = self.encoder.att_weights
            sum_output = summed_outputs + output

            feature_preds_dict = {
                f"{self.shapefuncs[i].name}": pred
                for i, pred in enumerate(feature_preds)
            }

            return {
                "output": sum_output,
                "importances": expl,
                "att_weights": att_testing_weights,
                "raw_embeddings": raw_embeddings,
                **feature_preds_dict,
            }

        else:
            x = self.encoder(inputs)
            x = self.transformer_mlp(x)
            output = self.output_layer(x)
            return output

    def _get_plotting_preds(self, training_dataset=False):
        """
        Get predictions for plotting.

        Args:
            training_data (bool, optional): If True, get predictions for training data; otherwise, get predictions for plotting data. Defaults to False.

        Returns:
            dictionary
        """
        if training_dataset:
            predictions = self.predict(self.training_dataset)
        else:
            predictions = self.predict(self.plotting_dataset)

        for key in ["output", "importances", "att_weights"]:
            predictions.pop(key, None)

        return predictions

    def plot_importances(self, title="importances"):
        visualize_importances(self, title)

    def plot_analysis(self):
        dataset = self._get_dataset(self.data)
        preds = self.predict(dataset)["output"].squeeze()
        visual_analysis(preds, self.data[self.target_name])

    def plot(self, port=8050, smooth=True, k=3, s=0.5):

        plot_NAMformer(self, port=port, smooth=smooth, k=k, s=s)
