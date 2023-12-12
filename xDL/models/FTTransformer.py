import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import *
import numpy as np
from xDL.backend.basemodel import BaseModel
from xDL.shapefuncs.transformer_encoder import FTTransformerEncoder
from xDL.shapefuncs.helper_nets.helper_funcs import build_cls_mlp
import seaborn as sns
from xDL.utils.graphing import generate_subplots
import warnings

# Filter out the specific warning by category
warnings.filterwarnings("ignore", category=UserWarning)


class FTTransformer(BaseModel):
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
        num_classes=1,
        binning_task="regression",
        explainable=True,
        batch_size=1024,
        num_encoding="PLE",
        n_bins_num=20,
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

        super(FTTransformer, self).__init__(
            data=data,
            y=y,
            activation=activation,
            dropout=dropout,
            binning_task=binning_task,
            batch_size=batch_size,
            num_encoding=num_encoding,
            n_bins_num=n_bins_num,
        )

        self.val_data = val_data
        self.val_split = val_split
        self.activation = activation
        self.dropout = dropout
        self.classification = classification
        self.explainable = explainable
        self.output_activation = output_activation
        self.hidden_units = hidden_units
        self.FEATURES = self.CAT_FEATURES + self.NUM_FEATURES

        num_classes = num_classes

        num_categories = [len(data[cat].unique()) + 1 for cat in self.CAT_FEATURES]
        numeric_categories = []
        for key in self.NUM_FEATURES:
            if num_encoding == "PLE":
                n_categories = n_bins_num
                # (
                #    next(iter(self.training_dataset))[0][key].numpy().shape[1]
                # )
            else:
                n_categories = n_bins_num  # next(iter(self.training_dataset))[0][key].numpy().max()

            numeric_categories.append(n_categories + 1)

        num_categories += numeric_categories

        # Initialise encoder
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = FTTransformerEncoder(
                categorical_features=self.CAT_FEATURES,
                numerical_features=self.NUM_FEATURES,
                num_categories=num_categories,
                embedding_dim=embedding_dim,
                depth=depth,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                explainable=self.explainable,
                data=self.data,
            )

        mlp_input_dim = embedding_dim * len(self.encoder.features)
        hidden_units = [mlp_input_dim // f for f in mlp_hidden_factors]

        self.final_mlp = build_cls_mlp(mlp_input_dim, mlp_hidden_factors, ff_dropout)

        self.output_layer = tf.keras.layers.Dense(
            num_classes, activation=output_activation
        )

        self.out_activation = output_activation
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        if self.encoder.explainable:
            x, expl = self.encoder(inputs)

            x = self.ln(x[:, 0, :])

            x = self.final_mlp(x)

            x = self.output_layer(x)

            if self.out_activation == "linear":
                output = x
            else:
                output = self.out_activation(x)
            att_testing_weights = self.encoder.att_weights

            return {
                "output": output,
                "importances": expl,
                "att_weights": att_testing_weights,
            }

        else:
            x = self.encoder(inputs)
            x = self.final_mlp(x)
            output = self.output_layer(x)
            return output

    def plot_importances(self, title="Importances"):
        """
        Plot feature importances.

        Args:
            all (bool, optional): If True, plot importances for all features; otherwise, plot average importances (default is False).
            title (str, optional): Title of the plot (default is "Importances").
        """

        importances = self.predict(self.training_dataset, verbose=0)["importances"]

        column_list = []
        for i, feature in enumerate(self.FEATURES):
            column_list.append(feature)
        importances = pd.DataFrame(importances[:, :-1], columns=column_list)
        average_importances = []
        for col_name in self.FEATURES:
            average_importances.append(importances.filter(like=col_name).sum(axis=1))
        importances = pd.DataFrame(
            {
                column_name: column_data
                for column_name, column_data in zip(self.FEATURES, average_importances)
            }
        )
        imps_sorted = importances.mean().sort_values(ascending=False)
        imps_sorted = imps_sorted / sum(imps_sorted)

        plt.figure(figsize=(6, 4))
        ax = imps_sorted.plot.bar()
        for i, p in enumerate(ax.patches):
            ax.annotate(
                str(np.round(p.get_height(), 3)),
                (p.get_x(), p.get_height() * 1.01),
                rotation=90,
                fontsize=12,
            )
            ax.annotate(
                str(imps_sorted.index[i]),
                (p.get_x() + 0.1, 0.01),
                rotation=90,
                fontsize=15,
            )

        ax.xaxis.set_tick_params(labelbottom=False)
        plt.title(title)

        plt.show()

    def plot_categorical_importances(self, title="Importances"):
        """
        Plot categorical feature importances.

        Args:
            title (str, optional): Title of the plot (default is "Importances").
            n_top_categories (int, optional): Number of top categories to consider (default is 5).
        """

        dataset = self._get_dataset(self.data, shuffle=False)
        importances = self.predict(dataset, verbose=0)["importances"]

        column_list = []
        for i, feature in enumerate(self.FEATURES):
            column_list.append(feature)
        importances = pd.DataFrame(importances[:, :-1], columns=column_list)
        average_importances = []
        for col_name in self.FEATURES:
            average_importances.append(importances.filter(like=col_name).sum(axis=1))
        importances = pd.DataFrame(
            {
                column_name: column_data
                for column_name, column_data in zip(self.FEATURES, average_importances)
            }
        )
        result_dict = {}
        imps_sorted = importances.mean().sort_values(ascending=False)

        for category in self.FEATURES:
            unique_vals = self.data[category].unique()
            for val in unique_vals:
                bsc = self.data[self.data[category] == val].index
                imps_value = importances.loc[bsc.values][category].mean()
                result_dict[val] = imps_value
        sort_dict = dict(sorted(result_dict.items(), key=lambda item: item[1]))
        sorted_df = pd.DataFrame([sort_dict])
        sorted_df = sorted_df / np.sum(sorted_df, axis=1)[0]
        imps_sorted = sorted_df.iloc[:, -5:].transpose()
        plt.figure(figsize=(12, 4))
        ax = imps_sorted.plot.bar(legend=None)
        for p in ax.patches:
            ax.annotate(
                str(np.round(p.get_height(), 4)), (p.get_x(), p.get_height() * 1.01)
            )
        plt.title(title)
        plt.show()

    def plot_heatmap_importances(self, cat1, cat2, title="Importances"):
        """
        Plot heatmap of feature importances for two categorical features.

        Args:
            cat1 (str): Name of the first categorical feature.
            cat2 (str): Name of the second categorical feature.
            title (str, optional): Title of the plot (default is "Importances").
        """

        dataset = self._get_dataset(self.data, shuffle=False)
        importances = self.predict(dataset, verbose=0)["importances"]

        column_list = []
        for i, feature in enumerate(self.FEATURES):
            column_list.append(feature)
        importances = pd.DataFrame(importances[:, :-1], columns=column_list)
        average_importances = []
        for col_name in self.FEATURES:
            average_importances.append(importances.filter(like=col_name).sum(axis=1))
        importances = pd.DataFrame(
            {
                column_name: column_data
                for column_name, column_data in zip(self.FEATURES, average_importances)
            }
        )
        result_dict = {}

        unique_vals = self.data[cat1].unique()
        for val1 in unique_vals:
            temp_dict = {}
            bsc1 = self.data[self.data[cat1] == val1].index
            cat_df = self.data.loc[bsc1.values]

            unique_vals2 = self.data[cat2].unique()
            for val2 in unique_vals2:
                bsc2 = cat_df[cat_df[cat2] == val2].index

                cat_values = importances.loc[bsc1.values]
                cat_values = importances.loc[bsc2.values]

                temp_dict[val2] = cat_values[cat2].sum()

            result_dict[val1] = temp_dict

        plotting_importances = pd.DataFrame(result_dict) / np.sum(
            pd.DataFrame(result_dict)
        )
        plotting_importances[plotting_importances == 0] = None
        fig, axs = plt.subplots(
            ncols=2, gridspec_kw=dict(width_ratios=[10, 0.5]), figsize=(4, 4)
        )
        sns.heatmap(plotting_importances, annot=True, fmt=".2%", cbar=False, ax=axs[0])
        fig.colorbar(axs[0].collections[0], cax=axs[1])
        plt.show()

    def plot_features(self):
        # Get all columns except the target column
        (
            plotting_datasets,
            plotting_data,
        ) = self.datamodule._generate_plotting_data_dense()
        columns_to_plot = [col for col in self.data.columns if col != self.y]

        # Create a separate plot for each column (except target) and overlay predictions
        for col in columns_to_plot:
            preds = self.predict(plotting_datasets[col], verbose=0)
            preds = preds["output"].squeeze()

            fig, ax = plt.subplots()

            ax.scatter(
                self.data[col],
                self.data[self.y],
                s=2,
                alpha=0.5,
                color="cornflowerblue",
            )

            if col in self.CAT_FEATURES:
                ax.scatter(
                    plotting_data[col],
                    preds,
                    color="crimson",
                    marker="x",
                )
            else:
                ax.plot(
                    plotting_data[col],
                    preds - np.mean(preds),
                    linewidth=2,
                    color="crimson",
                )

            ax.hist(
                self.data[col],
                bins=30,
                alpha=0.5,
                color="green",
                density=True,
            )

            ax.set_title(f"Plot for {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("target")

            plt.show()
