import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import *
import numpy as np
from xDL.backend.basemodel import BaseModel
import warnings

# Filter out the specific warning by category
warnings.filterwarnings("ignore", category=UserWarning)


class XplainMLP(BaseModel):
    def __init__(
        self,
        data,
        y,
        val_split=0.2,
        val_data=None,
        activation="relu",
        classification=False,
        dropout: float = 0.5,
        hidden_units=[128, 128, 64],
        output_activation="linear",
        num_classes=1,
        binning_task="regression",
        batch_size=1024,
        num_encoding="PLE",
        n_bins_num=20,
    ):
        """
        Initializes an instance of the XplainMLP class.

        Parameters:
        - data: Input data.
        - y: Target variable.
        - val_split: Validation split ratio (default: 0.2).
        - val_data: Validation data (default: None).
        - activation: Activation function for hidden layers (default: "relu").
        - classification: If True, indicates a classification task (default: False).
        - dropout: Dropout rate for regularization (default: 0.5).
        - hidden_units: List containing the number of units for hidden layers (default: [128, 128, 64]).
        - output_activation: Activation function for output layer (default: "linear").
        - num_classes: Number of classes in classification (default: 1).
        - binning_task: Task type, either "regression" or "classification" (default: "regression").
        - batch_size: Size of batches for training (default: 1024).
        - num_encoding: Encoding method for categorical variables (default: "PLE").
        - n_bins_num: Number of bins for encoding numerical variables (default: 20).

        Raises:
        - ValueError: If num_encoding is not one of the supported encodings.
        """

        super(XplainMLP, self).__init__(
            data=data,
            y=y,
            activation=activation,
            dropout=dropout,
            binning_task=binning_task,
            batch_size=batch_size,
            num_encoding=num_encoding,
            n_bins_num=n_bins_num,
        )

        if binning_task == "regression" and num_classes != 1:
            warnings.warn(
                f"If you perform a classification task, also consider the binning task. Current binning task is set to {binning_task}"
            )

        if binning_task == "classification" and num_classes == 1:
            warnings.warn(
                f"If you perform a regression task, also consider the binning task. Current binning task is set to {binning_task}"
            )

        supported_encodings = [
            "PLE",
            "one_hot",
            "one_hot_discretized",
            "one_hot_constant",
        ]
        if num_encoding not in supported_encodings:
            raise ValueError(
                f"please choose approriate interpretable encoding from {supported_encodings}"
            )

        self.val_data = val_data
        self.val_split = val_split
        self.activation = activation
        self.dropout = dropout
        self.classification = classification
        self.output_activation = output_activation
        self.hidden_units = hidden_units
        self.FEATURES = self.CAT_FEATURES + self.NUM_FEATURES

        num_classes = num_classes

        self.fully_connected = []
        for size in hidden_units:
            self.fully_connected.append(
                tf.keras.layers.Dense(size, activation=activation)
            )
            self.fully_connected.append(tf.keras.layers.Dropout(dropout))

        self.concat_layer = tf.keras.layers.Concatenate(axis=-1)
        self.output_layer = tf.keras.layers.Dense(
            num_classes, activation=output_activation
        )

        self.out_activation = output_activation
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        """
        Defines the forward pass of the model.

        Parameters:
        - inputs: Input data.

        Returns:
        - output: Model predictions.
        """

        x = [inputs[key] for key in self.feature_information.keys()]
        x = self.concat_layer(x)
        for layer in self.fully_connected:
            x = layer(x)
        output = self.output_layer(x)

        return output

    def plot_features(self):
        """
        Plots the features against the target variable and overlays predictions.
        """
        # Get all columns except the target column
        (
            plotting_datasets,
            plotting_data,
        ) = self.datamodule._generate_plotting_data_dense()
        columns_to_plot = [col for col in self.data.columns if col != self.y]

        # Create a separate plot for each column (except target) and overlay predictions
        for col in columns_to_plot:
            preds = self.predict(plotting_datasets[col], verbose=0)

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
