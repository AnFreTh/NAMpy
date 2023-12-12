import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf
from keras.layers import Add
from xDL.shapefuncs.helper_nets.layers import InterceptLayer, IdentityLayer
import pandas as pd
import warnings
from scipy.stats import ttest_ind
from xDL.shapefuncs.registry import ShapeFunctionRegistry

tfd = tfp.distributions
import numpy as np

from xDL.backend.basemodel import AdditiveBaseModel
from xDL.backend.families import *


import warnings

# Filter out the specific warning by category
warnings.filterwarnings("ignore", category=UserWarning)


class NAMLSS(AdditiveBaseModel):
    def __init__(
        self,
        formula,
        data,
        family,
        feature_dropout=0.01,
        val_split=0.2,
        val_data=None,
        batch_size=1024,
        binning_task="regression",
    ):
        """
        Initialize the NAMLSS model.

        Args:
            formula (str): Formula similar to mgc.
            data (pd.DataFrame): DataFrame that should contain X and y with named columns.
            family (str): Probability distribution family for the response variable. Must be one of ['Normal', 'Logistic', 'InverseGamma', 'Poisson', 'JohnsonSU', 'Gamma'].
            activation (str, optional): Activation function used in hidden layers (default is "relu").
            dropout (float, optional): Dropout coefficient (default is 0.01).
            feature_dropout (float, optional): Feature dropout rate (default is 0.01).

        Attributes:
            formula (str): The formula for feature transformations.
            data: The input data.
            family: The distribution family for the target variable.
            dropout (float): The dropout rate for model layers.
            feature_dropout (float): The feature dropout rate.
            val_data: Validation data to use.
            val_split (float): The validation data split ratio.
            activation (str): The activation function for model layers.
            feature_nets (list): List of all the feature nets for the numerical features
            output_layer (tf.keras.Layer). Convenience Layer that returns the input
            training_dataset (tf.data.Dataset): training dataset containing the transformed inputs
            validation_dataset (tf.data.Dataset): validation dataset containing the transformed inputs
            plotting_dataset (tf.data.Dataset): dataset containing the transformed inputs adapted for creating the plots
            inputs (dict): dictionary with all tf.keras.Inputs -> mapping from feature name to feature
            input_dict (dict): dictionary containg all the model specification -> mapping from feature to network type, network size, name, input
            NUM_FEATURES (list): Convenience list with all numerical features
            CAT_FEATURES (list): Convenience list with all categorical features

            named_feature_nets (list): List of named feature networks.
            y (str): Name of the target variable.
            feature_names (list): List of feature names.
            fit_intercept (bool): Whether to fit an intercept.
            hidden_layer_sizes (list): List of hidden layer sizes.

        Raises:
            ValueError: If the formula is not a string or if the family is not one of the supported distributions.
        """

        super(NAMLSS, self).__init__(
            formula=formula,
            data=data,
            feature_dropout=feature_dropout,
            val_data=val_data,
            val_split=val_split,
            batch_size=batch_size,
            binning_task=binning_task,
        )

        if not isinstance(formula, str):
            raise ValueError("The formula must be a string.")

        if family not in [
            "Normal",
            "Logistic",
            "InverseGamma",
            "Poisson",
            "JohnsonSU",
            "Gamma",
        ]:
            raise ValueError(
                "The family must be in ['Normal', 'Logistic', 'InverseGamma', 'Poisson', 'JohnsonSU', 'Gamma']. If you wish further distributions to be implemented please raise an Issue"
            )

        self.formula = formula

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

        self.feature_dropout = feature_dropout

        if self.fit_intercept:
            self.intercept_layer = InterceptLayer()

        shapefuncs = []
        for _, key in enumerate(self.input_dict):
            class_reference = ShapeFunctionRegistry.get_class(
                self.input_dict[key]["Network"]
            )
            if class_reference:
                shapefuncs.append(
                    class_reference(
                        inputs=self.input_dict[key]["Input"],
                        param_dict=self.input_dict[key]["hyperparams"],
                        name=key,
                        identifier=key,
                        output_dimension=self.family.dimension,
                    )
                )
            else:
                raise ValueError(
                    f"{self.input_dict[key]['Network']} not found in the registry"
                )

        self.feature_nets = []
        for idx, key in enumerate(self.input_dict.keys()):
            if "<>" in key:
                keys = key.split("<>")
                inputs = [self.inputs[k] for k in keys]
                name = "_._".join(keys)
                my_model = shapefuncs[idx].build(inputs, name=name)
            else:
                my_model = shapefuncs[idx].build(self.inputs[key], name=key)

            self.feature_nets.append(my_model)

        print("------------- Network architecture --------------")
        print(
            f"chosen distribution: {self.family._name}, distributional parameters: {self.family.param_names}"
        )
        for idx, net in enumerate(self.feature_nets):
            print(
                f"{net.name} -> {shapefuncs[idx].Network}(feature={net.name}, n_params={net.count_params()}) -> output dimension={self.family.dimension}"
            )

        self.output_layer = IdentityLayer(activation="linear")
        self.FeatureDropoutLayer = tf.keras.layers.Dropout(self.feature_dropout)

    def NegativeLogLikelihood(self, y_true, y_hat):
        """Negative LogLIkelihood Loss function

        Args:
            y_true (_type_): True Labels
            y_hat (_type_): Predicted Distribution

        Returns:
            _type_: negative Log likelihood of respective input distribution
        """
        return -y_hat.log_prob(tf.cast(y_true, dtype=tf.float32))

    def call(self, inputs, training=False):
        """
        Call function for the NAMLSS model.

        Args:
            inputs: Input tensors.

        Returns:
            Output tensor.
        """
        outputs = [network(inputs) for network in self.feature_nets]
        if training:
            outputs = [self.FeatureDropoutLayer(output) for output in outputs]
        summed_outputs = Add()(outputs)

        # Manage the intercept:
        if self.fit_intercept:
            summed_outputs = self.intercept_layer(summed_outputs)
        output = self.output_layer(summed_outputs)

        # Add probability Layer
        p_y = tfp.layers.DistributionLambda(lambda x: self.family.forward(x))(output)
        return p_y

    def _get_training_preds(self, mean=False):
        preds = [
            net.predict(self.training_dataset, verbose=0) for net in self.feature_nets
        ]
        preds = np.sum(preds, axis=0)
        preds = self.family.transform(preds)

        if mean:
            preds = preds[:, 0]

        return preds

    def _get_validation_preds(self):
        preds = [
            net.predict(self.validation_dataset, verbose=0) for net in self.feature_nets
        ]
        preds = np.sum(preds, axis=0)
        preds = self.family.transform(preds)

        return preds

    def _get_plotting_preds(self, training_data=False, mean=False):
        if training_data:
            preds = [
                net.predict(self.training_dataset, verbose=0)
                for net in self.feature_nets
            ]
        else:
            preds = [
                net.predict(self.plotting_dataset, verbose=0)
                for net in self.feature_nets
            ]

        return preds

    def plot_dist(self):
        preds = self._get_training_preds()
        self.family._plot_dist(preds)

    def plot(self, levels=50):
        fig, ax = plt.subplots(
            len(self.input_dict), self.family.dimension, figsize=(10, 12)
        )

        preds = self._get_plotting_preds()

        for idx in range(len(self.input_dict)):
            try:
                len(self.feature_nets[idx].input)
                if (
                    self.feature_nets[idx].input[0].dtype != float
                    or self.feature_nets[idx].input[1].dtype != float
                ):
                    continue
                else:
                    if len(self.feature_nets[idx].input) == 2:
                        min_feature0 = np.min(
                            self.data[self.feature_nets[idx].input[0].name]
                        )
                        max_feature0 = np.max(
                            self.data[self.feature_nets[idx].input[0].name]
                        )
                        min_feature1 = np.min(
                            self.data[self.feature_nets[idx].input[1].name]
                        )
                        max_feature1 = np.max(
                            self.data[self.feature_nets[idx].input[1].name]
                        )
                        x1_values = np.linspace(min_feature0, max_feature0, 100)
                        x2_values = np.linspace(min_feature1, max_feature1, 100)
                        X1, X2 = np.meshgrid(x1_values, x2_values)
                        grid_dataset = tf.data.Dataset.from_tensor_slices((X1, X2))

                        def add_feature_names_and_normalize(x1, x2):
                            # Normalize Longitude and Latitude features
                            feature0_normalized = (x1 - min_feature0) / (
                                max_feature0 - min_feature0
                            )
                            feature1_normalized = (x2 - min_feature1) / (
                                max_feature1 - min_feature1
                            )

                            return {
                                self.feature_nets[idx]
                                .input[0]
                                .name: feature0_normalized,
                                self.feature_nets[idx]
                                .input[1]
                                .name: feature1_normalized,
                            }

                        grid_dataset = grid_dataset.map(add_feature_names_and_normalize)
                        predictions = self.feature_nets[idx].predict(grid_dataset)

                        for j in range(self.family.dimension):
                            cs = ax[idx, j].contourf(
                                X1,
                                X2,
                                predictions[:, j].reshape(X1.shape),
                                extend="both",
                                levels=levels,
                            )
                            ax[idx, j].scatter(
                                self.data[self.feature_nets[idx].input[0].name],
                                self.data[self.feature_nets[idx].input[1].name],
                                c="black",
                                label="Scatter Points",
                                s=5,
                            )

                            plt.colorbar(cs, label="Predictions")
                            ax[idx, j].set_xlabel(self.feature_nets[idx].input[0].name)
                            ax[idx, j].set_ylabel(self.feature_nets[idx].input[1].name)
                    else:
                        continue

            except TypeError:
                ax[idx, 0].scatter(
                    self.data[self.feature_nets[idx].name],
                    self.data[self.y],  # - np.mean(self.data[self.y]),
                    s=2,
                    alpha=0.5,
                    color="cornflowerblue",
                )

                for j in range(self.family.dimension):
                    if self.feature_nets[idx].name in self.CAT_FEATURES:
                        ax[idx, j].scatter(
                            self.plotting_data[self.feature_nets[idx].name],
                            preds[idx][:, j],
                            linewidth=2,
                            color="crimson",
                            marker="x",
                        )
                        ax[idx, j].set_title(
                            f"Effect of {self.feature_nets[idx].name} on theta_{[j+1]}"
                        )
                        ax[idx, j].set_ylabel(f"theta_{[j+1]}")
                        ax[idx, j].grid(True)
                        # Data density histogram

                    else:
                        ax[idx, j].plot(
                            self.plotting_data[self.feature_nets[idx].name],
                            preds[idx][:, j],
                            linewidth=2,
                            color="crimson",
                        )
                        ax[idx, j].set_title(
                            f"Effect of {self.feature_nets[idx].name} on theta_{[j+1]}"
                        )
                        ax[idx, j].set_ylabel(f"theta_{[j+1]}")
                        ax[idx, j].grid(True)
                        # Data density histogram
        plt.tight_layout(pad=0.4, w_pad=0.3)
        plt.show()
