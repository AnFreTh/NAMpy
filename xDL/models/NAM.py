import tensorflow as tf
from keras.layers import Add
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xDL.backend.basemodel import AdditiveBaseModel
from xDL.shapefuncs.helper_nets.layers import InterceptLayer, IdentityLayer
from xDL.utils.graphing import generate_subplots
import warnings
from scipy.stats import ttest_ind
from xDL.shapefuncs.registry import ShapeFunctionRegistry

# Filter out the specific warning by category
warnings.filterwarnings("ignore", category=UserWarning)


class NAM(AdditiveBaseModel):
    """
    NAML Model Class for fitting a Neural Additive Model
    """

    def __init__(
        self,
        formula,
        data,
        feature_dropout=0.001,
        val_split=0.2,
        batch_size=1024,
        val_data=None,
        output_activation="linear",
        classification=False,
        binning_task="regression",
    ):
        """
        Initialize the NAM model.

        Args:
            formula (str): Formula similar to mgc.
            data (pd.DataFrame): DataFrame that should contain X and y with named columns.
            activation (str, optional): Activation function used in hidden layers (default is "relu").
            dropout (float, optional): Dropout coefficient (default is 0.01).
            feature_dropout (float, optional): Feature dropout rate (default is 0.01).
            val_data: Validation data to use. Defaults to None
            val_split (float): The validation data split ratio if no val-data is given
            output-activation (str): the final layer output activation just as in any neural network: Should be
            adapted to the used loss function
            classification (bool): If true, the output dimensions will be adjusted according to the number of classes

        Attributes:
            formula (str): The formula for feature transformations.
            data: The input data.
            dropout (float): The dropout rate for model layers.
            feature_dropout (float): The feature dropout rate.
            val_data: Validation data to use.
            val_split (float): The validation data split ratio.
            activation (str): The activation function for model layers.
            feature_nets (list): List of all the feature nets for the numerical features
            output_layer (tf.keras.Layer). Convenience Layer that returns the input
            classification (bool): If true, the output dimensions will be adjusted according to the number of classes
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
        """

        super(NAM, self).__init__(
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

        self.formula = formula
        self.val_data = val_data
        self.val_split = val_split
        self.feature_dropout = feature_dropout
        self.classification = classification

        if self.classification:
            num_classes = self.y.shape[1]
        else:
            num_classes = 1

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
                        output_dimension=num_classes,
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
        for idx, net in enumerate(self.feature_nets):
            print(
                f"{net.name} -> {shapefuncs[idx].Network}(feature={net.name}, n_params={net.count_params()}) -> output dimension={shapefuncs[idx].output_dimension}"
            )

        self.output_layer = IdentityLayer(activation=output_activation)
        self.FeatureDropoutLayer = tf.keras.layers.Dropout(self.feature_dropout)

    def call(self, inputs, training=False):
        """
        Call function for the NAM model.

        Args:
            inputs: Input tensors.

        Returns:
            Output tensor.
        """

        # outputs = [network(inputs) for network in self.feature_nets]
        outputs = [network(inputs) for network in self.feature_nets]

        if training:
            outputs = [self.FeatureDropoutLayer(output) for output in outputs]
        summed_outputs = Add()(outputs)
        # Manage the intercept:
        if self.fit_intercept:
            summed_outputs = self.intercept_layer(summed_outputs)
        output = self.output_layer(summed_outputs)
        return output

    def _get_training_preds(self, mean=True):
        """
        Get training set predictions.

        Args:
            mean (bool, optional): If True, return mean predictions; otherwise, return individual predictions. Defaults to True.

        Returns:
            numpy.ndarray: Predictions for the training set.
        """
        preds = self.predict(self.training_dataset)
        if mean:
            preds = np.mean(preds, axis=1)  # Calculate mean prediction

        return preds

    def _get_validation_preds(self):
        """
        Get validation set predictions.

        Returns:
            numpy.ndarray: Predictions for the validation set.
        """
        preds = self.predict(self.validation_dataset)
        return preds

    def _get_plotting_preds(self, training_data=False):
        """
        Get predictions for plotting.

        Args:
            training_data (bool, optional): If True, get predictions for training data; otherwise, get predictions for plotting data. Defaults to False.

        Returns:
            list of numpy.ndarray: List of predictions for plotting.
        """
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

    def feature_preds(self, X):
        """
        Get predictions for specific features.

        Args:
            X (numpy.ndarray or pd.DataFrame): Input data for features.

        Returns:
            list of numpy.ndarray: List of predictions for the given input data.
        """
        dataset = self._get_dataset(X)
        preds = [net.predict(dataset, verbose=0) for net in self.feature_nets]
        return preds

    def plot(self):
        """
        Plot the model's predictions.
        """

        # Generate subplots for visualization
        fig, axes = generate_subplots(len(self.input_dict), figsize=(10, 12))

        # Get plotting predictions
        preds = self._get_plotting_preds()

        for idx, ax in enumerate(axes.flat):
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

                        cs = ax.contourf(
                            X1,
                            X2,
                            predictions.reshape(X1.shape),
                            extend="both",
                            levels=25,
                        )
                        ax.scatter(
                            self.data[self.feature_nets[idx].input[0].name],
                            self.data[self.feature_nets[idx].input[1].name],
                            c="black",
                            label="Scatter Points",
                            s=5,
                        )

                        plt.colorbar(cs, label="Predictions")
                        ax.set_xlabel(self.feature_nets[idx].input[0].name)
                        ax.set_ylabel(self.feature_nets[idx].input[1].name)
                    else:
                        continue

            except TypeError:
                # Scatter plot of training data
                ax.scatter(
                    self.data[self.feature_nets[idx].name],
                    self.data[self.y],  # - np.mean(self.data[self.y]),
                    s=2,
                    alpha=0.5,
                    color="cornflowerblue",
                )
                # Line plot of predictions
                if self.feature_nets[idx].name in self.CAT_FEATURES:
                    ax.scatter(
                        self.plotting_data[self.feature_nets[idx].name],
                        preds[idx].squeeze(),
                        color="crimson",
                        marker="x",
                    )

                else:
                    ax.plot(
                        self.plotting_data[self.feature_nets[idx].name],
                        preds[idx].squeeze(),
                        linewidth=2,
                        color="crimson",
                    )
                # Data density histogram
                ax.hist(
                    self.data[self.feature_nets[idx].name],
                    bins=30,
                    alpha=0.5,
                    color="green",
                    density=True,
                )
                ax.set_ylabel(self.y)
                ax.set_xlabel(self.feature_nets[idx].name)

        plt.tight_layout(pad=0.4, w_pad=0.3)
        plt.show()

    def get_significance(self, permutations=5000):
        """computes pseudo permutation significance by comparing the complete prediction distribution with
          a prediction distribution that omits a feature (variable)
          So far without the intercept

        Returns:
            pd.DataFrame: DataFrame vontaining variable names, t-statistics and p-values
        """
        preds = self._get_plotting_preds(training_data=True)

        all_preds = np.sum(preds, axis=0)

        cols = list(self.input_dict.keys())

        result_df = pd.DataFrame(columns=["feature", "t-stat", "p_value"])

        for i in range(len(preds)):
            preds_without_feature_i = all_preds - preds[i]

            stat, p_value = ttest_ind(
                all_preds,
                preds_without_feature_i,
                equal_var=False,
                permutations=permutations,
            )
            res_df = pd.DataFrame(
                [[cols[i], np.round(stat, 4), np.round(p_value, 4)]],
                columns=["feature", "t-stat", "p_value"],
            )
            result_df = pd.concat([result_df, res_df], ignore_index=True)
        return result_df
