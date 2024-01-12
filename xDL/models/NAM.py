import tensorflow as tf
from xDL.shapefuncs.registry import ShapeFunctionRegistry
from xDL.shapefuncs.helper_nets.layers import InterceptLayer, IdentityLayer
import tensorflow as tf
from keras.layers import Add
import pandas as pd
import numpy as np
from xDL.backend.interpretable_basemodel import AdditiveBaseModel
from xDL.shapefuncs.helper_nets.layers import InterceptLayer, IdentityLayer
import warnings
from scipy.stats import ttest_ind
from xDL.shapefuncs.registry import ShapeFunctionRegistry
from xDL.visuals.plot_predictions import plot_additive_model
from xDL.visuals.plot_interactive import (
    visualize_regression_predictions,
    visualize_additive_model,
)
from xDL.visuals.analytics_plot import visual_analysis

# Filter out the specific warning by category
warnings.filterwarnings("ignore", category=UserWarning)


class NAM(AdditiveBaseModel):
    """
    Neural Additive Model (NAM) Class for fitting a Neural Additive Model.

    Inherits from AdditiveBaseModel and includes methods for building and training
    the model, along with methods for prediction and visualization.
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
        ...
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

        self.classification = classification
        self.output_activation = output_activation
        self.model_built = False

    def build(self, input_shape):
        """
        Build the model. This method should be called before training the model.
        """
        if self.model_built:
            return

        num_classes = self.y.shape[1] if self.classification else 1

        self._initialize_shapefuncs(num_classes)
        self._initialize_feature_nets()
        self._initialize_output_layer()

        self.model_built = True

        print("------------- Network architecture --------------")
        for idx, net in enumerate(self.feature_nets):
            print(
                f"{net.name} -> {self.shapefuncs[idx].Network}(feature={net.name}, n_params={net.count_params()}) -> output dimension={self.shapefuncs[idx].output_dimension}"
            )

    def _initialize_shapefuncs(self, num_classes):
        self.shapefuncs = []
        for _, key in enumerate(self.input_dict):
            class_reference = ShapeFunctionRegistry.get_class(
                self.input_dict[key]["Network"]
            )
            if not class_reference:
                raise ValueError(
                    f"{self.input_dict[key]['Network']} not found in the registry"
                )

            self.shapefuncs.append(
                class_reference(
                    inputs=self.input_dict[key]["Input"],
                    param_dict=self.input_dict[key]["hyperparams"],
                    name=key,
                    identifier=key,
                    output_dimension=num_classes,
                )
            )

    def _initialize_feature_nets(self):
        self.feature_nets = []
        for idx, key in enumerate(self.input_dict.keys()):
            if "<>" in key:
                inputs = [self.inputs[k + "_."] for k in key.split("<>")]
                name = "_._".join(key.split("<>"))
                my_model = self.shapefuncs[idx].build(inputs, name=name)
            else:
                my_model = self.shapefuncs[idx].build(self.inputs[key], name=key)
            self.feature_nets.append(my_model)

    def _initialize_output_layer(self):
        self.output_layer = IdentityLayer(activation=self.output_activation)
        self.FeatureDropoutLayer = tf.keras.layers.Dropout(self.feature_dropout)
        if self.fit_intercept:
            self.intercept_layer = InterceptLayer()

    def call(self, inputs, training=True):
        # Forward pass through feature networks
        feature_preds = [network(inputs) for network in self.feature_nets]

        # Apply dropout if in training mode
        if training:
            feature_preds_dropout = [
                self.FeatureDropoutLayer(output) for output in feature_preds
            ]
            summed_outputs = Add()(feature_preds_dropout)
        else:
            summed_outputs = Add()(feature_preds)

        # Add intercept if applicable
        if self.fit_intercept:
            summed_outputs = self.intercept_layer(summed_outputs)

        # Final output layer
        output = self.output_layer(summed_outputs)

        feature_preds_dict = {
            f"{self.feature_nets[i].name}": pred for i, pred in enumerate(feature_preds)
        }

        return {"output": output, **feature_preds_dict}

    def _get_plotting_preds(self, training_data=False):
        """
        Get predictions for plotting.

        Args:
            training_data (bool, optional): If True, get predictions for training data; otherwise, get predictions for plotting data. Defaults to False.

        Returns:
            dictionary
        """
        if training_data:
            return self.predict(self.training_dataset)["output"]
        else:
            preds = {}

            for net in self.feature_nets:
                if net.name.count("_._") == 1:
                    # Logic for nets with '_._' in their name
                    min_feature0 = np.min(self.data[net.input[0].name])
                    max_feature0 = np.max(self.data[net.input[0].name])
                    min_feature1 = np.min(self.data[net.input[1].name])
                    max_feature1 = np.max(self.data[net.input[1].name])

                    x1_values = np.linspace(min_feature0, max_feature0, 100)
                    x2_values = np.linspace(min_feature1, max_feature1, 100)
                    X1, X2 = np.meshgrid(x1_values, x2_values)

                    # Normalize features
                    X1_normalized = (X1 - min_feature0) / (max_feature0 - min_feature0)
                    X2_normalized = (X2 - min_feature1) / (max_feature1 - min_feature1)

                    # Create tf.data.Dataset from normalized inputs
                    grid_dataset = tf.data.Dataset.from_tensor_slices(
                        {
                            net.input[0].name: X1_normalized.flatten(),
                            net.input[1].name: X2_normalized.flatten(),
                        }
                    ).batch(
                        128
                    )  # Batch size can be adjusted

                    # Generate predictions
                    predictions = []
                    for batch in grid_dataset:
                        batch_predictions = net(batch, training=False).numpy()
                        predictions.extend(batch_predictions)

                    preds[net.name] = {
                        "predictions": np.array(predictions).reshape(X1.shape),
                        "X1": X1,
                        "X2": X2,
                    }
                else:
                    # Standard prediction logic
                    predictions = []
                    for inputs, _ in self.plotting_dataset:
                        prediction = net(inputs, training=False).numpy()
                        predictions.append(prediction)

                    preds[net.name] = np.concatenate(predictions, axis=0)

            return preds

    def plot(self, hist=True):
        plot_additive_model(self, hist=hist)

    def plot_single_effects(self, port=8050):
        visualize_regression_predictions(self, port=port)

    def plot_all_effects(self, port=8050):
        visualize_additive_model(self, port=port)

    def plot_analysis(self):
        dataset = self._get_dataset(self.data)
        preds = self.predict(dataset)["output"].squeeze()
        visual_analysis(preds, self.data[self.target_name])

    def get_significance(self, permutations=5000):
        """
        Computes pseudo permutation significance by comparing the complete prediction
        distribution with a prediction distribution that omits a feature (variable).

        Returns:
            pd.DataFrame: DataFrame containing variable names, t-statistics and p-values.
        """
        preds = self.predict(self.training_dataset)

        # Sum of all predictions (assuming they are numpy arrays or similar)
        all_preds = preds["output"]

        result_df = pd.DataFrame(columns=["feature", "t-stat", "p_value"])

        for feature, feature_preds in preds.items():
            if feature == "output":
                pass
            else:
                preds_without_feature = all_preds - feature_preds

                stat, p_value = ttest_ind(
                    all_preds,
                    preds_without_feature,
                    equal_var=False,
                    permutations=permutations,
                )
                res_df = pd.DataFrame(
                    [[feature, np.round(stat, 4), np.round(p_value, 4)]],
                    columns=["feature", "t-stat", "p_value"],
                )
                result_df = pd.concat([result_df, res_df], ignore_index=True)

        return result_df
