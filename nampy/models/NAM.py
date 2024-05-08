import tensorflow as tf
from nampy.shapefuncs.registry import ShapeFunctionRegistry
from nampy.shapefuncs.helper_nets.layers import InterceptLayer, IdentityLayer
import tensorflow as tf
from keras.layers import Add
import pandas as pd
import numpy as np
from nampy.backend.interpretable_basemodel import AdditiveBaseModel
from nampy.shapefuncs.helper_nets.layers import InterceptLayer, IdentityLayer
import warnings
from scipy.stats import ttest_ind
from nampy.shapefuncs.registry import ShapeFunctionRegistry
from nampy.visuals.plot_predictions import plot_additive_model
from nampy.visuals.plot_interactive import (
    visualize_regression_predictions,
    visualize_additive_model,
)
from nampy.visuals.plot_predictions import (
    plot_multi_output,
)
from nampy.visuals.analytics_plot import visual_analysis

# Filter out the specific warning by category
warnings.filterwarnings("ignore", category=UserWarning)


class NAM(AdditiveBaseModel):
    """
    Neural Additive Model (NAM) for interpretable machine learning, extending the AdditiveBaseModel
    class. It leverages the flexibility of neural networks for feature representation while maintaining
    the interpretability of additive models. This class facilitates building, training, and visualizing
    NAMs, which are particularly useful for regression and classification tasks.

    Attributes:
        formula (str): The formula specifying the model structure.
        data (pd.DataFrame): The dataset to be used for the model.
        feature_dropout (float): Dropout rate for feature regularization, used to prevent overfitting. Default is 0.001.
        val_split (float): Proportion of data to be used for validation, to monitor and prevent overfitting. Default is 0.2.
        batch_size (int): Batch size for training, impacting the speed and memory usage during model training. Default is 1024.
        val_data (pd.DataFrame or None): Optional separate validation dataset. If provided, it is used instead of splitting `data` based on `val_split`. Default is None.
        output_activation (str): Activation function for the output layer, determining the form of the model output. Default is "linear".
        classification (bool): Indicates whether the model is for classification (True) or regression (False). Default is False.
        binning_task (str): Specifies the task type for binning features, which can influence the model's interpretability and performance. Default is "regression".
        model_built (bool): Flag to check if the model has been built. This is used internally to prevent redundant model building.

    Methods:
        build: Constructs the neural network architecture for the NAM.
        call: Defines the forward pass for the NAM.
        _initialize_shapefuncs: Initializes shape functions for each feature based on the registry.
        _initialize_feature_nets: Sets up the neural networks for representing the features.
        _initialize_output_layer: Configures the output layer of the model.
        _get_plotting_preds: Prepares predictions for visualization purposes.
        plot: Visualizes the model's predictions and feature effects.
        _plot_single_effects: Generates plots for individual feature effects.
        _plot_all_effects: Produces plots for all possible feature effects and interactions.
        plot_analysis: Conducts and visualizes a statistical analysis of the model's predictions.
        pseudo_significance: Performs a pseudo permutation significance test to evaluate the importance of features.
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
        Initializes the NAM model with the specified configuration.

        Parameters:
            formula (str): The formula specifying the model structure.
            data (pd.DataFrame): The dataset to be used for the model.
            feature_dropout (float, optional): Dropout rate for feature regularization. Default is 0.001.
            val_split (float, optional): Proportion of data to be used for validation. Default is 0.2.
            batch_size (int, optional): Batch size for training. Default is 1024.
            val_data (pd.DataFrame or None, optional): Optional separate validation dataset. Default is None.
            output_activation (str, optional): Activation function for the output layer. Default is "linear".
            classification (bool, optional): Specifies if the model is for a classification task. Default is False.
            binning_task (str, optional): Specifies the task type for binning features. Default is "regression".
        """

        task = "classification" if classification else "regression"
        super(NAM, self).__init__(
            formula=formula,
            data=data,
            feature_dropout=feature_dropout,
            val_data=val_data,
            val_split=val_split,
            binning_task=binning_task,
            task=task,
            batch_size=batch_size
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

        self._initialize_shapefuncs(self.n_classes)
        self._initialize_feature_nets()
        self._initialize_output_layer()

        self.model_built = True

        print("------------- Network architecture --------------")
        for idx, net in enumerate(self.feature_nets):
            print(
                f"{net.name} -> {self.shapefuncs[idx].Network}(features={[inp.name.split(':')[1] for inp in net.inputs]}, n_params={net.count_params()}) -> output dimension={self.shapefuncs[idx].output_dimension}"
            )

    def _initialize_shapefuncs(self, num_classes):
        self.shapefuncs = []
        for key, value in self.feature_information.items():
            class_reference = ShapeFunctionRegistry.get_class(value["Network"])
            if not class_reference:
                raise ValueError(
                    f"specified network {value['Network']} for {key} not found in the registry"
                )

            identifier = [val["identifier"] for val in value["inputs"]]
            inps = [self.inputs[val] for val in identifier]
            params = {}
            params.update(value["shapefunc_args"])
            params.update({"Network": value["Network"]})
            self.shapefuncs.append(
                class_reference(
                    inputs=inps,
                    param_dict=params,
                    name=key,
                    identifier=key,
                    output_dimension=num_classes,
                )
            )

    def _initialize_feature_nets(self):
        self.feature_nets = []
        for idx, value in enumerate(self.feature_information.items()):
            key = value[0]
            feature = value[1]
            identifier = [val["identifier"] for val in feature["inputs"]]
            inps = [self.inputs[val] for val in identifier]
            if len(inps) > 1:
                my_model = self.shapefuncs[idx].build(inps, name=key)
            else:
                my_model = self.shapefuncs[idx].build(inps[0], name=key)

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
                if len(net.inputs) == 2:
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

    def plot(self, hist=True, port=8050, interactive=True, interaction=True):
        """NAM visualization function

        Args:
            port (int, optional): port used for dash/plotly. Defaults to 8050.
            interactive (bool, optional): if true, a dash/plotly plot is created. Defaults to True.
            interaction (bool, optional): if true, all pairwise feature interactions are plotted. Defaults to True.
        """
        if interactive:
            if interaction:
                self._plot_all_effects(port=port)
            else:
                self._plot_single_effects(port=port)
        else:
            if self.n_classes > 1:
                plot_multi_output(self, hist=hist, n_classes=self.n_classes)
            else:
                plot_additive_model(self, hist=hist)

    def _plot_single_effects(self, port=8050):
        visualize_regression_predictions(self, port=port)

    def _plot_all_effects(self, port=8050):
        visualize_additive_model(self, port=port)

    def plot_analysis(self):
        dataset = self._get_dataset(self.data)
        preds = self.predict(dataset)["output"].squeeze()
        visual_analysis(preds, self.data[self.target_name])

    def pseudo_significance(self, permutations=5000):
        """
        Computes pseudo permutation significance by comparing the complete prediction
        distribution with a prediction distribution that omits a feature (variable).

        Returns:
            pd.DataFrame: DataFrame containing variable names, t-statistics and p-values.
        """
        preds = self.predict(self.training_dataset)

        # Sum of all predictions
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
