import tensorflow_probability as tfp
import tensorflow as tf
from keras.layers import Add
from nampy.shapefuncs.helper_nets.layers import InterceptLayer
import warnings
from nampy.shapefuncs.registry import ShapeFunctionRegistry
import numpy as np
from nampy.backend.interpretable_basemodel import AdditiveBaseModel
from nampy.backend.families import *
from nampy.visuals.plot_predictions import (
    plot_additive_distributional_model,
)
from nampy.visuals.plot_distributional_interactive import (
    visualize_distributional_regression_predictions,
    visualize_distributional_additive_model,
)
from nampy.visuals.plot_distributions import visualize_distribution
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
        loss="nll",
        **distribution_params,
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

        self.model_built = False
        self.family = family
        self.loss_func = loss
        self.distributional_params = distribution_params

    def build(self, input_shape):
        """
        Build the model. This method should be called before training the model.
        """
        if self.model_built:
            return

        self._initialize_family()
        self._initialize_shapefuncs()
        self._initialize_feature_nets()
        self._initialize_output_layer()

        self.model_built = True

        print("------------- Network architecture --------------")
        for idx, net in enumerate(self.feature_nets):
            print(
                f"{net.name} -> {self.shapefuncs[idx].Network}(feature={[inp.name.split(':')[1] for inp in net.inputs]}, n_params={net.count_params()}) -> output dimension={self.shapefuncs[idx].output_dimension}"
            )

    def _initialize_family(self):
        distribution_classes = {
            "Normal": Normal,
            "Logistic": Logistic,
            "InverseGamma": InverseGamma,
            "Poisson": Poisson,
            "JohnsonSU": JohnsonSU,
            "Gamma": Gamma,
            "Beta": Beta,
            "Exponential": Exponential,
            "StudentT": StudentT,
            "Bernoulli": Bernoulli,
            "Chi2": Chi2,
            "Laplace": Laplace,
            "Cauchy": Cauchy,
            "Binomial": Binomial,
            "NegativeBinomial": NegativeBinomial,
            "Uniform": Uniform,
            "Weibull": Weibull,
        }

        if self.family in distribution_classes:
            # Pass additional distribution_params to the constructor of the distribution class
            self.family = distribution_classes[self.family](
                **self.distributional_params
            )
        else:
            raise ValueError("Unsupported family: {}".format(self.family))

    def _initialize_shapefuncs(self):
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
                    output_dimension=self.family.param_count,
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
        self.FeatureDropoutLayer = tf.keras.layers.Dropout(self.feature_dropout)
        if self.fit_intercept:
            self.intercept_layer = InterceptLayer()

    def Loss(self, y_true, y_hat):
        """Builds the Loss function for NAMLSS, one of NegativeLogLikelihood or KKL-Divergence

        Args:
            y_true (_type_): True Labels
            y_hat (_type_): Predicted Distribution

        Returns:
            _type_: negative Log likelihood of respective input distribution
        """
        # return self.family.negative_log_likelihood(y_true, y_hat)
        if self.loss_func:
            return -y_hat.log_prob(tf.cast(y_true, dtype=tf.float32))
        elif self.loss_func == "kld":
            return self.family.KL_divergence(y_true, y_hat)
        else:
            raise ValueError

    def call(self, inputs, training=False):
        """
        Call function for the NAMLSS model.

        Args:
            inputs: Input tensors.

        Returns:
            Output tensor.
        """
        feature_preds = [network(inputs) for network in self.feature_nets]
        if training:
            feature_preds_dropout = [
                self.FeatureDropoutLayer(output) for output in feature_preds
            ]
            summed_outputs = Add()(feature_preds_dropout)
        else:
            summed_outputs = Add()(feature_preds)

        # Manage the intercept:
        if self.fit_intercept:
            summed_outputs = self.intercept_layer(summed_outputs)

            # Add probability Layer
        p_y = tfp.layers.DistributionLambda(lambda x: self.family(x))(summed_outputs)
        feature_preds_dict = {
            f"{self.feature_nets[i].name}": pred for i, pred in enumerate(feature_preds)
        }

        return {"output": p_y, "summed_output": summed_outputs, **feature_preds_dict}

    def _get_plotting_preds(self, training_data=False):
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
                        "predictions": np.array(predictions).reshape(100, 100, 2),
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

    def plot_dist(self):
        preds = self.predict(self.training_dataset)["summed_output"]
        visualize_distribution(self.family, preds)

    def plot(self, port=8050, interactive=True, interaction=True):
        """NAMLSS visualization function

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
            plot_additive_distributional_model(self)

    def _plot_single_effects(self, port=8505):
        visualize_distributional_regression_predictions(self, port=port)

    def _plot_all_effects(self, port=8505):
        visualize_distributional_additive_model(self, port=port)
