import tensorflow as tf
from keras.callbacks import *
from nampy.backend.interpretable_basemodel import AdditiveBaseModel
import numpy as np
from nampy.shapefuncs.transformer_encoder import TransformerEncoder
from nampy.shapefuncs.helper_nets.layers import InterceptLayer, IdentityLayer
from nampy.shapefuncs.helper_nets.helper_funcs import build_cls_mlp
from nampy.shapefuncs.registry import ShapeFunctionRegistry
from nampy.backend.families import *
import warnings
from nampy.visuals.plot_predictions import plot_multi_output
from nampy.visuals.plot_importances import (
    visualize_importances,
    visualize_categorical_importances,
    visualize_heatmap_importances,
)
from nampy.visuals.plot_distributions import visualize_distribution

# Filter out the specific warning by category
warnings.filterwarnings("ignore", category=UserWarning)


class NATTLSS(AdditiveBaseModel):
    def __init__(
        self,
        formula,
        data,
        family,
        feature_dropout=0.001,
        val_split=0.2,
        val_data=None,
        activation="relu",
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        use_column_embedding: bool = False,
        mlp_hidden_factors: list = [2, 4],
        encoder=None,
        explainable=True,
        binning_task="regression",
        batch_size=1024,
        loss="nll",
        **distribution_params,
    ):
        """
        NATTLSS (Neural Adaptive Tabular Learning with Synthetic Sampling) model.

        This class defines the NATTLSS model for tabular data.

        Args:
            formula (str): The formula for feature transformations.
            data: The input data.
            family (str): The distribution family for the target variable.
            dropout (float, optional): The dropout rate for model layers (default is 0.1).
            feature_dropout (float, optional): The feature dropout rate (default is 0.001).
            val_split (float, optional): The validation data split ratio (default is 0.2).
            val_data (None or tuple, optional): Validation data to use instead of splitting.
            activation (str, optional): The activation function for model layers (default is "relu").
            embedding_dim (int, optional): The embedding dimension (default is 32).
            depth (int, optional): The depth of the transformer encoder (default is 4).
            heads (int, optional): The number of attention heads in the transformer encoder (default is 8).
            attn_dropout (float, optional): The attention dropout rate (default is 0.1).
            ff_dropout (float, optional): The feedforward dropout rate (default is 0.1).
            use_column_embedding (bool, optional): True to use column embeddings in the transformer encoder (default is False).
            mlp_hidden_factors (list, optional): The hidden layer factors for MLPs (default is [2, 4]).
            encoder (None or TabTransformerEncoder, optional): Predefined encoder to use (default is None).
            explainable (bool, optional): True to make the model explainable (default is True).

        Attributes:
            formula (str): The formula for feature transformations.
            data: The input data.
            family: The distribution family for the target variable.
            dropout (float): The dropout rate for model layers.
            feature_dropout (float): The feature dropout rate.
            val_data: Validation data to use.
            val_split (float): The validation data split ratio.
            activation (str): The activation function for model layers.
            TRANSFORMER_FEATURES (list): List of transformer features.
            encoder: The transformer encoder.
            transformer_mlp: The transformer MLP layer.
            mlp_final: The final MLP layer.
            feature_nets (list): List of all the feature nets for the numerical features
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

        super(NATTLSS, self).__init__(
            formula=formula,
            data=data,
            feature_dropout=feature_dropout,
            val_data=val_data,
            val_split=val_split,
            batch_size=batch_size,
            binning_task=binning_task,
        )

        # Initialization of parameters
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.use_column_embedding = use_column_embedding
        self.mlp_hidden_factors = mlp_hidden_factors
        self.encoder = encoder
        self.explainable = explainable
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
        self._initialize_transformer()
        self._initialize_transformer_mlp()
        self._initialize_shapefuncs()
        self._initialize_feature_nets()
        self._initialize_output_layer()

        self.model_built = True

        self.print_network_architecture(self.family.param_count)

    def print_network_architecture(self, num_classes):
        print("------------- Network architecture --------------")
        print(
            f"Transformer -> ({self.TRANSFORMER_FEATURES}, dims={self.embedding_dim}, depth={self.depth}, heads={self.heads}) -> MLP(input_dim={self.mlp_input_dim}) -> output dimension={num_classes}"
        )
        for idx, net in enumerate(self.feature_nets):
            print(
                f"{net.name} -> {self.shapefuncs[idx].Network}(features={[inp.name.split(':')[1] for inp in net.inputs]}, n_params={net.count_params()}) -> output dimension={self.shapefuncs[idx].output_dimension}"
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

    def _initialize_transformer(self):
        self.TRANSFORMER_FEATURES = []
        for key, feature in self.feature_information.items():
            if feature["Network"] == "Transformer":
                self.TRANSFORMER_FEATURES += [
                    val["identifier"] for val in feature["inputs"]
                ]

        # Initialise encoder
        if self.encoder:
            pass
        else:
            self.encoder = TransformerEncoder(
                self.TRANSFORMER_FEATURES,
                self.inputs,
                self.embedding_dim,
                self.depth,
                self.heads,
                self.attn_dropout,
                self.ff_dropout,
                self.use_column_embedding,
                explainable=self.explainable,
                data=self.data,
            )

        self.ln = tf.keras.layers.LayerNormalization()

    def _initialize_transformer_mlp(self):
        self.mlp_input_dim = self.embedding_dim * len(self.encoder.categorical)

        self.transformer_mlp = build_cls_mlp(
            self.mlp_input_dim, self.mlp_hidden_factors, self.ff_dropout
        )

        self.mlp_output_layer = tf.keras.layers.Dense(
            self.family.param_count,
            "linear",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.0001),
        )

    def _initialize_shapefuncs(self):
        self.shapefuncs = []
        for key, value in self.feature_information.items():
            if value["Network"] != "Transformer":
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
            if feature["Network"] != "Transformer":
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

    def call(self, inputs, training=False):
        """
        Model call function.

        Args:
            inputs: Input data.
            training (bool, optional): Training mode (default is False).

        Returns:
            dict: Dictionary containing model outputs. -> tf-probability ditsribution
        """
        if self.encoder.explainable:
            x, expl = self.encoder(inputs)

            x = self.ln(x[:, 0, :])
            x = self.transformer_mlp(x)

            outputs = [self.mlp_output_layer(x)]
            feature_preds = [network(inputs) for network in self.feature_nets]
            outputs += feature_preds

            if training:
                outputs = [self.FeatureDropoutLayer(output) for output in outputs]
            summed_outputs = tf.keras.layers.Add()(outputs)

            # Manage the intercept:
            if self.fit_intercept:
                summed_outputs = self.intercept_layer(summed_outputs)

            # Add probability Layer

            p_y = tfp.layers.DistributionLambda(lambda x: self.family(x))(
                summed_outputs
            )

            att_testing_weights = self.encoder.att_weights

            feature_preds_dict = {
                f"{self.feature_nets[i].name}": pred
                for i, pred in enumerate(feature_preds)
            }

            return {
                "output": p_y,
                "importances": expl,
                "att_weights": att_testing_weights,
                "summed_output": summed_outputs,
                **feature_preds_dict,
            }
        else:
            x = self.encoder(inputs)
            x = self.transformer_mlp(x)
            output = p_y
            return output

    def _get_plotting_preds(self, training_data=False):
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

    def plot(self):
        plot_multi_output(self)

    def plot_dist(self):
        preds = self.predict(self.training_dataset)["summed_output"]
        visualize_distribution(self.family, preds)

    def plot_importances(self, title="importances"):
        visualize_importances(self, title)

    def plot_categorical_importances(self, title="Importances"):
        visualize_categorical_importances(self, title)

    def plot_heatmap_importances(self, cat1, cat2):
        visualize_heatmap_importances(self, cat1, cat2)
