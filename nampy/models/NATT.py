import tensorflow as tf
from keras.callbacks import *
import numpy as np
from nampy.backend.interpretable_basemodel import AdditiveBaseModel
from nampy.shapefuncs.transformer_encoder import TransformerEncoder
from nampy.shapefuncs.helper_nets.layers import InterceptLayer, IdentityLayer
from nampy.shapefuncs.registry import ShapeFunctionRegistry
from nampy.visuals.plot_predictions import plot_additive_model
from nampy.visuals.plot_interactive import (
    visualize_regression_predictions,
    visualize_additive_model,
)
from nampy.visuals.plot_importances import (
    visualize_importances,
    visualize_categorical_importances,
    visualize_heatmap_importances,
)
from nampy.visuals.analytics_plot import visual_analysis
import warnings

# Filter out the specific warning by category
warnings.filterwarnings("ignore", category=UserWarning)


class NATT(AdditiveBaseModel):
    def __init__(
        self,
        formula,
        data,
        feature_dropout=0.001,
        val_split=0.2,
        val_data=None,
        classification=False,
        embedding_dim: int = 64,
        depth: int = 4,
        heads: int = 4,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        use_column_embedding: bool = False,
        encoder=None,
        explainable=True,
        output_activation="linear",
        binning_task="regression",
        batch_size=1024,
        transformer_mlp_sizes=[64, 32],
        transformer_mlp_dropout=0.1,
        transformer_mlp_activation="relu",
    ):
        """
        Initialize the NATT model.
        Args:
            formula (str): A formula string for data processing.
            data (pd.DataFrame): Input data as a Pandas DataFrame.
            dropout (float): Dropout rate (default is 0.1).
            feature_dropout (float): Feature dropout rate (default is 0.001).
            val_split (float): Validation data split ratio (default is 0.2).
            val_data (pd.DataFrame, optional): Validation data as a Pandas DataFrame (default is None).
            activation (str): Activation function for the model (default is "relu").
            classification (bool): Whether the problem is a classification task (default is False).
            embedding_dim (int): Dimension of embeddings (default is 32).
            depth (int): Depth of the transformer encoder (default is 4).
            heads (int): Number of attention heads (default is 8).
            attn_dropout (float): Attention dropout rate (default is 0.1).
            ff_dropout (float): Feedforward dropout rate (default is 0.1).
            use_column_embedding (bool): Whether to use column embeddings (default is False).
            encoder (object): Custom encoder for the model (default is None).
            explainable (bool): Whether the encoder is explainable (default is True).
            out_activation (callable): Output layer activation function (default is tf.math.sigmoid).

        Attributes:
            formula (str): The formula for feature transformations.
            data: The input data.
            dropout (float): The dropout rate for model layers.
            feature_dropout (float): The feature dropout rate.
            val_data: Validation data to use.
            val_split (float): The validation data split ratio.
            activation (str): The activation function for model layers.
            classification (bool): True if a classification task, False for regression.
            TRANSFORMER_FEATURES (list): List of transformer features.
            encoder: The transformer encoder.
            transformer_mlp: The transformer MLP layer.
            mlp_final: The final MLP layer.
            out_activation (callable): The output activation function.
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

        Raises:
            ValueError: If the formula is not a string.
        """

        task = "classification" if classification else "regression"
        super(NATT, self).__init__(
            formula=formula,
            data=data,
            feature_dropout=feature_dropout,
            val_data=val_data,
            val_split=val_split,
            batch_size=batch_size,
            binning_task=binning_task,
            task=task,
        )

        # Initialization of parameters
        self.classification = classification
        self.output_activation = output_activation
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.use_column_embedding = use_column_embedding
        self.encoder = encoder
        self.explainable = explainable
        self.model_built = False
        self.transformer_mlp_sizes = transformer_mlp_sizes
        self.transformer_mlp_dropout = transformer_mlp_dropout
        self.transformer_mlp_activation = transformer_mlp_activation

    def build(self, input_shape):
        """
        Build the model. This method should be called before training the model.
        """
        if self.model_built:
            return

        self._initialize_transformer()
        self._initialize_transformer_mlp(self.n_classes)
        self._initialize_shapefuncs(self.n_classes)
        self._initialize_feature_nets()
        self._initialize_output_layer()

        self.model_built = True

        self.print_network_architecture(self.n_classes)

    def print_network_architecture(self, num_classes):
        print("------------- Network architecture --------------")

        # Preparing Transformer line
        transformer_line = (
            f"Transformer -> ("
            f"{self.TRANSFORMER_FEATURES}, "
            f"dims={self.embedding_dim}, "
            f"depth={self.depth}, "
            f"heads={self.heads}"
            f") -> MLP(input_dim={self.embedding_dim}) -> output dimension={num_classes}"
        )
        print(transformer_line)

        # Iterating through feature networks
        for idx, net in enumerate(self.feature_nets):
            # Pre-calculate complex expressions for readability
            features = [inp.name.split(":")[1] for inp in net.inputs]
            net_params = net.count_params()
            output_dim = self.shapefuncs[idx].output_dimension

            # Constructing and printing each feature network line
            feature_net_line = (
                f"{net.name} -> {self.shapefuncs[idx].Network}("
                f"features={features}, "
                f"n_params={net_params}"
                f") -> output dimension={output_dim}"
            )
            print(feature_net_line)

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

    def _initialize_transformer_mlp(self, num_classes):
        self.mlp_input_dim = self.embedding_dim  # * len(self.encoder.categorical)

        mlp_layers = []
        for units in self.transformer_mlp_sizes:
            mlp_layers.append(tf.keras.layers.BatchNormalization()),
            mlp_layers.append(
                tf.keras.layers.Dense(
                    units,
                    activation=self.transformer_mlp_activation,
                )
            )
            mlp_layers.append(tf.keras.layers.Dropout(self.transformer_mlp_dropout))

        self.transformer_mlp = tf.keras.Sequential(mlp_layers)

        self.mlp_output_layer = tf.keras.layers.Dense(
            num_classes,
            "linear",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.0001),
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

    def _initialize_shapefuncs(self, num_classes):
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
                        output_dimension=num_classes,
                    )
                )

    def _initialize_output_layer(self):
        self.output_layer = IdentityLayer(activation=self.output_activation)
        self.FeatureDropoutLayer = tf.keras.layers.Dropout(self.feature_dropout)
        if self.fit_intercept:
            self.intercept_layer = InterceptLayer()

    def call(self, inputs):
        """
        Model call function.

        Args:
            inputs: Input data.
            training (bool, optional): Training mode (default is False).

        Returns:
            dict: Dictionary containing model outputs.
        """
        if self.encoder.explainable:
            x, expl = self.encoder(inputs)

            # only pass on [cls] token
            x = self.ln(x[:, 0, :])
            x = self.transformer_mlp(x)

            outputs = [self.mlp_output_layer(x)]
            feature_preds = [network(inputs) for network in self.feature_nets]
            outputs += feature_preds

            summed_outputs = tf.keras.layers.Add()(outputs)
            # Manage the intercept:
            if self.fit_intercept:
                summed_outputs = self.intercept_layer(summed_outputs)

            output = self.output_layer(summed_outputs)
            att_testing_weights = self.encoder.att_weights

            feature_preds_dict = {
                f"{self.feature_nets[i].name}": pred
                for i, pred in enumerate(feature_preds)
            }

            return {
                "output": output,
                "importances": expl,
                "att_weights": att_testing_weights,
                **feature_preds_dict,
            }
        else:
            x = self.encoder(inputs)
            x = self.transformer_mlp(x)

            self.ms = [self.output_layer(x)]
            self.ms += [network(inputs) for network in self.feature_nets]

            x = tf.keras.layers.Add()(self.ms)

            output = self.output_layer(x)
            return output

    def _get_plotting_preds(self, training_data=False):
        if training_data:
            return self.predict(self.training_dataset)
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

    def plot_single_effects(self, port=8050):
        visualize_regression_predictions(self, port=port)

    def plot_all_effects(self, port=8050):
        visualize_additive_model(self, port=port)

    def plot(self, hist=True):
        plot_additive_model(self, hist=hist)

    def plot_importances(self, title="importances"):
        visualize_importances(self, title)

    def plot_categorical_importances(self, title="Importances"):
        visualize_categorical_importances(self, title)

    def plot_heatmap_importances(self, cat1, cat2):
        visualize_heatmap_importances(self, cat1, cat2)

    def plot_analysis(self):
        dataset = self._get_dataset(self.data)
        preds = self.predict(dataset)["output"].squeeze()
        visual_analysis(preds, self.data[self.target_name])
