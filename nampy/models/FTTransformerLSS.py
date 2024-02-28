import tensorflow as tf
from keras.callbacks import *
from nampy.backend.black_box_basemodel import BaseModel
from nampy.shapefuncs.transformer_encoder import FTTransformerEncoder
from nampy.shapefuncs.helper_nets.helper_funcs import build_cls_mlp
from nampy.backend.families import *
from nampy.visuals.plot_importances import (
    visualize_importances,
    visualize_categorical_importances,
    visualize_heatmap_importances,
)
from nampy.visuals.plot_distributions import visualize_distribution
from nampy.visuals.analytics_plot import visual_analysis

import warnings

# Filter out the specific warning by category
warnings.filterwarnings("ignore", category=UserWarning)


class FTTransformerLSS(BaseModel):
    def __init__(
        self,
        data,
        y,
        family,
        dropout=0.5,
        val_split=0.2,
        val_data=None,
        activation="relu",
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        mlp_hidden_factors: list = [2, 4],
        hidden_units=[128, 128, 64],
        encoder=None,
        binning_task="regression",
        explainable=True,
        batch_size=1024,
        num_encoding="PLE",
        n_bins=20,
        loss="nll",
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

        super(FTTransformerLSS, self).__init__(
            data=data,
            y=y,
            activation=activation,
            dropout=dropout,
            binning_task=binning_task,
            batch_size=batch_size,
            num_encoding=num_encoding,
            n_bins=n_bins,
        )

        self.val_data = val_data
        self.val_split = val_split
        self.activation = activation
        self.dropout = dropout
        self.explainable = explainable
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
        self.loss_func = loss
        self.family = family

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

        self._initialize_family()
        self._initialize_transformer(num_categories)
        self._initialize_transformer_mlp()
        self._initialize_output_layer()

        self.model_built = True

    def _initialize_family(self):
        if self.family not in [
            "Normal",
            "Logistic",
            "InverseGamma",
            "Poisson",
            "JohnsonSU",
            "Gamma",
            "Beta",
            "Exponential",
            "StudentT",
            "Bernoulli",
            "Chi2",
            "Laplace",
            "Cauchy",
            "Binomial",
            "NegativeBinomial",
            "Uniform",
            "Weibull",
        ]:
            raise ValueError(
                "The family must be in ['Normal', 'Logistic', 'InverseGamma', 'Poisson', 'JohnsonSU', "
                "'Gamma', 'Beta', 'Exponential', 'StudentT', 'Bernoulli', 'Chi2', 'Laplace', 'Cauchy', "
                "'Binomial', 'NegativeBinomial', 'Uniform', 'Weibull']. If you wish further distributions "
                "to be implemented please raise an Issue"
            )

        if self.family == "Normal":
            self.family = Normal()
        elif self.family == "Logistic":
            self.family = Logistic()
        elif self.family == "InverseGamma":
            self.family = InverseGamma()
        elif self.family == "Poisson":
            self.family = Poisson()
        elif self.family == "JohnsonSU":
            self.family = JohnsonSU()
        elif self.family == "Gamma":
            self.family = Gamma()
        elif self.family == "Beta":
            self.family = Beta()
        elif self.family == "Exponential":
            self.family = Exponential()
        elif self.family == "StudentT":
            self.family = StudentT()
        elif self.family == "Bernoulli":
            self.family = Bernoulli()
        elif self.family == "Chi2":
            self.family = Chi2()
        elif self.family == "Laplace":
            self.family = Laplace()
        elif self.family == "cauchy":
            self.family = Cauchy()
        elif self.family == "Binomial":
            self.family = Binomial()
        elif self.family == "NegativeBinomial":
            self.family = NegativeBinomial()
        elif self.family == "Uniform":
            self.family = Uniform()
        elif self.family == "Weibull":
            self.family = Weibull()
        else:
            raise ValueError(
                "Something went wrong with the specified Family. Please documentation or get in contact via an Issue"
            )

    def _initialize_transformer(self, num_categories):
        # Initialise
        if self.encoder:
            pass
        else:
            self.encoder = FTTransformerEncoder(
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

    def _initialize_output_layer(self):
        self.output_layer = tf.keras.layers.Dense(
            self.family.param_count,
            use_bias=False,
        )

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

    def call(self, inputs):
        if self.encoder.explainable:
            x, expl = self.encoder(inputs)

            x = self.ln(x[:, 0, :])
            x = self.transformer_mlp(x)
            output = self.output_layer(x)
            p_y = tfp.layers.DistributionLambda(lambda x: self.family(x))(output)

            att_testing_weights = self.encoder.att_weights

            return {
                "output": p_y,
                "importances": expl,
                "att_weights": att_testing_weights,
                "params": output,
            }

        else:
            x = self.encoder(inputs)
            x = self.transformer_mlp(x)
            output = self.output_layer(x)
            p_y = tfp.layers.DistributionLambda(lambda x: self.family.forward(x))(
                output
            )

            return {
                "output": p_y,
                "params": output,
            }

    def plot_dist(self):
        preds = self.predict(self.training_dataset)["params"]
        visualize_distribution(self.family, preds)

    def plot_importances(self, title="importances"):
        visualize_importances(self, title)

    def plot_categorical_importances(self, title="Importances"):
        visualize_categorical_importances(self, title)

    def plot_heatmap_importances(self, cat1, cat2):
        visualize_heatmap_importances(self, cat1, cat2)

    def plot_analysis(self):
        dataset = self._get_dataset(self.data)
        preds = self.predict(dataset)["params"]
        visual_analysis(preds[:, 0], self.data[self.target_name])
