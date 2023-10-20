import tensorflow as tf
from keras.callbacks import *
from sklearn.model_selection import KFold
from xDL.utils.data_utils import *
from xDL.backend.basemodel import AdditiveBaseModel
from xDL.utils.graphing import *
from xDL.backend.transformer_encoder import TransformerEncoder
from xDL.backend.helper_nets.featurenets import *
from xDL.backend.families import *
import warnings
import seaborn as sns

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
            activation=activation,
            feature_dropout=feature_dropout,
            val_data=val_data,
            val_split=val_split,
            batch_size=batch_size,
            binning_task=binning_task,
        )

        if not isinstance(formula, str):
            raise ValueError(
                "The formula must be a string. See patsy for documentation"
            )

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
        self.val_data = val_data
        self.val_split = val_split
        self.activation = activation
        self.feature_dropout = feature_dropout

        self.TRANSFORMER_FEATURES = []
        for key, feature in self.input_dict.items():
            if feature["Network"] == "Transformer":
                self.TRANSFORMER_FEATURES += [input.name for input in feature["Input"]]

        # Initialise encoder
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = TransformerEncoder(
                self.TRANSFORMER_FEATURES,
                self.inputs,
                embedding_dim,
                depth,
                heads,
                attn_dropout,
                ff_dropout,
                use_column_embedding,
                explainable=explainable,
                data=self.data,
            )

        mlp_input_dim = embedding_dim * len(self.encoder.categorical)

        self.transformer_mlp = build_cls_mlp(
            mlp_input_dim, mlp_hidden_factors, ff_dropout
        )
        # self.output_layer = Dense(out_dim, activation=out_activation)
        self.mlp_final = build_cls_mlp(mlp_input_dim, mlp_hidden_factors, ff_dropout)

        ####################################
        self.output_layer = tf.keras.layers.Dense(
            self.family.dimension, "linear", use_bias=False
        )

        # self.num_mlp = [build_shape_funcs() for _ in range(len(self.NUM_FEATURES))]

        self.feature_nets = []

        if self.fit_intercept:
            self.intercept_layer = InterceptLayer()

        for _, key in enumerate(self.input_dict):
            if self.input_dict[key]["Network"] == "MLP":
                self.feature_nets.append(
                    eval(self.input_dict[key]["Network"])(
                        inputs=self.input_dict[key]["Input"],
                        param_dict=self.input_dict[key]["hyperparams"],
                        name=key,
                        output_dimension=self.family.dimension,
                    )
                )

        self.ln = tf.keras.layers.LayerNormalization()
        self.FeatureDropoutLayer = tf.keras.layers.Dropout(self.feature_dropout)
        self.identity_layer = IdentityLayer(activation="linear")

    def NegativeLogLikelihood(self, y_true, y_hat):
        """Negative LogLIkelihood Loss function

        Args:
            y_true (_type_): True Labels
            y_hat (_type_): Predicted Distribution

        Returns:
            _type_: negative Log likelihood of respective input distribution
        """
        return -y_hat.log_prob(tf.cast(y_true, dtype=tf.float32))

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
            x = self.mlp_final(x)
            x = self.transformer_mlp(x)

            outputs = [self.output_layer(x)]
            outputs += [network(inputs) for network in self.feature_nets]

            if training:
                outputs = [self.FeatureDropoutLayer(output) for output in outputs]
            summed_outputs = tf.keras.layers.Add()(outputs)

            # Manage the intercept:
            if self.fit_intercept:
                summed_outputs = self.intercept_layer(summed_outputs)
            output = self.identity_layer(summed_outputs)

            # Add probability Layer
            p_y = tfp.layers.DistributionLambda(lambda x: self.family.forward(x))(
                output
            )

            att_testing_weights = self.encoder.att_weights

            return {
                "output": p_y,
                "importances": expl,
                "att_weights": att_testing_weights,
            }
        else:
            x = self.encoder(inputs)
            x = self.mlp_final(x)
            output = p_y
            return output

    def plot(self):
        """
        Plot model predictions.

        Returns:
            None
        """
        fig, ax = plt.subplots(
            len(self.input_dict) - 1, self.family.dimension, figsize=(10, 12)
        )

        preds = self._get_plotting_preds()

        for idx in range(len(self.input_dict) - 1):
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
                                levels=25,
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
                    self.data[self.y] - np.mean(self.data[self.y]),
                    s=2,
                    alpha=0.5,
                    color="cornflowerblue",
                )
                for j in range(self.family.dimension):
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

    def plot_importances(self, title="Importances"):
        """
        Plot feature importances.

        Args:
            all (bool, optional): True to plot all importances (default is False).
            title (str, optional): The title of the plot (default is "Importances").

        Returns:
            None
        """
        importances = self.predict(self.training_dataset, verbose=0)["importances"]

        column_list = []
        for i, feature in enumerate(self.TRANSFORMER_FEATURES):
            column_list.extend([feature] * self.inputs[feature].shape[1])
        importances = pd.DataFrame(importances[:, :-1], columns=column_list)
        average_importances = []
        for col_name in self.TRANSFORMER_FEATURES:
            average_importances.append(importances.filter(like=col_name).sum(axis=1))
        importances = pd.DataFrame(
            {
                column_name: column_data
                for column_name, column_data in zip(
                    self.TRANSFORMER_FEATURES, average_importances
                )
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
            title (str, optional): The title of the plot (default is "Importances").
            n_top_categories (int, optional): The number of top categories to plot (default is 5).

        Returns:
            None
        """
        dataset = self._get_dataset(self.data, shuffle=False)
        importances = self.predict(dataset, verbose=0)["importances"]

        column_list = []
        for i, feature in enumerate(self.TRANSFORMER_FEATURES):
            column_list.extend([feature] * self.inputs[feature].shape[1])
        importances = pd.DataFrame(importances[:, :-1], columns=column_list)
        average_importances = []
        for col_name in self.TRANSFORMER_FEATURES:
            average_importances.append(importances.filter(like=col_name).sum(axis=1))
        importances = pd.DataFrame(
            {
                column_name: column_data
                for column_name, column_data in zip(
                    self.TRANSFORMER_FEATURES, average_importances
                )
            }
        )
        result_dict = {}
        imps_sorted = importances.mean().sort_values(ascending=False)

        for category in self.TRANSFORMER_FEATURES:
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
        Plot heatmap of feature importances.

        Args:
            cat1: The first categorical feature.
            cat2: The second categorical feature.
            title (str, optional): The title of the plot (default is "Importances").

        Returns:
            None
        """
        dataset = self._get_dataset(self.data, shuffle=False)
        importances = self.predict(dataset, verbose=0)["importances"]

        column_list = []
        for i, feature in enumerate(self.TRANSFORMER_FEATURES):
            column_list.extend([feature] * self.inputs[feature].shape[1])
        importances = pd.DataFrame(importances[:, :-1], columns=column_list)
        average_importances = []
        for col_name in self.TRANSFORMER_FEATURES:
            average_importances.append(importances.filter(like=col_name).sum(axis=1))
        importances = pd.DataFrame(
            {
                column_name: column_data
                for column_name, column_data in zip(
                    self.TRANSFORMER_FEATURES, average_importances
                )
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
