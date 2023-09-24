import tensorflow as tf
from keras.callbacks import *
from sklearn.model_selection import KFold
from xDL.utils.data_utils import *
from xDL.backend.basemodel import AdditiveBaseModel
from xDL.utils.graphing import *
from xDL.backend.transformer_encoder import TransformerEncoder
from xDL.backend.helper_nets.featurenets import *
import seaborn as sns

import warnings

# Filter out the specific warning by category
warnings.filterwarnings("ignore", category=UserWarning)


class NATT(AdditiveBaseModel):
    def __init__(
        self,
        formula,
        data,
        dropout=0.1,
        feature_dropout=0.001,
        val_split=0.2,
        val_data=None,
        activation="relu",
        classification=False,
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        use_column_embedding: bool = False,
        mlp_hidden_factors: list = [2, 4],
        encoder=None,
        explainable=True,
        out_activation="linear",
        binning_task="regression",
        batch_size=1024,
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
            mlp_hidden_factors (list): List of factors for hidden layer sizes (default is [2, 4]).
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

        super(NATT, self).__init__(
            formula=formula,
            data=data,
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

        self.formula = formula
        self.val_data = val_data
        self.val_split = val_split
        self.feature_dropout = feature_dropout
        self.classification = classification

        if self.classification:
            num_classes = self.y.shape[1]
        else:
            num_classes = 1

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
        self.out_activation = out_activation

        ####################################
        self.output_layer = tf.keras.layers.Dense(
            1,
            "linear",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.0001),
        )

        self.feature_nets = []
        for _, key in enumerate(self.input_dict):
            if self.input_dict[key]["Network"] == "MLP":
                self.feature_nets.append(
                    eval(self.input_dict[key]["Network"])(
                        inputs=self.input_dict[key]["Input"],
                        param_dict=self.input_dict[key]["hyperparams"],
                        name=key,
                        output_dimension=num_classes,
                    )
                )

        self.ln = tf.keras.layers.LayerNormalization()

    def _get_plotting_preds(self, training_data=False):
        """
        Get predictions for plotting.

        Args:
            training_data (bool, optional): If True, get predictions for training data;
            otherwise, get predictions for plotting data (default is False).

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

            self.ms = [self.output_layer(x)]
            self.ms += [network(inputs) for network in self.feature_nets]

            x = tf.keras.layers.Add()(self.ms)

            if self.out_activation == "linear":
                output = x
            else:
                output = self.out_activation(x)
            att_testing_weights = self.encoder.att_weights

            return {
                "output": output,
                "importances": expl,
                "att_weights": att_testing_weights,
            }
        else:
            x = self.encoder(inputs)
            x = self.transformer_mlp(x)

            self.ms = [self.output_layer(x)]
            self.ms += [network(inputs) for network in self.feature_nets]

            x = tf.keras.layers.Add()(self.ms)

            if self.out_activation == "linear":
                output = x
            else:
                output = self.out_activation(x)
            return output

    def plot(self):
        """
        Plot the model's predictions.
        """

        # Generate subplots for visualization
        fig, axes = generate_subplots(len(self.input_dict) - 1, figsize=(10, 12))

        # Get plotting predictions
        preds = self._get_plotting_preds()

        for idx, ax in enumerate(axes.flat):
            try:
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
                    self.data[self.y] - np.mean(self.data[self.y]),
                    s=2,
                    alpha=0.5,
                    color="cornflowerblue",
                )
                # Line plot of predictions
                if self.feature_nets[idx].name in self.CAT_FEATURES:
                    ax.scatter(
                        self.plotting_data[self.feature_nets[idx].name],
                        preds[idx].squeeze(),
                        linewidth=2,
                        color="crimson",
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

    def plot_importances(self, title="Importances"):
        """
        Plot feature importances.

        Args:
            all (bool, optional): If True, plot importances for all features; otherwise, plot average importances (default is False).
            title (str, optional): Title of the plot (default is "Importances").
        """

        importances = self.predict(self.training_dataset, verbose=0)["importances"]

        column_list = []
        for i, feature in enumerate(self.TRANSFORMER_FEATURES):
            column_list.extend([feature] * self.inputs[feature].shape[1])
        importances = pd.DataFrame(importances[:, 1:], columns=column_list)
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
            title (str, optional): Title of the plot (default is "Importances").
            n_top_categories (int, optional): Number of top categories to consider (default is 5).
        """

        dataset = self._get_dataset(self.data, shuffle=False)
        importances = self.predict(dataset, verbose=0)["importances"]

        column_list = []
        for i, feature in enumerate(self.TRANSFORMER_FEATURES):
            column_list.extend([feature] * self.inputs[feature].shape[1])
        importances = pd.DataFrame(importances[:, 1:], columns=column_list)
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
        Plot heatmap of feature importances for two categorical features.

        Args:
            cat1 (str): Name of the first categorical feature.
            cat2 (str): Name of the second categorical feature.
            title (str, optional): Title of the plot (default is "Importances").
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
