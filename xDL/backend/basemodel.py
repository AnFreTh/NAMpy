from xDL.utils.data_utils import DataModule
from xDL.utils.formulas import FormulaHandler
from keras.callbacks import *
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf


class AdditiveBaseModel(tf.keras.Model):
    def __init__(
        self,
        formula,
        data,
        feature_dropout,
        val_data=None,
        batch_size=1024,
        val_split=0.2,
        test_split=None,
        shuffle=True,
        binning_task="regression",
        n_bins_num=None,
        **kwargs,
    ):
        """
        Initialize the BaseModel.

        Args:
            formula (str): A formula for data processing.
            data (pd.DataFrame): Input data as a Pandas DataFrame.
            activation (str): Activation function for the model.
            dropout (float): Dropout rate.
            feature_dropout (float): Feature dropout rate.
            val_data (pd.DataFrame, optional): Validation data as a Pandas DataFrame.
            batch_size (int): Batch size for training (default is 1024).
            random_state (int): Random seed (default is 101).
            **kwargs: Additional keyword arguments.

        Attributes:
            formula (str): A formula for data processing.
            val_data (pd.DataFrame): Validation data as a Pandas DataFrame.
            data (pd.DataFrame): Input data as a Pandas DataFrame.
            activation (str): Activation function for the model.
            dropout (float): Dropout rate.
            feature_dropout (float): Feature dropout rate.
            training_dataset (tf.data.Dataset): Training dataset.
            validation_dataset (tf.data.Dataset): Validation dataset.
            plotting_dataset (tf.data.Dataset): Plotting dataset.
            named_feature_nets (list): List of named feature networks.
            y (str): Name of the target variable.
            feature_names (list): List of feature names.
            fit_intercept (bool): Whether to fit an intercept.
            hidden_layer_sizes (list): List of hidden layer sizes.

        Raises:
            NotImplementedError: If any of the methods is not implemented in the subclass.
        """

        super(AdditiveBaseModel, self).__init__(**kwargs)

        if binning_task not in ["regression", "classification"]:
            raise ValueError("choose a task of either 'regression' or 'classification'")

        self.formula = formula
        self.val_data = val_data
        self.data = data
        self.feature_dropout = feature_dropout
        self.binning_task = binning_task
        self.n_bins = n_bins_num

        (
            self.training_dataset,
            self.validation_dataset,
            self.test_dataset,
            self.plotting_dataset,
            self.named_feature_nets,
            self.y,
            self.feature_names,
            self.fit_intercept,
        ) = self._get_model_specifications(
            batch_size=batch_size,
            val_split=val_split,
            test_split=test_split,
            shuffle=shuffle,
        )

    def _get_model_specifications(self, batch_size, val_split, test_split, shuffle):
        """
        Get model specifications such as datasets, feature networks, and other attributes.

        Args:
            batch_size (int): Batch size for training.
            val_split (float): Validation data split ratio.
            random_state (int): Random seed.

        Returns:
            Tuple: A tuple containing model specifications.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """

        FH = FormulaHandler()
        (
            feature_names,
            target_name,
            named_feature_nets,
            intercept,
            self.feature_information,
        ) = FH._extract_formula_data(self.formula, self.data)

        helper_idx = feature_names + [target_name]
        self.data = self.data[helper_idx]

        self.input_dict = {}
        for idx, name in enumerate(FH.terms):
            if ":" in name:
                input_names = name.split(":")
                self.input_dict["_".join(input_names)] = {
                    "Network": named_feature_nets[idx],
                    "hyperparams": self.feature_information[input_names[0]],
                    "output_mode": self.feature_information[input_names[0]]["encoding"],
                }

            else:
                self.input_dict[name] = {
                    "Network": named_feature_nets[idx],
                    "hyperparams": self.feature_information[name],
                    "output_mode": self.feature_information[name]["encoding"],
                }

        self.NUM_FEATURES = []
        self.CAT_FEATURES = []
        for column in self.data:
            if pd.api.types.is_numeric_dtype(self.data[column]):
                if pd.api.types.is_integer_dtype(self.data[column].dtype):
                    self.CAT_FEATURES.append(column)
                else:
                    self.NUM_FEATURES.append(column)
            elif pd.api.types.is_string_dtype(
                self.data[column]
            ) or pd.api.types.is_object_dtype(self.data[column]):
                self.CAT_FEATURES.append(column)

        if target_name in self.NUM_FEATURES:
            self.NUM_FEATURES.remove(target_name)
        elif target_name in self.CAT_FEATURES:
            self.CAT_FEATURES.remove(target_name)

        self.datamodule = DataModule(
            self.data,
            input_dict={},
            feature_dictionary=self.feature_information,
            target_name=target_name,
        )

        self.datamodule.preprocess(
            validation_split=val_split,
            test_split=test_split,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        if self.val_data is None:
            val_dataset = self.datamodule.validation_dataset
        else:
            val_dataset = self.val_data
        train_dataset = self.datamodule.training_dataset
        test_dataset = self.datamodule.test_dataset

        plotting_dataset, self.plotting_data = self.datamodule._plotting_data()

        self.inputs = {}

        # Create tf.keras.Input for each feature in the dataset with appropriate shapes
        for inputs, labels in train_dataset.take(1):
            for feature_name, feature_value in inputs.items():
                self.inputs[feature_name] = tf.keras.Input(
                    shape=feature_value.shape[1:],
                    name=feature_name,
                )

        for idx, name in enumerate(FH.terms):
            if ":" in name:
                input_names = name.split(":")
                self.input_dict["_".join(input_names)]["Input"] = [
                    self.inputs[inp_name] for inp_name in input_names
                ]
            else:
                self.input_dict[name]["Input"] = [self.inputs[name]]

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            plotting_dataset,
            named_feature_nets,
            target_name,
            feature_names,
            intercept,
        )

    def _get_dataset(self, data, batch_size=512, shuffle=False):
        """
        Get a TF Dataset from input data.

        Args:
            data (pd.DataFrame): Input data as a Pandas DataFrame.
            batch_size (int): Batch size for the dataset (default is 512).
            shuffle (bool): Whether to shuffle the dataset (default is False).
            output_mode (str): Output mode for the dataset.

        Returns:
            tf.data.Dataset: A TensorFlow Dataset.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """

        datamodule = DataModule(
            data,
            input_dict={},
            feature_dictionary=self.feature_information,
            target_name=self.y,
        )
        datamodule.preprocess(
            validation_split=None,
            test_split=None,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        return datamodule.training_dataset

    def _build_model(self):
        """
        Build the machine learning model.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def _build_blackbox_interactions(self):
        """
        Build black-box interaction terms.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def _get_training_preds(self):
        """
        Get predictions on the training dataset.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def _get_validation_preds(self):
        """
        Get predictions on the validation dataset.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def _get_plotting_preds(self):
        """
        Get predictions on the plotting dataset.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def get_significance(self):
        """
        Calculate the significance of model features.

        Returns:
            pd.DataFrame: DataFrame containing feature significance statistics.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def call(self, inputs):
        """
        Define the forward pass of the model.

        Args:
            inputs: Model input.

        Returns:
            Model output.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def get_significance(self):
        ### !!!! TODO: Not adapted to TF.Datasets
        all_preds = self._get_training_preds()

        cols = self.cols

        result_df = pd.DataFrame(columns=["feature", "t-stat", "p_value"])

        if self.blackbox_interactions:
            cols.append("interaction")

        for i in range(len(self.data)):
            sig_feature = [i]
            indices = [j for j in range(len(self.data)) if j not in sig_feature]

            preds = [
                sum(self.ms[idx].predict(self.data[idx], verbose=0) for idx in indices)
            ]

            preds = np.array([preds[0][i] for i in range(len(self.X))])

            stat, p_value = ttest_ind(all_preds, preds[:, 0])
            res_df = pd.DataFrame(
                [[cols[i], np.round(stat, 4), np.round(p_value, 4)]],
                columns=["feature", "t-stat", "p_value"],
            )
            result_df = pd.concat([result_df, res_df], ignore_index=True)
        return result_df

    def analytics_plot(self):
        """creates an analytics plot for the additive models and compares true distribution to fitted ditsribution as well as residuals"""
        # Calculate residuals
        self.mu_preds = self._get_training_preds(mean=True).squeeze()
        y = target_list = []
        for _, target in self.training_dataset:
            target_list.append(target.numpy())

        target_list = [item for sublist in target_list for item in sublist]
        target_array = np.array(target_list)

        residuals = target_array - self.mu_preds

        # Create a gAMLSS-like plot
        sns.set(style="whitegrid", font_scale=0.8)
        plt.figure(figsize=(6, 6))

        # Histogram of residuals
        plt.subplot(2, 2, 1)
        sns.histplot(residuals, kde=True, color="blue")
        plt.axvline(x=0, color="red", linestyle="--")
        plt.xlabel("Residuals")
        plt.ylabel("Density")
        plt.title("Histogram of Residuals")

        # Q-Q plot of residuals
        plt.subplot(2, 2, 2)
        stats.probplot(residuals, plot=plt)
        plt.title("Q-Q Plot of Residuals")

        # Distribution of response variable
        plt.subplot(2, 2, 3)
        sns.histplot(self.data[self.y], kde=True, color="green")
        plt.xlabel("Y")
        plt.ylabel("Density")
        plt.title("Distribution of Response Variable")

        # Distribution of response variable
        plt.subplot(2, 2, 4)
        sns.histplot(self.mu_preds, kde=True, color="red")
        plt.xlabel("Y")
        plt.ylabel("Density")
        plt.title("Distribution of predictions")

        # Apply the same x-axis limits to both distribution plots
        plt.subplot(2, 2, 3)
        plt.xlim(-2, 4)
        plt.subplot(2, 2, 4)
        plt.xlim(-2, 4)
        plt.tight_layout()
        plt.show()


###############################################################################################


class BaseModel(tf.keras.Model):
    def __init__(
        self,
        data,
        y,
        activation,
        dropout,
        val_data=None,
        batch_size=1024,
        val_split=0.2,
        test_split=None,
        shuffle=True,
        binning_task="regression",
        num_encoding="normalized",
        n_bins_num=None,
        **kwargs,
    ):
        super(BaseModel, self).__init__(**kwargs)

        if binning_task not in ["regression", "classification"]:
            raise ValueError("choose a task of either 'regression' or 'classification'")

        self.val_data = val_data
        self.data = data
        self.activation = activation
        self.dropout = dropout
        self.y = y
        self.binning_task = binning_task
        self.num_encoding = num_encoding
        self.n_bins_num = n_bins_num

        (
            self.training_dataset,
            self.validation_dataset,
            self.test_dataset,
            self.y,
        ) = self._get_model_specifications(
            batch_size=batch_size,
            val_split=val_split,
            test_split=test_split,
            shuffle=shuffle,
        )

    def _get_model_specifications(self, batch_size, val_split, test_split, shuffle):
        """
        Get model specifications such as datasets, feature networks, and other attributes.

        Args:
            batch_size (int): Batch size for training.
            val_split (float): Validation data split ratio.
            random_state (int): Random seed.

        Returns:
            Tuple: A tuple containing model specifications.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """

        self.NUM_FEATURES = []
        self.CAT_FEATURES = []
        target_name = self.y
        feature_names = [col for col in self.data.columns if col != self.y]

        for column in feature_names:
            if pd.api.types.is_numeric_dtype(self.data[column]):
                if self.data[column].equals(self.data[column].astype(int)):
                    self.CAT_FEATURES.append(column)
                else:
                    self.NUM_FEATURES.append(column)
            elif pd.api.types.is_string_dtype(
                self.data[column]
            ) or pd.api.types.is_object_dtype(self.data[column]):
                self.CAT_FEATURES.append(column)

        self.input_dict = {}
        for idx, name in enumerate(feature_names):
            self.input_dict[name] = {
                "data_type": self.data[name].dtype,
                "Network": None,
            }

        if target_name in self.NUM_FEATURES:
            self.NUM_FEATURES.remove(target_name)
        elif target_name in self.CAT_FEATURES:
            self.CAT_FEATURES.remove(target_name)

        # get encoding types
        output_mode = []
        self.feature_information = {}
        for name in feature_names:
            self.feature_information[name] = {"data_type": self.data[name].dtype}
            if (
                np.issubdtype(self.data[name].dtype, np.integer)
                or self.data[name].dtype == "object"
            ):
                output_mode.append(["int"])
                self.feature_information[name]["encoding"] = "int"
            else:
                output_mode.append([self.num_encoding])
                self.feature_information[name]["encoding"] = self.num_encoding
                self.feature_information[name]["n_bins"] = self.n_bins_num

        self.datamodule = DataModule(
            self.data,
            input_dict={},
            feature_dictionary=self.feature_information,
            target_name=target_name,
        )
        self.datamodule.preprocess(
            validation_split=val_split,
            test_split=test_split,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        if self.val_data is None:
            val_dataset = self.datamodule.validation_dataset
        else:
            val_dataset = self.val_data
        train_dataset = self.datamodule.training_dataset
        test_dataset = self.datamodule.test_dataset

        self.inputs = {}

        # Create tf.keras.Input for each feature in the dataset with appropriate shapes
        for inputs, labels in train_dataset.take(1):
            for feature_name, feature_value in inputs.items():
                self.inputs[feature_name] = tf.keras.Input(
                    shape=feature_value.shape[1:],
                    name=feature_name,
                )

        for idx, name in enumerate(feature_names):
            if ":" in name:
                input_names = name.split(":")
                self.input_dict[name]["Input"] = [
                    self.inputs[inp_name] for inp_name in input_names
                ]
            else:
                self.input_dict[name]["Input"] = [self.inputs[name]]

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            target_name,
        )

    def _get_dataset(self, data, batch_size=512, shuffle=False):
        """
        Get a TF Dataset from input data.

        Args:
            data (pd.DataFrame): Input data as a Pandas DataFrame.
            batch_size (int): Batch size for the dataset (default is 512).
            shuffle (bool): Whether to shuffle the dataset (default is False).
            output_mode (str): Output mode for the dataset.

        Returns:
            tf.data.Dataset: A TensorFlow Dataset.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """

        datamodule = DataModule(
            data,
            input_dict={},
            feature_dictionary=self.feature_information,
            target_name=self.y,
        )
        datamodule.preprocess(
            validation_split=None,
            test_split=None,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        return datamodule.training_dataset

    def get_significance(self):
        """
        Calculate the significance of model features.

        Returns:
            pd.DataFrame: DataFrame containing feature significance statistics.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def call(self, inputs):
        """
        Define the forward pass of the model.

        Args:
            inputs: Model input.

        Returns:
            Model output.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def analytics_plot(self):
        """creates an analytics plot for the additive models and compares true distribution to fitted ditsribution as well as residuals"""
        # Calculate residuals
        self.mu_preds = tf.squeeze(self.predict(self.training_dataset))
        y = target_list = []
        for _, target in self.training_dataset:
            target_list.append(target.numpy())

        target_list = [item for sublist in target_list for item in sublist]
        target_array = np.array(target_list)

        residuals = target_array - self.mu_preds

        # Create a gAMLSS-like plot
        sns.set(style="whitegrid", font_scale=0.8)
        plt.figure(figsize=(6, 6))

        # Histogram of residuals
        plt.subplot(2, 2, 1)
        sns.histplot(residuals, kde=True, color="blue")
        plt.axvline(x=0, color="red", linestyle="--")
        plt.xlabel("Residuals")
        plt.ylabel("Density")
        plt.title("Histogram of Residuals")

        # Q-Q plot of residuals
        plt.subplot(2, 2, 2)
        stats.probplot(residuals, plot=plt)
        plt.title("Q-Q Plot of Residuals")

        # Distribution of response variable
        plt.subplot(2, 2, 3)
        sns.histplot(self.data[self.y], kde=True, color="green")
        plt.xlabel("Y")
        plt.ylabel("Density")
        plt.title("Distribution of Response Variable")

        # Distribution of response variable
        plt.subplot(2, 2, 4)
        sns.histplot(self.mu_preds, kde=True, color="red")
        plt.xlabel("Y")
        plt.ylabel("Density")
        plt.title("Distribution of predictions")

        # Apply the same x-axis limits to both distribution plots
        plt.subplot(2, 2, 3)
        plt.xlim(-2, 4)
        plt.subplot(2, 2, 4)
        plt.xlim(-2, 4)
        plt.tight_layout()
        plt.show()
