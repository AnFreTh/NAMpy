from nampy.utils.data_utils import DataModule
from nampy.formulas.formulas import FormulaHandler
from keras.callbacks import *
import pandas as pd
import tensorflow as tf
import numpy as np


class AdditiveBaseModel(tf.keras.Model):
    """
    Base model class for Additive models.

    This class provides a basic framework for building additive models
    with TensorFlow and Keras. It includes methods for preparing data,
    building the model architecture, and evaluating model performance.
    """

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
        task="regression",
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

        self._validate_task(binning_task, False)

        self._initialize_attributes(
            formula,
            data,
            feature_dropout,
            val_data,
            binning_task,
            n_bins_num,
            batch_size,
            task,
        )

        self._create_input_dictionary()
        self.n_classes = self._validate_task(task, True)
        self._extract_data_types()
        if self.val_data is not None:
            val_split = 0.0
        self._build_datasets(batch_size, val_split, test_split, shuffle)
        self._create_model_inputs()

    def _validate_task(self, task, classes):
        if task not in ["regression", "classification"]:
            raise ValueError("Task must be 'regression' or 'classification'")
        if classes:
            if task == "classification":
                unique_labels = np.unique(self.data[self.target_name])
                # Set num_classes to 1 for binary classification or non-classification tasks
                if len(unique_labels) == 2 or not self.task == "classification":
                    return 1
                else:
                    return len(unique_labels)
            else:
                return 1

    def _initialize_attributes(
        self,
        formula,
        data,
        feature_dropout,
        val_data,
        binning_task,
        n_bins,
        batch_size,
        task,
    ):
        self.formula = formula
        self.val_data = val_data
        self.data = data
        self.feature_dropout = feature_dropout
        self.binning_task = binning_task
        self.n_bins = n_bins
        self.task = task
        self.batch_size = batch_size

    def _create_input_dictionary(self):
        self.FH = FormulaHandler()
        (
            self.feature_names,
            self.target_name,
            self.fit_intercept,
            network_identifier,
            self.feature_information,
        ) = self.FH.extract_formula_data(self.formula, self.data)

        network_identifier.append(self.target_name)
        helper_idx = self.feature_names + [self.target_name]
        self.data = self.data[helper_idx]
        self.data.columns = network_identifier
        self.y = self.data[self.target_name]

    def _extract_data_types(self):
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

        if self.target_name in self.NUM_FEATURES:
            self.NUM_FEATURES.remove(self.target_name)
        elif self.target_name in self.CAT_FEATURES:
            self.CAT_FEATURES.remove(self.target_name)

    def _build_datasets(self, batch_size, val_split, test_split, shuffle):
        """
        Builds the datasets required for training, validation, and plotting.

        """
        self.datamodule = DataModule(
            self.data,
            input_dict={},
            feature_dictionary=self.feature_information,
            target_name=self.target_name,
            task=self.task,
        )
        self.datamodule.fit_transform(
            validation_split=val_split,
            test_split=test_split,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        self._assign_datasets()

    def _assign_datasets(self):
        """
        Assigns datasets to the respective class attributes.
        """

        if self.val_data is not None:
            val_data = self.val_data.copy()
            (
                feature_names,
                target_name,
                fit_intercept,
                network_identifier,
                feature_information,
            ) = self.FH.extract_formula_data(self.formula, val_data)

            network_identifier.append(self.target_name)
            helper_idx = self.feature_names + [self.target_name]
            val_data = val_data[helper_idx]
            val_data.columns = network_identifier
            y = val_data[self.target_name]

        else:
            val_data = None

        self.validation_dataset = (
            self.datamodule.validation_dataset
            if self.val_data is None
            else self.datamodule.transform(
                val_data.copy(),
                target_name=self.target_name,
                batch_size=self.batch_size,
                shuffle=False,
            )
        )
        self.training_dataset = self.datamodule.training_dataset
        self.test_dataset = self.datamodule.test_dataset
        self.plotting_dataset, self.plotting_data = self.datamodule._plotting_data()

    def _create_model_inputs(self):
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

        self.inputs = {}

        # Create tf.keras.Input for each feature in the dataset with appropriate shapes
        for inputs, labels in self.training_dataset.take(1):
            for feature_name, feature_value in inputs.items():
                self.inputs[feature_name] = tf.keras.layers.Input(
                    shape=feature_value.shape[1:],
                    name=feature_name,
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

        (
            feature_names,
            target_name,
            fit_intercept,
            network_identifier,
            feature_information,
        ) = self.FH.extract_formula_data(self.formula, data)

        network_identifier.append(self.target_name)
        helper_idx = self.feature_names + [self.target_name]
        data = data[helper_idx]
        data.columns = network_identifier
        y = data[self.target_name]

        dataset = self.datamodule.transform(
            data.copy(),
            target_name=self.target_name,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        return dataset
