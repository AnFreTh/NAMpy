from xDL.utils.data_utils import DataModule
from xDL.formulas.formulas import FormulaHandler
from keras.callbacks import *
import pandas as pd
import tensorflow as tf


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

        self._validate_task(binning_task)
        self._initialize_attributes(
            formula, data, feature_dropout, val_data, binning_task, n_bins_num
        )
        self._create_input_dictionary()
        self._extract_data_types()
        self._build_datasets(batch_size, val_split, test_split, shuffle)
        self._create_model_inputs()

    def _validate_task(self, task):
        if task not in ["regression", "classification"]:
            raise ValueError("Task must be 'regression' or 'classification'")

    def _initialize_attributes(
        self, formula, data, feature_dropout, val_data, task, n_bins
    ):
        self.formula = formula
        self.val_data = val_data
        self.data = data.copy()
        self.feature_dropout = feature_dropout
        self.binning_task = task
        self.n_bins = n_bins

    def _create_input_dictionary(self):
        FH = FormulaHandler()
        (
            self.feature_names,
            self.target_name,
            self.terms,
            self.fit_intercept,
            self.feature_information,
        ) = FH._extract_formula_data(self.formula, self.data)

        helper_idx = self.feature_names + [self.target_name]
        self.data = self.data[helper_idx]

        self.input_dict = {}
        for idx, name in enumerate(self.terms):
            if ":" in name:
                input_names = name.split(":")
                self.input_dict["<>".join(input_names)] = {
                    "Network": self.feature_information[input_names[0]]["Network"],
                    "hyperparams": self.feature_information[input_names[0]],
                }

            else:
                self.input_dict[name] = {
                    "Network": self.feature_information[name]["Network"],
                    "hyperparams": self.feature_information[name],
                }

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
        )
        self.datamodule.preprocess(
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
        self.validation_dataset = (
            self.datamodule.validation_dataset
            if self.val_data is None
            else self.val_data
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

        for idx, name in enumerate(self.terms):
            if ":" in name:
                input_names = name.split(":")
                self.input_dict["<>".join(input_names)]["Input"] = [
                    self.inputs[inp_name] for inp_name in input_names
                ]
            else:
                self.input_dict[name]["Input"] = [self.inputs[name]]

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
            target_name=self.target_name,
        )
        datamodule.preprocess(
            validation_split=None,
            test_split=None,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        return datamodule.training_dataset
