from nampy.utils.data_utils import DataModule
from keras.callbacks import *
import pandas as pd
import numpy as np
import tensorflow as tf


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
        task="regression",
        binning_task="regression",
        num_encoding="normalized",
        n_bins=None,
        **kwargs,
    ):
        super(BaseModel, self).__init__(**kwargs)

        self._validate_task(binning_task, False)
        self._initialize_attributes(
            data,
            val_data,
            binning_task,
            n_bins,
            activation,
            dropout,
            y,
            num_encoding,
            batch_size,
            task,
        )

        self._extract_data_types()
        self._create_input_dictionary()
        self.n_classes = self._validate_task(task, True)
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
        data,
        val_data,
        binning_task,
        n_bins,
        activation,
        dropout,
        y,
        num_encoding,
        batch_size,
        task,
    ):
        self.val_data = val_data
        self.data = data.copy()
        self.n_bins = n_bins
        self.activation = activation
        self.dropout = dropout
        self.target_name = y
        self.binning_task = binning_task
        self.num_encoding = num_encoding
        self.task = task
        self.batch_size = batch_size

    def _extract_data_types(self):
        self.NUM_FEATURES = []
        self.CAT_FEATURES = []
        self.feature_names = [
            col for col in self.data.columns if col != self.target_name
        ]

        for column in self.feature_names:
            if pd.api.types.is_numeric_dtype(self.data[column]):
                if pd.api.types.is_integer_dtype(self.data[column]):
                    self.CAT_FEATURES.append(column)
                elif pd.api.types.is_float_dtype(self.data[column]):
                    self.NUM_FEATURES.append(column)
                else:
                    raise ValueError(
                        f"The datatypes in your pd.DataFrame are not supported for column {column}"
                    )
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
        self.validation_dataset = (
            self.datamodule.validation_dataset
            if self.val_data is None
            else self.datamodule.transform(
                self.val_data.copy(),
                target_name=self.target_name,
                batch_size=self.batch_size,
                shuffle=False,
            )
        )
        self.training_dataset = self.datamodule.training_dataset
        self.test_dataset = self.datamodule.test_dataset
        self.plotting_dataset, self.plotting_data = self.datamodule._plotting_data()

    def _create_input_dictionary(self):
        self.input_dict = {}
        for idx, name in enumerate(self.feature_names):
            self.input_dict[name] = {
                "Network": None,
                "inputs": [
                    {
                        "feature_name": name,
                        "preprocessing": {"data_type": self.data[name].dtype},
                        "shapefunc_args": {},
                    }
                ],
            }

        # get encoding types
        output_mode = []
        self.feature_information = {}
        for name in self.feature_names:
            self.feature_information[name] = {}
            self.feature_information[name]["inputs"] = [
                {
                    "identifier": name,
                    "preprocessing": {"dtype": self.data[name].dtype},
                }
            ]
            self.feature_information[name]["Network"] = None
            self.feature_information[name]["shapefunc_args"] = {}
            if (
                np.issubdtype(self.data[name].dtype, np.integer)
                or self.data[name].dtype == "object"
            ):
                output_mode.append(["int"])
                self.feature_information[name]["inputs"][0]["preprocessing"][
                    "encoding"
                ] = "int"
            else:
                output_mode.append([self.num_encoding])
                self.feature_information[name]["inputs"][0]["preprocessing"][
                    "encoding"
                ] = self.num_encoding
                self.feature_information[name]["inputs"][0]["preprocessing"][
                    "n_bins"
                ] = self.n_bins

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
                self.inputs[feature_name] = tf.keras.Input(
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

        dataset = self.datamodule.transform(
            data.copy(),
            target_name=self.target_name,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        return dataset
