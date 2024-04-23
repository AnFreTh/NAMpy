import numpy as np
import tensorflow as tf
import pandas as pd
from .preprocessing_utils._periodic_linear_encoding import (
    PLE,
    OneHotBinning,
    IntegerBinning,
)
from .preprocessing_utils._cubic_expansion import CubicExpansion
from .preprocessing_utils._polynomial_expansion import PolynomialExpansion
from .preprocessing_utils._minmax import MinMaxEncodingLayer
from .preprocessing_utils._helper import (
    NoPreprocessingCatLayer,
    NoPreprocessingLayer,
    OneHotConstantBins,
)
import numbers
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical


class Preprocessor(tf.keras.layers.Layer):
    """
    A custom TensorFlow Keras layer for preprocessing features in a dataset.

    Args:
    feature_preprocessing_dict (dict): A dictionary specifying the preprocessing details for each feature.
    target_name (str): The name of the target feature.
    task (str, optional): The machine learning task type, e.g., "regression" or "classification". Default is "regression".
    tree_params (dict, optional): Parameters for tree-based preprocessing methods. Default is an empty dictionary.

    Attributes:
    feature_preprocessing_dict (dict): A dictionary specifying the preprocessing details for each feature.
    target_name (str): The name of the target feature.
    task (str): The machine learning task type.
    preprocessors (dict): A dictionary containing preprocessing layers for each feature.
    tree_params (dict): Parameters for tree-based preprocessing methods.

    Methods:
    call(data, target):
        Preprocess the input data according to the specified preprocessing methods.

    Example usage:
    preprocessor = Preprocessor(feature_preprocessing_dict, target_name, task="regression")
    preprocessed_data = preprocessor(data, target)

    Raises:
    ValueError: If the data types and preprocessing methods are not compatible.

    """

    def __init__(
        self,
        feature_preprocessing_dict: dict,
        target_name: str,
        task: str = "regression",
        tree_params: dict = {},
    ):
        """
        Initialize the Preprocessor instance.

        Args:
        feature_preprocessing_dict (dict): A dictionary specifying the preprocessing details for each feature.
        target_name (str): The name of the target feature.
        task (str, optional): The machine learning task type, e.g., "regression" or "classification". Default is "regression".
        tree_params (dict, optional): Parameters for tree-based preprocessing methods. Default is an empty dictionary.
        """

        super(Preprocessor, self).__init__()
        self.feature_preprocessing_dict = feature_preprocessing_dict
        self.target_name = target_name
        self.task = task
        self.preprocessors = {}
        self.tree_params = tree_params

        for _, val in self.feature_preprocessing_dict.items():
            for input in val["inputs"]:
                key = input["identifier"]
                preprocessing = input["preprocessing"]

                if preprocessing["encoding"] == "normalized":
                    if val["Network"] == "CubicSplineNet":
                        self.preprocessors[key] = CubicExpansion(
                            preprocessing["n_knots"]
                        )
                    elif val["Network"] == "PolynomialSplineNet":
                        self.preprocessors[key] = PolynomialExpansion(
                            preprocessing["degree"]
                        )
                    else:
                        self.preprocessors[key] = tf.keras.layers.Normalization()

                elif preprocessing["encoding"] == "min_max":
                    self.preprocessors[key] = MinMaxEncodingLayer()

                elif preprocessing["encoding"] == "hashing":
                    self.preprocessors[key] = tf.keras.layers.Hashing(
                        num_bins=preprocessing["n_bins"]
                    )

                elif preprocessing["encoding"] == "discretized":
                    self.preprocessors[key] = tf.keras.layers.Discretization(
                        num_bins=preprocessing["n_bins"]
                    )

                elif preprocessing["encoding"] == "int":
                    if preprocessing["dtype"] == object:
                        self.preprocessors[key] = tf.keras.layers.StringLookup(
                            output_mode="int"
                        )
                    elif preprocessing["dtype"] == float:
                        self.preprocessors[key] = IntegerBinning(
                            n_bins=preprocessing["n_bins"],
                            task=self.task,
                            tree_params=self.tree_params,
                        )
                    elif np.issubdtype(preprocessing["dtype"], np.integer):
                        self.preprocessors[key] = tf.keras.layers.IntegerLookup(
                            output_mode="int"
                        )  # NoPreprocessingLayer(type="int")

                elif preprocessing["encoding"] == "one_hot":
                    if preprocessing["dtype"] == object:
                        self.preprocessors[key] = tf.keras.layers.StringLookup(
                            output_mode="one_hot"
                        )
                    elif preprocessing["dtype"] == float:
                        self.preprocessors[key] = OneHotBinning(
                            n_bins=preprocessing["n_bins"],
                            task=self.task,
                            tree_params=self.tree_params,
                        )
                    elif np.issubdtype(preprocessing["dtype"], np.integer):
                        self.preprocessors[key] = NoPreprocessingCatLayer(
                            type="one_hot"
                        )

                elif preprocessing["encoding"] == "one_hot_occurences":
                    self.preprocessors[key] = tf.keras.layers.Discretization(
                        num_bins=preprocessing["n_bins"],
                        output_mode="one_hot",
                    )

                elif preprocessing["encoding"] == "one_hot_constant":
                    self.preprocessors[key] = OneHotConstantBins(
                        num_bins=preprocessing["n_bins"],
                    )

                elif preprocessing["encoding"] == "PLE":
                    self.preprocessors[key] = PLE(
                        n_bins=preprocessing["n_bins"],
                        task=self.task,
                        tree_params=self.tree_params,
                    )

                else:
                    self.preprocessors[key] = NoPreprocessingLayer()

    def fit(self, data, target):
        """
        Fit the preprocessing layers to the input data.

        Args:
        data (dict): A dictionary containing input features.
        target: The target feature.

        Raises:
        ValueError: If the data types and preprocessing methods are not compatible.
        """

        for key, feature in data.items():
            if key == self.target_name:
                continue
            try:
                feature.shape[1]
            except IndexError:
                feature = np.expand_dims(feature, 1)

            # adapt the preprocessing layers to the data
            try:
                self.preprocessors[key].adapt(feature)
            except TypeError:
                try:
                    self.preprocessors[key].adapt(feature, target)
                except:
                    raise ValueError(
                        "the datatypes and preprocessing are not functional together"
                    )

    def call(self, data):
        """
        Preprocess the input data according to the specified preprocessing methods.

        Args:
        data (dict): A dictionary containing input features.
        target: The target feature.

        Returns:
        dict: A dictionary containing the preprocessed features.

        Raises:
        ValueError: If the data types and preprocessing methods are not compatible.
        """

        dataset = {}
        print("--- Preprocessing ---")
        for key, feature in tqdm(data.items()):
            if key == self.target_name:
                continue
                # get data in correct shape
            try:
                feature.shape[1]
            except IndexError:
                feature = np.expand_dims(feature, 1)
            encoded_feature = self.preprocessors[key](feature)
            if encoded_feature.shape[0] == 1:
                encoded_feature = tf.transpose(encoded_feature)
            dataset[key] = np.array(encoded_feature)

        return dataset


class DataModule:
    """
    A class for managing and preprocessing data for machine learning tasks.

    Args:
    data (pandas.DataFrame): The input data as a pandas DataFrame.
    input_dict (dict): A dictionary specifying the preprocessing details for each feature.
    target_name (str): The name of the target feature.
    feature_dictionary (dict, optional): A dictionary specifying additional information about features. Default is an empty dictionary.
    task (str, optional): The machine learning task type, e.g., "regression" or "classification". Default is "regression".
    tree_params (dict, optional): Parameters for tree-based preprocessing methods. Default is an empty dictionary.

    Attributes:
    data (pandas.DataFrame): The input data as a pandas DataFrame.
    labels (pandas.Series): The target labels.
    input_dict (dict): A dictionary specifying the preprocessing details for each feature.
    task (str): The machine learning task type.
    target_name (str): The name of the target feature.
    feature_dictionary (dict): A dictionary specifying additional information about features.
    tree_params (dict): Parameters for tree-based preprocessing methods.
    preprocessing_called (bool): A flag indicating whether the preprocessing has been performed.
    encoder (Preprocessor): A preprocessing encoder for the data.

    Methods:
    preprocess(validation_split=0.2, test_split=None, batch_size=1024, shuffle=True):
        Perform data preprocessing and create TensorFlow datasets for training, validation, and testing.

    Raises:
    AssertionError: If an unsupported data type or encoding is encountered.
    RuntimeError: If preprocessing functions are called before the preprocessing is performed.

    Example usage:
    data_module = DataModule(data, input_dict, target_name, task="regression")
    data_module.preprocess(validation_split=0.2)
    """

    def __init__(
        self,
        data,
        input_dict,
        target_name,
        feature_dictionary={},
        task="regression",
        tree_params={},
    ):
        """
        Initialize the DataModule with input data and preprocessing information.

        Args:
        data (pandas.DataFrame): The input data as a pandas DataFrame.
        input_dict (dict): A dictionary specifying the preprocessing details for each feature.
        target_name (str): The name of the target feature.
        feature_dictionary (dict, optional): A dictionary specifying additional information about features. Default is an empty dictionary.
        task (str, optional): The machine learning task type, e.g., "regression" or "classification". Default is "regression".
        tree_params (dict, optional): Parameters for tree-based preprocessing methods. Default is an empty dictionary.
        """

        # Common data types including subtypes
        self.data = data

        common_datatypes = (int, float, str, bool, list, dict, tuple, object)
        for column, dtype in self.data.dtypes.items():
            if column == target_name:
                continue
            # Extract the main data type without the subtype
            main_dtype = np.dtype(dtype).type

            if any(
                issubclass(main_dtype, dt)
                or (isinstance(main_dtype, numbers.Integral) and issubclass(int, dt))
                or (isinstance(main_dtype, numbers.Real) and issubclass(float, dt))
                for dt in common_datatypes
            ):
                continue
            else:
                raise AssertionError(
                    f"Column '{column}' has an unsupported datatype: {dtype}"
                )

        # if target_name:
        #    self.labels = self.data.pop(target_name)

        # check for valid encoding
        supported_encodings = [
            None,
            "int",
            "one_hot",
            "one_hot_constant",
            "one_hot_occurences",
            "PLE",
            "normalized",
            "min_max",
            "hashing",
            "discretized",
        ]

        for key, value in feature_dictionary.items():
            for input in value["inputs"]:
                if input["preprocessing"]["encoding"] not in supported_encodings:
                    raise ValueError(
                        f"{input['feature_name']} has unsupported encoding {input['preprocessing']['encoding']}. Encoding must be in {supported_encodings}"
                    )

        self.input_dict = input_dict
        self.task = task
        self.labels = self.data[target_name]
        if self.task == "classification":
            unique_labels = np.unique(self.labels)

            if len(unique_labels) == 2:
                # Binary classification; ensure labels are 0 and 1
                self.labels = np.where(self.labels == unique_labels[0], 0, 1)
            elif len(unique_labels) > 2:
                # Multiclass classification; one-hot encode
                # First, ensure labels are integers starting from 0
                label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
                mapped_labels = np.vectorize(label_mapping.get)(self.labels)
                self.labels = to_categorical(mapped_labels)
        self.target_name = target_name
        self.feature_dictionary = feature_dictionary
        self.tree_params = tree_params
        self.preprocessing_called = False

        self.encoder = Preprocessor(feature_dictionary, target_name, task, tree_params)

    def _get_info(self):
        """
        Collect information about the data module and its preprocessing.

        This method updates the 'info' attribute with relevant information.
        """
        self.info = {}
        self.info["datapoints"] = len(self.data)
        self.info["preprocessors"] = self.encoder.preprocessors

    def _plotting_data(self):
        """
        Generate and preprocess data for visualization.

        Returns:
        tf.data.Dataset: A dataset containing preprocessed data for visualization.
        dict: Raw data used for visualization.

        Raises:
        RuntimeError: If called before preprocessing.
        """

        if not self.preprocessing_called:
            raise RuntimeError(
                "call_first_function must be called before call_second_function"
            )

        plotting_data = generate_plotting_data(self.data, 1000)

        dataset = {}
        plotting_labels = plotting_data.pop(self.target_name)
        for key, feature in tqdm(plotting_data.items()):
            if key == self.target_name:
                continue
            try:
                feature.shape[1]
            except IndexError:
                feature = np.expand_dims(feature, 1)

            encoded_feature = self.encoder.preprocessors[key](feature)

            if encoded_feature.shape[0] == 1:
                encoded_feature = tf.transpose(encoded_feature)
            dataset[key] = np.array(encoded_feature)

        self.plotting_dataset = tf.data.Dataset.from_tensor_slices(
            (dict(dataset), plotting_labels)
        )

        self.plotting_dataset = self.plotting_dataset.batch(1000)
        self.plotting_dataset = self.plotting_dataset.prefetch(1000)

        return self.plotting_dataset, plotting_data

    def fit_transform(
        self, validation_split=0.2, test_split=None, batch_size=1024, shuffle=True
    ):
        """
        Perform data preprocessing and create TensorFlow datasets for training, validation, and testing.

        Args:
        validation_split (float, optional): The fraction of data to use for validation. Default is 0.2.
        test_split (float, optional): The fraction of data to use for testing. Default is None.
        batch_size (int, optional): Batch size for training and validation datasets. Default is 1024.
        shuffle (bool, optional): Whether to shuffle the data. Default is True.

        Raises:
        RuntimeError: If preprocessing functions are called before the preprocessing is performed.
        """
        self.preprocessing_called = True
        self.validation_split = validation_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.encoder.fit(self.data, self.labels)
        self.dataset = self.encoder(self.data)
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (dict(self.dataset), self.labels)
        )

        if self.shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=len(self.data))

        # Calculate the split point for validation data
        if validation_split:
            num_samples = len(self.data)
            num_validation_samples = int(validation_split * num_samples)

            validation_dataset = self.dataset.take(num_validation_samples)
            training_dataset = self.dataset.skip(num_validation_samples)
            if test_split:
                num_test_samples = int(test_split * num_samples)
                test_dataset = self.dataset.take(num_test_samples)
                training_dataset = self.dataset.skip(num_test_samples)
                self.test_dataset = test_dataset.batch(batch_size).prefetch(batch_size)

            else:
                self.test_dataset = None

            # Batch and prefetch both datasets
            self.validation_dataset = validation_dataset.batch(batch_size).prefetch(
                batch_size
            )
            self.training_dataset = training_dataset.batch(batch_size).prefetch(
                batch_size
            )

        else:
            self.training_dataset = self.dataset.batch(batch_size)
            self.training_dataset = self.training_dataset.prefetch(batch_size)
            self.validation_dataset = None
            self.test_dataset = None

    def transform(self, data, target_name, batch_size, shuffle):
        assert self.preprocessing_called, "Preprocessing must be called first"
        if target_name in data.columns:
            labels = data.pop(target_name)
        else:
            labels = None
        dataset = self.encoder(data)
        dataset = tf.data.Dataset.from_tensor_slices((dict(dataset), labels))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(data))

        dataset = dataset.batch(batch_size).prefetch(batch_size)
        return dataset


def generate_plotting_data(df, num_samples):
    """
    Generates data for plotting purposes.

    Args:
        df (pd.DataFrame): Original data as a Pandas DataFrame.
        num_samples (int): Number of samples to generate.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing generated data.

    """
    new_data = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            if pd.api.types.is_integer_dtype(df[column].dtype):
                unique_values = sorted(df[column].unique())
                num_unique = len(unique_values)
                repetitions = num_samples // num_unique
                repeated_values = np.repeat(unique_values, repetitions)
                remaining_samples = num_samples - repetitions * num_unique
                new_data[column] = np.concatenate(
                    (repeated_values, unique_values[:remaining_samples])
                )

                new_data[column].astype(int)
            else:
                min_value = df[column].min()
                max_value = df[column].max()
                new_data[column] = np.linspace(min_value, max_value, num_samples)
        elif pd.api.types.is_string_dtype(df[column]):
            unique_values = sorted(df[column].unique())
            num_unique = len(unique_values)
            repetitions = num_samples // num_unique
            repeated_values = np.repeat(unique_values, repetitions)
            remaining_samples = num_samples - repetitions * num_unique
            new_data[column] = np.concatenate(
                (repeated_values, unique_values[:remaining_samples])
            )
        else:
            print(f"Unsupported column type: {column}")

    return pd.DataFrame(new_data)
