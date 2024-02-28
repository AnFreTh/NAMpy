import numpy as np
import tensorflow as tf
import pandas as pd
from keras.utils import to_categorical
from .preprocessing_utils._periodic_linear_encoding import *
from .preprocessing_utils._cubic_expansion import *
from .preprocessing_utils._polynomial_expansion import *
from .preprocessing_utils._minmax import *
from .preprocessing_utils._helper import *


def df_to_dataset(
    dataframe: pd.DataFrame,
    training_dataframe: pd.DataFrame,
    input_dict: dict,
    target: str = None,
    shuffle: bool = True,
    batch_size: int = 1024,
    validation_split=None,
    feature_information: dict = {},
    task="regression",
):
    """
    Converts a Pandas DataFrame into a TensorFlow Dataset.
    Performs all preprocessing steps before the actual modelling.
    Available preprocessing is: None, integer encoding, Periodic Linear encoding, normalization,
    min-max encoding, one-hot encoding

    Args:
        dataframe (pd.DataFrame): Input data as a Pandas DataFrame.
        input_dict (dict): A dictionary specifying the network type for each feature.
        target (str, optional): Name of the target column in the DataFrame (default is None).
        shuffle (bool, optional): Whether to shuffle the dataset (default is True).
        batch_size (int, optional): Batch size for the dataset (default is 1024).
        validation_split (float, optional): Fraction of data to use for validation (default is None).
        output_mode (str, optional): Output mode for encoding categorical data (default is "one_hot").

    Returns:
        tf.data.Dataset: A TensorFlow Dataset.
    """
    # Common data types including subtypes
    common_datatypes = (int, float, str, bool, list, dict, tuple, object)

    for column, dtype in dataframe.dtypes.items():
        if column == target:
            continue
        # Extract the main data type without the subtype
        main_dtype = np.dtype(dtype).type

        if any(issubclass(main_dtype, dt) for dt in common_datatypes):
            continue
        else:
            raise AssertionError(
                f"Column '{column}' has an unsupported datatype: {dtype}"
            )

    df = dataframe.copy()

    # check for valid encoding
    supported_encodings = [None, "int", "one_hot", "PLE", "normalized", "min_max"]
    for key, value in df.items():
        if key == target:
            continue
        if feature_information[key]["encoding"] not in supported_encodings:
            raise AssertionError(
                f"encoding {feature_information[key]['encoding']} for variable {key} is not supported"
            )

    if target:
        labels = df.pop(target)
        dataset = {}
        for key, value in df.items():
            if (
                feature_information[key]["encoding"] == "int"
                or feature_information[key]["encoding"] == "one_hot"
                or feature_information[key]["encoding"] == "PLE"
            ):
                if value.dtype == object:
                    value = np.expand_dims(value, 1)
                    lookup_class = tf.keras.layers.StringLookup
                    # Create a lookup layer which will turn strings into integer indices
                    lookup = lookup_class(
                        output_mode=feature_information[key]["encoding"]
                    )
                    # Learn the set of possible string values and assign them a fixed integer index
                    lookup.adapt(training_dataframe[key])
                    # Turn the string input into integer indices
                    encoded_feature = lookup(value)

                elif value.dtype == float:
                    encoded_feature = numerical_feature_binning(
                        training_dataframe,
                        key,
                        value,
                        labels,
                        target,
                        n_bins=feature_information[key]["n_bins"],
                        task=task,
                        encoding_type=feature_information[key]["encoding"],
                    )

                elif np.issubdtype(value.dtype, np.integer):
                    if feature_information[key]["encoding"] == "one_hot":
                        encoded_feature = to_categorical(value)
                    else:
                        encoded_feature = np.expand_dims(value, 1) + 1

            elif feature_information[key]["encoding"] == "normalized":
                if feature_information[key]["Network"] == "CubicSplineNet":
                    expander = CubicExpansion(feature_information[key]["n_knots"][0])
                    encoded_feature = expander.expand(value)
                elif feature_information[key]["Network"] == "PolynomialSplineNet":
                    expander = PolynomialExpansion(
                        feature_information[key]["degree"][0]
                    )
                    encoded_feature = expander.expand(value)
                else:
                    normalizer = tf.keras.layers.Normalization
                    norm = normalizer()
                    norm.adapt(training_dataframe[key])
                    encoded_feature = tf.transpose(norm(value))

            elif feature_information[key]["encoding"] == "normalized":
                min_max_layer = MinMaxEncodingLayer()
                encoded_feature = min_max_layer(value)

            elif feature_information[key]["encoding"] is None:
                encoded_feature = np.expand_dims(value, 1)

            else:
                raise ValueError(
                    f"the preprocessing type for variable {key}: {value.dtype} is not supported"
                )

            dataset[key] = np.array(encoded_feature)  #

        dataset = tf.data.Dataset.from_tensor_slices((dict(dataset), labels))
    else:
        dataset = {}
        for key, value in df.items():
            dataset[key] = np.array(value)[:, tf.newaxis]

        dataset = tf.data.Dataset.from_tensor_slices(dict(dataset))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))

    # Calculate the split point for validation data
    if validation_split:
        num_samples = len(dataframe)
        num_validation_samples = int(validation_split * num_samples)

        validation_dataset = dataset.take(num_validation_samples)
        training_dataset = dataset.skip(num_validation_samples)

        # Batch and prefetch both datasets
        validation_dataset = validation_dataset.batch(batch_size).prefetch(batch_size)
        training_dataset = training_dataset.batch(batch_size).prefetch(batch_size)

        return training_dataset, validation_dataset

    else:
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size)
        return dataset
