import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import bisect


def df_to_dataset(
    dataframe: pd.DataFrame,
    input_dict: dict,
    target: str = None,
    shuffle: bool = True,
    batch_size: int = 1024,
    validation_split=None,
    output_mode="one_hot",
):
    """
    Converts a Pandas DataFrame into a TensorFlow Dataset.

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

    df = dataframe.copy()
    if target:
        labels = df.pop(target)
        dataset = {}
        for key, value in df.items():
            if value.dtype == object:
                if output_mode == "int":
                    value = np.expand_dims(value, 1)
                    lookup_class = tf.keras.layers.StringLookup
                    # Create a lookup layer which will turn strings into integer indices
                    lookup = lookup_class(output_mode=output_mode)
                    # Learn the set of possible string values and assign them a fixed integer index
                    lookup.adapt(value)
                    # Turn the string input into integer indices
                    encoded_feature = lookup(value)

                else:
                    lookup_class = tf.keras.layers.StringLookup
                    # Create a lookup layer which will turn strings into integer indices
                    lookup = lookup_class(output_mode=output_mode)
                    # Learn the set of possible string values and assign them a fixed integer index
                    lookup.adapt(value)
                    # Turn the string input into integer indices
                    encoded_feature = lookup(value)

            elif value.dtype == float:
                if key in input_dict:
                    if input_dict[key]["Network"] == "CubicSplineNet":
                        expander = CubicExpansion(input_dict[key]["Sizes"][0])
                        encoded_feature = expander.expand(value)

                    elif input_dict[key]["Network"] == "PolySplineNet":
                        expander = PolynomialExpansion(input_dict[key]["Sizes"][0])
                        encoded_feature = expander.expand(value)

                    else:
                        normalizer = tf.keras.layers.Normalization
                        norm = normalizer()
                        norm.adapt(value)
                        encoded_feature = tf.transpose(norm(value))
                else:
                    normalizer = tf.keras.layers.Normalization
                    norm = normalizer()
                    norm.adapt(value)
                    encoded_feature = tf.transpose(norm(value))

            elif value.dtype == int:
                encoded_feature = np.expand_dims(value, 1)

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


# get the caterogical prep layer for transformer
def build_categorical_prep(data: pd.DataFrame, categorical_features: list):
    """
    Builds categorical preparation layers for transformer models.

    Args:
        data (pd.DataFrame): Input data as a Pandas DataFrame.
        categorical_features (list): List of categorical feature names.

    Returns:
        dict: A dictionary of categorical preparation layers.
    """

    category_prep_layers = {}
    for c in tqdm(categorical_features):
        lookup = tf.keras.layers.StringLookup(vocabulary=data[c].unique())
        category_prep_layers[c] = lookup

    return category_prep_layers


def generate_plotting_data(df, num_samples, new_data={}):
    """
    Generates data for plotting purposes.

    Args:
        df (pd.DataFrame): Original data as a Pandas DataFrame.
        num_samples (int): Number of samples to generate.
        new_data (dict, optional): Additional data to include (default is {}).

    Returns:
        pd.DataFrame: A Pandas DataFrame containing generated data.

    """

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].equals(df[column].astype(int)):
                unique_values = sorted(df[column].unique())
                num_unique = len(unique_values)
                repetitions = num_samples // num_unique
                repeated_values = np.repeat(unique_values, repetitions)
                remaining_samples = num_samples - repetitions * num_unique
                new_data[column] = np.concatenate(
                    (repeated_values, unique_values[:remaining_samples])
                )
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


# min max preprocessing keras layer
class MinMaxEncodingLayer(tf.keras.layers.Layer):
    """
    Custom Keras layer for min-max scaling of input data.

    This layer scales input values to the range [-1, 1] using min-max scaling.

    Args:
        min_value (float): Minimum value for scaling.
        max_value (float): Maximum value for scaling.

    Returns:
        tf.Tensor: Scaled tensor in the range [-1, 1].

    Example:
        min_max_layer = MinMaxEncodingLayer(min_value=0, max_value=1)
        scaled_data = min_max_layer(inputs)
    """

    def __init__(self, min_value, max_value, **kwargs):
        super(MinMaxEncodingLayer, self).__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        # Apply min-max scaling to the range [-1, 1]
        encoded = 2 * (inputs - self.min_value) / (self.max_value - self.min_value) - 1
        return encoded

    def get_config(self):
        config = super(MinMaxEncodingLayer, self).get_config()
        config.update({"min_value": self.min_value, "max_value": self.max_value})
        return config


class PolynomialExpansion:
    """
    Polynomial expansion utility for feature transformation.

    This class performs polynomial expansion of input features up to a specified degree.

    Args:
        degree (int): Degree of polynomial expansion.

    Returns:
        np.ndarray: Array containing expanded polynomial features.

    Example:
        poly_expander = PolynomialExpansion(degree=2)
        expanded_features = poly_expander.expand(inputs)
    """

    def __init__(self, degree, **kwargs):
        super(PolynomialExpansion, self).__init__(**kwargs)
        self.degree = degree

    def expand(self, inputs):
        # Assuming inputs is a 2D tensor of shape (batch_size, input_dim)

        # Expand the polynomial terms
        polynomial_terms = []

        for d in range(1, self.degree + 1):
            expanded_term = inputs**d
            polynomial_terms.append(expanded_term)

        # Concatenate the polynomial terms along the feature dimension
        expanded_features = np.stack(polynomial_terms, 1)

        return expanded_features


class CubicExpansion:
    def __init__(self, num_knots, **kwargs):
        super(CubicExpansion, self).__init__(**kwargs)
        self.num_knots = num_knots

    def get_FS(self, xk):
        """
        Create matrix F required to build the spline base and the penalizing matrix S,
        based on a set of knots xk (ascending order). Pretty much directly from p.201 in Wood (2017)
        :param xk: knots (for now always np.linspace(x.min(), x.max(), n_knots)
        """
        k = len(xk)
        h = np.diff(xk)
        h_shift_up = h.copy()[1:]

        D = np.zeros((k - 2, k))
        np.fill_diagonal(D, 1 / h[: k - 2])
        np.fill_diagonal(D[:, 1:], (-1 / h[: k - 2] - 1 / h_shift_up))
        np.fill_diagonal(D[:, 2:], 1 / h_shift_up)

        B = np.zeros((k - 2, k - 2))
        np.fill_diagonal(B, (h[: k - 2] + h_shift_up) / 3)
        np.fill_diagonal(B[:, 1:], h_shift_up[k - 3] / 6)
        np.fill_diagonal(B[1:, :], h_shift_up[k - 3] / 6)
        F_minus = np.linalg.inv(B) @ D
        F = np.vstack([np.zeros(k), F_minus, np.zeros(k)])
        S = D.T @ np.linalg.inv(B) @ D
        return F, S

    def expand(self, x):
        """

        :param x: x values to be evalutated
        :param n_knots: number of knots
        :return:
        """

        xk = np.linspace(x.min(), x.max(), self.num_knots)
        n = len(x)
        k = len(xk)
        F, S = self.get_FS(xk)
        base = np.zeros((n, k))
        for i in range(0, len(x)):
            # find interval in which x[i] lies
            # and evaluate basis function from p.201 in Wood (2017)
            j = bisect.bisect_left(xk, x[i])
            x_j = xk[j - 1]
            x_j1 = xk[j]
            h = x_j1 - x_j
            a_jm = (x_j1 - x[i]) / h
            a_jp = (x[i] - x_j) / h
            c_jm = ((x_j1 - x[i]) ** 3 / h - h * (x_j1 - x[i])) / 6
            c_jp = ((x[i] - x_j) ** 3 / h - h * (x[i] - x_j)) / 6
            base[i, :] = c_jm * F[j - 1, :] + c_jp * F[j, :]
            base[i, j - 1] += a_jm
            base[i, j] += a_jp
        return base
