import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import bisect
import re
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from keras.utils import to_categorical


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
                    expander = CubicExpansion(feature_information[key]["n_knots"])
                    encoded_feature = expander.expand(
                        value, training_dataframe[key], training_dataframe[target]
                    )
                else:
                    normalizer = tf.keras.layers.Normalization
                    norm = normalizer()
                    norm.adapt(training_dataframe[key])
                    encoded_feature = tf.transpose(norm(value))

            else:
                raise ValueError(
                    f"the datatype for variable {key}: {value.dtype} is not supported"
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

    def expand(self, x, train_x, train_target):
        """

        :param x: x values to be evalutated
        :param n_knots: number of knots
        :return:
        """

        # xk = get_optimal_knots(train_x, train_target, n_bins=self.num_knots)
        xk = np.linspace(x.min(), x.max(), self.num_knots)

        # xk.append(train_x.min())
        # xk.append(train_x.max())
        # xk = np.sort(list(set(xk)))

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


def tree_to_code(tree, feature_names):
    """
    Convert a scikit-learn decision tree into a list of conditions.

    Args:
        tree (sklearn.tree.DecisionTreeRegressor or sklearn.tree.DecisionTreeClassifier):
            The decision tree model to be converted.
        feature_names (list of str): The names of the features used in the tree.
        Y (array-like): The target values associated with the tree.

    Returns:
        list of str: A list of conditions representing the decision tree paths.

    Example:
        # Convert a decision tree into a list of conditions
        tree_conditions = tree_to_code(tree_model, feature_names, target_values)
    """

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    pathto = dict()
    my_list = []

    global k
    k = 0

    def recurse(node, depth, parent):
        global k
        indent = "  " * depth

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # name = df_name + "[" + "'" + feature_name[node]+ "'" + "]"
            name = feature_name[node]
            threshold = tree_.threshold[node]
            s = "{} <= {} ".format(name, threshold, node)
            if node == 0:
                pathto[node] = "(" + s + ")"
            else:
                pathto[node] = "(" + pathto[parent] + ")" + " & " + "(" + s + ")"

            recurse(tree_.children_left[node], depth + 1, node)
            s = "{} > {}".format(name, threshold)
            if node == 0:
                pathto[node] = s
            else:
                pathto[node] = "(" + pathto[parent] + ")" + " & " + "(" + s + ")"
            recurse(tree_.children_right[node], depth + 1, node)
        else:
            k = k + 1
            my_list.append(pathto[parent])
            # print(k,')',pathto[parent], tree_.value[node])

    recurse(0, 1, 0)

    return my_list


def numerical_feature_binning(
    training_dataframe,
    feature_name,
    feature,
    target,
    target_name,
    n_bins=20,
    tree_params={},
    task="regression",
    encoding_type="int",
):
    """
    Perform numerical feature binning using decision tree-based discretization.

    Args:
        feature (array-like): The numerical feature to be binned.
        target (array-like): The target values associated with the feature.
        n_bins (int, optional): The maximum number of bins. Defaults to 20.
        tree_params (dict, optional): Additional parameters to be passed to the decision tree model.
        task (str, optional): The machine learning task, either "regression" or "classification".
                             Defaults to "regression".
        encoding_type (str, optional): The type of encoding to use for the bins, either "int" for integer
                                    encoding or "one_hot" for one-hot encoding. Defaults to "int".

    Returns:
        tf.Tensor: The binned and encoded feature as a TensorFlow tensor.

    Raises:
        ValueError: If the specified task or encoding type is not supported.

    Example:
        # Bin and encode a numerical feature for regression task.
        binned_feature = numerical_feature_binning(
            feature, target, n_bins=10, task="regression", encoding_type="int"
        )
    """
    # adjust for min and max nodes
    if task == "regression":
        dt = DecisionTreeRegressor(max_leaf_nodes=n_bins, **tree_params)
    elif task == "classification":
        dt = DecisionTreeClassifier(max_leaf_nodes=n_bins, **tree_params)
    else:
        raise ValueError("This task is not supported")

    dt.fit(
        np.expand_dims(training_dataframe[feature_name], 1),
        training_dataframe[target_name],
    )

    conditions = tree_to_code(dt, ["feature"])

    result_list = []
    for idx, cond in enumerate(conditions):
        if encoding_type == "int" or encoding_type == "PLE":
            result_list.append(eval(cond) * (idx + 1))
        elif encoding_type == "one_hot":
            result_list.append(eval(cond) * 1)
        else:
            raise ValueError("This encoding type is not supported")

    if encoding_type == "int":
        encoded_feature = np.expand_dims(np.sum(np.stack(result_list).T, axis=1), 1)
        return tf.cast(tf.convert_to_tensor(encoded_feature), dtype=tf.int64)
    elif encoding_type == "one_hot":
        encoded_feature = np.stack(result_list).T

        return tf.cast(tf.convert_to_tensor(encoded_feature), dtype=tf.int64)
    elif encoding_type == "PLE":
        encoded_feature = np.expand_dims(np.sum(np.stack(result_list).T, axis=1), 1)

        encoded_feature = tf.cast(
            tf.convert_to_tensor(encoded_feature) - 1, dtype=tf.int64
        )

        pattern = r"-?\d+\.\d+"  # This pattern matches integers and floats
        # Initialize an empty list to store the extracted numbers
        locations = []
        # Iterate through the strings and extract numbers
        for string in conditions:
            matches = re.findall(pattern, string)
            locations.extend(matches)

        locations = [float(number) for number in locations]

        locations = list(set(locations))

        locations = locations = np.sort(locations)

        ple_encoded_feature = np.zeros(
            (len(feature), tf.reduce_max(encoded_feature).numpy() + 1)
        )

        for idx in range(len(encoded_feature)):
            ple_encoded_feature[idx][encoded_feature[idx]] = (
                feature[idx] - locations[(encoded_feature[idx].numpy() - 2)[0]]
            ) / (
                locations[(encoded_feature[idx].numpy() - 1)[0]]
                - locations[(encoded_feature[idx].numpy() - 2)[0]]
            )
            ple_encoded_feature[idx, : encoded_feature[idx].numpy()[0]] = 1

        if ple_encoded_feature.shape[1] == 1:
            return tf.zeros([len(feature), n_bins])

        else:
            return tf.cast(tf.convert_to_tensor(ple_encoded_feature), dtype=tf.float32)


def get_optimal_knots(feature, target, n_bins=20, tree_params={}, task="regression"):
    if task == "regression":
        dt = DecisionTreeRegressor(max_leaf_nodes=n_bins, **tree_params)
    elif task == "classification":
        dt = DecisionTreeClassifier(max_leaf_nodes=n_bins, **tree_params)
    else:
        raise ValueError("This task is not supported")

    dt.fit(np.expand_dims(feature, 1), target)

    conditions = tree_to_code(dt, ["feature"], target)

    pattern = r"-?\d+\.\d+"  # This pattern matches integers and floats

    # Initialize an empty list to store the extracted numbers
    locations = [np.min(feature)]

    # Iterate through the strings and extract numbers
    for string in conditions:
        matches = re.findall(pattern, string)
        locations.extend(matches)
    locations.append(np.max(feature))
    # Convert the extracted numbers to floats
    locations = [float(number) for number in locations]

    return list(set(locations))
