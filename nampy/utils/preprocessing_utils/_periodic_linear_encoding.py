import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import bisect
import re
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from keras.utils import to_categorical


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


class PLE(tf.keras.layers.Layer):
    """
    Perform Periodic Linear Encoding (PLE) for numerical feature binning using decision tree-based discretization.

    Args:
        training_dataframe (pd.DataFrame): The training dataset containing the feature and target values.
        feature_name (str): The name of the numerical feature to be binned.
        target (array-like): The target values associated with the feature.
        target_name (str): The name of the target variable.
        n_bins (int, optional): The maximum number of bins. Defaults to 20.
        tree_params (dict, optional): Additional parameters to be passed to the decision tree model.
        task (str, optional): The machine learning task, either "regression" or "classification".
                            Defaults to "regression".

    Raises:
        ValueError: If the specified task is not supported.

    Example:
        # Create an instance of PLE
        ple_encoder = PLE(training_dataframe, 'feature_name', feature, target, 'target_name', n_bins=10, task="regression")

        # Transform your data using the fitted encoder
        encoded_feature = ple_encoder.transform(feature)
    """

    def __init__(
        self, n_bins=20, tree_params={}, task="regression", conditions=None, **kwargs
    ):
        super(PLE, self).__init__(**kwargs)

        self.task = task
        self.tree_params = tree_params
        self.n_bins = n_bins
        self.conditions = conditions
        self.pattern = (
            r"-?\d+\.?\d*[eE]?[+-]?\d*"  # This pattern matches integers and floats
        )

    def adapt(self, feature, target):
        if self.task == "regression":
            dt = DecisionTreeRegressor(max_leaf_nodes=self.n_bins)
        elif self.task == "classification":
            dt = DecisionTreeClassifier(max_leaf_nodes=self.n_bins)
        else:
            raise ValueError("This task is not supported")

        dt.fit(feature, target)

        self.conditions = tree_to_code(dt, ["feature"])

    def call(self, feature):
        if feature.shape == (feature.shape[0], 1):
            feature = np.squeeze(feature, axis=1)
        else:
            feature = feature
        result_list = []
        for idx, cond in enumerate(self.conditions):
            result_list.append(eval(cond) * (idx + 1))

        encoded_feature = np.expand_dims(np.sum(np.stack(result_list).T, axis=1), 1)

        encoded_feature = tf.cast(
            tf.convert_to_tensor(encoded_feature) - 1, dtype=tf.int64
        )

        # Initialize an empty list to store the extracted numbers
        locations = []
        # Iterate through the strings and extract numbers
        for string in self.conditions:
            matches = re.findall(self.pattern, string)
            locations.extend(matches)

        locations = [float(number) for number in locations]
        locations = list(set(locations))
        locations = np.sort(locations)

        ple_encoded_feature = np.zeros((len(feature), locations.shape[0] + 1))
        if locations[-1] > np.max(feature):
            locations[-1] = np.max(feature)

        for idx in range(len(encoded_feature)):

            if feature[idx] >= locations[-1]:
                ple_encoded_feature[idx][encoded_feature[idx]] = feature[idx]
                ple_encoded_feature[idx, : encoded_feature[idx].numpy()[0]] = 1
            elif feature[idx] <= locations[0]:
                ple_encoded_feature[idx][encoded_feature[idx]] = feature[idx]

            else:

                ple_encoded_feature[idx][encoded_feature[idx]] = (
                    feature[idx] - locations[(encoded_feature[idx].numpy() - 1)[0]]
                ) / (
                    locations[(encoded_feature[idx].numpy())[0]]
                    - locations[(encoded_feature[idx].numpy() - 1)[0]]
                )

                ple_encoded_feature[idx, : encoded_feature[idx].numpy()[0]] = 1

        if ple_encoded_feature.shape[1] == 1:
            return tf.zeros([len(feature), self.n_bins])

        else:
            return tf.cast(tf.convert_to_tensor(ple_encoded_feature), dtype=tf.float32)


class OneHotBinning(tf.keras.layers.Layer):
    def __init__(
        self, n_bins=20, tree_params={}, task="regression", conditions=None, **kwargs
    ):
        super(OneHotBinning, self).__init__(**kwargs)

        self.task = task
        self.tree_params = tree_params
        self.n_bins = n_bins
        self.conditions = conditions

    def build(self, input_shape):
        super(OneHotBinning, self).build(input_shape)

    def call(self, feature):
        if feature.shape == (feature.shape[0], 1):
            feature = np.squeeze(feature, axis=1)
        else:
            feature = feature

        result_list = []
        for idx, cond in enumerate(self.conditions):
            result_list.append(eval(cond) * 1)

        encoded_feature = np.stack(result_list).T

        return tf.cast(tf.convert_to_tensor(encoded_feature), dtype=tf.int64)

    def adapt(self, feature, target):
        if self.task == "regression":
            dt = DecisionTreeRegressor(max_leaf_nodes=self.n_bins)
        elif self.task == "classification":
            dt = DecisionTreeClassifier(max_leaf_nodes=self.n_bins)
        else:
            raise ValueError("This task is not supported")

        dt.fit(feature, target)
        self.conditions = tree_to_code(dt, ["feature"])


class IntegerBinning(tf.keras.layers.Layer):
    def __init__(
        self, n_bins=20, tree_params={}, task="regression", conditions=None, **kwargs
    ):
        super(IntegerBinning, self).__init__(**kwargs)

        self.task = task
        self.tree_params = tree_params
        self.n_bins = n_bins
        self.conditions = conditions

    def build(self, input_shape):
        super(IntegerBinning, self).build(input_shape)

    def call(self, feature):
        if feature.shape == (feature.shape[0], 1):
            feature = np.squeeze(feature, axis=1)
        else:
            feature = feature

        result_list = []
        for idx, cond in enumerate(self.conditions):
            result_list.append(eval(cond) * (idx + 1))

        encoded_feature = tf.expand_dims(
            tf.reduce_sum(tf.stack(result_list, axis=1), axis=1), axis=-1
        )

        encoded_feature = tf.cast(encoded_feature, dtype=tf.int64)

        return encoded_feature

    def adapt(self, feature, target):
        if self.task == "regression":
            dt = DecisionTreeRegressor(max_leaf_nodes=self.n_bins)
        elif self.task == "classification":
            dt = DecisionTreeClassifier(max_leaf_nodes=self.n_bins)
        else:
            raise ValueError("This task is not supported")

        dt.fit(feature, target)

        self.conditions = tree_to_code(dt, ["feature"])


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
