import numpy as np
import bisect
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import re
from ._periodic_linear_encoding import tree_to_code
import tensorflow as tf


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


class CubicExpansion(tf.keras.layers.Layer):
    def __init__(self, num_knots, **kwargs):
        super(CubicExpansion, self).__init__(**kwargs)
        self.num_knots = num_knots

    def build(self, input_shape):
        super(CubicExpansion, self).build(input_shape)

    def adapt(self, x):
        # xk = get_optimal_knots(train_x, train_target, n_bins=self.num_knots)
        self.xk = np.linspace(x.min(), x.max(), self.num_knots)

        # xk.append(train_x.min())
        # xk.append(train_x.max())
        # xk = np.sort(list(set(xk)))

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

    def call(self, x):
        """

        :param x: x values to be evalutated
        :param n_knots: number of knots
        :return:
        """

        F, S = self.get_FS(self.xk)
        n = len(x)
        k = len(self.xk)
        base = np.zeros((n, k))
        for i in range(0, len(x)):
            # find interval in which x[i] lies
            # and evaluate basis function from p.201 in Wood (2017)
            j = bisect.bisect_left(self.xk, x[i])
            x_j = self.xk[j - 1]
            x_j1 = self.xk[j]
            h = x_j1 - x_j
            a_jm = (x_j1 - x[i]) / h
            a_jp = (x[i] - x_j) / h
            c_jm = ((x_j1 - x[i]) ** 3 / h - h * (x_j1 - x[i])) / 6
            c_jp = ((x[i] - x_j) ** 3 / h - h * (x[i] - x_j)) / 6
            base[i, :] = c_jm * F[j - 1, :] + c_jp * F[j, :]
            base[i, j - 1] += a_jm
            base[i, j] += a_jp
        return base
