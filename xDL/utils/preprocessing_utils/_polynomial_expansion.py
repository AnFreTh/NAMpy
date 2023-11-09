import numpy as np
import tensorflow as tf


class PolynomialExpansion(tf.keras.layers.Layer):
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

    def build(self, input_shape):
        super(PolynomialExpansion, self).build(input_shape)

    def adapt(self, inputs):
        pass

    def call(self, inputs):
        # Assuming inputs is a 2D tensor of shape (batch_size, input_dim)

        # Expand the polynomial terms
        polynomial_terms = []

        for d in range(1, self.degree + 1):
            expanded_term = inputs**d
            polynomial_terms.append(expanded_term)

        # Concatenate the polynomial terms along the feature dimension
        expanded_features = np.stack(polynomial_terms, 1)

        return np.squeeze(expanded_features, axis=-1)
