from patsy import dmatrices, dmatrix, demo_data
from patsy import ModelDesc
import pandas as pd
import numpy as np
import patsy
import re
import tensorflow as tf


class MLP(object):
    """
    Multi-Layer Perceptron (MLP) transformer.

    This class defines an MLP transformer that can be used with patsy for feature transformations.
    Since the transformations happen inside the tf_dataset this is just an empty shell and returns no transformation but simplyn the raw input

    """

    def __init__(self):
        pass

    def memorize_chunk(self, x, sizes: list = [128, 128, 64], activation: str = "relu"):
        """
        Memorize a chunk of data.

        Args:
            x: The input data.
            sizes (list): A list specifying the hidden layer sizes of the MLP.
            activation (str): The activation function to use in the MLP.

        Returns:
            None
        """
        pass

    def memorize_finish(self):
        """
        Finalize the memorization process.

        Returns:
            None
        """
        pass

    def transform(self, x, sizes: list = [128, 128, 64], activation: str = "relu"):
        """
        Transform data using the MLP.

        Args:
            x: The input data.
            sizes (list): A list specifying the hidden layer sizes of the MLP.
            activation (str): The activation function to use in the MLP.

        Returns:
            x: The untransformed data.
        """
        return x


class CubicSplineNet(object):
    def __init__(self):
        pass

    def memorize_chunk(self, x, n_knots):
        pass

    def memorize_finish(self):
        pass

    def transform(self, x, n_knots):
        return x


class PolySplineNet(object):
    def __init__(self):
        pass

    def memorize_chunk(self, x, n_knots):
        pass

    def memorize_finish(self):
        pass

    def transform(self, x, n_knots):
        return x


class Transformer(object):
    """
    Transformer for categorical features.

    This class defines a transformer for categorical features that can be used with patsy for feature transformations.
    Since the transformations happen inside the tf_dataset this is just an empty shell and returns no transformation but simplyn the raw input

    """

    def __init__(self):
        pass

    def memorize_chunk(self, x):
        """
        Memorize a chunk of data.

        Args:
            x: The input data.

        Returns:
            None
        """
        pass

    def memorize_finish(self):
        """
        Finalize the memorization process.

        Returns:
            None
        """
        pass

    def transform(self, x):
        """
        Transform data using the Transformer.

        Args:
            x: The input data.

        Returns:
            x: The transformed data.
        """
        assert x.dtype != float, print(
            "please only specify categorical features to the transformer input in NATT"
        )
        return x


MLP = patsy.stateful_transform(MLP)
CubicSplineNet = patsy.stateful_transform(CubicSplineNet)
PolySplineNet = patsy.stateful_transform(PolySplineNet)
Transformer = patsy.stateful_transform(Transformer)

formula_handler = {}


class FormulaHandler:
    """
    FormulaHandler class for extracting formula data.

    This class handles the extraction of formula data for feature transformations.

    Attributes:
        formula (str): The formula string to be processed.
        matrices (tuple): A tuple containing design matrices from patsy.
        feature_nets (list): A list of feature network types.
        intercept (bool): A boolean indicating whether an intercept term is present.
        feature_names (list): A list of feature names.
        interaction_terms (list): A list of interaction term matches.
        single_terms (list): A list of single terms.
        terms (list): A list of all terms.

    Methods:
        _extract_formula_data(self, formula, data: pd.DataFrame, missing="drop"): Extracts formula data.
    """

    def __init__(self):
        pass

    def _extract_formula_data(self, formula, data: pd.DataFrame, missing="drop"):
        """
        Extract formula data.

        Args:
            formula (str): The formula string to be processed.
            data (pd.DataFrame): The input data.
            missing (str, optional): A string indicating how to handle missing data (default is "drop").

        Returns:
            Tuple containing extracted formula data.
        """

        self.formula = formula.replace(" ", "")

        if isinstance(formula, tuple(formula_handler.keys())):
            return formula_handler[type(formula)]

        if missing == "drop":
            data.dropna()
            data.reset_index(drop=True)

        if data is not None:
            self.matrices = dmatrices(formula, data=data, return_type="dataframe")

        idx = self.matrices[1].design_info.term_name_slices.values()
        sl_idx = [list(idx)[i] for i in range(len(idx))]

        X = [self.matrices[1].iloc[:, sl_idx[i]] for i in range(len(sl_idx))]
        self.formula = "".join(self.formula.split("~")[1:])
        split_formula = self.formula.split("+")[1:]

        self.split_formula = split_formula

        if sum(X[0].columns == "Intercept") == 1:
            self.feature_nets = ["MLP"] + list(
                filter(None, [feature.split("(")[0] for feature in self.split_formula])
            )
        else:
            self.feature_nets = list(
                filter(None, [feature.split("(")[0] for feature in self.split_formula])
            )

        assert self.feature_nets.count("Transformer") <= 1, print(
            "please specify only one Transformer network at the moment"
        )

        self.hidden_layer_sizes = [
            re.findall(r"\[(.*?)\]", self.split_formula[i])
            if re.findall(r"\[(.*?)\]", self.split_formula[i]) != []
            else ["128, 64, 32"]
            for i in range(len(self.split_formula))
        ]

        self.hidden_layer_sizes = [
            self.hidden_layer_sizes[i][0].split(",")
            for i in range(len(self.hidden_layer_sizes))
        ]

        self.hidden_layer_sizes = [
            [int(j) for j in self.hidden_layer_sizes[i]]
            for i in range(len(self.hidden_layer_sizes))
        ]

        if sum(X[0].columns == "Intercept") == 1:
            intercept = True
        else:
            intercept = False

        self.feature_names = [
            re.search(r"\((\w+)", item).group(1) for item in self.split_formula
        ]

        self.interaction_terms = []
        for item in self.split_formula:
            if ":" in item:
                matches = re.findall(r"\((\w+)", item)
                self.interaction_terms.append(matches)

        self.single_terms = []  #
        for item in self.split_formula:
            if ":" not in item:
                matches = re.findall(r"\((\w+)", item)
                self.single_terms.append(matches[0])

        terms = []
        for item in self.split_formula:
            matches = re.findall(r"\((\w+)", item)
            terms.append(matches)

        self.terms = [":".join(term) for term in terms]

        all_features = []
        for sublist in terms:
            all_features.extend(sublist)

        return (
            all_features,
            self.matrices[0].columns[0],
            self.feature_nets,
            intercept,
        )
