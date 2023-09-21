import pandas as pd
import re
import numpy as np

default_MLP = {
    "Network": "MLP",
    "sizes": [128, 64, 32],
    "dropout": 0.5,
    "activation": "relu",
}

default_Transformer = {
    "Network": "Transformer",
    "ff_dropout": 0.1,
    "dropout": 0.5,
    "attn_dropout": 0.1,
    "heads": 8,
    "depth": 4,
    "embedding_dim": 32,
}

default_CubicSplineNet = {
    "Network": "CubicSplineNet",
    "n_knots": 15,
    "activation": "linear",
    "l1_regularizer": 0.005,
    "l2_regularizer": 0.005,
    "l2_activity_regularizer": 0.005,
}


def merge_default_into_defined(defined_dict, default_dict):
    for key, value in default_dict.items():
        if key not in defined_dict:
            defined_dict[key] = value


def convert_string_to_value(input_dict: dict):
    for key, value in input_dict.items():
        if (
            isinstance(value, int)
            or isinstance(value, float)
            or isinstance(value, list)
        ):
            # If the value is already an integer, leave it unchanged
            continue
        elif value.startswith("[") and value.endswith("]"):
            # If the value looks like a list (enclosed in square brackets), convert it to a list
            try:
                input_dict[key] = eval(
                    value
                )  # Using eval to safely convert the string to a list
            except Exception as e:
                print(f"Error converting {key} value to a list: {e}")
        else:
            # If the value is not a list, check if it's an integer or float and convert accordingly
            try:
                numeric_value = float(value)
                if numeric_value.is_integer():
                    input_dict[key] = int(numeric_value)
                else:
                    input_dict[key] = numeric_value
            except ValueError:
                # If the conversion fails, leave the value as a string
                pass

    return input_dict


def extract_MLP(input: str):
    feature_dict = {}
    feature_dict["Network"] = "MLP"

    pattern = r",(?![^\[\]]*\])"

    # Split the input string using the pattern
    feature_list = re.split(pattern, input)

    # Remove leading and trailing spaces from each split part
    feature_list = [part.strip() for part in feature_list]

    for feature in feature_list[1:]:
        key, value = feature.split("=")
        feature_dict[key] = value

    merge_default_into_defined(feature_dict, default_MLP)

    return feature_list[0], convert_string_to_value(feature_dict)


def extract_Transformer(input: str):
    feature_dict = {}
    feature_dict["Network"] = "Transformer"

    pattern = r",(?![^\[\]]*\])"

    # Split the input string using the pattern
    feature_list = re.split(pattern, input)

    # Remove leading and trailing spaces from each split part
    feature_list = [part.strip() for part in feature_list]

    for feature in feature_list[1:]:
        key, value = feature.split("=")
        feature_dict[key] = value

    merge_default_into_defined(feature_dict, default_Transformer)

    return feature_list[0], convert_string_to_value(feature_dict)


def extract_CubicSplineNet(input: str):
    feature_dict = {}
    feature_dict["Network"] = "CubicSplineNet"

    pattern = r",(?![^\[\]]*\])"

    # Split the input string using the pattern
    feature_list = re.split(pattern, input)

    # Remove leading and trailing spaces from each split part
    feature_list = [part.strip() for part in feature_list]

    for feature in feature_list[1:]:
        key, value = feature.split("=")
        feature_dict[key] = value

    merge_default_into_defined(feature_dict, default_CubicSplineNet)

    return feature_list[0], convert_string_to_value(feature_dict)


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

        if not "~" in formula:
            raise ValueError(
                "~ must indicate dependent variable. Formula must be of form: y ~ MLP(x1) + MLP(x2) ..."
            )
        self.data = data
        self.formula = formula.replace(" ", "")
        y = "".join(self.formula.split("~")[0])

        if missing == "drop":
            data.dropna()
            data.reset_index(drop=True)

        self.formula = "".join(self.formula.split("~")[1:])
        split_formula = self.formula.split("+")[1:]

        self.split_formula = split_formula

        if "-1" in self.formula:
            intercept = False
            self.feature_nets = list(
                filter(None, [feature.split("(")[0] for feature in self.split_formula])
            )

        else:
            intercept = True
            self.feature_nets = ["MLP"] + list(
                filter(None, [feature.split("(")[0] for feature in self.split_formula])
            )

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

        self.main_dict = {}
        pattern = r"(\w+)\(([^)]+)\)(?=\+|$|:)"

        matches = re.findall(pattern, self.formula)
        for network, input_string in matches:
            modified_string = network + "(" + "'" + input_string + "'" + ")"

            feature_name, feature_dict = eval("extract_" + modified_string)

            feature_dict["dtype"] = self.data[feature_name].dtype

            if not "encoding" in feature_dict.keys():
                if self.data[feature_name].dtype == object:
                    if feature_dict["Network"] == "Transformer":
                        feature_dict["encoding"] = "int"
                    else:
                        feature_dict["encoding"] = "one_hot"
                elif self.data[feature_name].dtype == float:
                    feature_dict["encoding"] = "normalized"
                elif np.issubdtype(self.data[feature_name].dtype, np.integer):
                    if feature_dict["Network"] == "Transformer":
                        feature_dict["encoding"] = "int"
                    else:
                        feature_dict["encoding"] = "one_hot"
            self.main_dict[feature_name] = feature_dict

        return (
            all_features,
            y,
            self.feature_nets,
            intercept,
            self.main_dict,
        )
