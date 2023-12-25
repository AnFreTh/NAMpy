import pandas as pd
import re
import numpy as np
import ast


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

        if missing == "drop":
            data.dropna()
            data.reset_index(drop=True)

        self.data = data
        my_string = formula

        y = "".join(formula.split("~")[0])
        y = y.replace(" ", "")

        if "+1" in my_string:
            intercept = True
            formula = formula.split("+1")[1]
        elif "-1" in my_string:
            intercept = False
            formula = formula.split("-1")[1]
        else:
            print(
                "please specify whether you want an intercept or not via '+1' or '-1'"
            )

        main_dict = {}

        split_formula = "".join(my_string.split("~")[1:])
        split_formula = split_formula.split("+")[1:]
        terms = []
        for item in split_formula:
            matches = re.findall(r"\((\w+)", item)
            matches = [match for match in matches if match in self.data.columns]
            a = ["" in self.data.columns]
            if len(matches) > 0:
                terms.append(matches)

        feature_names = [item for sublist in terms for item in sublist]
        terms = [":".join(term) for term in terms]

        my_string = formula.split("+")[1:]
        splitting_string = []
        for input in my_string:
            if ":" in input:
                inputs = input.split(":")
                splitting_string += inputs
            else:
                splitting_string.append(input)

        for input in splitting_string:
            feature_dict = {}

            shapefunc_name = input.split("(")[0]
            feature_name = input.split("(")[1].split(";")[0]

            if ")" in feature_name:
                feature_name = feature_name.replace(")", "")

            feature_name = feature_name.replace(" ", "")
            shapefunc_name = shapefunc_name.replace(" ", "")

            input = input.split(";")
            input = input[1:]

            feature_dict["Network"] = shapefunc_name
            for feature in input[:-1]:
                feature = feature.replace(" ", "")
                key, value = feature.split("=")

                try:
                    feature_dict[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    feature_dict[key] = value

            try:
                feature = input[-1].replace(" ", "")
                key, value = feature.split("=")
                if value[-1] == ")":
                    value = value[:-1]
                    try:
                        feature_dict[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        feature_dict[key] = value

                else:
                    try:
                        feature_dict[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        feature_dict[key] = value
            except IndexError:
                pass

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

            main_dict[feature_name] = feature_dict
            if feature_name not in self.data.columns:
                raise ValueError(
                    f"\n Something went wrong with the variable names. \n Variable {feature_name} is not in the provided dataframes column {self.data.columns}. \n Make sure that you do not have any white spaces in your column names."
                )

        return feature_names, y, terms, intercept, main_dict
