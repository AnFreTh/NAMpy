import pandas as pd
import re
import numpy as np
import ast


class FormulaHandler:
    def __init__(self):
        pass

    def _remove_suffix(self, text, suffix):
        if text.endswith(suffix):
            return text[: -len(suffix)]
        return text

    def _get_intercept(self, formula):
        my_string = formula

        y = "".join(formula.split("~")[0]).replace(" ", "")

        if "+1" in my_string:
            intercept = True
            formula = formula.split("+")[1]
        elif "-1" in my_string:
            intercept = False
            formula = formula.split("-")[1]
        else:
            print(
                "please specify whether you want an intercept or not via '+1' or '-1'"
            )

        return intercept, y

    def _get_terms_and_feature_names(self, formula):
        split_formula = "".join(formula.split("~")[1:])
        split_formula = split_formula.split("+")[1:]
        terms = []
        for item in split_formula:
            matches = re.findall(r"\((\w+)", item)
            matches = [match for match in matches if match in self.data.columns]
            terms.append(matches)

        feature_names = [item for sublist in terms for item in sublist]
        terms = [":".join(term) for term in terms]

        return terms, feature_names

    def _extract_formula_data(self, formula, data: pd.DataFrame, missing="drop"):
        if not "~" in formula:
            raise ValueError(
                "~ must indicate dependent variable. Formula must be of form: y ~ MLP(x1) + MLP(x2) ..."
            )

        if missing == "drop":
            data = data.dropna().reset_index(drop=True)

        self.data = data

        intercept, y = self._get_intercept(formula)
        terms, feature_names = self._get_terms_and_feature_names(formula)

        my_string = formula.split("+")[1:]
        splitting_string = []
        for input in my_string:
            if ":" in input:
                inputs = input.split(":")
                splitting_string.extend([part + "_." for part in inputs])
            else:
                splitting_string.append(input)

        main_dict = {}
        for input in splitting_string:
            feature_dict = {}
            shapefunc_name = input.split("(")[0]
            feature_identifier = input.split("(")[1].split(";")[0]

            if ")" in feature_identifier:
                feature_identifier = feature_identifier.replace(")", "")
            feature_identifier = feature_identifier.replace(" ", "")
            shapefunc_name = shapefunc_name.replace(" ", "")
            feature_name = self._remove_suffix(feature_identifier, "_.")

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

            main_dict[feature_identifier] = feature_dict
            if feature_name not in self.data.columns:
                raise ValueError(
                    f"\n Something went wrong with the variable names. \n Variable {feature_name} is not in the provided dataframes column {self.data.columns}. \n Make sure that you do not have any white spaces in your column names."
                )

        return feature_names, y, terms, intercept, main_dict
