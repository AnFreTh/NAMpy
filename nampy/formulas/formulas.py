import pandas as pd
import re
import numpy as np
import ast


class FormulaHandler_old:
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
        if intercept:
            main_dict["Intercept"] = True
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
                    print(
                        "first",
                    )
                    feature_dict[key] = value

            try:
                feature = input[-1].replace(" ", "")
                key, value = feature.split("=")
                if value[-1] == ")":
                    value = value[:-1]
                    try:
                        feature_dict[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        print("second", value)
                        feature_dict[key] = value

                else:
                    try:
                        feature_dict[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        print("third", value)
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


class FormulaHandler:
    def __init__(self):
        pass

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

    def _parse_feature_options(self, options_str):
        # Initialize preprocessing options and shape function arguments
        preprocessing = {}
        shapefunc_args = {}

        # Split options by semicolon and process each option
        for option in options_str.split(";"):
            if "=" in option:
                key, value = option.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"')  # Remove any double quotes

                # Check if value is a list and convert it
                if value.startswith("[") and value.endswith("]"):
                    value = [int(x.strip()) for x in value.strip("[]").split(",")]
                elif value.isdigit():
                    value = int(value)  # Convert numeric strings to integers

                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass

                # Determine if option is preprocessing or shape function argument
                if key in ["encoding", "n_bins"]:
                    preprocessing[key] = value
                else:
                    shapefunc_args[key] = value

        return preprocessing, shapefunc_args

    def _parse_feature(self, feature_str):
        # Extract network name, feature name, and options
        network_name, rest = feature_str.split("(", 1)
        feature_name, options_str = (
            rest.split(")", 1)[0].split(";", 1)
            if ";" in rest
            else (rest.split(")", 1)[0], None)
        )

        preprocessing, shapefunc_args = {}, {}
        if options_str:
            preprocessing, shapefunc_args = self._parse_feature_options(options_str)

        return {
            "network_name": network_name.strip(),  # "MLP", "MyNet1", "MyNet2", etc.
            "feature_name": feature_name.strip(),
            "preprocessing": preprocessing,
            "shapefunc_args": shapefunc_args,
        }

    def _process_interaction(self, interaction_str):
        # Split interaction terms and parse each feature
        features = interaction_str.split(":")
        inputs = [self._parse_feature(feature) for feature in features]
        shapefunc_name = inputs[0]["network_name"]
        # delete network name from inputs
        for feature in inputs:
            feature.pop("network_name")

        # Aggregate shape function arguments from all features
        shapefunc_args = {}
        for feature in inputs:
            shapefunc_args.update(feature.pop("shapefunc_args"))

        return inputs, shapefunc_args, shapefunc_name

    def _process_shapefunction(self, s):
        shapefunc_dict = {}

        if ":" in s:  # Interaction terms present
            inputs, shapefunc_args, shapefunc_name = self._process_interaction(s)
        else:  # Single feature or features separated by "+"
            features = s.split("+")
            inputs = [self._parse_feature(feature.strip()) for feature in features]
            shapefunc_name = inputs[0]["network_name"]
            # delete network name from inputs
            for feature in inputs:
                feature.pop("network_name")
            # Aggregate shape function arguments from all features
            shapefunc_args = {}
            for feature in inputs:
                shapefunc_args.update(feature.pop("shapefunc_args"))

        # Assign inputs and shape function arguments to the shape function dictionary
        shapefunc_dict["inputs"] = inputs
        shapefunc_dict["Network"] = shapefunc_name
        if shapefunc_args:
            shapefunc_dict["shapefunc_args"] = shapefunc_args
        else:
            shapefunc_dict["shapefunc_args"] = {}

        return shapefunc_dict

    def extract_formula_data(self, formula: str, data: pd.DataFrame):
        intercept, y = self._get_intercept(formula)
        processed_data = {}
        for idx, mystr in enumerate(formula.split("+")[1:]):
            processed_data[f"Shapefunction{idx}"] = self._process_shapefunction(
                mystr.strip()
            )

        feature_names = [
            input["feature_name"]
            for value in processed_data.values()
            for input in value["inputs"]
        ]

        network_identifier = []

        for key, value in processed_data.items():
            for input in value["inputs"]:
                input["preprocessing"]["dtype"] = data[input["feature_name"]].dtype
                # first create identifier
                input["identifier"] = f"{key}:{input['feature_name']}"
                network_identifier.append(f"{key}:{input['feature_name']}")
                # create encoding standard values based on dtypes
                if not "encoding" in input["preprocessing"]:
                    if input["preprocessing"]["dtype"] == object:
                        if value["Network"] == "Transformer":
                            input["preprocessing"]["encoding"] = "int"
                        else:
                            input["preprocessing"]["encoding"] = "one_hot"
                    elif input["preprocessing"]["dtype"] == float:
                        input["preprocessing"]["encoding"] = "normalized"
                    elif np.issubdtype(input["preprocessing"]["dtype"], np.integer):
                        if value["Network"] == "Transformer":
                            input["preprocessing"]["encoding"] = "int"
                        else:
                            input["preprocessing"]["encoding"] = "one_hot"

        return feature_names, y, intercept, network_identifier, processed_data
