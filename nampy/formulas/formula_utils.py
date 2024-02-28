from itertools import combinations
import pandas as pd


def all_features_additive_model(df: pd.DataFrame, target: str, intercept: bool = False):
    """helper function that creates an additive model for every feature in a pd.dataframe

    Args:
        df (pd.DataFrame): dataframe
        target (str): target variable
        intercept (bool, optional): if an intercept is fit or not. Defaults to False.

    Returns:
        str: formula string that can be plugged into a nampy model
    """
    assert target in df.columns, f"Make sure that {target} is in the dataframe columns"

    my_formula = f"{target} ~"
    if intercept:
        my_formula += " 1"
    else:
        my_formula += " - 1"
    for col in df.columns:
        if col == target:
            pass
        else:
            my_formula += f" + MLP({col})"

    return my_formula


def all_features_pairwsie_interactions_additive_model(
    df: pd.DataFrame, target: str, intercept: bool = False
):
    """helper function that creates an additive model for every feature and pairwise feature interactions between every feature in a pd.dataframe

    Args:
        df (pd.DataFrame): dataframe
        target (str): target variable
        intercept (bool, optional): if an intercept is fit or not. Defaults to False.

    Returns:
        str: formula string that can be plugged into a nampy model
    """
    assert target in df.columns, f"Make sure that {target} is in the dataframe columns"

    my_formula = all_features_additive_model(df, target, intercept)

    df = df.drop(target, axis=1)
    # Using itertools.combinations to get unique pairwise combinations
    pairwise_combinations = list(combinations(df.columns, 2))

    for comb in pairwise_combinations:
        my_formula += f" + MLP({comb[0]}):MLP({comb[1]})"

    return my_formula
