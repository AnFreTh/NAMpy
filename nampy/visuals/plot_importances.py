import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def visualize_importances(model, title="Importances"):
    """
    Plot feature importances.
    Args:
        all (bool, optional): If True, plot importances for all features; otherwise, plot average importances (default is False).
        title (str, optional): Title of the plot (default is "Importances").
    """

    # Assert that the model has the 'encoder' attribute
    assert hasattr(model, "encoder"), "Model does not have an 'encoder' attribute"

    dataset = model._get_dataset(model.data, shuffle=False)
    importances = model.predict(dataset, verbose=0)["importances"]
    column_list = model.TRANSFORMER_FEATURES
    # column_list = []
    # for i, feature in enumerate(model.TRANSFORMER_FEATURES):
    #    column_list.extend([feature] * model.inputs[feature].shape[1])
    importances = pd.DataFrame(importances[:, 1:], columns=column_list)
    average_importances = []
    for col_name in model.TRANSFORMER_FEATURES:
        average_importances.append(importances.filter(like=col_name).sum(axis=1))
    importances = pd.DataFrame(
        {
            column_name: column_data
            for column_name, column_data in zip(
                model.TRANSFORMER_FEATURES, average_importances
            )
        }
    )
    imps_sorted = importances.mean().sort_values(ascending=False)
    imps_sorted = imps_sorted / sum(imps_sorted)
    plt.figure(figsize=(6, 4))
    ax = imps_sorted.plot.bar()
    for i, p in enumerate(ax.patches):
        ax.annotate(
            str(np.round(p.get_height(), 3)),
            (p.get_x(), p.get_height() * 1.01),
            rotation=90,
            fontsize=12,
        )
        ax.annotate(
            str(imps_sorted.index[i]),
            (p.get_x() + 0.1, 0.01),
            rotation=90,
            fontsize=15,
        )
    ax.xaxis.set_tick_params(labelbottom=False)
    plt.title(title)
    plt.show()


def visualize_categorical_importances(model, title="Importances"):
    """
    Plot categorical feature importances.
    Args:
        title (str, optional): Title of the plot (default is "Importances").
        n_top_categories (int, optional): Number of top categories to consider (default is 5).
    """

    # Assert that the model has the 'encoder' attribute
    assert hasattr(model, "encoder"), "Model does not have an 'encoder' attribute"

    dataset = model._get_dataset(model.data, shuffle=False)
    importances = model.predict(dataset, verbose=0)["importances"]
    column_list = []
    for i, feature in enumerate(model.TRANSFORMER_FEATURES):
        column_list.extend([feature] * model.inputs[feature].shape[1])
    importances = pd.DataFrame(importances[:, 1:], columns=column_list)
    average_importances = []
    for col_name in model.TRANSFORMER_FEATURES:
        average_importances.append(importances.filter(like=col_name).sum(axis=1))
    importances = pd.DataFrame(
        {
            column_name: column_data
            for column_name, column_data in zip(
                model.TRANSFORMER_FEATURES, average_importances
            )
        }
    )
    result_dict = {}
    imps_sorted = importances.mean().sort_values(ascending=False)
    for category in model.TRANSFORMER_FEATURES:
        unique_vals = model.data[category].unique()
        for val in unique_vals:
            bsc = model.data[model.data[category] == val].index
            imps_value = importances.loc[bsc.values][category].mean()
            result_dict[val] = imps_value
    sort_dict = dict(sorted(result_dict.items(), key=lambda item: item[1]))
    sorted_df = pd.DataFrame([sort_dict])
    sorted_df = sorted_df / np.sum(sorted_df, axis=1)[0]
    imps_sorted = sorted_df.iloc[:, -5:].transpose()
    plt.figure(figsize=(12, 4))
    ax = imps_sorted.plot.bar(legend=None)
    for p in ax.patches:
        ax.annotate(
            str(np.round(p.get_height(), 4)), (p.get_x(), p.get_height() * 1.01)
        )
    plt.title(title)
    plt.show()


def visualize_heatmap_importances(model, cat1, cat2):
    """
    Plot heatmap of feature importances for two categorical features.
    Args:
        cat1 (str): Name of the first categorical feature.
        cat2 (str): Name of the second categorical feature.
        title (str, optional): Title of the plot (default is "Importances").
    """

    # Assert that the model has the 'encoder' attribute
    assert hasattr(model, "encoder"), "Model does not have an 'encoder' attribute"

    dataset = model._get_dataset(model.data, shuffle=False)
    importances = model.predict(dataset, verbose=0)["importances"]
    column_list = []
    for i, feature in enumerate(model.TRANSFORMER_FEATURES):
        column_list.extend([feature] * model.inputs[feature].shape[1])
    importances = pd.DataFrame(importances[:, :-1], columns=column_list)
    average_importances = []
    for col_name in model.TRANSFORMER_FEATURES:
        average_importances.append(importances.filter(like=col_name).sum(axis=1))
    importances = pd.DataFrame(
        {
            column_name: column_data
            for column_name, column_data in zip(
                model.TRANSFORMER_FEATURES, average_importances
            )
        }
    )
    result_dict = {}

    for key, value in model.feature_information.items():
        for val in value["inputs"]:
            if val["feature_name"] == cat1:
                cat1 = val["identifier"]
            if val["feature_name"] == cat2:
                cat2 = val["identifier"]

    unique_vals = model.data[cat1].unique()
    for val1 in unique_vals:
        temp_dict = {}
        bsc1 = model.data[model.data[cat1] == val1].index
        cat_df = model.data.loc[bsc1.values]
        unique_vals2 = model.data[cat2].unique()
        for val2 in unique_vals2:
            bsc2 = cat_df[cat_df[cat2] == val2].index
            cat_values = importances.loc[bsc1.values]
            cat_values = importances.loc[bsc2.values]
            temp_dict[val2] = cat_values[cat2].sum()
        result_dict[val1] = temp_dict
    plotting_importances = pd.DataFrame(result_dict) / np.sum(pd.DataFrame(result_dict))
    plotting_importances[plotting_importances == 0] = None
    fig, axs = plt.subplots(
        ncols=2, gridspec_kw=dict(width_ratios=[10, 0.5]), figsize=(4, 4)
    )
    sns.heatmap(plotting_importances, annot=True, fmt=".2%", cbar=False, ax=axs[0])
    fig.colorbar(axs[0].collections[0], cax=axs[1])
    plt.show()
