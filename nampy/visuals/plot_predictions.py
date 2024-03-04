import matplotlib.pyplot as plt
import numpy as np
from nampy.backend.interpretable_basemodel import AdditiveBaseModel


def plot_interaction(ax, preds, key, data):
    """
    Plots the interaction effect of two features.

    Args:
    preds (dict): A dictionary containing predictions.
    key (str): The key in the dictionary to identify the interaction effect.
    data (pd.DataFrame): The dataset containing the features.
    ax: The matplotlib Axes object where the plot will be drawn.

    Raises:
    AssertionError: If the provided key does not indicate an interaction effect.
    """
    # Ensure the key indicates an interaction effect

    assert (
        len(key) == 2
    ), f"There went something wrong. nampy only supports plotting of pairwrise interaction effects, however, there were {len(key)} input features passed for plotting."

    # Split the key to get individual feature names
    feature1, feature2 = key[0], key[1]

    # Plot the contour of predictions
    cs = ax.contourf(
        preds["X1"],
        preds["X2"],
        preds["predictions"].reshape(preds["X1"].shape),
        extend="both",
        levels=25,
    )

    # Add a scatter plot of the actual data points
    ax.scatter(
        data[feature1],
        data[feature2],
        c="black",
        label="Scatter Points",
        s=5,
    )

    # Add a color bar to indicate prediction values
    plt.colorbar(cs, ax=ax, label="Predictions")

    # Set the labels for the axes
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)


def plot_interaction_lss(ax, preds, key, data, param_idx):
    """
    Plots the interaction effect of two features.

    Args:
    preds (dict): A dictionary containing predictions.
    key (str): The key in the dictionary to identify the interaction effect.
    data (pd.DataFrame): The dataset containing the features.
    ax: The matplotlib Axes object where the plot will be drawn.

    Raises:
    AssertionError: If the provided key does not indicate an interaction effect.
    """
    # Ensure the key indicates an interaction effect

    assert (
        len(key) == 2
    ), f"There went something wrong. nampy only supports plotting of pairwrise interaction effects, however, there were {len(key)} input features passed for plotting."

    # Split the key to get individual feature names
    feature1, feature2 = key[0], key[1]

    # Plot the contour of predictions
    cs = ax.contourf(
        preds["X1"],
        preds["X2"],
        preds["predictions"][..., param_idx].reshape(preds["X1"].shape),
        extend="both",
        levels=25,
    )

    # Add a scatter plot of the actual data points
    ax.scatter(
        data[feature1],
        data[feature2],
        c="black",
        label="Scatter Points",
        s=5,
    )

    # Add a color bar to indicate prediction values
    plt.colorbar(cs, ax=ax, label="Predictions")

    # Set the labels for the axes
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)


def plot_categorical_feature(
    ax, data, key, target_name, predictions, plotting_data, hist=False
):
    """
    Plots the categorical feature effect.

    Args:
    ax: Matplotlib Axes object where the plot will be drawn.
    data (pd.DataFrame): The dataset containing the features and target.
    key (str): The key for the feature in the dataset.
    target_name (str): The name of the target variable in the dataset.
    predictions: The model's predictions.
    plotting_data: Data used for plotting, separate from the training or test data.
    hist (bool): Whether to include a histogram for the feature. Default is False.
    """
    # Scatter plot of actual data points

    ax.scatter(
        data[key],
        data[target_name],
        s=2,
        alpha=0.5,
        color="cornflowerblue",
    )

    # Scatter plot of predictions
    ax.scatter(
        plotting_data[key],
        predictions.squeeze(),
        color="crimson",
        marker="x",
    )

    # Optional histogram for data density
    if hist:
        ax.hist(
            data[key],
            bins=30,
            alpha=0.5,
            color="green",
            density=True,
        )

    # Setting labels for the axes
    ax.set_ylabel(target_name)
    ax.set_xlabel(key)


def plot_continuous_feature(
    ax, data, key, target_name, predictions, plotting_data, hist=False
):
    """
    Plots the continuous feature effect.

    Args:
    ax: Matplotlib Axes object where the plot will be drawn.
    data (pd.DataFrame): The dataset containing the features and target.
    key (str): The key for the feature in the dataset.
    target_name (str): The name of the target variable in the dataset.
    predictions: The model's predictions.
    plotting_data: Data used for plotting, separate from the training or test data.
    hist (bool): Whether to include a histogram for the feature. Default is False.
    """
    ax.scatter(
        data[key],
        data[target_name],
        s=2,
        alpha=0.5,
        color="cornflowerblue",
    )
    ax.plot(
        plotting_data[key],
        predictions.squeeze(),
        linewidth=2,
        color="crimson",
    )
    if hist:
        ax.hist(
            data[key],
            bins=30,
            alpha=0.5,
            color="green",
            density=True,
        )
    ax.set_ylabel(target_name)
    ax.set_xlabel(key)


def plot_additive_model(
    model,
    hist: bool = False,
):
    """
    Plot the model's predictions for both categorical and continuous features,
    including interactions.

    Args:
    model:
    """
    # Assert that the model has the 'encoder' attribute
    assert isinstance(
        model, AdditiveBaseModel
    ), "Model does not have an 'encoder' attribute"

    # Assert that the model has the '_get_training_preds' method implemented
    assert callable(
        getattr(model, "_get_plotting_preds", None)
    ), "Model does not implement '_get_plotting_preds' method"
    # Number of subplots required

    preds = model._get_plotting_preds()
    num_plots = len(preds)

    # Calculate number of rows and columns for subplots
    nrows = int(np.ceil(num_plots / 2))
    ncols = 2 if num_plots > 1 else 1

    # Generate subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 12))
    axs = axs.flatten()  # Flatten in case of a grid

    for idx, (key, value) in enumerate(preds.items()):
        shapefunc_keys = [k for k in model.inputs.keys() if k.startswith(key)]

        count = len(shapefunc_keys)
        ax = axs[idx]  # Get the current Axes instance on the grid

        if count == 2:
            plot_interaction(ax, value, shapefunc_keys, model.data)
        elif count == 1:
            if shapefunc_keys[0] in model.CAT_FEATURES:
                plot_categorical_feature(
                    ax,
                    model.data,
                    shapefunc_keys[0],
                    model.target_name,
                    value,
                    model.plotting_data,
                    hist,
                )
            else:
                plot_continuous_feature(
                    ax,
                    model.data,
                    shapefunc_keys[0],
                    model.target_name,
                    value,
                    model.plotting_data,
                    hist,
                )

        else:
            pass

        # Adjust layout for each subplot
        ax.set_title(f"Effect of {key}")

    plt.tight_layout(pad=0.4, w_pad=0.3)
    plt.show()


def plot_multi_output(
    model,
    n_classes: int = 2,
    hist: bool = False,
):
    """
    Plot the model's predictions for both categorical and continuous features,
    including interactions.

    Args:
    model:
    """
    # Assert that the model has the 'encoder' attribute
    assert isinstance(
        model, AdditiveBaseModel
    ), "Model does not have an 'encoder' attribute"

    if hasattr(model, "family"):
        n_classes = model.family.param_count

    # Assert that the model has the '_get_training_preds' method implemented
    assert callable(
        getattr(model, "_get_plotting_preds", None)
    ), "Model does not implement '_get_plotting_preds' method"
    # Number of subplots required

    preds = model._get_plotting_preds()
    nrows = len(preds)
    ncols = n_classes

    # Generate subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 12))

    for dist_param in range(n_classes):
        for idx, (key, value) in enumerate(preds.items()):
            shapefunc_keys = [k for k in model.inputs.keys() if k.startswith(key)]
            count = len(shapefunc_keys)
            ax = axs[idx, dist_param]  # Get the current Axes instance on the grid

            if count == 2:
                plot_interaction_lss(
                    ax, preds[key], shapefunc_keys, model.data, dist_param
                )
            elif count == 1:
                if key in model.CAT_FEATURES:
                    plot_categorical_feature(
                        ax,
                        model.data,
                        shapefunc_keys[0],
                        model.target_name,
                        preds[key][:, dist_param],
                        model.plotting_data,
                        hist,
                    )
                else:
                    plot_continuous_feature(
                        ax,
                        model.data,
                        shapefunc_keys[0],
                        model.target_name,
                        preds[key][:, dist_param],
                        model.plotting_data,
                        hist,
                    )
            else:
                pass

            # Adjust layout for each subplot
            ax.set_title(f"Effect of {key}")

    plt.tight_layout(pad=0.4, w_pad=0.3)
    plt.show()
