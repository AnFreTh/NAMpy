import math
import numpy as np
from matplotlib import pyplot as plt


def choose_subplot_dimensions(k):
    if k < 4:
        return k, 1
    elif k < 11:
        return math.ceil(k / 2), 2
    else:
        # I've chosen to have a maximum of 5 columns
        return math.ceil(k / 5), 5


def generate_subplots(k, row_wise=False, figsize=(16, 16)):
    """
    Generate a grid of subplots for plotting multiple figures.

    Args:
        k (int): Number of subplots to create.
        row_wise (bool, optional): Arrange subplots in row-wise order (True) or column-wise order (False). Defaults to False.
        figsize (tuple, optional): Figure size in inches, as a tuple (width, height). Defaults to (16, 16).

    Returns:
        tuple: A tuple containing the generated matplotlib Figure and an array of Axes objects.

    Example:
        fig, axes = generate_subplots(4, row_wise=True)
        axes[0].plot(x1, y1)
        axes[1].plot(x2, y2)
        # Continue plotting on other axes as needed.
        plt.show()
    """
    nrow, ncol = choose_subplot_dimensions(k)
    # Choose your share X and share Y parameters as you wish:
    figure, axes = plt.subplots(
        nrow,
        ncol,
        sharex=False,
        sharey=False,
        figsize=figsize,
    )

    # Check if it's an array. If there's only one plot, it's just an Axes obj
    if not isinstance(axes, np.ndarray):
        return figure, np.atleast_1d(axes)
    else:
        # Choose the traversal you'd like: 'F' is col-wise, 'C' is row-wise
        axes = axes.flatten(order=("C" if row_wise else "F"))

        # Delete any unused axes from the figure, so that they don't show
        # blank x- and y-axis lines
        for idx, ax in enumerate(axes[k:]):
            figure.delaxes(ax)

            # Turn ticks on for the last ax in each column, wherever it lands
            idx_to_turn_on_ticks = idx + k - ncol if row_wise else idx + k - 1
            for tk in axes[idx_to_turn_on_ticks].get_xticklabels():
                tk.set_visible(True)

        axes = axes[:k]
        return figure, axes
