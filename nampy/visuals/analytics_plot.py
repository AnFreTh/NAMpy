import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np


def visual_analysis(preds, targets):
    """creates an analytics plot for the additive models and compares true distribution to fitted ditsribution as well as residuals"""

    residuals = targets - preds
    # Create a gAMLSS-like plot
    sns.set(style="whitegrid", font_scale=0.8)
    plt.figure(figsize=(6, 6))
    # Histogram of residuals
    plt.subplot(2, 2, 1)
    sns.histplot(residuals, kde=True, color="blue")
    plt.axvline(x=0, color="red", linestyle="--")
    plt.xlabel("Residuals")
    plt.ylabel("Density")
    plt.title("Histogram of Residuals")
    # Q-Q plot of residuals
    plt.subplot(2, 2, 2)
    stats.probplot(residuals, plot=plt)
    plt.title("Q-Q Plot of Residuals")
    # Distribution of response variable
    plt.subplot(2, 2, 3)
    sns.histplot(targets, kde=True, color="green")
    plt.xlabel("Y")
    plt.ylabel("Density")
    plt.title("Distribution of Response Variable")
    # Distribution of response variable
    plt.subplot(2, 2, 4)
    sns.histplot(preds, kde=True, color="red")
    plt.xlabel("Y")
    plt.ylabel("Density")
    plt.title("Distribution of predictions")
    # Apply the same x-axis limits to both distribution plots
    plt.subplot(2, 2, 3)
    plt.xlim(-2, 4)
    plt.subplot(2, 2, 4)
    plt.xlim(-2, 4)
    plt.tight_layout()
    plt.show()
