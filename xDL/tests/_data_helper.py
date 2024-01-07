import numpy as np
import pandas as pd


def data_gen():
    n_samples = 1000
    # Simulate continuous feature (e.g., uniformly distributed float values)
    continuous_feature = np.random.uniform(0, 10, n_samples)

    # Simulate categorical feature (e.g., string values)
    categories = ["Category1", "Category2", "Category3"]
    categorical_feature = np.random.choice(categories, n_samples)

    # Simulate integer feature (e.g., integers in a range)
    integer_feature = np.random.randint(0, 100, n_samples)

    # Simulate target variable (e.g., normally distributed float values)
    target = np.random.normal(5, 2, n_samples)  # Mean = 5, Std = 2

    # Create a DataFrame
    data = pd.DataFrame(
        {
            "ContinuousFeature": continuous_feature,
            "CategoricalFeature": categorical_feature,
            "IntegerFeature": integer_feature,
            "target": target,
        }
    )

    return data
