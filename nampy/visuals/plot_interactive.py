import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go
from nampy.backend.interpretable_basemodel import AdditiveBaseModel
from scipy.interpolate import UnivariateSpline


def visualize_regression_predictions(model, datapoints=True, port=8050):
    """
    Launches a Dash app to visualize regression model predictions for each feature from a given dictionary.

    Parameters:
        predictions_dict (dict): Dictionary with feature names as keys and lists of predictions as values.
        X (pd.DataFrame): DataFrame containing the features of the dataset.
        feature_names (list of str): List of feature names in the dataset.
        port (int): Port number for the Dash app.
    """

    assert isinstance(
        model, AdditiveBaseModel
    ), "Model is not an AdditiveBaseModel instance"

    # Assert that the model has the '_get_training_preds' method implemented
    assert callable(
        getattr(model, "_get_plotting_preds", None)
    ), "Model does not implement '_get_plotting_preds' method"

    predictions_dict = model._get_plotting_preds()

    app = dash.Dash(__name__)

    keys = [
        key
        for idx, key in enumerate(predictions_dict.keys())
        if len(model.feature_nets[idx].inputs) == 1
    ]

    data_keys = [
        next(k for k in model.inputs.keys() if k.startswith(key))
        for idx, key in enumerate(predictions_dict.keys())
        if len(model.feature_nets[idx].inputs) == 1
    ]

    app.layout = html.Div(
        [
            html.H1("Regression Model Prediction Visualization"),
            html.Label("Select Feature:"),
            dcc.Dropdown(
                id="feature-dropdown",
                # Use the computed data_keys for the dropdown labels, but keep the value as the original key
                options=[
                    {"label": data_key, "value": key}
                    for data_key, key in zip(data_keys, keys)
                ],
                value=keys[
                    0
                ],  # The value still needs to be the original key to match with predictions_dict
            ),
            dcc.Graph(id="prediction-plot"),
        ]
    )

    @app.callback(
        Output("prediction-plot", "figure"), [Input("feature-dropdown", "value")]
    )
    def update_graph(selected_feature):
        if selected_feature not in predictions_dict:
            return go.Figure().update_layout(
                title=f"Data not available for {selected_feature}"
            )

        # extract data key to access model.data -> the renamed dataframe
        data_key = [k for k in model.inputs.keys() if k.startswith(selected_feature)][0]

        feature_range = model.plotting_data[data_key]
        predictions = predictions_dict.get(selected_feature, []).squeeze()

        fig = go.Figure()
        feature_name = data_key.split(":")[1]

        # Add trace for actual data points
        if datapoints:

            fig.add_trace(
                go.Scatter(
                    x=model.data[data_key],
                    y=model.data[model.target_name],
                    mode="markers",
                    name="Data Points",
                    marker=dict(size=3, color="cornflowerblue"),
                    hoverinfo="text",
                    hovertext=[
                        f"{feature_name}: {x}<br>Target: {y}"
                        for x, y in zip(
                            model.data[data_key], model.data[model.target_name]
                        )
                    ],
                )
            )

        # Check if predictions are categorical (integers)
        if selected_feature in model.CAT_FEATURES:  # integer or unsigned integer
            plot_mode = "markers"
        else:
            plot_mode = "lines"

        # Add trace for predictions
        fig.add_trace(
            go.Scatter(
                x=feature_range,
                y=predictions,
                mode=plot_mode,
                name="Predictions",
                line=(
                    dict(width=3, color="crimson")
                    if plot_mode == "lines"
                    else dict(color="crimson")
                ),
            )
        )

        fig.update_layout(
            title="Predictions with Data Points",
            xaxis_title=feature_name,
            yaxis_title="Predicted Value",
            hovermode="closest",  # Adjust hover mode if needed
        )

        return fig

    app.run_server(debug=True, port=port)


def visualize_additive_model(
    model,
    datapoints=True,
    port=8050,
):
    assert isinstance(
        model, AdditiveBaseModel
    ), "Model is not an AdditiveBaseModel instance"

    # Assert that the model has the '_get_training_preds' method implemented
    assert callable(
        getattr(model, "_get_plotting_preds", None)
    ), "Model does not implement '_get_plotting_preds' method"

    predictions_dict = model._get_plotting_preds()

    app = dash.Dash(__name__)

    # Separate keys for individual and interaction effects
    individual_keys = [
        key
        for idx, key in enumerate(predictions_dict.keys())
        if len(model.feature_nets[idx].inputs) == 1
    ]
    interaction_keys = [
        key
        for idx, key in enumerate(predictions_dict.keys())
        if len(model.feature_nets[idx].inputs) == 2
    ]

    individual_data_keys = [
        next(k for k in model.inputs.keys() if k.startswith(key))
        for idx, key in enumerate(individual_keys)
        if len(model.feature_nets[idx].inputs) == 1
    ]

    app.layout = html.Div(
        [
            html.H1("Model Effect Visualization"),
            html.Div(
                [
                    html.Label("Select Feature for Individual Effect:"),
                    dcc.Dropdown(
                        id="individual-feature-dropdown",
                        options=[
                            {"label": data_key, "value": key}
                            for data_key, key in zip(
                                individual_data_keys, individual_keys
                            )
                        ],
                        value=individual_keys[0] if individual_keys else None,
                    ),
                    dcc.Graph(id="individual-effect-plot"),
                ]
            ),
            html.Div(
                [
                    html.Label("Select Features for Interaction Effect:"),
                    dcc.Dropdown(
                        id="interaction-feature-dropdown",
                        options=[
                            {"label": name, "value": name} for name in interaction_keys
                        ],
                        value=interaction_keys[0] if interaction_keys else None,
                    ),
                    dcc.Graph(id="interaction-effect-plot"),
                ]
            ),
        ]
    )

    @app.callback(
        Output("individual-effect-plot", "figure"),
        [Input("individual-feature-dropdown", "value")],
    )
    def update_individual_effect_plot(selected_feature):
        if selected_feature not in predictions_dict:
            return go.Figure().update_layout(
                title=f"Data not available for {selected_feature}"
            )

        data_key = [k for k in model.inputs.keys() if k.startswith(selected_feature)][0]

        feature_range = model.plotting_data[data_key]
        predictions = predictions_dict.get(selected_feature, []).squeeze()

        feature_name = data_key.split(":")[1]

        fig = go.Figure()

        if selected_feature in model.CAT_FEATURES:  # integer or unsigned integer
            plot_mode = "markers"
        else:
            plot_mode = "lines"

        # Add trace for actual data points
        # Hoverinfo can be adjusted as per the data available
        if datapoints:
            fig.add_trace(
                go.Scatter(
                    x=model.data[data_key],
                    y=model.data[model.target_name],
                    mode="markers",
                    name="Data Points",
                    marker=dict(
                        size=3, color="cornflowerblue"
                    ),  # Adjust the size of the scatter points
                    hoverinfo="text",  # Adjust as per the data you want to show on hover
                    hovertext=[
                        f"{feature_name}: {x}<br>Target: {y}"
                        for x, y in zip(
                            model.data[data_key], model.data[model.target_name]
                        )
                    ],
                )
            )

        # Add trace for predictions
        # Ensure this is added after the scatter trace so that it's drawn on top
        fig.add_trace(
            go.Scatter(
                x=feature_range,
                y=predictions,
                mode=plot_mode,
                name="Predictions",
                line=(
                    dict(width=3, color="crimson")
                    if plot_mode == "lines"
                    else dict(color="crimson")
                ),  # Adjust the width of the line
            )
        )

        fig.update_layout(
            title="Predictions with Data Points",
            xaxis_title=feature_name,
            yaxis_title="Predicted Value",
            hovermode="closest",  # Adjust hover mode if needed
        )

        return fig

    @app.callback(
        Output("interaction-effect-plot", "figure"),
        [Input("interaction-feature-dropdown", "value")],
    )
    def update_interaction_effect_plot(selected_feature):
        if selected_feature not in predictions_dict:
            return go.Figure(data=[go.Surface()]).update_layout(
                title="Data not available for interaction effect"
            )

        data_keys = [k for k in model.inputs.keys() if k.startswith(selected_feature)]

        pred_data = predictions_dict[selected_feature]
        X1, X2, predictions = (
            pred_data["X1"],
            pred_data["X2"],
            pred_data.get("predictions", []).reshape(pred_data["X1"].shape),
        )

        # Create a meshgrid if X1 and X2 are 1D arrays
        if X1.ndim == 1 and X2.ndim == 1:
            X1, X2 = np.meshgrid(X1, X2)

        fig = go.Figure()

        # Add surface plot for predictions
        fig.add_trace(
            go.Surface(z=predictions, x=X1, y=X2, opacity=0.7)
        )  # Adjust opacity here

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title=data_keys[0].split(":")[1], range=[np.min(X1), np.max(X1)]
                ),
                yaxis=dict(
                    title=data_keys[1].split(":")[1], range=[np.min(X2), np.max(X2)]
                ),
                zaxis=dict(title="Prediction"),
            )
        )
        return fig

    app.run_server(debug=True, port=port)


def plot_NAMformer(model, datapoints=True, port=8050, smooth=True, k=5, s=0.3):
    assert callable(
        getattr(model, "_get_plotting_preds", None)
    ), "Model does not implement '_get_plotting_preds' method"

    predictions_dict = model._get_plotting_preds()
    predictions_dict.pop("raw_embeddings", None)

    app = dash.Dash(__name__)

    # Convert keys to a list to support indexing
    keys = list(predictions_dict.keys())

    app.layout = html.Div(
        [
            html.H1("Regression Model Prediction Visualization"),
            html.Label("Select Feature:"),
            dcc.Dropdown(
                id="feature-dropdown",
                options=[{"label": key, "value": key} for key in keys],
                value=(
                    keys[0] if keys else None
                ),  # Set default value to the first key if available
            ),
            dcc.Graph(id="prediction-plot"),
        ]
    )

    @app.callback(
        Output("prediction-plot", "figure"), [Input("feature-dropdown", "value")]
    )
    def update_graph(selected_feature):
        if selected_feature not in predictions_dict:
            return go.Figure().update_layout(
                title=f"Data not available for {selected_feature}"
            )

        predictions = predictions_dict.get(selected_feature, []).flatten()

        fig = go.Figure()

        if datapoints:
            fig.add_trace(
                go.Scatter(
                    x=model.data[selected_feature],
                    y=model.data[model.target_name]
                    - np.mean(model.data[model.target_name]),
                    mode="markers",
                    name="Data Points",
                    marker=dict(size=3, color="cornflowerblue"),
                )
            )

        plot_mode = (
            "markers"
            if selected_feature in getattr(model, "CAT_FEATURES", [])
            else "lines"
        )

        fig.add_trace(
            go.Scatter(
                x=model.plotting_data[selected_feature],
                y=predictions,
                mode=plot_mode,
                name="Predictions",
                line=(
                    dict(width=3, color="crimson")
                    if plot_mode == "lines"
                    else dict(color="crimson")
                ),
            )
        )

        if smooth and selected_feature not in getattr(model, "CAT_FEATURES", []):
            combined = np.vstack((model.plotting_data[selected_feature], predictions)).T

            # Sort the combined array by feature values
            combined_sorted = combined[np.argsort(combined[:, 0])]

            # Find unique feature values and average the predicted values for duplicates
            unique_features, indices = np.unique(
                combined_sorted[:, 0], return_index=True
            )
            averaged_predictions = np.array(
                [
                    np.mean(combined_sorted[indices == index, 1])
                    for index in range(len(unique_features))
                ]
            )

            # Use unique feature values and their averaged predicted values for spline interpolation
            spline = UnivariateSpline(unique_features, averaged_predictions, s=s, k=k)

            # Generate a smooth range of feature values
            smooth_feature_range = np.linspace(
                unique_features.min(), unique_features.max(), 1000
            )

            # Evaluate the spline over the smooth feature range to get smooth predicted values
            smooth_predicted_values = spline(smooth_feature_range)

            fig.add_trace(
                go.Scatter(
                    x=smooth_feature_range,
                    y=smooth_predicted_values,
                    mode=plot_mode,
                    name="Smoothed Predictions",
                    opacity=0.5,
                    line=(
                        dict(width=2, color="black")
                        if plot_mode == "lines"
                        else dict(color="black")
                    ),
                )
            )

        fig.update_layout(
            title="Predictions with Data Points",
            xaxis_title=selected_feature,
            yaxis_title="Predicted Value",
            hovermode="closest",
        )

        return fig

    app.run_server(debug=True, port=port)
