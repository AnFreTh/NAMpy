import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from xDL.backend.interpretable_basemodel import AdditiveBaseModel


def visualize_distributional_regression_predictions(model, datapoints=True, port=8050):
    """
    Launches a Dash app to visualize regression model predictions for each feature from a given dictionary.

    Parameters:
        predictions_dict (dict): Dictionary with feature names as keys and lists of predictions as values.
        X (pd.DataFrame): DataFrame containing the features of the dataset.
        feature_names (list of str): List of feature names in the dataset.
        port (int): Port number for the Dash app.
    """
    # Assert that the model has the 'encoder' attribute
    assert isinstance(
        model, AdditiveBaseModel
    ), "Model does not have an 'encoder' attribute"

    assert hasattr(model, "family")

    # Assert that the model has the '_get_training_preds' method implemented
    assert callable(
        getattr(model, "_get_plotting_preds", None)
    ), "Model does not implement '_get_plotting_preds' method"

    predictions_dict = model._get_plotting_preds()

    app = dash.Dash(__name__)

    keys = [key for key in predictions_dict.keys()]

    app.layout = html.Div(
        [
            html.H1("Regression Model Prediction Visualization"),
            html.Label("Select Feature:"),
            dcc.Dropdown(
                id="feature-dropdown",
                options=[{"label": name, "value": name} for name in keys],
                value=keys[0],
            ),
            dcc.Graph(id="prediction-plot"),
        ]
    )

    @app.callback(
        Output("prediction-plot", "figure"), [Input("feature-dropdown", "value")]
    )
    def update_graph(selected_feature):
        if (
            selected_feature not in predictions_dict
            or selected_feature not in model.plotting_data
        ):
            return go.Figure().update_layout(
                title=f"Data not available for {selected_feature}"
            )

        feature_range = model.plotting_data[selected_feature]
        predictions = predictions_dict.get(selected_feature, []).squeeze()

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
                    x=model.data[selected_feature],
                    y=model.data[model.target_name],
                    mode="markers",
                    name="Data Points",
                    marker=dict(
                        size=3, color="cornflowerblue"
                    ),  # Adjust the size of the scatter points
                    hoverinfo="text",  # Adjust as per the data you want to show on hover
                    hovertext=[
                        f"{selected_feature}: {x}<br>Target: {y}"
                        for x, y in zip(
                            model.data[selected_feature], model.data[model.target_name]
                        )
                    ],
                )
            )

        # Add trace for predictions
        # Ensure this is added after the scatter trace so that it's drawn on top
        for idx in range(predictions.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=feature_range,
                    y=predictions[:, idx],
                    mode=plot_mode,
                    name=model.family.param_names[idx],
                    line=dict(width=3) if plot_mode == "lines" else dict(),
                )
            )

        fig.update_layout(
            title="Predictions with Data Points",
            xaxis_title=selected_feature,
            yaxis_title="Predicted Value",
            hovermode="closest",  # Adjust hover mode if needed
        )

        return fig

    app.run_server(debug=True, port=port)


def visualize_distributional_additive_model(
    model,
    datapoints=True,
    port=8050,
):
    # Assert that the model has the 'encoder' attribute
    assert isinstance(
        model, AdditiveBaseModel
    ), "Model does not have an 'encoder' attribute"

    # Assert that the model has the '_get_training_preds' method implemented
    assert callable(
        getattr(model, "_get_plotting_preds", None)
    ), "Model does not implement '_get_plotting_preds' method"

    predictions_dict = model._get_plotting_preds()

    app = dash.Dash(__name__)

    # Separate keys for individual and interaction effects
    individual_keys = [key for key in predictions_dict.keys() if "_._" not in key]
    interaction_keys = [key for key in predictions_dict.keys() if "_._" in key]

    app.layout = html.Div(
        [
            html.H1("Model Effect Visualization"),
            html.Div(
                [
                    html.Label("Select Feature for Individual Effect:"),
                    dcc.Dropdown(
                        id="individual-feature-dropdown",
                        options=[
                            {"label": name, "value": name} for name in individual_keys
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
                    html.Label("Select Parameter Index:"),
                    dcc.Dropdown(
                        id="parameter-index-dropdown",
                        options=[
                            {"label": f"{model.family.param_names[i]}", "value": i}
                            for i in range(model.family.param_count)
                        ],
                        value=0,  # Default value
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
        if (
            selected_feature not in predictions_dict
            or selected_feature not in model.plotting_data
        ):
            return go.Figure().update_layout(
                title=f"Data not available for {selected_feature}"
            )

        feature_range = model.plotting_data[selected_feature]
        predictions = predictions_dict.get(selected_feature, []).squeeze()

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
                    x=model.data[selected_feature],
                    y=model.data[model.target_name],
                    mode="markers",
                    name="Data Points",
                    marker=dict(
                        size=3, color="cornflowerblue"
                    ),  # Adjust the size of the scatter points
                    hoverinfo="text",  # Adjust as per the data you want to show on hover
                    hovertext=[
                        f"{selected_feature}: {x}<br>Target: {y}"
                        for x, y in zip(
                            model.data[selected_feature], model.data[model.target_name]
                        )
                    ],
                )
            )

        # Add trace for predictions
        # Ensure this is added after the scatter trace so that it's drawn on top
        for idx in range(predictions.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=feature_range,
                    y=predictions[:, idx],
                    mode=plot_mode,
                    name=model.family.param_names[idx],
                    line=dict(width=3) if plot_mode == "lines" else dict(),
                )
            )

        fig.update_layout(
            title="Predictions with Data Points",
            xaxis_title=selected_feature,
            yaxis_title="Predicted Value",
            hovermode="closest",  # Adjust hover mode if needed
        )

        return fig

    @app.callback(
        Output("interaction-effect-plot", "figure"),
        [
            Input("interaction-feature-dropdown", "value"),
            Input("parameter-index-dropdown", "value"),
        ],
    )
    def update_interaction_effect_plot(selected_feature, selected_param_idx):
        if selected_feature not in predictions_dict:
            return go.Figure(data=[go.Surface()]).update_layout(
                title="Data not available for interaction effect"
            )

        fig = go.Figure()
        pred_data = predictions_dict[selected_feature]

        X1, X2, predictions = (
            pred_data["X1"],
            pred_data["X2"],
            pred_data.get("predictions", [])[..., selected_param_idx].reshape(
                pred_data["X1"].shape
            ),
        )

        if X1.ndim == 1 and X2.ndim == 1:
            X1, X2 = np.meshgrid(X1, X2)

        fig.add_trace(
            go.Surface(z=predictions, x=X1, y=X2, opacity=0.7)  # Adjust opacity here
        )

        named_keys = selected_feature.split("_._")
        # if datapoints:
        #    actual_X1, actual_X2, actual_Z = (
        #        model.data[named_keys[0]],
        #        model.data[named_keys[1]],
        #        model.data[model.target_name],
        #    )
        #    fig.add_trace(
        #        go.Scatter3d(
        #            x=actual_X1,
        #            y=actual_X2,
        #            z=actual_Z,
        #            mode="markers",
        #            marker=dict(size=3, color="cornflowerblue"),
        #            name="Actual Data",
        #        )
        #    )

        fig.update_layout(
            scene=dict(
                xaxis=dict(title=named_keys[0], range=[np.min(X1), np.max(X1)]),
                yaxis=dict(title=named_keys[1], range=[np.min(X2), np.max(X2)]),
                zaxis=dict(title="Prediction"),
            )
        )
        return fig

    app.run_server(debug=True, port=port)
