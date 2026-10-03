# src/forecasting.py

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


FEATURES = [
    "store_nbr",
    "family_code",
    "onpromotion",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_52",
    "rolling_mean_4",
    "rolling_mean_8",
    "month",
    "week_of_year"
]


def wape(y_true, y_pred):
    """
    Weighted Absolute Percentage Error.
    """

    return (
        np.abs(y_true - y_pred).sum()
        / np.abs(y_true).sum()
        * 100
    )


def evaluate_predictions(y_true, y_pred):
    """
    Calculate forecasting metrics.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    wape_score = wape(
        np.asarray(y_true),
        np.asarray(y_pred)
    )

    return {
        "MAE": mae,
        "WAPE": wape_score
    }


def baseline_forecast(test_df):
    """
    Naive forecast:
    next week's demand = previous week's demand.
    """

    predictions = test_df["lag_1"]

    metrics = evaluate_predictions(
        test_df["sales"],
        predictions
    )

    return predictions, metrics


def train_model(train_df):
    """
    Train HistGradientBoosting model.
    """

    X_train = train_df[FEATURES]
    y_train = train_df["sales"]

    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def predict_model(model, test_df):
    """
    Generate ML forecasts.
    """

    X_test = test_df[FEATURES]

    predictions = model.predict(X_test)

    predictions = np.clip(
        predictions,
        0,
        None
    )

    metrics = evaluate_predictions(
        test_df["sales"],
        predictions
    )

    return predictions, metrics


def create_forecast_results(
    test_df,
    baseline_pred,
    ml_pred
):
    """
    Create a structured prediction output.
    """

    results = test_df[
        [
            "week",
            "store_nbr",
            "family",
            "sales"
        ]
    ].copy()

    results["baseline_prediction"] = (
        np.asarray(baseline_pred)
    )

    results["ml_prediction"] = ml_pred

    results["absolute_error"] = np.abs(
        results["sales"]
        - results["ml_prediction"]
    )

    return results