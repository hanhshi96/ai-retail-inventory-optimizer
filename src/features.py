# src/features.py

from src.config import TEST_WEEKS


def create_features(weekly_sales):
    """
    Create time-series features for weekly forecasting.
    """

    df = weekly_sales.copy()

    group_cols = [
        "store_nbr",
        "family"
    ]

    # Lag features
    df["lag_1"] = (
        df.groupby(group_cols)["sales"]
        .shift(1)
    )

    df["lag_2"] = (
        df.groupby(group_cols)["sales"]
        .shift(2)
    )

    df["lag_4"] = (
        df.groupby(group_cols)["sales"]
        .shift(4)
    )

    df["lag_52"] = (
        df.groupby(group_cols)["sales"]
        .shift(52)
    )

    # Rolling means
    df["rolling_mean_4"] = (
        df.groupby(group_cols)["sales"]
        .transform(
            lambda x:
            x.shift(1).rolling(4).mean()
        )
    )

    df["rolling_mean_8"] = (
        df.groupby(group_cols)["sales"]
        .transform(
            lambda x:
            x.shift(1).rolling(8).mean()
        )
    )

    # Calendar features
    df["month"] = df["week"].dt.month

    df["week_of_year"] = (
        df["week"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    # Encode family
    df["family_code"] = (
        df["family"]
        .astype("category")
        .cat.codes
    )

    df = df.dropna().reset_index(drop=True)

    return df


def time_based_split(df):
    """
    Use the final N weeks as test data.
    """

    all_weeks = sorted(
        df["week"].unique()
    )

    test_weeks = all_weeks[-TEST_WEEKS:]

    train_df = df[
        ~df["week"].isin(test_weeks)
    ].copy()

    test_df = df[
        df["week"].isin(test_weeks)
    ].copy()

    return train_df, test_df