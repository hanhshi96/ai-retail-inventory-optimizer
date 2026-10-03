# src/data.py

import pandas as pd
from huggingface_hub import hf_hub_download

from src.config import REPO_ID, START_DATE


def download_dataset():
    """
    Download train and store metadata from Hugging Face.
    """

    train_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="train.csv",
        repo_type="dataset"
    )

    stores_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="stores.csv",
        repo_type="dataset"
    )

    return train_path, stores_path


def load_data():
    """
    Load sales and store metadata.
    """

    train_path, stores_path = download_dataset()

    sales = pd.read_csv(
        train_path,
        parse_dates=["date"]
    )

    stores = pd.read_csv(stores_path)

    return sales, stores


def create_weekly_sales(sales):
    """
    Convert daily sales into weekly sales at:
    store × product-family × week level.
    """

    sales_work = sales[
        sales["date"] >= START_DATE
    ].copy()

    sales_work["week"] = (
        sales_work["date"]
        .dt.to_period("W-SUN")
        .apply(lambda x: x.start_time)
    )

    weekly_sales = (
        sales_work
        .groupby(
            ["week", "store_nbr", "family"],
            as_index=False
        )
        .agg(
            sales=("sales", "sum"),
            onpromotion=("onpromotion", "sum")
        )
    )

    weekly_sales = weekly_sales.sort_values(
        ["store_nbr", "family", "week"]
    ).reset_index(drop=True)

    # Remove incomplete final week
    last_date = sales_work["date"].max()

    last_week_start = (
        last_date
        .to_period("W-SUN")
        .start_time
    )

    weekly_sales = weekly_sales[
        weekly_sales["week"] < last_week_start
    ].copy()

    return weekly_sales