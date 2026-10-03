# src/inventory.py

import numpy as np

from src.config import (
    SAFETY_FACTOR,
    SHORTAGE_THRESHOLD,
    OVERSTOCK_THRESHOLD,
    RANDOM_SEED
)


def create_inventory_snapshot(
    results,
    stores
):
    """
    Create simulated inventory snapshot
    based on the final forecast week.
    """

    decision_week = results["week"].max()

    snapshot = results[
        results["week"] == decision_week
    ].copy()

    snapshot = snapshot.rename(
        columns={
            "ml_prediction":
            "forecast_demand"
        }
    )

    snapshot = snapshot[
        [
            "week",
            "store_nbr",
            "family",
            "forecast_demand"
        ]
    ]

    snapshot["forecast_demand"] = (
        snapshot["forecast_demand"]
        .clip(lower=0)
        .round()
        .astype(int)
    )

    snapshot = snapshot.merge(
        stores,
        on="store_nbr",
        how="left"
    )

    return snapshot


def simulate_inventory(snapshot):
    """
    Generate reproducible synthetic inventory levels.
    """

    snapshot = snapshot.copy()

    snapshot = (
        snapshot
        .sort_values(
            ["store_nbr", "family"]
        )
        .reset_index(drop=True)
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    inventory_factor = rng.uniform(
        0.6,
        1.8,
        size=len(snapshot)
    )

    snapshot["current_inventory"] = (
        snapshot["forecast_demand"]
        * inventory_factor
    ).round().astype(int)

    snapshot["target_inventory"] = (
        snapshot["forecast_demand"]
        * SAFETY_FACTOR
    ).round().astype(int)

    snapshot["inventory_ratio"] = (
        snapshot["current_inventory"]
        / snapshot[
            "target_inventory"
        ].clip(lower=1)
    )

    return snapshot


def classify_inventory(ratio):
    """
    Convert inventory ratio into business status.
    """

    if ratio < SHORTAGE_THRESHOLD:
        return "SHORTAGE"

    if ratio > OVERSTOCK_THRESHOLD:
        return "OVERSTOCK"

    return "BALANCED"


def calculate_inventory_gaps(snapshot):
    """
    Calculate shortage and transferable surplus.
    """

    snapshot = snapshot.copy()

    snapshot["inventory_status"] = (
        snapshot["inventory_ratio"]
        .apply(classify_inventory)
    )

    snapshot["shortage_units"] = np.maximum(
        snapshot["target_inventory"]
        - snapshot["current_inventory"],
        0
    )

    snapshot["surplus_units"] = np.where(
        snapshot["inventory_status"]
        == "OVERSTOCK",

        np.maximum(
            snapshot["current_inventory"]
            - snapshot["target_inventory"],
            0
        ),

        0
    )

    return snapshot