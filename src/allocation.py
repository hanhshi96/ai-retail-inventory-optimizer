# src/allocation.py

import numpy as np
import pandas as pd


def build_reallocation_plan(snapshot_df):
    """
    Match surplus stores with shortage stores.

    Priority:
    1. Same city
    2. Same state
    3. Other state
    """

    transfers = []

    donor_remaining = (
        snapshot_df["surplus_units"]
        .to_dict()
    )

    families = (
        snapshot_df["family"]
        .unique()
    )

    for family in families:

        family_df = snapshot_df[
            snapshot_df["family"] == family
        ]

        receivers = (
            family_df[
                family_df["shortage_units"] > 0
            ]
            .sort_values(
                "shortage_units",
                ascending=False
            )
        )

        donors = family_df[
            family_df["surplus_units"] > 0
        ]

        for receiver_idx, receiver in receivers.iterrows():

            remaining_need = int(
                receiver["shortage_units"]
            )

            candidates = []

            for donor_idx, donor in donors.iterrows():

                available = donor_remaining.get(
                    donor_idx,
                    0
                )

                if available <= 0:
                    continue

                if (
                    donor["store_nbr"]
                    == receiver["store_nbr"]
                ):
                    continue

                if donor["city"] == receiver["city"]:

                    priority = 0
                    priority_label = "SAME_CITY"

                elif (
                    donor["state"]
                    == receiver["state"]
                ):

                    priority = 1
                    priority_label = "SAME_STATE"

                else:

                    priority = 2
                    priority_label = "OTHER_STATE"

                candidates.append({
                    "donor_idx": donor_idx,
                    "priority": priority,
                    "priority_label":
                        priority_label,
                    "available": available,
                    "donor": donor
                })

            candidates = sorted(
                candidates,
                key=lambda x: (
                    x["priority"],
                    -x["available"]
                )
            )

            for candidate in candidates:

                if remaining_need <= 0:
                    break

                donor_idx = (
                    candidate["donor_idx"]
                )

                donor = candidate["donor"]

                available = (
                    donor_remaining[
                        donor_idx
                    ]
                )

                transfer_units = min(
                    remaining_need,
                    available
                )

                if transfer_units <= 0:
                    continue

                transfers.append({
                    "family": family,
                    "from_store":
                        donor["store_nbr"],
                    "to_store":
                        receiver["store_nbr"],
                    "from_city":
                        donor["city"],
                    "to_city":
                        receiver["city"],
                    "from_state":
                        donor["state"],
                    "to_state":
                        receiver["state"],
                    "units":
                        int(transfer_units),
                    "route_priority":
                        candidate[
                            "priority_label"
                        ]
                })

                donor_remaining[
                    donor_idx
                ] -= transfer_units

                remaining_need -= (
                    transfer_units
                )

    return pd.DataFrame(transfers)

def apply_transfer_plan(
    snapshot,
    transfer_plan
):
    """
    Apply incoming/outgoing transfers
    and calculate post-allocation inventory.
    """

    outgoing = (
        transfer_plan
        .groupby(
            ["from_store", "family"],
            as_index=False
        )["units"]
        .sum()
        .rename(columns={
            "from_store": "store_nbr",
            "units": "outgoing_units"
        })
    )

    incoming = (
        transfer_plan
        .groupby(
            ["to_store", "family"],
            as_index=False
        )["units"]
        .sum()
        .rename(columns={
            "to_store": "store_nbr",
            "units": "incoming_units"
        })
    )

    post_snapshot = snapshot.merge(
        outgoing,
        on=["store_nbr", "family"],
        how="left"
    )

    post_snapshot = post_snapshot.merge(
        incoming,
        on=["store_nbr", "family"],
        how="left"
    )

    post_snapshot[
        [
            "outgoing_units",
            "incoming_units"
        ]
    ] = (
        post_snapshot[
            [
                "outgoing_units",
                "incoming_units"
            ]
        ]
        .fillna(0)
    )

    post_snapshot["post_inventory"] = (
        post_snapshot["current_inventory"]
        - post_snapshot["outgoing_units"]
        + post_snapshot["incoming_units"]
    )

    post_snapshot[
        "post_shortage_units"
    ] = np.maximum(
        post_snapshot["target_inventory"]
        - post_snapshot["post_inventory"],
        0
    )

    return post_snapshot


def calculate_allocation_kpis(
    snapshot,
    transfer_plan,
    post_snapshot
):
    """
    Calculate business impact metrics.
    """

    initial_shortage = (
        snapshot["shortage_units"]
        .sum()
    )

    initial_surplus = (
        snapshot["surplus_units"]
        .sum()
    )

    total_transferred = (
        transfer_plan["units"]
        .sum()
    )

    remaining_shortage = (
        post_snapshot[
            "post_shortage_units"
        ]
        .sum()
    )

    shortage_coverage = (
        total_transferred
        / initial_shortage
        * 100
    )

    surplus_utilization = (
        total_transferred
        / initial_surplus
        * 100
    )

    return {
        "initial_shortage":
            initial_shortage,

        "initial_surplus":
            initial_surplus,

        "units_reallocated":
            total_transferred,

        "remaining_shortage":
            remaining_shortage,

        "shortage_coverage":
            shortage_coverage,

        "surplus_utilization":
            surplus_utilization,

        "number_of_transfers":
            len(transfer_plan)
    }