import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast_results(
    results,
    save_path=None
):
    weekly_result = (
        results
        .groupby("week", as_index=False)
        .agg(
            actual_sales=("sales", "sum"),
            predicted_sales=("ml_prediction", "sum")
        )
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        weekly_result["week"],
        weekly_result["actual_sales"],
        marker="o",
        label="Actual"
    )

    plt.plot(
        weekly_result["week"],
        weekly_result["predicted_sales"],
        marker="o",
        label="Predicted"
    )

    plt.title("Actual vs Predicted Weekly Sales")
    plt.xlabel("Week")
    plt.ylabel("Sales")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

    plt.show()


def plot_shortage_before_after(
    initial_shortage,
    remaining_shortage,
    save_path=None
):
    before_after = pd.DataFrame({
        "Stage": [
            "Before Reallocation",
            "After Reallocation"
        ],
        "Shortage Units": [
            initial_shortage,
            remaining_shortage
        ]
    })

    plt.figure(figsize=(7, 5))

    plt.bar(
        before_after["Stage"],
        before_after["Shortage Units"]
    )

    plt.title(
        "Inventory Shortage Before vs After Reallocation"
    )

    plt.ylabel("Shortage Units")
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

    plt.show()