"""
Extended Sandbox Analysis Script

This script performs the same analysis as sandbox.py but uses the extended
50-point clinical scenarios with controlled model bias patterns.

Model A: Underestimates below euglycemic, overestimates above (SAFER)
Model B: Overestimates below euglycemic, underestimates above (DANGEROUS)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from new_metric.cega import clarke_error_grid
from new_metric.common import get_standard_madex_params
from new_metric.common.clinical_data import get_extended_clinical_scenarios
from new_metric.madex import graph_vs_mse, mean_adjusted_exponent_error

# Get standard MADEX parameters (a=125, b=55, c=100)
MADEX_PARAMS = get_standard_madex_params()


def ensure_output_folder(folder: str) -> None:
    """Create output folder if it doesn't exist."""
    if not os.path.exists(folder):
        os.makedirs(folder)


def run_extended_analysis():
    """Run comprehensive analysis using extended 50-point scenarios."""

    output_folder = "extended_outcome"
    ensure_output_folder(output_folder)

    # Get extended scenarios repository
    repo = get_extended_clinical_scenarios()

    # Demonstrate MADEX calculation with verbose output for one example
    print("=" * 60)
    print("MADEX Calculation Example (single point)")
    print("=" * 60)
    _ = mean_adjusted_exponent_error([50], [40], verbose=True)
    print()

    # Generate comparison graphs for specific reference values
    print("Generating MADEX vs MSE comparison graphs...")
    for ref_value in [50, 120, 300]:
        graph_vs_mse(ref_value, 40, action="save", save_folder=output_folder)
    print(f"  Saved comparison graphs to {output_folder}/")
    print()

    # Collect all results
    all_results = {}
    all_scenario_data = []

    print("=" * 60)
    print("EXTENDED SCENARIO ANALYSIS (50 points per scenario)")
    print("=" * 60)

    for scenario_id in repo.get_scenario_ids():
        scenario = repo.get_scenario(scenario_id)
        y_true = scenario.y_true.tolist()
        y_pred_a = scenario.model_predictions["Model_A"].tolist()
        y_pred_b = scenario.model_predictions["Model_B"].tolist()
        description = scenario.description
        scenario_name = f"Scenario {scenario_id}: {scenario.name}"

        print(f"\n{scenario_name}")
        print(f"  Description: {description}")
        print(f"  Data points: {len(y_true)}")
        print(f"  Reference range: {min(y_true):.0f} - {max(y_true):.0f} mg/dL")

        # Store scenario data for Excel export
        scenario_df_data = {
            "Reference Values": y_true,
            f"{scenario_id} Model A": y_pred_a,
            f"{scenario_id} Model B": y_pred_b,
        }
        all_scenario_data.append(pd.DataFrame(scenario_df_data))

        # Analyze Model A
        model_a_key = f"{scenario_id} - Model A"
        all_results[model_a_key] = analyze_model(
            y_true, y_pred_a, model_a_key, output_folder
        )

        # Analyze Model B
        model_b_key = f"{scenario_id} - Model B"
        all_results[model_b_key] = analyze_model(
            y_true, y_pred_b, model_b_key, output_folder
        )

        # Print comparison
        print(
            f"\n  Model A - MSE: {all_results[model_a_key]['MSE']:.2f}, MADEX: {all_results[model_a_key]['MADEX']:.2f}"
        )
        print(
            f"  Model B - MSE: {all_results[model_b_key]['MSE']:.2f}, MADEX: {all_results[model_b_key]['MADEX']:.2f}"
        )

        madex_diff = (
            all_results[model_b_key]["MADEX"] - all_results[model_a_key]["MADEX"]
        )
        if abs(madex_diff) > 0.01:
            better = "Model A" if madex_diff > 0 else "Model B"
            print(f"  MADEX difference: {abs(madex_diff):.2f} ({better} is better)")

    # Export all scenario data to Excel
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)

    # Combine all scenario data
    combined_data = pd.concat(all_scenario_data, axis=1)
    combined_data.to_excel(f"{output_folder}/extended_scenarios.xlsx", index=False)
    print(f"  Saved scenario data to {output_folder}/extended_scenarios.xlsx")

    # Export results summary
    results_df = pd.DataFrame.from_dict(all_results, orient="index")
    results_df.to_excel(f"{output_folder}/extended_scenario_results.xlsx")
    print(f"  Saved results to {output_folder}/extended_scenario_results.xlsx")

    # Generate summary visualizations
    generate_summary_plots(results_df, output_folder)

    # Print final summary
    print("\n" + "=" * 60)
    print("SUMMARY: MADEX vs MSE Comparison")
    print("=" * 60)
    print(results_df[["MSE", "RMSE", "MADEX", "RMADEX"]].to_string())

    return results_df


def analyze_model(y_true: list, y_pred: list, name: str, output_folder: str) -> dict:
    """Analyze a single model's predictions and return metrics."""
    results = {}

    # Clarke Error Grid Analysis
    plot, zones = clarke_error_grid(y_true, y_pred, name)
    results["A"] = zones[0]
    results["B"] = zones[1]
    results["C"] = zones[2]
    results["D"] = zones[3]
    results["E"] = zones[4]

    # Save CEGA plot
    safe_name = name.replace(" ", "_").replace("-", "_")
    plot.savefig(f"{output_folder}/{safe_name}_cega.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Calculate MSE metrics
    mse = mean_squared_error(y_true, y_pred)
    results["MSE"] = round(mse, 2)
    results["RMSE"] = round(np.sqrt(mse), 2)

    # Calculate MADEX metrics using standard parameters (a=125, b=55, c=100)
    madex = mean_adjusted_exponent_error(
        y_true,
        y_pred,
        center=MADEX_PARAMS.a,
        critical_range=MADEX_PARAMS.b,
        slope=MADEX_PARAMS.c,
    )
    results["MADEX"] = round(madex, 2)
    results["RMADEX"] = round(np.sqrt(madex), 2)

    return results


def generate_summary_plots(results_df: pd.DataFrame, output_folder: str) -> None:
    """Generate summary comparison plots."""

    # Plot 1: Clarke zones comparison (all scenarios)
    fig, ax = plt.subplots(figsize=(14, 8))
    results_df[["A", "B", "C", "D", "E"]].plot(kind="bar", ax=ax)
    ax.set_xlabel("Scenario - Model")
    ax.set_ylabel("Percentage in Zone")
    ax.set_title("Clarke Error Grid Zone Distribution (Extended Scenarios)")
    ax.legend(title="Zone", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(
        f"{output_folder}/extended_scenarios_zones.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(
        f"  Saved zone distribution plot to {output_folder}/extended_scenarios_zones.png"
    )

    # Plot 2: MSE vs MADEX comparison
    fig, ax = plt.subplots(figsize=(12, 10))
    results_df[["MSE", "MADEX"]].plot(kind="barh", ax=ax)
    ax.set_xlabel("Error Value")
    ax.set_ylabel("Scenario - Model")
    ax.set_title("MSE vs MADEX Comparison (Extended Scenarios)")
    plt.tight_layout()
    fig.savefig(
        f"{output_folder}/extended_compare_mse_madex.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(
        f"  Saved MSE vs MADEX comparison to {output_folder}/extended_compare_mse_madex.png"
    )

    # Plot 3: RMSE vs RMADEX comparison
    fig, ax = plt.subplots(figsize=(12, 10))
    results_df[["RMSE", "RMADEX"]].plot(kind="barh", ax=ax)
    ax.set_xlabel("Root Error Value")
    ax.set_ylabel("Scenario - Model")
    ax.set_title("RMSE vs RMADEX Comparison (Extended Scenarios)")
    plt.tight_layout()
    fig.savefig(
        f"{output_folder}/extended_compare_rmse_rmadex.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(
        f"  Saved RMSE vs RMADEX comparison to {output_folder}/extended_compare_rmse_rmadex.png"
    )

    # Plot 4: Model A vs Model B MADEX comparison by scenario
    model_a_data = results_df[results_df.index.str.contains("Model A")]["MADEX"]
    model_b_data = results_df[results_df.index.str.contains("Model B")]["MADEX"]

    scenario_names = [idx.replace(" - Model A", "") for idx in model_a_data.index]

    x = np.arange(len(scenario_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        x - width / 2,
        model_a_data.values,
        width,
        label="Model A (Safer)",
        color="steelblue",
    )
    bars2 = ax.bar(
        x + width / 2,
        model_b_data.values,
        width,
        label="Model B (Dangerous)",
        color="coral",
    )

    ax.set_xlabel("Scenario")
    ax.set_ylabel("MADEX Value")
    ax.set_title("MADEX Comparison: Model A vs Model B by Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(
        f"{output_folder}/extended_model_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(
        f"  Saved model comparison plot to {output_folder}/extended_model_comparison.png"
    )


if __name__ == "__main__":
    results = run_extended_analysis()
