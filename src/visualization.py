from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  
    sns = None

from src.model import classify_crop_viability


FEATURE_COLUMNS = [
    "i1_nutrient_level_balance",
    "i2_soil_context",
    "i3_atmospheric_support",
    "i4_atmospheric_stress",
    "i5_water_support",
    "i6_water_deficit",
    "i7_growth_readiness",
    "i8_biotic_env_stress",
    "i9_management_context",
]


def ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_crop_trajectory(simulation: dict[str, np.ndarray], crop_label: str, row_id: int) -> plt.Figure:
    time = np.arange(len(simulation["LGo (LongTerm_Growth_Outlook)"]))
    lgo = simulation["LGo (LongTerm_Growth_Outlook)"]
    lir = simulation["LIr (LongTerm_Immediate_Risk)"]
    lcv = simulation["LCv (LongTerm_Crop_Viability)"]
    crop_state = classify_crop_viability(float(lcv[-1]))

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(time, lgo, linestyle="--", linewidth=2.5, label="Growth outlook")
    ax.plot(time, lir, linestyle="--", linewidth=2.5, label="Immediate risk")
    ax.plot(time, lcv, linestyle="--", linewidth=2.5, label="Crop viability")
    ax.set_xlabel("Time Steps", fontsize=12)
    ax.set_ylabel("Levels", fontsize=12)
    ax.set_xlim(0, len(time))
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Row {row_id} | {crop_label} | State: {crop_state}", fontsize=12)
    ax.legend(loc="upper right", fontsize=10, frameon=True, edgecolor="black")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    return fig


def plot_crop_state_distribution(results_df: pd.DataFrame) -> plt.Figure:
    crop_state_counts = results_df["crop_state"].value_counts()
    state_order = ["Healthy", "Moderate", "Warning", "Critical"]
    crop_state_counts = crop_state_counts.reindex(state_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(crop_state_counts.index, crop_state_counts.values)
    ax.set_xlabel("Crop State")
    ax.set_ylabel("Number of Cases")
    ax.set_title("Distribution of Final Crop Viability States")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.tight_layout()
    return fig


def plot_growth_risk_scatter(results_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    x = results_df["LGo (LongTerm_Growth_Outlook)"]
    y = results_df["LIr (LongTerm_Immediate_Risk)"]

    ax.scatter(x, y, alpha=0.5, edgecolors="black", linewidths=0.3)
    coefficients = np.polyfit(x, y, 1)
    trend_line = np.poly1d(coefficients)
    ax.plot(x, trend_line(x), linestyle="--", linewidth=2, label="Trend Line")
    ax.set_xlabel("Growth Outlook (LGO)", fontsize=12)
    ax.set_ylabel("Immediate Risk (LIR)", fontsize=12)
    ax.set_title("Growth vs Risk Relationship", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_viability_distribution(results_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(results_df["LCv (LongTerm_Crop_Viability)"], bins=20)
    ax.set_title("Distribution of Crop Viability")
    ax.set_xlabel("Crop Viability (LCV)")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_feature_vs_viability_grid(feature_df: pd.DataFrame, results_df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for ax, column in zip(axes.flatten(), FEATURE_COLUMNS):
        x = feature_df[column]
        y = results_df["LCv (LongTerm_Crop_Viability)"]
        ax.scatter(x, y, alpha=0.5)

        coefficients = np.polyfit(x, y, 1)
        trend_line = np.poly1d(coefficients)
        color = "green" if coefficients[0] > 0 else "red"
        sorted_idx = np.argsort(x)
        ax.plot(x.iloc[sorted_idx], trend_line(x.iloc[sorted_idx]), linestyle="--", linewidth=2, color=color)
        ax.set_title(column)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("All Features vs Crop Viability", fontsize=16)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(feature_df: pd.DataFrame, results_df: pd.DataFrame) -> plt.Figure:
    heat_df = pd.concat(
        [
            feature_df[FEATURE_COLUMNS],
            results_df[
                [
                    "LGo (LongTerm_Growth_Outlook)",
                    "LIr (LongTerm_Immediate_Risk)",
                    "LCv (LongTerm_Crop_Viability)",
                ]
            ],
        ],
        axis=1,
    )

    corr = heat_df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))

    if sns is not None:
        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
    else:
        image = ax.imshow(corr, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        for row_idx in range(corr.shape[0]):
            for col_idx in range(corr.shape[1]):
                ax.text(col_idx, row_idx, f"{corr.iloc[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax)

    ax.set_title("Feature vs Crop Outcome Correlation Heatmap")
    fig.tight_layout()
    return fig


def plot_crop_health_quadrant(results_df: pd.DataFrame) -> plt.Figure:
    x = results_df["LGo (LongTerm_Growth_Outlook)"]
    y = results_df["LIr (LongTerm_Immediate_Risk)"]
    c = results_df["LCv (LongTerm_Crop_Viability)"]
    x_cut = x.median()
    y_cut = y.median()

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(x, y, c=c, cmap="viridis", alpha=0.8, edgecolors="black", linewidths=0.3)
    ax.axvline(x_cut, linestyle="--", linewidth=1.5)
    ax.axhline(y_cut, linestyle="--", linewidth=1.5)

    ax.text(x_cut + 0.12, y_cut - 0.08, "High Growth /\n Low Risk\nHealthy Zone", fontsize=11, weight="bold")
    ax.text(x_cut + 0.12, y_cut + 0.05, "High Growth /\n High Risk\nWarning Zone", fontsize=11, weight="bold")
    ax.text(x_cut - 0.32, y_cut - 0.08, "Low Growth / Low Risk\nModerate Zone", fontsize=11, weight="bold")
    ax.text(x_cut - 0.32, y_cut + 0.05, "Low Growth /\n High Risk\nCritical Zone", fontsize=11, weight="bold")

    colorbar = fig.colorbar(scatter)
    colorbar.set_label("Crop Viability (LCV)")
    ax.set_xlabel("Long-Term Growth Outlook (LGO)")
    ax.set_ylabel("Long-Term Immediate Risk (LIR)")
    ax.set_title("Crop Health Quadrant: Growth Outlook vs Risk Levels")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def calculate_feature_importance(feature_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    targets = {
        "LCv (LongTerm_Crop_Viability)": "Crop Viability",
        "LGo (LongTerm_Growth_Outlook)": "Growth Outlook",
        "LIr (LongTerm_Immediate_Risk)": "Immediate Risk",
    }

    rows = []
    for feature in FEATURE_COLUMNS:
        for target_col, target_name in targets.items():
            correlation = np.corrcoef(feature_df[feature], results_df[target_col])[0, 1]
            rows.append(
                {
                    "feature": feature,
                    "target": target_name,
                    "correlation": correlation,
                    "importance": abs(correlation),
                }
            )

    return pd.DataFrame(rows).sort_values(by="importance", ascending=False)


def plot_top_feature_importance(importance_df: pd.DataFrame) -> plt.Figure:
    df_plot = importance_df[importance_df["target"] == "Crop Viability"].head(8)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_plot["feature"], df_plot["importance"])
    ax.set_xlabel("Correlation Strength (|r|)")
    ax.set_title("Top Drivers of Crop Viability")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_feature_impact_direction(importance_df: pd.DataFrame) -> plt.Figure:
    df_plot = importance_df[importance_df["target"] == "Crop Viability"].head(8)
    colors = ["green" if correlation > 0 else "red" for correlation in df_plot["correlation"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_plot["feature"], df_plot["correlation"], color=colors)
    ax.set_xlabel("Correlation (Positive / Negative Effect)")
    ax.set_title("Feature Impact Direction on Crop Viability")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path
