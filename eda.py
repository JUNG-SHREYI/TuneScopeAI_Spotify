"""TuneScope AI - Exploratory Data Analysis helpers.

This module generates the data tables and chart images used by templates/eda.html.
It never changes the input DataFrame; imputations are calculated on a deep copy.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


POPULARITY_BINS = [-0.001, 20, 40, 60, 80, 100]
POPULARITY_LABELS = [
    "Very Low (0–20)",
    "Low (21–40)",
    "Moderate (41–60)",
    "High (61–80)",
    "Very High (81–100)",
]


def dataframe_to_html(df, table_class="eda-table", precision=2):
    """Convert a DataFrame into a safely escaped HTML table."""
    if df is None:
        df = pd.DataFrame()
    return df.to_html(
        classes=table_class,
        index=False,
        border=0,
        escape=True,
        na_rep="—",
        float_format=lambda value: f"{value:.{precision}f}",
    )


def _figure_to_base64(fig):
    """Save a Matplotlib figure as a base64-encoded PNG and close it."""
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=130,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _make_histogram(series, title, x_label):
    """Create a TuneScope-themed histogram for one numeric feature."""
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None

    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    fig.patch.set_facecolor("#111a2d")
    ax.set_facecolor("#111a2d")
    ax.hist(values, bins=30, color="#4de4e2", edgecolor="#17233a", alpha=0.92)
    ax.set_title(title, color="#eef3ff", fontsize=11, pad=12, loc="left")
    ax.set_xlabel(x_label, color="#91a5cb", fontsize=9)
    ax.set_ylabel("Number of tracks", color="#91a5cb", fontsize=9)
    ax.tick_params(colors="#91a5cb", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#263653")
    ax.grid(axis="y", color="#263653", alpha=0.55, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _figure_to_base64(fig)


def _make_popularity_chart(category_table):
    """Create a bar chart for popularity category counts."""
    if category_table is None or category_table.empty:
        return None

    labels = category_table["Popularity Category"].astype(str).tolist()
    counts = category_table["Track Count"].astype(int).tolist()
    palette = ["#7d78eb", "#8d83f5", "#4da9d8", "#4de4e2", "#55e6aa", "#f2b966"]

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    fig.patch.set_facecolor("#111a2d")
    ax.set_facecolor("#111a2d")
    bars = ax.bar(labels, counts, color=palette[:len(labels)])
    ax.set_title("Track counts by popularity category", color="#eef3ff", fontsize=11, pad=12, loc="left")
    ax.set_ylabel("Number of tracks", color="#91a5cb", fontsize=9)
    ax.tick_params(axis="x", colors="#91a5cb", labelsize=8, rotation=18)
    ax.tick_params(axis="y", colors="#91a5cb", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#263653")
    ax.grid(axis="y", color="#263653", alpha=0.55, linewidth=0.7)
    ax.set_axisbelow(True)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#eef3ff",
            fontsize=8,
        )

    fig.tight_layout()
    return _figure_to_base64(fig)


def create_missing_values_table(df):
    """Create missing count, percentage, and status for every original column."""
    total_rows = len(df)
    missing_counts = df.isna().sum()
    result = pd.DataFrame({
        "Feature": [str(column) for column in df.columns],
        "Missing Count": missing_counts.values.astype(int),
        "Missing Percentage (%)": [
            (int(count) / total_rows * 100) if total_rows else 0.0
            for count in missing_counts.values
        ],
    })
    result["Status"] = np.where(
        result["Missing Count"] > 0,
        "Missing values found",
        "Complete",
    )
    return result.sort_values(
        "Missing Count", ascending=False, kind="stable"
    ).reset_index(drop=True)


def prepare_imputed_copy(df):
    """Impute missing cells on a copy; return the copy and a report table."""
    imputed = df.copy(deep=True)
    rows = []

    for column in df.columns:
        missing_before = int(df[column].isna().sum())
        if missing_before == 0:
            continue

        col = imputed[column]
        if is_numeric_dtype(col) and not is_bool_dtype(col):
            numeric_col = pd.to_numeric(col, errors="coerce")
            median = numeric_col.median(skipna=True)
            if pd.isna(median):
                fill_value = 0
                method = "Median unavailable; filled with 0"
            else:
                fill_value = float(median)
                method = "Median imputation"
            imputed[column] = numeric_col.fillna(fill_value)
            display_fill = f"{fill_value:.4f}"
        else:
            mode_values = col.mode(dropna=True)
            if not mode_values.empty:
                fill_value = mode_values.iloc[0]
                method = "Mode imputation"
            else:
                fill_value = "Unknown"
                method = "No observed value; filled with 'Unknown'"
            imputed[column] = col.fillna(fill_value)
            display_fill = str(fill_value)

        rows.append({
            "Feature": str(column),
            "Data Type": str(df[column].dtype),
            "Missing Before": missing_before,
            "Imputation Method": method,
            "Value Used": display_fill,
            "Missing After": int(imputed[column].isna().sum()),
        })

    report_columns = [
        "Feature", "Data Type", "Missing Before", "Imputation Method",
        "Value Used", "Missing After",
    ]
    return imputed, pd.DataFrame(rows, columns=report_columns)


def create_descriptive_statistics(df):
    """Calculate descriptive statistics for numeric columns."""
    numeric_columns = [
        column for column in df.select_dtypes(include=[np.number]).columns
        if not str(column).strip().lower().startswith("unnamed:")
    ]
    if not numeric_columns:
        return pd.DataFrame()

    stats = df[numeric_columns].describe().T.reset_index().rename(columns={"index": "Feature"})
    stats = stats.rename(columns={
        "count": "Count",
        "mean": "Mean",
        "std": "Std Dev",
        "min": "Minimum",
        "25%": "25%",
        "50%": "Median",
        "75%": "75%",
        "max": "Maximum",
    })
    return stats.round(3)


def create_popularity_categories(df):
    """Group popularity scores into five project-defined categories."""
    columns = ["Popularity Category", "Score Range", "Track Count", "Percentage (%)"]
    if "popularity" not in df.columns:
        return pd.DataFrame(columns=columns)

    popularity = pd.to_numeric(df["popularity"], errors="coerce")
    categories = pd.cut(
        popularity,
        bins=POPULARITY_BINS,
        labels=POPULARITY_LABELS,
        include_lowest=True,
        right=True,
    )
    counts = categories.value_counts(sort=False).reindex(POPULARITY_LABELS, fill_value=0)
    total = len(df)
    ranges = ["0–20", "21–40", "41–60", "61–80", "81–100"]

    result = pd.DataFrame({
        "Popularity Category": POPULARITY_LABELS,
        "Score Range": ranges,
        "Track Count": [int(counts[label]) for label in POPULARITY_LABELS],
        "Percentage (%)": [
            round(int(counts[label]) / total * 100, 2) if total else 0.0
            for label in POPULARITY_LABELS
        ],
    })

    unavailable = int((popularity.isna() | ~popularity.between(0, 100)).sum())
    if unavailable:
        result.loc[len(result)] = [
            "Missing / outside 0–100",
            "Unavailable",
            unavailable,
            round(unavailable / total * 100, 2) if total else 0.0,
        ]
    return result


def feature_distribution_charts(df):
    """Make histograms for common Spotify audio features present in the CSV."""
    preferred = [
        ("popularity", "Popularity", "Popularity score"),
        ("danceability", "Danceability", "Danceability score"),
        ("energy", "Energy", "Energy score"),
        ("valence", "Valence", "Valence score"),
        ("tempo", "Tempo", "Tempo (BPM)"),
        ("acousticness", "Acousticness", "Acousticness score"),
        ("loudness", "Loudness", "Loudness (dB)"),
        ("speechiness", "Speechiness", "Speechiness score"),
        ("instrumentalness", "Instrumentalness", "Instrumentalness score"),
        ("liveness", "Liveness", "Liveness score"),
        ("duration_ms", "Track Duration", "Duration (minutes)"),
    ]
    charts = []
    for column, title, x_label in preferred:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if column == "duration_ms":
            values = values / 60000.0
        image = _make_histogram(values, title, x_label)
        if image:
            charts.append({"title": title, "column": column, "chart": image})

    if charts:
        return charts

    excluded = {"key", "mode", "time_signature"}
    for column in df.select_dtypes(include=[np.number]).columns:
        name = str(column).strip().lower()
        if name.startswith("unnamed:") or name in excluded:
            continue
        image = _make_histogram(
            df[column], str(column).replace("_", " ").title(), str(column)
        )
        if image:
            charts.append({
                "title": str(column).replace("_", " ").title(),
                "column": str(column),
                "chart": image,
            })
        if len(charts) >= 8:
            break
    return charts


def build_eda_context(raw_df):
    """Build the complete context dictionary expected by templates/eda.html."""
    if raw_df is None or raw_df.empty:
        raise ValueError("The Spotify dataset is empty.")

    missing_df = create_missing_values_table(raw_df)
    imputed_df, imputation_df = prepare_imputed_copy(raw_df)
    descriptive_df = create_descriptive_statistics(imputed_df)
    popularity_df = create_popularity_categories(raw_df)
    distributions = feature_distribution_charts(imputed_df)
    popularity_chart = _make_popularity_chart(popularity_df)

    missing_count = int(raw_df.isna().sum().sum())
    columns_with_missing = int((raw_df.isna().sum() > 0).sum())
    imputed_cells = int(imputation_df["Missing Before"].sum()) if not imputation_df.empty else 0

    return {
        "total_rows": len(raw_df),
        "total_columns": len(raw_df.columns),
        "missing_count": missing_count,
        "columns_with_missing": columns_with_missing,
        "imputed_cells": imputed_cells,
        "missing_table": dataframe_to_html(missing_df),
        "imputation_table": dataframe_to_html(imputation_df) if not imputation_df.empty else "",
        "descriptive_table": dataframe_to_html(descriptive_df) if not descriptive_df.empty else "",
        "popularity_table": dataframe_to_html(popularity_df),
        "distributions": distributions,
        "popularity_chart": popularity_chart,
        "no_missing_values": missing_count == 0,
        "no_imputations": imputation_df.empty,
        "no_numeric_features": descriptive_df.empty,
        "has_popularity": "popularity" in raw_df.columns,
    }
