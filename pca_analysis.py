"""TuneScope AI - Principal Component Analysis (PCA) module.

PCA is fitted on standardized numeric audio-related columns. The original
Data/dataset.csv is never modified. Popularity and metadata/text identifiers
are intentionally excluded from PCA inputs.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Use consistent feature ordering across runs and make it clear which columns
# are being used for music-characteristic PCA.
PREFERRED_PCA_FEATURES = [
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]


def _table_html(frame: pd.DataFrame, *, table_class: str = "pca-table") -> str:
    """Convert a DataFrame to a styled, escaped HTML table."""
    return frame.to_html(
        classes=table_class,
        index=False,
        border=0,
        escape=True,
        na_rep="—",
    )


def _figure_base64(fig: plt.Figure) -> str:
    """Encode a Matplotlib figure as a PNG for a Flask template."""
    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    buffer.seek(0)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    plt.close(fig)
    return encoded


def build_pca_context(raw_df: pd.DataFrame) -> dict[str, Any]:
    """Fit PCA and create the tables/chart data required by ``pca.html``.

    Returns the transformed DataFrame as ``transformed_df`` so the Flask route
    can paginate it without sending all 114,000 records to the browser at once.
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("The Spotify dataset is empty or could not be loaded.")

    missing_columns = [
        column for column in PREFERRED_PCA_FEATURES if column not in raw_df.columns
    ]
    selected_features = [
        column for column in PREFERRED_PCA_FEATURES if column in raw_df.columns
    ]
    if len(selected_features) < 2:
        raise ValueError(
            "PCA needs at least two supported numeric audio features. "
            f"Available features: {', '.join(selected_features) or 'none'}."
        )

    # Convert explicitly to numeric. Boolean values (explicit) become 0/1.
    numeric = raw_df[selected_features].copy()
    for column in numeric.columns:
        if pd.api.types.is_bool_dtype(numeric[column]):
            numeric[column] = numeric[column].astype("float64")
        else:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    # Completely empty columns cannot be median-imputed meaningfully.
    all_missing_columns = numeric.columns[numeric.isna().all()].tolist()
    if all_missing_columns:
        numeric = numeric.drop(columns=all_missing_columns)

    if numeric.shape[1] < 2:
        raise ValueError(
            "PCA needs at least two numeric features containing usable values."
        )

    # Impute only in memory on this working copy; never write to the source CSV.
    imputer = SimpleImputer(strategy="median")
    imputed_values = imputer.fit_transform(numeric)
    used_features = list(numeric.columns)

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(imputed_values)

    # Fit all available components first so the full cumulative curve reaches 100%.
    pca = PCA(svd_solver="full")
    all_scores = pca.fit_transform(scaled_values)

    explained_ratio = np.asarray(pca.explained_variance_ratio_, dtype=float)
    cumulative_ratio = np.cumsum(explained_ratio)
    n_components = int(np.searchsorted(cumulative_ratio, 0.95, side="left") + 1)
    n_components = min(max(n_components, 1), len(explained_ratio))

    # ------------------------------------------------------------
    # 1. Cumulative explained variance chart
    # ------------------------------------------------------------
    component_labels = [f"PC{i}" for i in range(1, len(explained_ratio) + 1)]
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor="white")
    ax.plot(
        component_labels,
        cumulative_ratio * 100,
        marker="o",
        linewidth=2,
        markersize=5,
        color="#2586cc",
    )
    ax.axhline(95, color="#df8f35", linestyle="--", linewidth=1.4, label="95% variance")
    ax.axvline(n_components - 1, color="#df8f35", linestyle=":", linewidth=1.2)
    ax.set_title("PCA Cumulative Explained Variance", fontsize=12, pad=10)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_ylim(0, 103)
    ax.grid(True, alpha=0.22)
    ax.tick_params(axis="x", labelrotation=42, labelsize=8)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    explained_variance_chart = _figure_base64(fig)

    # ------------------------------------------------------------
    # 2. Explained variance table (matching the requested headers)
    # ------------------------------------------------------------
    variance_table_df = pd.DataFrame({
        "Principal Component": component_labels,
        "Explained Variance": np.asarray(pca.explained_variance_, dtype=float),
        "Explained Variance (%)": explained_ratio * 100,
        "Cumulative Variance (%)": cumulative_ratio * 100,
    })
    variance_table_df["Explained Variance"] = variance_table_df["Explained Variance"].map(lambda v: f"{v:.6f}")
    variance_table_df["Explained Variance (%)"] = variance_table_df["Explained Variance (%)"].map(lambda v: f"{v:.4f}%")
    variance_table_df["Cumulative Variance (%)"] = variance_table_df["Cumulative Variance (%)"].map(lambda v: f"{v:.4f}%")
    variance_table = _table_html(variance_table_df)

    # ------------------------------------------------------------
    # 3. PCA feature loadings, only components needed for 95% variance
    #    Loading = eigenvector coefficient * sqrt(eigenvalue).
    # ------------------------------------------------------------
    loadings_values = pca.components_[:n_components, :].T * np.sqrt(
        pca.explained_variance_[:n_components]
    )
    loading_columns = [f"PC{i}" for i in range(1, n_components + 1)]
    loadings_df = pd.DataFrame(loadings_values, columns=loading_columns)
    loadings_df.insert(0, "Feature", used_features)
    numeric_loading_columns = loadings_df.columns[1:]
    loadings_df[numeric_loading_columns] = loadings_df[numeric_loading_columns].round(5)
    loadings_table = _table_html(loadings_df)

    # ------------------------------------------------------------
    # 4. Transformed data for all original records, reduced to 95% PCs
    # ------------------------------------------------------------
    selected_scores = all_scores[:, :n_components]
    transformed_columns = [f"PC{i}" for i in range(1, n_components + 1)]
    transformed_df = pd.DataFrame(selected_scores, columns=transformed_columns)
    transformed_df.insert(0, "Record Number", np.arange(1, len(transformed_df) + 1))
    transformed_df[transformed_columns] = transformed_df[transformed_columns].round(5)

    return {
        "total_rows": int(len(raw_df)),
        "total_features": int(len(used_features)),
        "features": used_features,
        "missing_feature_columns": missing_columns,
        "dropped_all_missing_features": all_missing_columns,
        "component_count": int(len(explained_ratio)),
        "components_for_95": n_components,
        "variance_95_actual": float(cumulative_ratio[n_components - 1] * 100),
        "explained_variance_chart": explained_variance_chart,
        "variance_table": variance_table,
        "loadings_table": loadings_table,
        "loading_component_names": loading_columns,
        "transformed_df": transformed_df,
    }
