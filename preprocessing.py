"""TuneScope AI - categorical feature encoding demonstrations.

Demonstrates One-Hot Encoding, Ordinal Encoding and Target Encoding using
categories present in the Spotify tracks dataset. Target encoding is learned
from the training split only and displayed on a held-out test preview.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# Features treated as categorical for encoding. Numeric-looking Spotify fields
# (key, mode, time_signature) are represented as categories here, not as ranks.
ENCODING_FEATURES = [
    "track_genre",
    "explicit",
    "key",
    "mode",
    "time_signature",
]
TARGET_FEATURE = "popularity"
CATEGORY_FEATURE = "track_genre"
PREVIEW_ROWS = 12


def _as_category_strings(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert selected columns to clean strings and represent missing values."""
    result = pd.DataFrame(index=df.index)
    for column in columns:
        values = df[column].astype("object")
        values = values.where(values.notna(), "(Missing)")
        result[column] = values.map(str)
    return result


def _table_html(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Create a styled HTML table matching TuneScope's existing CSS."""
    shown = df if max_rows is None else df.head(max_rows)
    if shown is None or shown.empty:
        return '<p class="empty-state">No rows are available for this table.</p>'
    return shown.to_html(
        classes="eda-table",
        index=False,
        border=0,
        escape=True,
        na_rep="—",
    )


def _get_ohe_encoder() -> OneHotEncoder:
    """Build a version-compatible sparse OneHotEncoder."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.uint8,
        )
    except TypeError:  # Compatibility with scikit-learn < 1.2
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
            dtype=np.uint8,
        )


def build_preprocessing_context(df: pd.DataFrame) -> dict:
    """Return all HTML-ready results for the Preprocessing page.

    Parameters
    ----------
    df:
        The original Spotify dataset returned by data_loader.load_data().
        This function never mutates the supplied DataFrame.
    """
    if df is None or df.empty:
        raise ValueError("The Spotify dataset is empty. Check Data/dataset.csv.")

    missing_features = [column for column in ENCODING_FEATURES if column not in df.columns]
    if missing_features:
        raise ValueError(
            "The dataset is missing columns required for encoding: "
            + ", ".join(missing_features)
        )
    if TARGET_FEATURE not in df.columns:
        raise ValueError("The dataset needs a 'popularity' column for target encoding.")

    # -------------------------------
    # Feature metadata
    # -------------------------------
    categorical_values = _as_category_strings(df, ENCODING_FEATURES)
    feature_summary_rows = []
    for column in ENCODING_FEATURES:
        feature_summary_rows.append(
            {
                "Input Feature": column,
                "Unique Categories": int(categorical_values[column].nunique(dropna=False)),
                "Missing Values": int(df[column].isna().sum()),
            }
        )
    feature_summary = pd.DataFrame(feature_summary_rows)

    # -------------------------------
    # 1. One-Hot Encoding
    # -------------------------------
    onehot_encoder = _get_ohe_encoder()
    onehot_matrix = onehot_encoder.fit_transform(categorical_values)
    onehot_columns = onehot_encoder.get_feature_names_out(ENCODING_FEATURES).tolist()

    # Sparse transform keeps the memory footprint small even with all 114 genres.
    onehot_preview_values = onehot_matrix[:PREVIEW_ROWS].toarray()
    onehot_preview = pd.DataFrame(
        onehot_preview_values,
        columns=onehot_columns,
    )
    onehot_preview.insert(0, "Source Row", [int(value) + 1 for value in df.index[: len(onehot_preview)]])

    onehot_summary = pd.DataFrame(
        [
            {
                "Input Feature": column,
                "Number of Categories": int(len(categories)),
                "Generated Columns": int(len(categories)),
                "Encoding": "One column per category (0/1)",
            }
            for column, categories in zip(ENCODING_FEATURES, onehot_encoder.categories_)
        ]
    )

    # -------------------------------
    # 2. Ordinal Encoding
    # -------------------------------
    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=np.int32,
    )
    ordinal_values = ordinal_encoder.fit_transform(categorical_values)
    ordinal_columns = [f"{column}_encoded" for column in ENCODING_FEATURES]
    ordinal_preview = pd.DataFrame(
        ordinal_values[:PREVIEW_ROWS].astype(int),
        columns=ordinal_columns,
    )
    ordinal_preview.insert(0, "Source Row", [int(value) + 1 for value in df.index[: len(ordinal_preview)]])
    for col in ENCODING_FEATURES:
        ordinal_preview.insert(
            ordinal_preview.columns.get_loc(f"{col}_encoded"),
            col,
            df[col].reset_index(drop=True).iloc[: len(ordinal_preview)].astype("string").fillna("(Missing)").to_numpy(),
        )

    mapping_rows = []
    for column, categories in zip(ENCODING_FEATURES, ordinal_encoder.categories_):
        for code, category in enumerate(categories):
            mapping_rows.append(
                {
                    "Feature": column,
                    "Original Category": str(category),
                    "Ordinal Code": int(code),
                }
            )
    ordinal_mapping = pd.DataFrame(mapping_rows)

    # -------------------------------
    # 3. Target Encoding: track_genre -> mean popularity
    # -------------------------------
    popularity = pd.to_numeric(df[TARGET_FEATURE], errors="coerce")
    genre = df[CATEGORY_FEATURE].astype("object").where(df[CATEGORY_FEATURE].notna(), "(Missing)").astype(str)
    valid_mask = popularity.notna()
    valid_indices = np.flatnonzero(valid_mask.to_numpy())
    if len(valid_indices) < 2:
        raise ValueError("At least two non-missing popularity values are required for target encoding.")

    train_indices, test_indices = train_test_split(
        valid_indices,
        test_size=0.20,
        random_state=42,
    )

    training_frame = pd.DataFrame(
        {
            "Genre": genre.iloc[train_indices].to_numpy(),
            "Popularity": popularity.iloc[train_indices].to_numpy(),
        }
    )
    global_mean = float(training_frame["Popularity"].mean())
    target_mapping = training_frame.groupby("Genre", dropna=False)["Popularity"].agg(
        [("Training Records", "size"), ("Mean Popularity", "mean")]
    ).reset_index()
    target_mapping = target_mapping.rename(columns={"Genre": "Track Genre"})
    target_mapping["Mean Popularity"] = target_mapping["Mean Popularity"].round(4)
    target_mapping["Training Records"] = target_mapping["Training Records"].astype(int)
    target_map = training_frame.groupby("Genre")["Popularity"].mean().to_dict()

    # Show held-out test records only so their target values were not used to
    # construct the category means shown in the encoded preview.
    test_preview_indices = test_indices[:PREVIEW_ROWS]
    target_preview = pd.DataFrame(
        {
            "Source Row": test_preview_indices + 1,
            "Track Name": (
                df.iloc[test_preview_indices]["track_name"].fillna("(Missing)").astype(str).to_numpy()
                if "track_name" in df.columns
                else ["(Not available)"] * len(test_preview_indices)
            ),
            "Track Genre": genre.iloc[test_preview_indices].to_numpy(),
            "Actual Popularity": popularity.iloc[test_preview_indices].to_numpy(),
            "Target Encoded Popularity": [
                round(float(target_map.get(genre.iloc[index], global_mean)), 4)
                for index in test_preview_indices
            ],
            "Data Split": ["Test"] * len(test_preview_indices),
        }
    )
    target_mapping = target_mapping.sort_values(
        "Mean Popularity", ascending=False
    ).reset_index(drop=True)

    # Build compact categorical column names for display in the one-hot table.
    return {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "encoding_features": ENCODING_FEATURES,
        "feature_summary_table": _table_html(feature_summary),
        "onehot_output_columns": int(len(onehot_columns)),
        "onehot_preview_table": _table_html(onehot_preview),
        "onehot_summary_table": _table_html(onehot_summary),
        "ordinal_preview_table": _table_html(ordinal_preview),
        "ordinal_mapping_table": _table_html(ordinal_mapping),
        "target_preview_table": _table_html(target_preview),
        "target_mapping_table": _table_html(target_mapping),
        "target_global_mean": round(global_mean, 4),
        "target_train_records": int(len(train_indices)),
        "target_test_records": int(len(test_indices)),
        "target_category_count": int(genre.nunique(dropna=False)),
    }
