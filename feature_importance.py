"""TuneScope AI: compare tree-based feature importance across classifiers.

The popularity score is converted into project-defined categories. Popularity is
never passed to the models as a feature, preventing direct target leakage.
Each model's importance scores are normalized to percentages before comparison.
"""

import base64
import io
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_IMPORT_ERROR = None
except Exception as exc:  # keep this page usable if the optional package is unavailable
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = str(exc)

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_IMPORT_ERROR = None
except Exception as exc:  # keep this page usable if the optional package is unavailable
    LGBMClassifier = None
    LIGHTGBM_IMPORT_ERROR = str(exc)


POPULARITY_BINS = [-0.001, 20, 40, 60, 80, 100.001]
POPULARITY_LABELS = ["Very Low", "Low", "Moderate", "High", "Very High"]

AUDIO_FEATURES = [
    "duration_ms", "explicit", "danceability", "energy", "key",
    "loudness", "mode", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "time_signature",
]

PALETTE = {
    "background": "#111a2d",
    "text": "#eef3ff",
    "muted": "#91a5cb",
    "cyan": "#4de4e2",
    "violet": "#a994ff",
    "blue": "#79b8ff",
    "green": "#55e6aa",
    "amber": "#ffc16b",
    "border": "#263653",
}


def _figure_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=135,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _dataframe_to_html(df, precision=2):
    if df is None or df.empty:
        df = pd.DataFrame({"Information": ["No results available."]})
    return df.to_html(
        classes="eda-table",
        index=False,
        border=0,
        escape=True,
        na_rep="—",
        float_format=lambda value: f"{value:.{precision}f}",
    )


def _prepare_data(raw_df):
    if raw_df is None or raw_df.empty:
        raise ValueError("The Spotify dataset is empty.")
    if "popularity" not in raw_df.columns:
        raise ValueError("The dataset must contain a 'popularity' column.")

    available = [name for name in AUDIO_FEATURES if name in raw_df.columns]
    if len(available) < 3:
        raise ValueError("At least three supported Spotify audio-feature columns are required.")

    work = raw_df.copy(deep=True)
    popularity = pd.to_numeric(work["popularity"], errors="coerce")
    popularity = popularity.replace([np.inf, -np.inf], np.nan)
    valid = popularity.notna() & popularity.between(0, 100)
    X = work.loc[valid, available].copy()
    scores = popularity.loc[valid]

    for column in X.columns:
        if pd.api.types.is_bool_dtype(X[column]):
            X[column] = X[column].astype("float64")
        else:
            X[column] = pd.to_numeric(X[column], errors="coerce")
        X[column] = X[column].replace([np.inf, -np.inf], np.nan)

    features = [column for column in X.columns if X[column].notna().any()]
    X = X[features]
    if len(features) < 3:
        raise ValueError("Too few usable numeric audio features remain after validation.")

    categories = pd.cut(
        scores,
        bins=POPULARITY_BINS,
        labels=POPULARITY_LABELS,
        include_lowest=True,
        right=True,
    )
    keep = categories.notna()
    X = X.loc[keep].reset_index(drop=True)
    categories = categories.loc[keep].astype(str).reset_index(drop=True)
    y_codes = pd.Categorical(categories, categories=POPULARITY_LABELS, ordered=True).codes
    keep_codes = y_codes >= 0
    X = X.loc[keep_codes].reset_index(drop=True)
    y = pd.Series(y_codes[keep_codes], name="popularity_category", dtype="int32")

    counts = y.value_counts()
    low_count = [label for code, label in enumerate(POPULARITY_LABELS) if counts.get(code, 0) < 2]
    if low_count:
        raise ValueError("Not enough rows to split these popularity categories: " + ", ".join(low_count))
    if len(X) < 30:
        raise ValueError("At least 30 valid popularity records are required.")
    return X, y, features


def _model_specs():
    """Return comparable, reproducible classifiers and explanatory model settings."""
    specs = [
        ("AdaBoost", AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)),
        ("Random Forest", RandomForestClassifier(
            n_estimators=120,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )),
        ("Gradient Boosting", GradientBoostingClassifier(
            n_estimators=60,
            learning_rate=0.10,
            max_depth=2,
            subsample=0.80,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
        )),
    ]
    if XGBClassifier is not None:
        specs.append(("XGBoost", XGBClassifier(
            objective="multi:softprob",
            num_class=len(POPULARITY_LABELS),
            n_estimators=90,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=4,
            verbosity=0,
        )))
    if LGBMClassifier is not None:
        specs.append(("LightGBM", LGBMClassifier(
            objective="multiclass",
            num_class=len(POPULARITY_LABELS),
            n_estimators=120,
            learning_rate=0.08,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            importance_type="gain",
            random_state=42,
            n_jobs=4,
            verbosity=-1,
        )))
    return specs


def _style_axis(ax):
    ax.set_facecolor(PALETTE["background"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["border"])
    ax.grid(axis="x", color=PALETTE["border"], alpha=0.5, linewidth=0.65)
    ax.set_axisbelow(True)


def _important_features_chart(top_df):
    chart_df = top_df.sort_values("Average Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    fig.patch.set_facecolor(PALETTE["background"])
    _style_axis(ax)
    colors = [PALETTE["violet"] if i == len(chart_df) - 1 else PALETTE["cyan"] for i in range(len(chart_df))]
    ax.barh(chart_df["Feature"], chart_df["Average Importance"], color=colors, alpha=0.96)
    ax.set_xlabel("Average normalized importance (%)", color=PALETTE["muted"], fontsize=9)
    ax.set_ylabel("")
    ax.set_title("Top Features Across Tree-Based Models", color=PALETTE["text"], fontsize=12, loc="left", pad=13)
    for y_pos, value in enumerate(chart_df["Average Importance"]):
        ax.text(value, y_pos, f"  {value:.2f}%", va="center", ha="left", color=PALETTE["text"], fontsize=8)
    fig.tight_layout()
    return _figure_to_base64(fig)


def _comparison_chart(top_df, model_names):
    chart_df = top_df.head(8).sort_values("Average Importance", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.4, 6.0))
    fig.patch.set_facecolor(PALETTE["background"])
    _style_axis(ax)
    colors = [PALETTE["cyan"], PALETTE["violet"], PALETTE["blue"], PALETTE["green"], PALETTE["amber"]]
    y = np.arange(len(chart_df))
    n_models = max(1, len(model_names))
    bar_height = min(0.78 / n_models, 0.18)
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_height
    for i, model_name in enumerate(model_names):
        if model_name not in chart_df.columns:
            continue
        ax.barh(
            y + offsets[i], chart_df[model_name].to_numpy(dtype=float),
            height=bar_height * 0.9, color=colors[i % len(colors)],
            label=model_name, alpha=0.96,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(chart_df["Feature"], color=PALETTE["muted"], fontsize=8)
    ax.set_xlabel("Normalized importance (%)", color=PALETTE["muted"], fontsize=9)
    ax.set_title("Model-by-Model Importance Comparison", color=PALETTE["text"], fontsize=12, loc="left", pad=13)
    ax.legend(loc="lower right", facecolor="#17233a", edgecolor=PALETTE["border"], labelcolor=PALETTE["text"], fontsize=8, framealpha=0.95)
    fig.tight_layout()
    return _figure_to_base64(fig)


def build_feature_importance_context(raw_df):
    """Train importance models and return graph/table content for the page."""
    X, y, features = _prepare_data(raw_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )

    importance_columns = {}
    skipped_models = []
    model_settings = []
    for model_name, classifier in _model_specs():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(X_train, y_train)
            raw_importance = np.asarray(pipeline.named_steps["classifier"].feature_importances_, dtype=float)
            raw_importance = np.nan_to_num(raw_importance, nan=0.0, posinf=0.0, neginf=0.0)
            raw_importance = np.maximum(raw_importance, 0.0)
            total = raw_importance.sum()
            if total > 0:
                normalized = raw_importance / total * 100.0
            else:
                normalized = np.zeros(len(features), dtype=float)
            importance_columns[model_name] = normalized
            model_settings.append({
                "Model": model_name,
                "Training Records": f"{len(X_train):,}",
                "Testing Records": f"{len(X_test):,}",
                "Features Used": len(features),
                "Importance Scale": "Normalized to 100%",
            })
        except Exception as exc:
            skipped_models.append({"Model": model_name, "Reason": str(exc)[:220]})

    if not importance_columns:
        details = "; ".join(f"{row['Model']}: {row['Reason']}" for row in skipped_models)
        raise ValueError("Feature importance could not be calculated for any model. " + details)

    model_names = list(importance_columns.keys())
    full_df = pd.DataFrame({"Feature": features, **importance_columns})
    full_df["Average Importance"] = full_df[model_names].mean(axis=1)
    full_df = full_df.sort_values("Average Importance", ascending=False).reset_index(drop=True)
    full_df.insert(0, "Rank", np.arange(1, len(full_df) + 1))

    top_df = full_df.loc[:, ["Rank", "Feature", "Average Importance"]].head(10).copy()
    top_display = top_df.copy()
    top_display["Average Importance"] = top_display["Average Importance"].map(lambda value: f"{value:.2f}%")

    comparison_columns = ["Rank", "Feature"] + model_names + ["Average Importance"]
    comparison_df = full_df.loc[:, comparison_columns].copy()
    for column in model_names + ["Average Importance"]:
        comparison_df[column] = comparison_df[column].map(lambda value: f"{value:.2f}%")

    graph = _important_features_chart(top_df)
    comparison_graph = _comparison_chart(full_df, model_names)

    availability_notes = []
    if XGBClassifier is None:
        availability_notes.append("XGBoost is unavailable in this Python environment: " + (XGBOOST_IMPORT_ERROR or "package could not be imported"))
    if LGBMClassifier is None:
        availability_notes.append("LightGBM is unavailable in this Python environment: " + (LIGHTGBM_IMPORT_ERROR or "package could not be imported"))
    for item in skipped_models:
        availability_notes.append(f"{item['Model']} could not be trained: {item['Reason']}")

    notes = (
        "Each model's feature-importance values are normalized so the model's features sum to 100%. "
        "Different algorithms calculate importance differently, so compare these scores as relative rankings, "
        "not as causal effects. Popularity is used only to define the target categories and is not an input feature."
    )

    # Table column names are converted to safe HTML by pandas' default escaping.
    model_settings_df = pd.DataFrame(model_settings)
    skipped_df = pd.DataFrame(skipped_models, columns=["Model", "Reason"])

    return {
        "total_records": int(len(raw_df)),
        "used_records": int(len(X)),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "feature_count": int(len(features)),
        "features": features,
        "model_names": model_names,
        "top_features_table": _dataframe_to_html(top_display),
        "comparison_table": _dataframe_to_html(comparison_df),
        "model_settings_table": _dataframe_to_html(model_settings_df),
        "skipped_models_table": _dataframe_to_html(skipped_df) if not skipped_df.empty else None,
        "important_features_chart": graph,
        "comparison_chart": comparison_graph,
        "notes": notes,
        "availability_notes": availability_notes,
    }
