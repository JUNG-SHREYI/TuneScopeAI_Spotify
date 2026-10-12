"""TuneScope AI - Gradient Boosting popularity-category classification.

The target is a project-defined category derived from Spotify's popularity score.
Popularity itself is excluded from predictor inputs to prevent direct target leakage.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


POPULARITY_BINS = [-0.001, 20, 40, 60, 80, 100.001]
POPULARITY_LABELS = ["Very Low", "Low", "Moderate", "High", "Very High"]
POPULARITY_RANGES = ["0–20", "21–40", "41–60", "61–80", "81–100"]

# Popularity is the target; IDs and text metadata are excluded from predictors.
AUDIO_FEATURES = [
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

FEATURE_DESCRIPTIONS = {
    "duration_ms": "Track duration in milliseconds",
    "explicit": "Whether the track is marked explicit (0/1)",
    "danceability": "Estimated suitability for dancing",
    "energy": "Perceptual intensity and activity",
    "key": "Estimated musical key (numeric code)",
    "loudness": "Average track loudness in decibels",
    "mode": "Major/minor mode indicator",
    "speechiness": "Presence of spoken words",
    "acousticness": "Confidence that the track is acoustic",
    "instrumentalness": "Likelihood that the track has no vocals",
    "liveness": "Likelihood of a live audience recording",
    "valence": "Musical positivity conveyed by the track",
    "tempo": "Estimated tempo in beats per minute",
    "time_signature": "Estimated beats per bar",
}

PALETTE = {
    "background": "#111a2d",
    "text": "#eef3ff",
    "muted": "#91a5cb",
    "cyan": "#4de4e2",
    "violet": "#a994ff",
    "border": "#263653",
}


def dataframe_to_html(df, precision=4):
    """Render a dataframe as a safe, styled HTML table."""
    if df is None:
        df = pd.DataFrame()
    return df.to_html(
        classes="eda-table",
        index=False,
        border=0,
        escape=True,
        na_rep="—",
        float_format=lambda value: f"{value:.{precision}f}",
    )


def _figure_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=125,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _style_axes(ax):
    ax.set_facecolor(PALETTE["background"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["border"])
    ax.grid(color=PALETTE["border"], alpha=0.48, linewidth=0.6)
    ax.set_axisbelow(True)


def _prepare_data(raw_df):
    """Prepare features and categorical target on a copy of the raw data."""
    if raw_df is None or raw_df.empty:
        raise ValueError("The Spotify dataset is empty.")
    if "popularity" not in raw_df.columns:
        raise ValueError("The dataset must contain a 'popularity' column.")

    available_features = [name for name in AUDIO_FEATURES if name in raw_df.columns]
    if len(available_features) < 3:
        raise ValueError(
            "At least three supported audio features are required. "
            "Expected fields include danceability, energy, loudness, and tempo."
        )

    work = raw_df.copy(deep=True)
    popularity = pd.to_numeric(work["popularity"], errors="coerce")
    popularity = popularity.replace([np.inf, -np.inf], np.nan)
    valid = popularity.notna() & popularity.between(0, 100)

    X = work.loc[valid, available_features].copy()
    y_score = popularity.loc[valid]

    for column in X.columns:
        if pd.api.types.is_bool_dtype(X[column]):
            X[column] = X[column].astype("float64")
        else:
            X[column] = pd.to_numeric(X[column], errors="coerce")
        X[column] = X[column].replace([np.inf, -np.inf], np.nan)

    usable_features = [column for column in X.columns if X[column].notna().any()]
    X = X[usable_features]
    if len(usable_features) < 3:
        raise ValueError("Too few usable numeric audio features remain after validation.")

    categories = pd.cut(
        y_score,
        bins=POPULARITY_BINS,
        labels=POPULARITY_LABELS,
        include_lowest=True,
        right=True,
    )
    keep = categories.notna()
    X = X.loc[keep].reset_index(drop=True)
    categories = categories.loc[keep].astype(str).reset_index(drop=True)

    counts = categories.value_counts()
    missing_classes = [label for label in POPULARITY_LABELS if counts.get(label, 0) < 2]
    if missing_classes:
        raise ValueError(
            "Not enough records in these popularity categories for a stratified split: "
            + ", ".join(missing_classes)
        )
    if len(X) < 30:
        raise ValueError("At least 30 valid popularity records are required.")

    return X, categories, usable_features


def _make_model():
    """Build a reproducible, moderately sized Gradient Boosting pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", GradientBoostingClassifier(
            n_estimators=60,
            learning_rate=0.10,
            max_depth=2,
            subsample=0.80,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
        )),
    ])


def _confusion_matrix_chart(cm):
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])
    image = ax.imshow(cm, interpolation="nearest", cmap="viridis")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.yaxis.set_tick_params(color=PALETTE["muted"], labelsize=8)
    plt.setp(colorbar.ax.get_yticklabels(), color=PALETTE["muted"])

    ax.set(
        xticks=np.arange(len(POPULARITY_LABELS)),
        yticks=np.arange(len(POPULARITY_LABELS)),
        xticklabels=POPULARITY_LABELS,
        yticklabels=POPULARITY_LABELS,
        xlabel="Predicted popularity category",
        ylabel="Actual popularity category",
        title="Gradient Boosting Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right", rotation_mode="anchor")
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    ax.title.set_color(PALETTE["text"])
    ax.title.set_fontsize(11)

    threshold = cm.max() / 2 if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                f"{cm[row, col]:,}",
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "#15233b",
                fontsize=8,
            )
    for spine in ax.spines.values():
        spine.set_color(PALETTE["border"])
    fig.tight_layout()
    return _figure_to_base64(fig)


def _feature_importance_chart(importance_df):
    chart_df = importance_df.head(14).sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    fig.patch.set_facecolor(PALETTE["background"])
    _style_axes(ax)
    ax.barh(chart_df["Feature"], chart_df["Importance"], color=PALETTE["cyan"], alpha=0.9)
    ax.set_xlabel("Feature importance", color=PALETTE["muted"], fontsize=9)
    ax.set_ylabel("")
    ax.set_title(
        "Gradient Boosting — Feature Importance",
        color=PALETTE["text"], fontsize=11, loc="left", pad=12
    )
    fig.tight_layout()
    return _figure_to_base64(fig)


def build_gradient_boosting_context(raw_df):
    """Train the model and generate content for gradient_boosting.html."""
    X, y, feature_names = _prepare_data(raw_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = _make_model()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    accuracy = accuracy_score(y_test, predicted)
    balanced_accuracy = balanced_accuracy_score(y_test, predicted)
    macro_precision = precision_score(
        y_test, predicted, labels=POPULARITY_LABELS, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        y_test, predicted, labels=POPULARITY_LABELS, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_test, predicted, labels=POPULARITY_LABELS, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predicted, labels=POPULARITY_LABELS, average="weighted", zero_division=0
    )

    classifier = model.named_steps["classifier"]
    info_df = pd.DataFrame([
        {"Property": "Algorithm", "Value": "Gradient Boosting Classifier"},
        {"Property": "Learning task", "Value": "Multi-class classification"},
        {"Property": "Target", "Value": "Popularity category derived from popularity score"},
        {"Property": "Category ranges", "Value": "Very Low (0–20), Low (21–40), Moderate (41–60), High (61–80), Very High (81–100)"},
        {"Property": "Number of boosting stages", "Value": str(classifier.n_estimators)},
        {"Property": "Learning rate", "Value": str(classifier.learning_rate)},
        {"Property": "Loss function", "Value": str(classifier.loss)},
        {"Property": "Maximum tree depth", "Value": str(classifier.max_depth)},
        {"Property": "Subsample per stage", "Value": str(classifier.subsample)},
        {"Property": "Features considered per split", "Value": str(classifier.max_features)},
        {"Property": "Minimum samples per leaf", "Value": str(classifier.min_samples_leaf)},
        {"Property": "Train/test split", "Value": "80% training / 20% testing, stratified"},
        {"Property": "Random state", "Value": "42 (reproducible split and model)"},
        {"Property": "Input feature count", "Value": str(len(feature_names))},
        {"Property": "Training records", "Value": f"{len(X_train):,}"},
        {"Property": "Testing records", "Value": f"{len(X_test):,}"},
    ])

    features_df = pd.DataFrame([
        {
            "Feature": name,
            "Data type": "Boolean (converted to 0/1)" if name == "explicit" else "Numeric",
            "Description": FEATURE_DESCRIPTIONS.get(name, "Numeric audio feature"),
        }
        for name in feature_names
    ])

    performance_df = pd.DataFrame([
        {"Metric": "Test Accuracy", "Score": f"{accuracy * 100:.2f}%"},
        {"Metric": "Balanced Accuracy", "Score": f"{balanced_accuracy * 100:.2f}%"},
        {"Metric": "Macro Precision", "Score": f"{macro_precision * 100:.2f}%"},
        {"Metric": "Macro Recall", "Score": f"{macro_recall * 100:.2f}%"},
        {"Metric": "Macro F1 Score", "Score": f"{macro_f1 * 100:.2f}%"},
        {"Metric": "Weighted F1 Score", "Score": f"{weighted_f1 * 100:.2f}%"},
    ])

    cm = confusion_matrix(y_test, predicted, labels=POPULARITY_LABELS)
    cm_df = pd.DataFrame(
        cm,
        columns=[f"Predicted: {label}" for label in POPULARITY_LABELS],
    )
    cm_df.insert(0, "Actual category", POPULARITY_LABELS)

    report = classification_report(
        y_test,
        predicted,
        labels=POPULARITY_LABELS,
        target_names=POPULARITY_LABELS,
        output_dict=True,
        zero_division=0,
    )
    report_rows = []
    for label in POPULARITY_LABELS:
        row = report[label]
        report_rows.append({
            "Class": label,
            "Precision": row["precision"],
            "Recall": row["recall"],
            "F1 Score": row["f1-score"],
            "Support": int(row["support"]),
        })
    for label, display in [("macro avg", "Macro Average"), ("weighted avg", "Weighted Average")]:
        row = report[label]
        report_rows.append({
            "Class": display,
            "Precision": row["precision"],
            "Recall": row["recall"],
            "F1 Score": row["f1-score"],
            "Support": int(row["support"]),
        })
    report_df = pd.DataFrame(report_rows)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": classifier.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    importance_display = importance_df.copy()
    importance_display["Importance (%)"] = importance_display["Importance"].map(lambda value: f"{value * 100:.2f}%")
    importance_display["Importance"] = importance_display["Importance"].map(lambda value: f"{value:.6f}")
    importance_display = importance_display[["Feature", "Importance (%)", "Importance"]]

    target_counts = y.value_counts().reindex(POPULARITY_LABELS, fill_value=0)
    class_distribution_df = pd.DataFrame({
        "Popularity Category": POPULARITY_LABELS,
        "Score Range": POPULARITY_RANGES,
        "Records": [int(target_counts[label]) for label in POPULARITY_LABELS],
        "Percentage": [f"{target_counts[label] / len(y) * 100:.2f}%" for label in POPULARITY_LABELS],
    })

    return {
        "total_records": int(len(X)),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "feature_count": int(len(feature_names)),
        "class_count": len(POPULARITY_LABELS),
        "accuracy_display": f"{accuracy * 100:.2f}%",
        "macro_f1_display": f"{macro_f1 * 100:.2f}%",
        "model_info_table": dataframe_to_html(info_df),
        "features_table": dataframe_to_html(features_df),
        "performance_table": dataframe_to_html(performance_df),
        "confusion_matrix_table": dataframe_to_html(cm_df, precision=0),
        "confusion_matrix_chart": _confusion_matrix_chart(cm),
        "feature_importance_table": dataframe_to_html(importance_display),
        "feature_importance_chart": _feature_importance_chart(importance_df),
        "classification_report_table": dataframe_to_html(report_df, precision=4),
        "class_distribution_table": dataframe_to_html(class_distribution_df),
    }
