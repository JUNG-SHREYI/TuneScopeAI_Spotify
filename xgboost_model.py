"""TuneScope AI - XGBoost popularity-category classification.

Popularity is converted into project-defined categories and is never included
among the input features, preventing direct target leakage.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier  # noqa: F401
    _XGBOOST_AVAILABLE = False


POPULARITY_BINS = [-0.001, 20, 40, 60, 80, 100.001]
POPULARITY_LABELS = ["Very Low", "Low", "Moderate", "High", "Very High"]
POPULARITY_RANGES = ["0–20", "21–40", "41–60", "61–80", "81–100"]

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
    """Render a DataFrame as a safely escaped TuneScope table."""
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
    """Create numeric inputs and the popularity-category target on a copy."""
    if raw_df is None or raw_df.empty:
        raise ValueError("The Spotify dataset is empty.")
    if "popularity" not in raw_df.columns:
        raise ValueError("The dataset must contain a 'popularity' column.")

    available_features = [column for column in AUDIO_FEATURES if column in raw_df.columns]
    if len(available_features) < 3:
        raise ValueError(
            "At least three supported audio features are required. "
            "Expected fields include danceability, energy, loudness, and tempo."
        )

    work = raw_df.copy(deep=True)
    popularity = pd.to_numeric(work["popularity"], errors="coerce")
    popularity = popularity.replace([np.inf, -np.inf], np.nan)
    valid_target = popularity.notna() & popularity.between(0, 100)

    X = work.loc[valid_target, available_features].copy()
    scores = popularity.loc[valid_target]

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
        scores,
        bins=POPULARITY_BINS,
        labels=POPULARITY_LABELS,
        include_lowest=True,
        right=True,
    )
    keep = categories.notna()
    X = X.loc[keep].reset_index(drop=True)
    categories = categories.loc[keep].astype(str).reset_index(drop=True)

    class_codes = pd.Categorical(categories, categories=POPULARITY_LABELS, ordered=True).codes
    valid_codes = class_codes >= 0
    X = X.loc[valid_codes].reset_index(drop=True)
    y = pd.Series(class_codes[valid_codes], name="popularity_category", dtype="int32")

    counts = y.value_counts()
    missing_classes = [
        label for code, label in enumerate(POPULARITY_LABELS)
        if counts.get(code, 0) < 2
    ]
    if missing_classes:
        raise ValueError(
            "Not enough rows for a stratified training/test split in these categories: "
            + ", ".join(missing_classes)
        )
    if len(X) < 30:
        raise ValueError("At least 30 valid popularity records are required.")

    return X, y, usable_features


def _make_model():
    """Build a reproducible multiclass XGBoost model."""
    classifier = XGBClassifier(
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
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", classifier),
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
        title="XGBoost Confusion Matrix",
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
                col, row, f"{cm[row, col]:,}",
                ha="center", va="center",
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
    ax.barh(chart_df["Feature"], chart_df["Importance"], color=PALETTE["violet"], alpha=0.95)
    ax.set_xlabel("Feature importance", color=PALETTE["muted"], fontsize=9)
    ax.set_ylabel("")
    ax.set_title("XGBoost — Feature Importance", color=PALETTE["text"], fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    return _figure_to_base64(fig)


def build_xgboost_context(raw_df):
    """Train the XGBoost classifier and create the HTML context."""
    X, y_codes, feature_names = _prepare_data(raw_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_codes,
        test_size=0.20,
        random_state=42,
        stratify=y_codes,
    )

    model = _make_model()
    model.fit(X_train, y_train)
    predicted_codes = model.predict(X_test).astype(int)

    labels_as_strings = np.asarray(POPULARITY_LABELS, dtype=object)
    y_test_names = labels_as_strings[y_test.to_numpy(dtype=int)]
    predicted_names = labels_as_strings[predicted_codes]

    accuracy = accuracy_score(y_test, predicted_codes)
    balanced_accuracy = balanced_accuracy_score(y_test, predicted_codes)
    macro_precision = precision_score(y_test, predicted_codes, labels=np.arange(5), average="macro", zero_division=0)
    macro_recall = recall_score(y_test, predicted_codes, labels=np.arange(5), average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, predicted_codes, labels=np.arange(5), average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, predicted_codes, labels=np.arange(5), average="weighted", zero_division=0)

    classifier = model.named_steps["classifier"]
    features_df = pd.DataFrame({
        "Feature": feature_names,
        "Description": [FEATURE_DESCRIPTIONS.get(name, "Numeric audio feature") for name in feature_names],
        "Data Type": [str(raw_df[name].dtype) for name in feature_names],
    })
    info_df = pd.DataFrame([
        {"Setting": "Algorithm", "Value": "XGBoost (Extreme Gradient Boosting)"},
        {"Setting": "Task", "Value": "Multiclass classification of popularity categories"},
        {"Setting": "Target", "Value": "Popularity grouped into five project-defined categories"},
        {"Setting": "Objective", "Value": "multi:softprob"},
        {"Setting": "Evaluation metric", "Value": "mlogloss"},
        {"Setting": "Number of estimators", "Value": classifier.n_estimators},
        {"Setting": "Maximum tree depth", "Value": classifier.max_depth},
        {"Setting": "Learning rate", "Value": classifier.learning_rate},
        {"Setting": "Subsample", "Value": classifier.subsample},
        {"Setting": "Column sampling per tree", "Value": classifier.colsample_bytree},
        {"Setting": "Tree method", "Value": classifier.tree_method},
        {"Setting": "Missing-value treatment", "Value": "Training-fold median imputation"},
        {"Setting": "Train/test split", "Value": "80% training / 20% testing, stratified"},
        {"Setting": "Random state", "Value": classifier.random_state},
    ])

    performance_df = pd.DataFrame([
        {"Metric": "Test Accuracy", "Score": f"{accuracy * 100:.2f}%"},
        {"Metric": "Balanced Accuracy", "Score": f"{balanced_accuracy * 100:.2f}%"},
        {"Metric": "Macro Precision", "Score": f"{macro_precision * 100:.2f}%"},
        {"Metric": "Macro Recall", "Score": f"{macro_recall * 100:.2f}%"},
        {"Metric": "Macro F1 Score", "Score": f"{macro_f1 * 100:.2f}%"},
        {"Metric": "Weighted F1 Score", "Score": f"{weighted_f1 * 100:.2f}%"},
    ])

    cm = confusion_matrix(y_test, predicted_codes, labels=np.arange(len(POPULARITY_LABELS)))
    cm_df = pd.DataFrame(
        cm,
        columns=[f"Predicted: {label}" for label in POPULARITY_LABELS],
    )
    cm_df.insert(0, "Actual category", POPULARITY_LABELS)

    report = classification_report(
        y_test,
        predicted_codes,
        labels=np.arange(len(POPULARITY_LABELS)),
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

    full_counts = y_codes.value_counts().reindex(range(len(POPULARITY_LABELS)), fill_value=0)
    distribution_df = pd.DataFrame({
        "Popularity Category": POPULARITY_LABELS,
        "Score Range": POPULARITY_RANGES,
        "Records": [int(full_counts[code]) for code in range(len(POPULARITY_LABELS))],
        "Percentage": [f"{full_counts[code] / len(y_codes) * 100:.2f}%" for code in range(len(POPULARITY_LABELS))],
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
        "class_distribution_table": dataframe_to_html(distribution_df),
    }
