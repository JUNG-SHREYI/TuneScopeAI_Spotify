"""TuneScope AI - regression and popularity-classification model utilities.

The module trains Linear, Lasso, Ridge, and Logistic Regression models on
Spotify audio features. The raw input DataFrame is never modified.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Popularity groups are project-defined categories, not official Spotify labels.
POPULARITY_BINS = [-0.001, 20, 40, 60, 80, 100.001]
POPULARITY_LABELS = ["Very Low", "Low", "Moderate", "High", "Very High"]
POPULARITY_RANGES = ["0–20", "21–40", "41–60", "61–80", "81–100"]

# Numeric features related to a track's audio. Popularity itself is deliberately
# excluded from X so the classifier/regressors cannot read the answer directly.
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

_PALETTE = {
    "background": "#111a2d",
    "text": "#eef3ff",
    "muted": "#91a5cb",
    "cyan": "#4de4e2",
    "violet": "#a994ff",
    "border": "#263653",
    "green": "#55e6aa",
}


def dataframe_to_html(df, table_class="eda-table", precision=4):
    """Convert a DataFrame to an escaped HTML table using the shared CSS table class."""
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
    ax.set_facecolor(_PALETTE["background"])
    ax.tick_params(colors=_PALETTE["muted"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(_PALETTE["border"])
    ax.grid(color=_PALETTE["border"], alpha=0.55, linewidth=0.65)
    ax.set_axisbelow(True)


def _prepare_model_data(raw_df):
    """Return a numeric feature frame and popularity target from a safe copy."""
    if raw_df is None or raw_df.empty:
        raise ValueError("The Spotify dataset is empty.")
    if "popularity" not in raw_df.columns:
        raise ValueError("The dataset must contain a 'popularity' column.")

    feature_names = [column for column in AUDIO_FEATURES if column in raw_df.columns]
    if len(feature_names) < 3:
        raise ValueError(
            "At least three supported audio features are required. "
            "Check duration_ms, danceability, energy, loudness, tempo, and related columns."
        )

    work = raw_df.copy(deep=True)
    target = pd.to_numeric(work["popularity"], errors="coerce")
    target = target.replace([np.inf, -np.inf], np.nan)
    valid_target = target.notna() & target.between(0, 100)

    X = work.loc[valid_target, feature_names].copy()
    y = target.loc[valid_target].astype(float).copy()

    for column in feature_names:
        if pd.api.types.is_bool_dtype(X[column]):
            X[column] = X[column].astype("float64")
        else:
            X[column] = pd.to_numeric(X[column], errors="coerce")
        X[column] = X[column].replace([np.inf, -np.inf], np.nan)

    # A feature that is entirely missing cannot be median-imputed meaningfully.
    usable_features = [column for column in X.columns if X[column].notna().any()]
    X = X[usable_features]
    if X.empty or X.shape[1] < 3:
        raise ValueError("Too few usable numeric audio features remain after validation.")

    if len(X) < 20:
        raise ValueError("At least 20 valid popularity records are required to train and test models.")

    categories = pd.cut(
        y,
        bins=POPULARITY_BINS,
        labels=POPULARITY_LABELS,
        include_lowest=True,
        right=True,
    )
    valid_categories = categories.notna()
    X = X.loc[valid_categories].reset_index(drop=True)
    y = y.loc[valid_categories].reset_index(drop=True)
    categories = categories.loc[valid_categories].astype(str).reset_index(drop=True)

    # Pipeline fits imputation/scaling only on the training data to avoid leakage.
    return X, y, categories, list(X.columns)


def _numeric_model_pipeline(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def _regression_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R² Score": float(r2_score(y_true, y_pred)),
    }


def _metrics_table(metrics):
    return dataframe_to_html(
        pd.DataFrame([{"Metric": label, "Value": value} for label, value in metrics.items()]),
        table_class="eda-table",
        precision=4,
    )


def _coefficients(model_pipeline, feature_names):
    model = model_pipeline.named_steps["model"]
    coefficients = np.asarray(model.coef_, dtype=float).ravel()
    result = pd.DataFrame({
        "Audio Feature": feature_names,
        "Slope Coefficient (standardized features)": coefficients,
        "Effect": ["Positive" if value > 0 else "Negative" if value < 0 else "Near zero" for value in coefficients],
        "Absolute Coefficient": np.abs(coefficients),
    })
    result = result.sort_values("Absolute Coefficient", ascending=False).reset_index(drop=True)
    return result


def _model_intercept(model_pipeline):
    model = model_pipeline.named_steps["model"]
    intercept = np.asarray(model.intercept_).ravel()
    return float(intercept[0] if len(intercept) else 0.0)


def _equation(intercept, coefficient_df, feature_names):
    """Build a readable equation; coefficients are for standardized predictors z(feature)."""
    coef_map = dict(zip(
        coefficient_df["Audio Feature"],
        coefficient_df["Slope Coefficient (standardized features)"],
    ))
    equation = f"Predicted popularity = {intercept:.4f}"
    for feature in feature_names:
        coefficient = float(coef_map.get(feature, 0.0))
        sign = "+" if coefficient >= 0 else "−"
        equation += f" {sign} {abs(coefficient):.4f} × z({feature})"
    return equation


def _coefficients_for_display(coefficient_df):
    show = coefficient_df.copy()
    show["Slope Coefficient (standardized features)"] = show[
        "Slope Coefficient (standardized features)"
    ].round(5)
    show["Absolute Coefficient"] = show["Absolute Coefficient"].round(5)
    return show


def _actual_vs_predicted_chart(y_true, y_pred, title):
    y_values = np.asarray(y_true, dtype=float)
    predictions = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(42)
    if len(y_values) > 4500:
        selected = rng.choice(len(y_values), size=4500, replace=False)
        plot_y, plot_pred = y_values[selected], predictions[selected]
    else:
        plot_y, plot_pred = y_values, predictions

    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    fig.patch.set_facecolor(_PALETTE["background"])
    _style_axes(ax)
    ax.scatter(plot_y, plot_pred, s=12, alpha=0.38, color=_PALETTE["cyan"], edgecolors="none")
    ax.plot([0, 100], [0, 100], linestyle="--", linewidth=1.8, color=_PALETTE["violet"], label="Ideal prediction")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Actual popularity", color=_PALETTE["muted"])
    ax.set_ylabel("Predicted popularity", color=_PALETTE["muted"])
    ax.set_title(title, color=_PALETTE["text"], fontsize=11, loc="left", pad=12)
    ax.legend(frameon=False, labelcolor=_PALETTE["muted"], fontsize=8)
    fig.tight_layout()
    return _figure_to_base64(fig)


def _lasso_residual_chart(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residuals = actual - predicted
    rng = np.random.default_rng(42)
    if len(actual) > 4500:
        indices = rng.choice(len(actual), size=4500, replace=False)
        plotted_pred, plotted_residuals = predicted[indices], residuals[indices]
    else:
        plotted_pred, plotted_residuals = predicted, residuals

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.0))
    fig.patch.set_facecolor(_PALETTE["background"])
    _style_axes(ax1)
    _style_axes(ax2)
    ax1.scatter(plotted_pred, plotted_residuals, s=11, alpha=0.36, color=_PALETTE["cyan"], edgecolors="none")
    ax1.axhline(0, color=_PALETTE["violet"], linestyle="--", linewidth=1.6)
    ax1.set_xlabel("Predicted popularity", color=_PALETTE["muted"])
    ax1.set_ylabel("Residual (actual − predicted)", color=_PALETTE["muted"])
    ax1.set_title("Residuals vs predicted", color=_PALETTE["text"], fontsize=10, loc="left")
    ax2.hist(residuals, bins=32, color=_PALETTE["violet"], edgecolor=_PALETTE["background"], alpha=0.9)
    ax2.set_xlabel("Residual", color=_PALETTE["muted"])
    ax2.set_ylabel("Number of tracks", color=_PALETTE["muted"])
    ax2.set_title("Residual distribution", color=_PALETTE["text"], fontsize=10, loc="left")
    fig.tight_layout()
    return _figure_to_base64(fig)


def _confusion_matrix_chart(cm, labels):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    fig.patch.set_facecolor(_PALETTE["background"])
    ax.set_facecolor(_PALETTE["background"])
    image = ax.imshow(cm, cmap="Purples", interpolation="nearest")
    ax.set_title("Logistic Regression — Confusion Matrix", color=_PALETTE["text"], fontsize=11, loc="left", pad=12)
    ax.set_xlabel("Predicted category", color=_PALETTE["muted"])
    ax.set_ylabel("Actual category", color=_PALETTE["muted"])
    ax.set_xticks(np.arange(len(labels), dtype=int), labels=labels, rotation=20, ha="right", color=_PALETTE["muted"], fontsize=8)
    ax.set_yticks(np.arange(len(labels), dtype=int), labels=labels, color=_PALETTE["muted"], fontsize=8)
    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, f"{cm[row, col]:,}", ha="center", va="center", fontsize=8,
                    color="white" if cm[row, col] > threshold else "#1a1830")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _figure_to_base64(fig)


def _popularity_distribution_chart(categories):
    counts = categories.value_counts().reindex(POPULARITY_LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    fig.patch.set_facecolor(_PALETTE["background"])
    _style_axes(ax)
    palette = ["#7770d9", "#8f81ed", "#6d9fe9", "#4de4e2", "#55e6aa"]
    bars = ax.bar(POPULARITY_LABELS, [int(counts.get(label, 0)) for label in POPULARITY_LABELS], color=palette)
    ax.set_title("Popularity category distribution", color=_PALETTE["text"], fontsize=11, loc="left", pad=12)
    ax.set_ylabel("Number of tracks", color=_PALETTE["muted"])
    ax.tick_params(axis="x", colors=_PALETTE["muted"], labelsize=8, rotation=15)
    ax.tick_params(axis="y", colors=_PALETTE["muted"], labelsize=8)
    for bar in bars:
        height = int(bar.get_height())
        ax.annotate(f"{height:,}", (bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha="center", color=_PALETTE["text"], fontsize=8)
    fig.tight_layout()
    return _figure_to_base64(fig)


def _classification_report_table(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=POPULARITY_LABELS,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in POPULARITY_LABELS:
        values = report.get(label, {})
        rows.append({
            "Class": label,
            "Precision": float(values.get("precision", 0.0)),
            "Recall": float(values.get("recall", 0.0)),
            "F1 Score": float(values.get("f1-score", 0.0)),
            "Support": int(values.get("support", 0)),
        })
    for label in ("macro avg", "weighted avg"):
        values = report.get(label, {})
        rows.append({
            "Class": label.title(),
            "Precision": float(values.get("precision", 0.0)),
            "Recall": float(values.get("recall", 0.0)),
            "F1 Score": float(values.get("f1-score", 0.0)),
            "Support": int(values.get("support", 0)),
        })
    return dataframe_to_html(pd.DataFrame(rows), precision=4)


def _category_distribution_table(categories):
    counts = categories.value_counts().reindex(POPULARITY_LABELS, fill_value=0)
    total = len(categories)
    table = pd.DataFrame({
        "Popularity Category": POPULARITY_LABELS,
        "Score Range": POPULARITY_RANGES,
        "Track Count": [int(counts[label]) for label in POPULARITY_LABELS],
        "Percentage (%)": [float(counts[label] / total * 100) if total else 0.0 for label in POPULARITY_LABELS],
    })
    return dataframe_to_html(table, precision=2)


def _record_split_table(train_count, test_count):
    total = train_count + test_count
    return dataframe_to_html(pd.DataFrame([
        {"Dataset": "Training", "Records": int(train_count), "Percentage (%)": train_count / total * 100 if total else 0},
        {"Dataset": "Testing", "Records": int(test_count), "Percentage (%)": test_count / total * 100 if total else 0},
        {"Dataset": "Total", "Records": int(total), "Percentage (%)": 100.0 if total else 0},
    ]), precision=2)


def _model_context(name, model, X_train, X_test, y_train, y_test, feature_names, chart_title):
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    coef_df = _coefficients(model, feature_names)
    intercept = _model_intercept(model)
    metric_dict = _regression_metrics(y_test, predicted)
    return {
        "predicted": predicted,
        "chart": _actual_vs_predicted_chart(y_test, predicted, chart_title),
        "coefficients_df": coef_df,
        "coefficients_table": dataframe_to_html(_coefficients_for_display(coef_df), precision=5),
        "intercept": intercept,
        "equation": _equation(intercept, coef_df, feature_names),
        "metrics": metric_dict,
        "metrics_table": _metrics_table(metric_dict),
        "train_count": len(X_train),
        "test_count": len(X_test),
        "split_table": _record_split_table(len(X_train), len(X_test)),
    }


def build_regressions_context(raw_df):
    """Fit requested models and return context for templates/regressions.html."""
    X, y, popularity_categories, feature_names = _prepare_model_data(raw_df)

    # Regression uses a common reproducible hold-out split for fair model comparison.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    linear = _model_context(
        "Linear Regression",
        _numeric_model_pipeline(LinearRegression()),
        X_train, X_test, y_train, y_test, feature_names,
        "Linear Regression — Actual vs Predicted Popularity",
    )

    lasso_model = _numeric_model_pipeline(Lasso(alpha=0.1, max_iter=10000, selection="cyclic"))
    lasso_model.fit(X_train, y_train)
    lasso_predicted = lasso_model.predict(X_test)
    lasso_coefficients = _coefficients(lasso_model, feature_names)
    lasso_intercept = _model_intercept(lasso_model)
    lasso_metrics = _regression_metrics(y_test, lasso_predicted)
    lasso = {
        "predicted": lasso_predicted,
        "residual_chart": _lasso_residual_chart(y_test, lasso_predicted),
        "coefficients_table": dataframe_to_html(_coefficients_for_display(lasso_coefficients), precision=5),
        "intercept": lasso_intercept,
        "equation": _equation(lasso_intercept, lasso_coefficients, feature_names),
        "metrics_table": _metrics_table(lasso_metrics),
        "train_count": len(X_train),
        "test_count": len(X_test),
        "split_table": _record_split_table(len(X_train), len(X_test)),
        "alpha": 0.1,
        "max_iter": 10000,
        "model_setting": "Lasso Regression with L1 regularization (alpha = 0.1), median imputation, and standard-scaled audio features.",
    }

    # Logistic Regression classifies a track into a project-defined popularity group.
    category_counts = popularity_categories.value_counts()
    stratify_values = popularity_categories if category_counts.min() >= 2 else None
    X_cls_train, X_cls_test, cat_train, cat_test = train_test_split(
        X,
        popularity_categories,
        test_size=0.20,
        random_state=42,
        stratify=stratify_values,
    )
    logistic_model = _numeric_model_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    )
    logistic_model.fit(X_cls_train, cat_train)
    cat_predicted = logistic_model.predict(X_cls_test)
    cm = confusion_matrix(cat_test, cat_predicted, labels=POPULARITY_LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {label}" for label in POPULARITY_LABELS],
        columns=[f"Predicted {label}" for label in POPULARITY_LABELS],
    ).reset_index(names="Category")
    logistic_accuracy = float(accuracy_score(cat_test, cat_predicted))
    full_distribution = _category_distribution_table(popularity_categories)
    distribution_chart = _popularity_distribution_chart(popularity_categories)
    logistic = {
        "confusion_chart": _confusion_matrix_chart(cm, POPULARITY_LABELS),
        "confusion_table": dataframe_to_html(cm_df, precision=0),
        "classification_report_table": _classification_report_table(cat_test, cat_predicted),
        "accuracy": logistic_accuracy,
        "accuracy_percent": logistic_accuracy * 100.0,
        "train_count": len(X_cls_train),
        "test_count": len(X_cls_test),
        "split_table": _record_split_table(len(X_cls_train), len(X_cls_test)),
        "category_distribution_table": full_distribution,
        "category_distribution_chart": distribution_chart,
        "model_setting": "Multiclass Logistic Regression with balanced class weights, median imputation, standard scaling, and max_iter = 1000.",
    }

    ridge_model = _numeric_model_pipeline(Ridge(alpha=1.0))
    ridge_model.fit(X_train, y_train)
    ridge_predicted = ridge_model.predict(X_test)
    ridge_coefficients = _coefficients(ridge_model, feature_names)
    ridge_intercept = _model_intercept(ridge_model)
    ridge_metrics = _regression_metrics(y_test, ridge_predicted)
    ridge_analysis = ridge_coefficients.copy()
    absolute_total = float(ridge_analysis["Absolute Coefficient"].sum())
    ridge_analysis["Relative Influence (%)"] = (
        ridge_analysis["Absolute Coefficient"] / absolute_total * 100.0
        if absolute_total else 0.0
    )
    ridge_analysis = ridge_analysis.rename(columns={
        "Slope Coefficient (standardized features)": "Standardized Coefficient",
        "Effect": "Direction",
        "Absolute Coefficient": "Absolute Coefficient",
    })
    ridge_analysis = ridge_analysis[[
        "Audio Feature", "Standardized Coefficient", "Direction", "Relative Influence (%)"
    ]].round(5)
    if ridge_metrics["R² Score"] >= 0.50:
        graph_interpretation = (
            "The points follow the ideal diagonal reasonably well, suggesting the audio features capture "
            "some popularity variation. Points far from the diagonal are prediction errors."
        )
    elif ridge_metrics["R² Score"] >= 0.15:
        graph_interpretation = (
            "The graph shows some relationship between actual and predicted popularity, but considerable "
            "spread from the ideal diagonal remains. Audio features alone provide limited-to-moderate predictive information."
        )
    else:
        graph_interpretation = (
            "The predictions are widely spread around the ideal diagonal. With these features, the model explains "
            "only a limited share of popularity variation; popularity can also depend on artist, promotion, release timing, and other unavailable factors."
        )

    ridge = {
        "chart": _actual_vs_predicted_chart(y_test, ridge_predicted, "Ridge Regression — Actual vs Predicted Popularity"),
        "graph_interpretation": graph_interpretation,
        "coefficients_table": dataframe_to_html(_coefficients_for_display(ridge_coefficients), precision=5),
        "audio_analysis_table": dataframe_to_html(ridge_analysis, precision=5),
        "intercept": ridge_intercept,
        "equation": _equation(ridge_intercept, ridge_coefficients, feature_names),
        "metrics_table": _metrics_table(ridge_metrics),
        "metrics": ridge_metrics,
        "train_count": len(X_train),
        "test_count": len(X_test),
        "split_table": _record_split_table(len(X_train), len(X_test)),
        "alpha": 1.0,
    }

    category_definitions = pd.DataFrame({
        "Popularity Category": POPULARITY_LABELS,
        "Score Range": POPULARITY_RANGES,
        "Meaning": [
            "Lower popularity score", "Below-mid popularity score", "Middle popularity score",
            "Above-mid popularity score", "Highest popularity score",
        ],
    })

    return {
        "total_records": len(X),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "linear": linear,
        "lasso": lasso,
        "logistic": logistic,
        "ridge": ridge,
        "popularity_categories_table": dataframe_to_html(category_definitions, precision=0),
        "category_labels": POPULARITY_LABELS,
    }
