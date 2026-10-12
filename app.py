"""TuneScope AI: raw dataset viewer, EDA, and regression models."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from data_loader import DATASET_PATH, load_data
from eda import build_eda_context
from regressions import build_regressions_context
from preprocessing import build_preprocessing_context
from adaboost import build_adaboost_context
from random_forest import build_random_forest_context
from gradient_boosting import build_gradient_boosting_context
from xgboost_model import build_xgboost_context
from lightgbm_model import build_lightgbm_context
from feature_importance import build_feature_importance_context
from pca_analysis import build_pca_context
from clustering import build_clustering_context
from model_evaluation import build_evaluation_context
from ml_engineering import build_ml_engineering_context

app = Flask(__name__)
ROWS_PER_PAGE = 100
_REGRESSION_CACHE = {"signature": None, "context": None}
_PREPROCESSING_CACHE = {"signature": None, "context": None}
_ADABOOST_CACHE = {"signature": None, "context": None}
_RANDOM_FOREST_CACHE = {"signature": None, "context": None}
_GRADIENT_BOOSTING_CACHE = {"signature": None, "context": None}
_XGBOOST_CACHE = {"signature": None, "context": None}
_LIGHTGBM_CACHE = {"signature": None, "context": None}
_FEATURE_IMPORTANCE_CACHE = {"signature": None, "context": None}
_PCA_CACHE = {"signature": None, "context": None}
_CLUSTERING_CACHE = {"signature": None, "context": None}
_EVALUATION_CACHE = {"signature": None, "context": None}
_ML_ENGINEERING_CACHE = {"signature": None, "context": None}


def display_value(value):
    """Make values safe and readable in the raw-data table."""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


@app.route("/")
def dataset_page():
    """Display the original CSV as a paginated, unmodified table."""
    try:
        df = load_data()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        return render_template("index.html", error=str(exc), active_page="dataset"), 500

    total_rows = len(df)
    total_pages = max(1, math.ceil(total_rows / ROWS_PER_PAGE))
    requested_page = request.args.get("page", default=1, type=int)
    page = min(max(requested_page or 1, 1), total_pages)
    start = (page - 1) * ROWS_PER_PAGE
    page_df = df.iloc[start:start + ROWS_PER_PAGE]
    columns = [str(column) for column in df.columns]
    rows = [[display_value(value) for value in row] for row in page_df.itertuples(index=False, name=None)]

    return render_template(
        "index.html", error=None, active_page="dataset", columns=columns, rows=rows,
        page=page, total_pages=total_pages, total_rows=total_rows,
        first_row=start + 1 if total_rows else 0,
        last_row=min(start + len(rows), total_rows),
    )


@app.route("/eda")
def eda_page():
    """Render the five EDA sections."""
    try:
        raw_df = load_data()
        context = build_eda_context(raw_df)
        return render_template("eda.html", active_page="eda", error=None, **context)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        return render_template("eda.html", active_page="eda", error=str(exc)), 500
    except Exception as exc:
        app.logger.exception("EDA page failed")
        return render_template("eda.html", active_page="eda", error=f"EDA could not be generated: {exc}"), 500


@app.route("/regressions")
def regressions_page():
    """Render Linear, Lasso, Logistic, and Ridge Regression results.

    Model output is cached until the source CSV changes, to avoid retraining on
    every refresh. The cache key comes from the file modification time and size.
    """
    try:
        path = Path(DATASET_PATH)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if _REGRESSION_CACHE["signature"] != signature or _REGRESSION_CACHE["context"] is None:
            raw_df = load_data()
            context = build_regressions_context(raw_df)
            _REGRESSION_CACHE["context"] = context
            _REGRESSION_CACHE["signature"] = signature
        else:
            context = _REGRESSION_CACHE["context"]
        return render_template("regressions.html", active_page="regressions", error=None, **context)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        return render_template("regressions.html", active_page="regressions", error=str(exc)), 500
    except Exception as exc:
        app.logger.exception("Regression page failed")
        return render_template("regressions.html", active_page="regressions", error=f"Regression models could not be generated: {exc}"), 500


@app.route("/preprocessing")
def preprocessing_page():
    """Display One-Hot, Ordinal and Target Encoding examples."""
    try:
        path = Path(DATASET_PATH)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if (
            _PREPROCESSING_CACHE["signature"] != signature
            or _PREPROCESSING_CACHE["context"] is None
        ):
            raw_df = load_data()
            context = build_preprocessing_context(raw_df)
            _PREPROCESSING_CACHE["context"] = context
            _PREPROCESSING_CACHE["signature"] = signature
        else:
            context = _PREPROCESSING_CACHE["context"]

        return render_template(
            "preprocessing.html",
            active_page="preprocessing",
            error=None,
            **context,
        )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        return render_template(
            "preprocessing.html",
            active_page="preprocessing",
            error=str(exc),
        ), 500
    except Exception as exc:
        app.logger.exception("Preprocessing page failed")
        return render_template(
            "preprocessing.html",
            active_page="preprocessing",
            error=f"Preprocessing could not be generated: {exc}",
        ), 500



@app.route("/adaboost")
def adaboost_page():
    """Train AdaBoost once per source-dataset version and display its results."""
    try:
        path = Path(DATASET_PATH)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if _ADABOOST_CACHE["signature"] != signature or _ADABOOST_CACHE["context"] is None:
            raw_df = load_data()
            context = build_adaboost_context(raw_df)
            _ADABOOST_CACHE["context"] = context
            _ADABOOST_CACHE["signature"] = signature
        else:
            context = _ADABOOST_CACHE["context"]
        return render_template("adaboost.html", active_page="adaboost", error=None, **context)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        return render_template("adaboost.html", active_page="adaboost", error=str(exc)), 500
    except Exception as exc:
        app.logger.exception("AdaBoost page failed")
        return render_template(
            "adaboost.html",
            active_page="adaboost",
            error=f"AdaBoost results could not be generated: {exc}",
        ), 500


@app.route("/random_forest")
@app.route("/random-forest")
def random_forest_page():
    """Train Random Forest once per source-dataset version and display its results."""
    try:
        path = Path(DATASET_PATH)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if (
            _RANDOM_FOREST_CACHE["signature"] != signature
            or _RANDOM_FOREST_CACHE["context"] is None
        ):
            raw_df = load_data()
            context = build_random_forest_context(raw_df)
            _RANDOM_FOREST_CACHE["context"] = context
            _RANDOM_FOREST_CACHE["signature"] = signature
        else:
            context = _RANDOM_FOREST_CACHE["context"]

        return render_template(
            "random_forest.html",
            active_page="random_forest",
            error=None,
            **context,
        )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        return render_template(
            "random_forest.html",
            active_page="random_forest",
            error=str(exc),
        ), 500
    except Exception as exc:
        app.logger.exception("Random Forest page failed")
        return render_template(
            "random_forest.html",
            active_page="random_forest",
            error=f"Random Forest results could not be generated: {exc}",
        ), 500


@app.route("/gradient_boosting")
@app.route("/gradient-boosting")
def gradient_boosting_page():
    """Train Gradient Boosting once per source-dataset version and show results."""
    try:
        path = Path(DATASET_PATH)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if (
            _GRADIENT_BOOSTING_CACHE["signature"] != signature
            or _GRADIENT_BOOSTING_CACHE["context"] is None
        ):
            raw_df = load_data()
            context = build_gradient_boosting_context(raw_df)
            _GRADIENT_BOOSTING_CACHE["context"] = context
            _GRADIENT_BOOSTING_CACHE["signature"] = signature
        else:
            context = _GRADIENT_BOOSTING_CACHE["context"]

        return render_template(
            "gradient_boosting.html",
            active_page="gradient_boosting",
            error=None,
            **context,
        )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        return render_template(
            "gradient_boosting.html",
            active_page="gradient_boosting",
            error=str(exc),
        ), 500
    except Exception as exc:
        app.logger.exception("Gradient Boosting page failed")
        return render_template(
            "gradient_boosting.html",
            active_page="gradient_boosting",
            error=f"Gradient Boosting results could not be generated: {exc}",
        ), 500


@app.route("/xgboost")
@app.route("/xg-boost")
def xgboost_page():
    """Train XGBoost once per source-dataset version and display its results."""
    try:
        path = Path(DATASET_PATH)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if _XGBOOST_CACHE["signature"] != signature or _XGBOOST_CACHE["context"] is None:
            raw_df = load_data()
            context = build_xgboost_context(raw_df)
            _XGBOOST_CACHE["context"] = context
            _XGBOOST_CACHE["signature"] = signature
        else:
            context = _XGBOOST_CACHE["context"]

        return render_template(
            "xgboost.html",
            active_page="xgboost",
            error=None,
            **context,
        )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        return render_template(
            "xgboost.html",
            active_page="xgboost",
            error=str(exc),
        ), 500
    except Exception as exc:
        app.logger.exception("XGBoost page failed")
        return render_template(
            "xgboost.html",
            active_page="xgboost",
            error=f"XGBoost results could not be generated: {exc}",
        ), 500


@app.route("/lightgbm")
@app.route("/light-gbm")
def lightgbm_page():
    """Train LightGBM once per source-dataset version and display its results."""
    try:
        path = Path(DATASET_PATH)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if (
            _LIGHTGBM_CACHE["signature"] != signature
            or _LIGHTGBM_CACHE["context"] is None
        ):
            raw_df = load_data()
            context = build_lightgbm_context(raw_df)
            _LIGHTGBM_CACHE["context"] = context
            _LIGHTGBM_CACHE["signature"] = signature
        else:
            context = _LIGHTGBM_CACHE["context"]

        return render_template(
            "lightgbm.html",
            active_page="lightgbm",
            error=None,
            **context,
        )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        return render_template(
            "lightgbm.html",
            active_page="lightgbm",
            error=str(exc),
        ), 500
    except Exception as exc:
        app.logger.exception("LightGBM page failed")
        return render_template(
            "lightgbm.html",
            active_page="lightgbm",
            error=f"LightGBM results could not be generated: {exc}",
        ), 500


@app.route("/feature-importance")
@app.route("/feature_importance")
def feature_importance_page():
    """Display averaged feature rankings and model-by-model comparisons."""
    try:
        path = Path(DATASET_PATH)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if (
            _FEATURE_IMPORTANCE_CACHE["signature"] != signature
            or _FEATURE_IMPORTANCE_CACHE["context"] is None
        ):
            raw_df = load_data()
            context = build_feature_importance_context(raw_df)
            _FEATURE_IMPORTANCE_CACHE["context"] = context
            _FEATURE_IMPORTANCE_CACHE["signature"] = signature
        else:
            context = _FEATURE_IMPORTANCE_CACHE["context"]

        return render_template(
            "feature_importance.html",
            active_page="feature_importance",
            error=None,
            **context,
        )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        return render_template(
            "feature_importance.html",
            active_page="feature_importance",
            error=str(exc),
        ), 500
    except Exception as exc:
        app.logger.exception("Feature Importance page failed")
        return render_template(
            "feature_importance.html",
            active_page="feature_importance",
            error=f"Feature importance could not be generated: {exc}",
        ), 500


@app.route("/pca")
@app.route("/pca-analysis")
def pca_page():
    """Show PCA variance, loadings, and paginated transformed records."""
    try:
        path = Path(DATASET_PATH)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if _PCA_CACHE["signature"] != signature or _PCA_CACHE["context"] is None:
            raw_df = load_data()
            context = build_pca_context(raw_df)
            _PCA_CACHE["context"] = context
            _PCA_CACHE["signature"] = signature
        context = _PCA_CACHE["context"]

        # Paginate transformed records so the browser only receives 100 rows
        # per request while every original record remains available.
        transformed_df = context["transformed_df"]
        rows_per_page = 100
        total_rows = len(transformed_df)
        total_pages = max(1, (total_rows + rows_per_page - 1) // rows_per_page)
        requested_page = request.args.get("page", default=1, type=int) or 1
        page = min(max(requested_page, 1), total_pages)
        start = (page - 1) * rows_per_page
        end = min(start + rows_per_page, total_rows)
        page_df = transformed_df.iloc[start:end]
        transformed_table = page_df.to_html(
            classes="pca-table", index=False, border=0, escape=True,
            float_format=lambda value: f"{value:.5f}",
        )

        render_context = {key: value for key, value in context.items() if key != "transformed_df"}
        render_context.update({
            "transformed_table": transformed_table,
            "transformed_page": page,
            "transformed_total_pages": total_pages,
            "transformed_total_rows": total_rows,
            "transformed_first_row": start + 1 if total_rows else 0,
            "transformed_last_row": end,
        })
        return render_template(
            "pca.html", active_page="pca", error=None, **render_context
        )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        return render_template("pca.html", active_page="pca", error=str(exc)), 500
    except Exception as exc:
        app.logger.exception("PCA page failed")
        return render_template(
            "pca.html", active_page="pca",
            error=f"PCA results could not be generated: {exc}",
        ), 500


@app.route("/clustering")
@app.route("/unsupervised")
def clustering_page():
    """CO4 – Unsupervised Learning & Dimensionality Reduction."""
    try:
        path = Path(DATASET_PATH)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if _CLUSTERING_CACHE["signature"] != signature or _CLUSTERING_CACHE["context"] is None:
            raw_df = load_data()
            context = build_clustering_context(raw_df)
            _CLUSTERING_CACHE["context"] = context
            _CLUSTERING_CACHE["signature"] = signature
        else:
            context = _CLUSTERING_CACHE["context"]
        return render_template("clustering.html", active_page="clustering", error=None, **context)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        return render_template("clustering.html", active_page="clustering", error=str(exc)), 500
    except Exception as exc:
        app.logger.exception("Clustering page failed")
        return render_template(
            "clustering.html", active_page="clustering",
            error=f"Clustering results could not be generated: {exc}"), 500


@app.route("/evaluation")
@app.route("/model-evaluation")
def evaluation_page():
    """CO5 – Model Evaluation and Optimization."""
    try:
        path = Path(DATASET_PATH)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if _EVALUATION_CACHE["signature"] != signature or _EVALUATION_CACHE["context"] is None:
            raw_df = load_data()
            context = build_evaluation_context(raw_df)
            _EVALUATION_CACHE["context"] = context
            _EVALUATION_CACHE["signature"] = signature
        else:
            context = _EVALUATION_CACHE["context"]
        return render_template("evaluation.html", active_page="evaluation", error=None, **context)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        return render_template("evaluation.html", active_page="evaluation", error=str(exc)), 500
    except Exception as exc:
        app.logger.exception("Evaluation page failed")
        return render_template(
            "evaluation.html", active_page="evaluation",
            error=f"Evaluation results could not be generated: {exc}"), 500


@app.route("/ml-engineering")
@app.route("/deployment")
def ml_engineering_page():
    """CO6 – ML Engineering and Deployment."""
    try:
        path = Path(DATASET_PATH)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if _ML_ENGINEERING_CACHE["signature"] != signature or _ML_ENGINEERING_CACHE["context"] is None:
            raw_df = load_data()
            context = build_ml_engineering_context(raw_df)
            _ML_ENGINEERING_CACHE["context"] = context
            _ML_ENGINEERING_CACHE["signature"] = signature
        else:
            context = _ML_ENGINEERING_CACHE["context"]
        return render_template("ml_engineering.html", active_page="ml_engineering", error=None, **context)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        return render_template("ml_engineering.html", active_page="ml_engineering", error=str(exc)), 500
    except Exception as exc:
        app.logger.exception("ML Engineering page failed")
        return render_template(
            "ml_engineering.html", active_page="ml_engineering",
            error=f"ML Engineering results could not be generated: {exc}"), 500


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
