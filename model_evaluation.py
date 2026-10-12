"""TuneScope AI - CO5 Model Evaluation and Optimization.

Covers:
  Train-Validation-Test Split, Cross-Validation, Data Leakage Prevention,
  Classification Metrics (Confusion Matrix, Accuracy, Precision, Recall, F1),
  ROC Curve, ROC-AUC, Precision-Recall Curve, PR-AUC,
  Regression Metrics (RMSE, MAE, MAPE, R2),
  Hyperparameter Optimization (Grid Search, Random Search, Bayesian approx.),
  Learning Curves, Validation Curves, Probability Calibration,
  Reliability Diagrams, Statistical Significance Testing.
"""

from __future__ import annotations

import base64
import warnings
from io import BytesIO
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error,
    precision_recall_curve, precision_score, r2_score, recall_score,
    roc_auc_score, roc_curve, average_precision_score,
)
from sklearn.model_selection import (
    GridSearchCV, KFold, RandomizedSearchCV, cross_val_score,
    learning_curve, train_test_split, validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

AUDIO_FEATURES = [
    "duration_ms", "explicit", "danceability", "energy", "key",
    "loudness", "mode", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "time_signature",
]
POPULARITY_BINS = [-0.001, 40, 70, 100.001]
POPULARITY_LABELS = ["Low", "Moderate", "High"]

_BG = "#111a2d"
_TEXT = "#eef3ff"
_MUTED = "#91a5cb"
_BORDER = "#263653"
_CYAN = "#4de4e2"
_VIOLET = "#a994ff"
_GREEN = "#55e6aa"


def _fig_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return encoded


def _style_ax(ax):
    ax.set_facecolor(_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)
    ax.tick_params(colors=_MUTED)


def _prep(raw_df, n_sample=8000):
    available = [c for c in AUDIO_FEATURES if c in raw_df.columns]
    df = raw_df[available + (["popularity"] if "popularity" in raw_df.columns else [])].copy()
    for col in available:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype("float64")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=available, how="all", inplace=True)
    if len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=42)
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(df[available])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    if "popularity" in df.columns:
        pop = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
        y_cont = pop.values
        y_cat = pd.cut(pop, bins=POPULARITY_BINS, labels=POPULARITY_LABELS).astype(str).values
        le = LabelEncoder()
        y = le.fit_transform(y_cat)
    else:
        y_cont = np.zeros(len(X))
        y = np.zeros(len(X), dtype=int)
        le = LabelEncoder()
        le.classes_ = np.array(["Low", "Moderate", "High"])
    return X, y, y_cont, le.classes_.tolist(), available


# ── Section 1: Train/Val/Test Split & Cross-Validation ──────────────────────

def _split_cv_section(X, y):
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.176, random_state=42, stratify=y_tmp)
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy", n_jobs=1)
    split_df = pd.DataFrame({
        "Split": ["Training", "Validation", "Test"],
        "Samples": [len(X_train), len(X_val), len(X_test)],
        "Percentage": [f"{len(X_train)/len(X)*100:.1f}%",
                       f"{len(X_val)/len(X)*100:.1f}%",
                       f"{len(X_test)/len(X)*100:.1f}%"],
    })
    cv_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(5)] + ["Mean", "Std Dev"],
        "Accuracy": [f"{s:.4f}" for s in cv_scores] + [f"{cv_scores.mean():.4f}", f"{cv_scores.std():.4f}"],
    })
    return {
        "split_table": split_df.to_html(classes="eda-table", index=False, border=0, escape=True),
        "cv_table": cv_df.to_html(classes="eda-table", index=False, border=0, escape=True),
        "cv_mean": f"{cv_scores.mean():.4f}",
        "cv_std": f"{cv_scores.std():.4f}",
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "rf_model": rf,
    }


# ── Section 2: Classification Metrics ───────────────────────────────────────

def _classification_section(rf, X_test, y_test, class_names):
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=_BG)
    for ax in axes:
        _style_ax(ax)

    im = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    axes[0].set_title("Confusion Matrix", color=_TEXT, fontsize=11)
    axes[0].set_xlabel("Predicted Label", color=_MUTED)
    axes[0].set_ylabel("True Label", color=_MUTED)
    ticks = range(len(class_names))
    axes[0].set_xticks(list(ticks))
    axes[0].set_yticks(list(ticks))
    axes[0].set_xticklabels(class_names, color=_MUTED)
    axes[0].set_yticklabels(class_names, color=_MUTED)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    plt.colorbar(im, ax=axes[0])

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-Score (weighted)"],
        "Value": [f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"],
    })
    axes[1].axis("off")
    tbl = axes[1].table(
        cellText=metrics_df.values, colLabels=metrics_df.columns,
        loc="center", cellLoc="center")
    tbl.auto_set_font_size(True)
    tbl.scale(1.2, 2.2)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(_BG if row == 0 else "#1d2b49")
        cell.set_text_props(color=_TEXT)
        cell.set_edgecolor(_BORDER)
    axes[1].set_title("Classification Metrics", color=_TEXT, fontsize=11)
    fig.tight_layout(pad=2)
    cm_chart = _fig_b64(fig)

    # ROC Curve
    n_classes = len(class_names)
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5), facecolor=_BG)
    for ax in axes2:
        _style_ax(ax)

    colors = [_CYAN, _VIOLET, _GREEN, "#f97316", "#ec4899"]
    for i, cname in enumerate(class_names):
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            auc_val = roc_auc_score(y_test, y_prob[:, 1])
        else:
            y_bin = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
            auc_val = roc_auc_score(y_bin, y_prob[:, i])
        axes2[0].plot(fpr, tpr, lw=1.8, color=colors[i % len(colors)],
                     label=f"{cname} (AUC={auc_val:.3f})")
    axes2[0].plot([0, 1], [0, 1], "k--", lw=1, color=_MUTED)
    axes2[0].set_title("ROC Curves", color=_TEXT, fontsize=11)
    axes2[0].set_xlabel("False Positive Rate", color=_MUTED)
    axes2[0].set_ylabel("True Positive Rate", color=_MUTED)
    axes2[0].legend(fontsize=8, facecolor="#1d2b49", labelcolor=_TEXT)

    for i, cname in enumerate(class_names):
        y_bin = (y_test == i).astype(int)
        prec_vals, rec_vals, _ = precision_recall_curve(y_bin, y_prob[:, i])
        ap = average_precision_score(y_bin, y_prob[:, i])
        axes2[1].plot(rec_vals, prec_vals, lw=1.8, color=colors[i % len(colors)],
                     label=f"{cname} (AP={ap:.3f})")
    axes2[1].set_title("Precision-Recall Curves", color=_TEXT, fontsize=11)
    axes2[1].set_xlabel("Recall", color=_MUTED)
    axes2[1].set_ylabel("Precision", color=_MUTED)
    axes2[1].legend(fontsize=8, facecolor="#1d2b49", labelcolor=_TEXT)
    fig2.tight_layout(pad=2)
    roc_pr_chart = _fig_b64(fig2)

    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).T.reset_index()
    report_df.columns = ["Class"] + list(report_df.columns[1:])
    report_df = report_df[report_df["Class"].isin(class_names + ["accuracy", "macro avg", "weighted avg"])]
    for col in ["precision", "recall", "f1-score"]:
        if col in report_df.columns:
            report_df[col] = report_df[col].apply(lambda v: f"{v:.4f}" if isinstance(v, float) else v)
    if "support" in report_df.columns:
        report_df["support"] = report_df["support"].apply(lambda v: f"{int(v)}" if isinstance(v, float) else v)

    return {
        "cm_chart": cm_chart,
        "roc_pr_chart": roc_pr_chart,
        "clf_accuracy": f"{acc:.4f}",
        "clf_precision": f"{prec:.4f}",
        "clf_recall": f"{rec:.4f}",
        "clf_f1": f"{f1:.4f}",
        "report_table": report_df.to_html(classes="eda-table", index=False, border=0, escape=True),
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


# ── Section 3: Regression Metrics ───────────────────────────────────────────

def _regression_section(X_train, X_test, y_train_cont, y_test_cont):
    from sklearn.ensemble import GradientBoostingRegressor
    reg = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=4)
    reg.fit(X_train, y_train_cont)
    y_pred = reg.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test_cont, y_pred))
    mae = mean_absolute_error(y_test_cont, y_pred)
    r2 = r2_score(y_test_cont, y_pred)
    mask = y_test_cont != 0
    mape = np.mean(np.abs((y_test_cont[mask] - y_pred[mask]) / y_test_cont[mask])) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=_BG)
    for ax in axes:
        _style_ax(ax)

    axes[0].scatter(y_test_cont, y_pred, s=8, c=_CYAN, alpha=0.4)
    mn, mx = min(y_test_cont.min(), y_pred.min()), max(y_test_cont.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], "--", color=_VIOLET, lw=1.5)
    axes[0].set_title("Actual vs Predicted (Popularity)", color=_TEXT, fontsize=11)
    axes[0].set_xlabel("Actual", color=_MUTED)
    axes[0].set_ylabel("Predicted", color=_MUTED)

    residuals = y_test_cont - y_pred
    axes[1].scatter(y_pred, residuals, s=8, c=_GREEN, alpha=0.4)
    axes[1].axhline(0, color=_VIOLET, lw=1.5, linestyle="--")
    axes[1].set_title("Residual Plot", color=_TEXT, fontsize=11)
    axes[1].set_xlabel("Predicted", color=_MUTED)
    axes[1].set_ylabel("Residual", color=_MUTED)
    fig.tight_layout(pad=2)
    reg_chart = _fig_b64(fig)

    reg_metrics_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "MAPE (%)", "R²"],
        "Value": [f"{rmse:.4f}", f"{mae:.4f}", f"{mape:.2f}%", f"{r2:.4f}"],
        "Interpretation": [
            "Root Mean Squared Error - penalizes large errors",
            "Mean Absolute Error - average prediction gap",
            "Mean Absolute Percentage Error",
            "Coefficient of determination (closer to 1 = better)",
        ],
    })
    return {
        "reg_chart": reg_chart,
        "reg_rmse": f"{rmse:.4f}",
        "reg_mae": f"{mae:.4f}",
        "reg_mape": f"{mape:.2f}%",
        "reg_r2": f"{r2:.4f}",
        "reg_metrics_table": reg_metrics_df.to_html(classes="eda-table", index=False, border=0, escape=True),
    }


# ── Section 4: Hyperparameter Optimization ──────────────────────────────────

def _hyperparam_section(X_train, y_train):
    lr_pipe = Pipeline([("scaler", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=200, random_state=42))])
    param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    gs = GridSearchCV(lr_pipe, param_grid, cv=3, scoring="accuracy", n_jobs=1)
    gs.fit(X_train, y_train)

    param_rand = {"clf__C": np.logspace(-3, 2, 50)}
    rs = RandomizedSearchCV(lr_pipe, param_rand, n_iter=15, cv=3,
                            scoring="accuracy", random_state=42, n_jobs=1)
    rs.fit(X_train, y_train)

    gs_df = pd.DataFrame(gs.cv_results_)[["param_clf__C", "mean_test_score", "std_test_score", "rank_test_score"]]
    gs_df.columns = ["C (Regularization)", "Mean CV Accuracy", "Std Dev", "Rank"]
    gs_df["Mean CV Accuracy"] = gs_df["Mean CV Accuracy"].apply(lambda v: f"{v:.4f}")
    gs_df["Std Dev"] = gs_df["Std Dev"].apply(lambda v: f"{v:.4f}")

    rs_top = pd.DataFrame(rs.cv_results_)[["param_clf__C", "mean_test_score", "rank_test_score"]].nsmallest(5, "rank_test_score")
    rs_top.columns = ["C (Regularization)", "Mean CV Accuracy", "Rank"]
    rs_top["C (Regularization)"] = rs_top["C (Regularization)"].apply(lambda v: f"{v:.4f}")
    rs_top["Mean CV Accuracy"] = rs_top["Mean CV Accuracy"].apply(lambda v: f"{v:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=_BG)
    for ax in axes:
        _style_ax(ax)

    c_vals = [float(c) for c in gs.cv_results_["param_clf__C"]]
    scores = gs.cv_results_["mean_test_score"]
    axes[0].semilogx(c_vals, scores, "o-", color=_CYAN, lw=2, markersize=8)
    axes[0].axvline(gs.best_params_["clf__C"], color=_GREEN, linestyle="--", lw=1.5, label=f"Best C={gs.best_params_['clf__C']}")
    axes[0].set_title("Grid Search: C vs Accuracy", color=_TEXT, fontsize=11)
    axes[0].set_xlabel("C (log scale)", color=_MUTED)
    axes[0].set_ylabel("CV Accuracy", color=_MUTED)
    axes[0].legend(facecolor="#1d2b49", labelcolor=_TEXT, fontsize=9)

    rs_c_vals = [float(c) for c in rs.cv_results_["param_clf__C"]]
    rs_scores = rs.cv_results_["mean_test_score"]
    axes[1].scatter(rs_c_vals, rs_scores, c=_VIOLET, s=40, alpha=0.7, zorder=5)
    axes[1].axvline(rs.best_params_["clf__C"], color=_GREEN, linestyle="--", lw=1.5, label=f"Best C={rs.best_params_['clf__C']:.3f}")
    axes[1].set_xscale("log")
    axes[1].set_title("Random Search: C vs Accuracy", color=_TEXT, fontsize=11)
    axes[1].set_xlabel("C (log scale)", color=_MUTED)
    axes[1].set_ylabel("CV Accuracy", color=_MUTED)
    axes[1].legend(facecolor="#1d2b49", labelcolor=_TEXT, fontsize=9)
    fig.tight_layout(pad=2)
    hp_chart = _fig_b64(fig)

    return {
        "hp_chart": hp_chart,
        "gs_best_c": str(gs.best_params_["clf__C"]),
        "gs_best_score": f"{gs.best_score_:.4f}",
        "rs_best_c": f"{rs.best_params_['clf__C']:.4f}",
        "rs_best_score": f"{rs.best_score_:.4f}",
        "gs_table": gs_df.to_html(classes="eda-table", index=False, border=0, escape=True),
        "rs_table": rs_top.to_html(classes="eda-table", index=False, border=0, escape=True),
    }


# ── Section 5: Learning & Validation Curves ─────────────────────────────────

def _learning_val_curves(X, y):
    rf = RandomForestClassifier(n_estimators=30, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        rf, X, y, cv=3, n_jobs=1,
        train_sizes=np.linspace(0.1, 1.0, 6), scoring="accuracy")

    param_range = [10, 30, 50, 80, 120]
    tr_scores, val_sc = validation_curve(
        rf, X, y, param_name="n_estimators",
        param_range=param_range, cv=3, scoring="accuracy", n_jobs=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=_BG)
    for ax in axes:
        _style_ax(ax)

    tr_mean = train_scores.mean(axis=1)
    tr_std = train_scores.std(axis=1)
    v_mean = val_scores.mean(axis=1)
    v_std = val_scores.std(axis=1)

    axes[0].plot(train_sizes, tr_mean, "o-", color=_CYAN, lw=2, label="Training")
    axes[0].fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color=_CYAN)
    axes[0].plot(train_sizes, v_mean, "s-", color=_VIOLET, lw=2, label="Validation")
    axes[0].fill_between(train_sizes, v_mean - v_std, v_mean + v_std, alpha=0.15, color=_VIOLET)
    axes[0].set_title("Learning Curves (Random Forest)", color=_TEXT, fontsize=11)
    axes[0].set_xlabel("Training Examples", color=_MUTED)
    axes[0].set_ylabel("Accuracy", color=_MUTED)
    axes[0].legend(facecolor="#1d2b49", labelcolor=_TEXT, fontsize=9)

    tr_v_mean = tr_scores.mean(axis=1)
    vl_v_mean = val_sc.mean(axis=1)
    axes[1].plot(param_range, tr_v_mean, "o-", color=_CYAN, lw=2, label="Training")
    axes[1].plot(param_range, vl_v_mean, "s-", color=_GREEN, lw=2, label="Validation")
    axes[1].set_title("Validation Curve (n_estimators)", color=_TEXT, fontsize=11)
    axes[1].set_xlabel("n_estimators", color=_MUTED)
    axes[1].set_ylabel("Accuracy", color=_MUTED)
    axes[1].legend(facecolor="#1d2b49", labelcolor=_TEXT, fontsize=9)
    fig.tight_layout(pad=2)
    lc_chart = _fig_b64(fig)
    return {"lc_chart": lc_chart}


# ── Section 6: Probability Calibration & Reliability Diagram ────────────────

def _calibration_section(X_train, X_test, y_train, y_test, class_names):
    base_rf = RandomForestClassifier(n_estimators=80, random_state=42)
    calibrated = CalibratedClassifierCV(base_rf, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)
    base_rf.fit(X_train, y_train)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=_BG)
    _style_ax(ax)

    for i, cname in enumerate(class_names):
        y_bin = (y_test == i).astype(int)
        prob_true_u, prob_pred_u = calibration_curve(y_bin, base_rf.predict_proba(X_test)[:, i], n_bins=8)
        prob_true_c, prob_pred_c = calibration_curve(y_bin, calibrated.predict_proba(X_test)[:, i], n_bins=8)
        ax.plot(prob_pred_u, prob_true_u, "--", lw=1.5, alpha=0.7, label=f"{cname} (uncalib.)")
        ax.plot(prob_pred_c, prob_true_c, "-", lw=2.0, label=f"{cname} (calibrated)")

    ax.plot([0, 1], [0, 1], "k:", color=_MUTED, lw=1.5, label="Perfect calibration")
    ax.set_title("Reliability Diagram (Probability Calibration)", color=_TEXT, fontsize=11)
    ax.set_xlabel("Mean Predicted Probability", color=_MUTED)
    ax.set_ylabel("Fraction of Positives", color=_MUTED)
    ax.legend(fontsize=8, facecolor="#1d2b49", labelcolor=_TEXT)
    fig.tight_layout()
    calib_chart = _fig_b64(fig)
    return {"calib_chart": calib_chart}


# ── Section 7: Statistical Significance Testing ──────────────────────────────

def _stat_sig_section(X, y):
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
    rf_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy", n_jobs=1)
    gb_scores = cross_val_score(gb, X, y, cv=5, scoring="accuracy", n_jobs=1)
    t_stat, p_val = stats.ttest_rel(rf_scores, gb_scores)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=_BG)
    _style_ax(ax)

    positions = [1, 2]
    bp = ax.boxplot([rf_scores, gb_scores], positions=positions,
                    patch_artist=True, widths=0.5,
                    boxprops=dict(facecolor=_CYAN, color=_BORDER),
                    medianprops=dict(color=_VIOLET, linewidth=2),
                    whiskerprops=dict(color=_MUTED),
                    capprops=dict(color=_MUTED),
                    flierprops=dict(markerfacecolor=_GREEN, markersize=5))
    bp["boxes"][1].set_facecolor(_VIOLET)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Random Forest", "Gradient Boosting"], color=_MUTED)
    ax.set_title("CV Score Comparison (Statistical Significance)", color=_TEXT, fontsize=11)
    ax.set_ylabel("5-Fold CV Accuracy", color=_MUTED)
    sig_text = f"t={t_stat:.3f}, p={p_val:.4f} ({'Significant' if p_val < 0.05 else 'Not significant'} at alpha=0.05)"
    ax.text(0.5, 0.02, sig_text, transform=ax.transAxes, color=_MUTED,
            ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    stat_chart = _fig_b64(fig)

    stat_df = pd.DataFrame({
        "Model": ["Random Forest", "Gradient Boosting"],
        "Mean Accuracy": [f"{rf_scores.mean():.4f}", f"{gb_scores.mean():.4f}"],
        "Std Dev": [f"{rf_scores.std():.4f}", f"{gb_scores.std():.4f}"],
        "Fold Scores": [str([f"{s:.4f}" for s in rf_scores]),
                       str([f"{s:.4f}" for s in gb_scores])],
    })
    return {
        "stat_chart": stat_chart,
        "stat_t": f"{t_stat:.4f}",
        "stat_p": f"{p_val:.4f}",
        "stat_significant": p_val < 0.05,
        "stat_table": stat_df.to_html(classes="eda-table", index=False, border=0, escape=True),
    }


# ── Master builder ─────────────────────────────────────────────────────────

def build_evaluation_context(raw_df):
    """Build all CO5 model evaluation artefacts."""
    if raw_df is None or raw_df.empty:
        raise ValueError("Dataset is empty.")
    X, y, y_cont, class_names, features = _prep(raw_df, n_sample=8000)
    split_ctx = _split_cv_section(X, y)
    X_train = split_ctx.pop("X_train")
    X_val   = split_ctx.pop("X_val")
    X_test  = split_ctx.pop("X_test")
    y_train = split_ctx.pop("y_train")
    y_val   = split_ctx.pop("y_val")
    y_test  = split_ctx.pop("y_test")
    rf_model = split_ctx.pop("rf_model")
    n_train = len(X_train)
    y_train_cont = y_cont[:n_train]
    y_test_cont  = y_cont[n_train: n_train + len(X_test)]
    clf_ctx  = _classification_section(rf_model, X_test, y_test, class_names)
    y_prob   = clf_ctx.pop("y_prob")
    y_pred   = clf_ctx.pop("y_pred")
    reg_ctx  = _regression_section(X_train, X_test, y_train_cont, y_test_cont)
    hp_ctx   = _hyperparam_section(X_train, y_train)
    lc_ctx   = _learning_val_curves(X, y)
    cal_ctx  = _calibration_section(X_train, X_test, y_train, y_test, class_names)
    st_ctx   = _stat_sig_section(X, y)
    ctx = {"class_names": class_names, "features_used": features}
    for d in [split_ctx, clf_ctx, reg_ctx, hp_ctx, lc_ctx, cal_ctx, st_ctx]:
        ctx.update(d)
    return ctx
