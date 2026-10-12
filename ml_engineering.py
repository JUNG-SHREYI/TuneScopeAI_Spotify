"""TuneScope AI - CO6 ML Engineering and Deployment.

Covers:
  ML Engineering Fundamentals, Notebook-to-Production Transition,
  Retraining Strategies, Monitoring & Logging, Model Packaging (joblib/ONNX),
  FastAPI / REST API concepts, Real-Time Prediction Services,
  Feature Stores, Training-Serving Consistency, Point-in-Time Correctness,
  MLflow (simulated), DVC, Data & Model Versioning,
  Drift Detection, Model Performance Monitoring, Model Observability,
  End-to-End ML Pipeline, Batch Scoring, CI/CD for ML,
  Model Retraining Pipeline.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pickle
import time
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

AUDIO_FEATURES = [
    "duration_ms", "explicit", "danceability", "energy", "key",
    "loudness", "mode", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "time_signature",
]
POPULARITY_BINS  = [-0.001, 40, 70, 100.001]
POPULARITY_LABELS = ["Low", "Moderate", "High"]

_BG    = "#111a2d"
_TEXT  = "#eef3ff"
_MUTED = "#91a5cb"
_BORDER= "#263653"
_CYAN  = "#4de4e2"
_VIOLET= "#a994ff"
_GREEN = "#55e6aa"
_RED   = "#ef4444"

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_STORE = PROJECT_DIR / "model_store"
MODEL_STORE.mkdir(exist_ok=True)


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


def _prep(raw_df, n_sample=6000):
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
        y_cat = pd.cut(pop, bins=POPULARITY_BINS, labels=POPULARITY_LABELS).astype(str).values
        le = LabelEncoder()
        y = le.fit_transform(y_cat)
    else:
        y = np.zeros(len(X), dtype=int)
    return X, y, available


# ── Section 1: Model Packaging (joblib) ─────────────────────────────────────

def _packaging_section(X_train, y_train, X_test, y_test, feature_names):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
    ])
    raw_X_train = X_train
    pipeline.steps[0][1].fit(raw_X_train)
    pipeline.steps[1][1].fit(raw_X_train)
    pipeline.steps[2][1].fit(X_train, y_train)

    model_path = MODEL_STORE / "tunescope_rf_pipeline.joblib"
    t0 = time.perf_counter()
    joblib.dump(pipeline, model_path, compress=3)
    save_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    loaded_pipeline = joblib.load(model_path)
    load_time = time.perf_counter() - t1

    y_pred = pipeline.steps[2][1].predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_size = model_path.stat().st_size / 1024

    meta = {
        "model_name": "TuneScope RF Classifier",
        "version": "1.0.0",
        "features": feature_names,
        "n_classes": 3,
        "accuracy_test": round(acc, 4),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "framework": "scikit-learn",
        "packaging": "joblib",
    }
    meta_path = MODEL_STORE / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    pkg_df = pd.DataFrame({
        "Property": ["Model Path", "Model Size (KB)", "Save Time (ms)", "Load Time (ms)", "Test Accuracy", "Metadata"],
        "Value": [str(model_path.name), f"{model_size:.1f}", f"{save_time*1000:.1f}",
                  f"{load_time*1000:.1f}", f"{acc:.4f}", str(meta_path.name)],
    })
    return {
        "pkg_table": pkg_df.to_html(classes="eda-table", index=False, border=0, escape=True),
        "pkg_accuracy": f"{acc:.4f}",
        "pkg_size": f"{model_size:.1f}",
        "model_path": str(model_path),
        "pipeline": pipeline,
        "meta": meta,
    }


# ── Section 2: Simulated MLflow Experiment Tracking ─────────────────────────

def _mlflow_sim_section(X_train, y_train, X_test, y_test):
    experiments = []
    configs = [
        {"name": "RF_baseline", "model": "RandomForest", "n_estimators": 50, "max_depth": None},
        {"name": "RF_tuned",    "model": "RandomForest", "n_estimators": 150, "max_depth": 8},
        {"name": "GB_v1",       "model": "GradientBoosting", "n_estimators": 100, "max_depth": 4},
        {"name": "GB_v2",       "model": "GradientBoosting", "n_estimators": 200, "max_depth": 6},
    ]
    for cfg in configs:
        t0 = time.perf_counter()
        if cfg["model"] == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                random_state=42, n_jobs=1)
        else:
            model = GradientBoostingClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                random_state=42)
        model.fit(X_train, y_train)
        duration = time.perf_counter() - t0
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        run_id = hashlib.md5(cfg["name"].encode()).hexdigest()[:8]
        experiments.append({
            "Run ID": run_id,
            "Experiment": cfg["name"],
            "Model": cfg["model"],
            "n_estimators": cfg["n_estimators"],
            "max_depth": str(cfg["max_depth"]),
            "Accuracy": f"{acc:.4f}",
            "F1 (weighted)": f"{f1:.4f}",
            "Train Time (s)": f"{duration:.2f}",
            "Status": "FINISHED",
        })

    exp_df = pd.DataFrame(experiments)
    best_run = exp_df.loc[exp_df["Accuracy"].astype(float).idxmax()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=_BG)
    for ax in axes:
        _style_ax(ax)

    colors = [_CYAN, _VIOLET, _GREEN, "#f97316"]
    names = exp_df["Experiment"].tolist()
    accs = [float(a) for a in exp_df["Accuracy"]]
    f1s  = [float(f) for f in exp_df["F1 (weighted)"]]

    x = np.arange(len(names))
    axes[0].bar(x, accs, color=colors, alpha=0.85, width=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, color=_MUTED, fontsize=8)
    axes[0].set_title("MLflow Experiments – Accuracy", color=_TEXT, fontsize=11)
    axes[0].set_ylabel("Accuracy", color=_MUTED)
    for xi, yi in zip(x, accs):
        axes[0].text(xi, yi + 0.002, f"{yi:.3f}", ha="center", color=_TEXT, fontsize=8)

    axes[1].bar(x, f1s, color=colors, alpha=0.85, width=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15, color=_MUTED, fontsize=8)
    axes[1].set_title("MLflow Experiments – F1 Score", color=_TEXT, fontsize=11)
    axes[1].set_ylabel("F1 (weighted)", color=_MUTED)
    for xi, yi in zip(x, f1s):
        axes[1].text(xi, yi + 0.002, f"{yi:.3f}", ha="center", color=_TEXT, fontsize=8)

    fig.tight_layout(pad=2)
    mlflow_chart = _fig_b64(fig)

    return {
        "mlflow_table": exp_df.to_html(classes="eda-table", index=False, border=0, escape=True),
        "mlflow_chart": mlflow_chart,
        "mlflow_best_run": best_run["Experiment"],
        "mlflow_best_acc": best_run["Accuracy"],
    }


# ── Section 3: Drift Detection & Monitoring ──────────────────────────────────

def _drift_section(X):
    rng = np.random.default_rng(42)
    n = min(2000, len(X))
    X_ref = X[:n]
    drift_factor = 0.4
    X_prod = X_ref + rng.normal(0, drift_factor, X_ref.shape)

    from scipy.stats import ks_2samp
    feat_count = min(6, X_ref.shape[1])
    drift_results = []
    for i in range(feat_count):
        stat, pval = ks_2samp(X_ref[:, i], X_prod[:, i])
        drift_results.append({
            "Feature Index": f"F{i+1}",
            "KS Statistic": f"{stat:.4f}",
            "p-value": f"{pval:.4f}",
            "Drift Detected": "YES" if pval < 0.05 else "NO",
        })
    drift_df = pd.DataFrame(drift_results)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=_BG)
    for ax in axes:
        _style_ax(ax)

    feat_idx = 0
    axes[0].hist(X_ref[:, feat_idx], bins=40, color=_CYAN, alpha=0.65, label="Reference (Training)", density=True)
    axes[0].hist(X_prod[:, feat_idx], bins=40, color=_RED, alpha=0.65, label="Production", density=True)
    axes[0].set_title(f"Feature Distribution Drift (F{feat_idx+1})", color=_TEXT, fontsize=11)
    axes[0].set_xlabel("Feature Value", color=_MUTED)
    axes[0].set_ylabel("Density", color=_MUTED)
    axes[0].legend(fontsize=9, facecolor="#1d2b49", labelcolor=_TEXT)

    ks_stats = [float(r["KS Statistic"]) for r in drift_results]
    x = np.arange(len(ks_stats))
    bar_colors = [_RED if float(r["p-value"]) < 0.05 else _GREEN for r in drift_results]
    axes[1].bar(x, ks_stats, color=bar_colors, alpha=0.85)
    axes[1].axhline(0.1, color=_VIOLET, linestyle="--", lw=1.5, label="Drift threshold (0.1)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"F{i+1}" for i in range(feat_count)], color=_MUTED)
    axes[1].set_title("KS Statistic per Feature (Red=Drift)", color=_TEXT, fontsize=11)
    axes[1].set_ylabel("KS Statistic", color=_MUTED)
    axes[1].legend(fontsize=9, facecolor="#1d2b49", labelcolor=_TEXT)
    fig.tight_layout(pad=2)
    drift_chart = _fig_b64(fig)

    n_drifted = sum(1 for r in drift_results if r["Drift Detected"] == "YES")
    return {
        "drift_chart": drift_chart,
        "drift_table": drift_df.to_html(classes="eda-table", index=False, border=0, escape=True),
        "drift_n_features": feat_count,
        "drift_n_drifted": n_drifted,
    }


# ── Section 4: Batch Scoring & End-to-End Pipeline ──────────────────────────

def _pipeline_section(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    steps = [
        {"Step": "Data Loading", "Component": "data_loader.py", "Status": "DONE", "Output": f"{len(X)} records"},
        {"Step": "Preprocessing", "Component": "SimpleImputer + StandardScaler", "Status": "DONE", "Output": f"{len(feature_names)} features"},
        {"Step": "Train/Test Split", "Component": "train_test_split(80/20)", "Status": "DONE", "Output": f"Train={len(X_train)}, Test={len(X_test)}"},
        {"Step": "Model Training", "Component": "RandomForestClassifier", "Status": "DONE", "Output": "100 trees"},
        {"Step": "Model Evaluation", "Component": "accuracy_score, f1_score", "Status": "DONE", "Output": "See metrics"},
        {"Step": "Model Packaging", "Component": "joblib.dump()", "Status": "DONE", "Output": "model_store/"},
        {"Step": "Batch Scoring", "Component": "pipeline.predict(X_batch)", "Status": "DONE", "Output": f"{len(X_test)} predictions"},
        {"Step": "Drift Monitoring", "Component": "KS-test", "Status": "DONE", "Output": "Feature drift report"},
    ]
    pipeline_df = pd.DataFrame(steps)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)

    batch_size = 500
    batch_X = X_test[:batch_size]
    t0 = time.perf_counter()
    batch_preds = rf.predict(batch_X)
    batch_time = time.perf_counter() - t0

    batch_df = pd.DataFrame({
        "Metric": ["Batch Size", "Predictions/sec", "Batch Latency (ms)", "Unique Classes Predicted"],
        "Value": [
            str(batch_size),
            f"{batch_size / batch_time:.0f}",
            f"{batch_time * 1000:.2f}",
            str(len(set(batch_preds))),
        ],
    })

    cicd_steps = [
        {"Stage": "1. Code Commit", "Action": "git push origin main", "Tool": "Git/GitHub", "Status": "TRIGGERED"},
        {"Stage": "2. Unit Tests", "Action": "pytest tests/", "Tool": "pytest", "Status": "PASS"},
        {"Stage": "3. Data Validation", "Action": "Great Expectations checks", "Tool": "GE", "Status": "PASS"},
        {"Stage": "4. Model Training", "Action": "python train.py --config cfg.yaml", "Tool": "sklearn", "Status": "DONE"},
        {"Stage": "5. Model Evaluation", "Action": "Threshold: accuracy > 0.70", "Tool": "mlflow", "Status": "PASS"},
        {"Stage": "6. Model Registry", "Action": "mlflow.register_model()", "Tool": "MLflow", "Status": "REGISTERED"},
        {"Stage": "7. Staging Deploy", "Action": "docker build & push", "Tool": "Docker", "Status": "DEPLOYED"},
        {"Stage": "8. Integration Test", "Action": "curl /predict endpoint", "Tool": "FastAPI", "Status": "PASS"},
        {"Stage": "9. Production Promote", "Action": "kubectl rollout update", "Tool": "K8s", "Status": "LIVE"},
    ]
    cicd_df = pd.DataFrame(cicd_steps)

    retrain_strategies = [
        {"Strategy": "Scheduled Retraining", "Trigger": "Daily/Weekly cron job", "Pros": "Predictable", "Cons": "May retrain when not needed"},
        {"Strategy": "Drift-Based Retraining", "Trigger": "KS-stat > threshold", "Pros": "Responsive to data shift", "Cons": "Requires monitoring infra"},
        {"Strategy": "Performance-Based", "Trigger": "Accuracy drop > 5%", "Pros": "Direct business impact", "Cons": "Needs ground truth labels"},
        {"Strategy": "Hybrid", "Trigger": "Drift OR schedule", "Pros": "Best of both", "Cons": "More complex logic"},
    ]
    retrain_df = pd.DataFrame(retrain_strategies)

    return {
        "pipeline_table": pipeline_df.to_html(classes="eda-table", index=False, border=0, escape=True),
        "batch_table": batch_df.to_html(classes="eda-table", index=False, border=0, escape=True),
        "cicd_table": cicd_df.to_html(classes="eda-table", index=False, border=0, escape=True),
        "retrain_table": retrain_df.to_html(classes="eda-table", index=False, border=0, escape=True),
        "batch_latency": f"{batch_time*1000:.2f}",
        "batch_throughput": f"{batch_size/batch_time:.0f}",
    }


# ── Section 5: FastAPI REST API Blueprint ─────────────────────────────────

def _api_blueprint_section():
    api_code = '''# FastAPI Prediction Service – TuneScope AI
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np

app = FastAPI(title="TuneScope AI Prediction Service")
model = joblib.load("model_store/tunescope_rf_pipeline.joblib")

class TrackFeatures(BaseModel):
    duration_ms: float; danceability: float; energy: float
    loudness: float; speechiness: float; acousticness: float
    instrumentalness: float; liveness: float; valence: float; tempo: float

@app.post("/predict")
def predict(features: TrackFeatures):
    X = np.array([[features.duration_ms, features.danceability,
                   features.energy, features.loudness, features.speechiness,
                   features.acousticness, features.instrumentalness,
                   features.liveness, features.valence, features.tempo]])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()
    return {"prediction": pred, "confidence": round(float(prob), 4)}

@app.get("/health")
def health(): return {"status": "healthy"}'''

    endpoints = [
        {"Endpoint": "GET /", "Method": "GET", "Description": "API info & version", "Auth": "None"},
        {"Endpoint": "GET /health", "Method": "GET", "Description": "Health check", "Auth": "None"},
        {"Endpoint": "POST /predict", "Method": "POST", "Description": "Single track prediction", "Auth": "API Key"},
        {"Endpoint": "POST /batch-predict", "Method": "POST", "Description": "Batch scoring (up to 1000)", "Auth": "API Key"},
        {"Endpoint": "GET /model/info", "Method": "GET", "Description": "Current model metadata", "Auth": "API Key"},
        {"Endpoint": "GET /metrics", "Method": "GET", "Description": "Model performance metrics", "Auth": "Admin"},
    ]
    feat_store_rows = [
        {"Feature Group": "audio_features", "Features": "danceability, energy, loudness, tempo, valence", "Freshness": "Daily"},
        {"Feature Group": "popularity_signals", "Features": "streams_7d, playlist_adds, chart_rank", "Freshness": "Hourly"},
        {"Feature Group": "artist_stats", "Features": "follower_count, avg_popularity, genre", "Freshness": "Weekly"},
        {"Feature Group": "track_metadata", "Features": "duration_ms, explicit, release_year, key", "Freshness": "Static"},
    ]
    return {
        "api_code": api_code,
        "api_endpoints_table": pd.DataFrame(endpoints).to_html(classes="eda-table", index=False, border=0, escape=True),
        "feat_store_table": pd.DataFrame(feat_store_rows).to_html(classes="eda-table", index=False, border=0, escape=True),
    }


# ── Master builder ──────────────────────────────────────────────────────────

def build_ml_engineering_context(raw_df):
    """Build all CO6 ML Engineering artefacts."""
    if raw_df is None or raw_df.empty:
        raise ValueError("Dataset is empty.")
    X, y, feature_names = _prep(raw_df, n_sample=6000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pkg_ctx  = _packaging_section(X_train, y_train, X_test, y_test, feature_names)
    pipeline = pkg_ctx.pop("pipeline")
    meta     = pkg_ctx.pop("meta")
    mlf_ctx  = _mlflow_sim_section(X_train, y_train, X_test, y_test)
    drft_ctx = _drift_section(X)
    pipe_ctx = _pipeline_section(X, y, feature_names)
    api_ctx  = _api_blueprint_section()
    ctx = {"features_used": feature_names, "model_meta": meta}
    for d in [pkg_ctx, mlf_ctx, drft_ctx, pipe_ctx, api_ctx]:
        ctx.update(d)
    return ctx
