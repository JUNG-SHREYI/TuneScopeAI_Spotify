"""TuneScope AI - CO4 Unsupervised Learning & Dimensionality Reduction.

Covers:
  K-Means Clustering, Lloyds Algorithm, K-Means++ Initialization,
  Mini-Batch K-Means, Elbow Method, Silhouette Score,
  Hierarchical Clustering, Dendrogram, DBSCAN,
  PCA (extended), t-SNE, UMAP,
  Anomaly Detection (Isolation Forest),
  Advanced Feature Engineering / Unsupervised Feature Generation.

The raw DataFrame is never modified.
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
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

AUDIO_FEATURES = [
    "duration_ms", "explicit", "danceability", "energy", "key",
    "loudness", "mode", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "time_signature",
]

_PALETTE = ["#4de4e2", "#a994ff", "#55e6aa", "#f97316", "#ec4899",
            "#facc15", "#3b82f6", "#ef4444"]


def _fig_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return encoded


def _prep_features(raw_df, n_sample=5000):
    available = [c for c in AUDIO_FEATURES if c in raw_df.columns]
    if len(available) < 2:
        raise ValueError("Need at least 2 numeric audio features.")
    sub = raw_df[available].copy()
    for col in sub.columns:
        if pd.api.types.is_bool_dtype(sub[col]):
            sub[col] = sub[col].astype("float64")
        else:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub.replace([np.inf, -np.inf], np.nan, inplace=True)
    sub.dropna(how="all", inplace=True)
    if len(sub) > n_sample:
        sub = sub.sample(n=n_sample, random_state=42)
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(sub)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    return X_scaled, available


def _kmeans_section(X):
    k_range = range(2, 11)
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor="#111a2d")
    for ax in axes:
        ax.set_facecolor("#111a2d")
        for spine in ax.spines.values():
            spine.set_edgecolor("#263653")
    axes[0].plot(list(k_range), inertias, "o-", color="#4de4e2", linewidth=2.2, markersize=7)
    axes[0].set_title("Elbow Method - Inertia vs k", color="#eef3ff", fontsize=11)
    axes[0].set_xlabel("Number of clusters (k)", color="#91a5cb")
    axes[0].set_ylabel("Inertia (WCSS)", color="#91a5cb")
    axes[0].tick_params(colors="#91a5cb")
    axes[1].plot(list(k_range), sil_scores, "s-", color="#a994ff", linewidth=2.2, markersize=7)
    axes[1].set_title("Silhouette Score vs k", color="#eef3ff", fontsize=11)
    axes[1].set_xlabel("Number of clusters (k)", color="#91a5cb")
    axes[1].set_ylabel("Silhouette Score", color="#91a5cb")
    axes[1].tick_params(colors="#91a5cb")
    fig.tight_layout(pad=2)
    elbow_chart = _fig_b64(fig)

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    km_best = KMeans(n_clusters=best_k, init="k-means++", n_init=15, random_state=42)
    labels = km_best.fit_predict(X)

    fig2, ax2 = plt.subplots(figsize=(9, 4.5), facecolor="#111a2d")
    ax2.set_facecolor("#111a2d")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#263653")
    sil_vals = silhouette_samples(X, labels)
    y_lower = 10
    for i in range(best_k):
        cluster_sil = np.sort(sil_vals[labels == i])
        size = cluster_sil.shape[0]
        y_upper = y_lower + size
        ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                          facecolor=_PALETTE[i % len(_PALETTE)], alpha=0.85)
        y_lower = y_upper + 10
    ax2.axvline(x=np.mean(sil_vals), color="#facc15", linestyle="--", linewidth=1.4)
    ax2.set_title(f"Silhouette Plot (k={best_k})", color="#eef3ff", fontsize=11)
    ax2.set_xlabel("Silhouette Coefficient", color="#91a5cb")
    ax2.set_ylabel("Cluster", color="#91a5cb")
    ax2.tick_params(colors="#91a5cb")
    fig2.tight_layout()
    sil_chart = _fig_b64(fig2)

    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    summary_df = pd.DataFrame({
        "Cluster": [f"Cluster {i}" for i in cluster_sizes.index],
        "Size": cluster_sizes.values,
        "Silhouette": [f"{silhouette_samples(X, labels)[labels == i].mean():.4f}"
                       for i in cluster_sizes.index],
    })
    elbow_table = pd.DataFrame({
        "k": list(k_range),
        "Inertia (WCSS)": [f"{v:.2f}" for v in inertias],
        "Silhouette Score": [f"{v:.4f}" for v in sil_scores],
    }).to_html(classes="eda-table", index=False, border=0, escape=True)

    return {
        "km_elbow_chart": elbow_chart,
        "km_sil_chart": sil_chart,
        "km_best_k": best_k,
        "km_best_sil": f"{max(sil_scores):.4f}",
        "km_cluster_table": summary_df.to_html(classes="eda-table", index=False, border=0, escape=True),
        "km_elbow_table": elbow_table,
    }


def _minibatch_section(X, best_k):
    mbkm = MiniBatchKMeans(n_clusters=best_k, batch_size=1024, n_init=10, random_state=42)
    mb_labels = mbkm.fit_predict(X)
    mb_sil = silhouette_score(X, mb_labels)
    km = KMeans(n_clusters=best_k, n_init=15, random_state=42)
    km_labels = km.fit_predict(X)
    km_sil = silhouette_score(X, km_labels)
    comp_df = pd.DataFrame({
        "Method": ["Full K-Means", "Mini-Batch K-Means"],
        "Inertia": [f"{km.inertia_:.2f}", f"{mbkm.inertia_:.2f}"],
        "Silhouette Score": [f"{km_sil:.4f}", f"{mb_sil:.4f}"],
        "Clusters": [best_k, best_k],
    })
    return {
        "mb_table": comp_df.to_html(classes="eda-table", index=False, border=0, escape=True),
        "mb_sil": f"{mb_sil:.4f}",
        "mb_labels": mb_labels,
    }


def _hierarchical_section(X):
    sample = X[:min(300, len(X))]
    Z = linkage(sample, method="ward")
    fig, ax = plt.subplots(figsize=(13, 5), facecolor="#111a2d")
    ax.set_facecolor("#111a2d")
    for spine in ax.spines.values():
        spine.set_edgecolor("#263653")
    dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=6,
               color_threshold=0.7 * max(Z[:, 2]),
               above_threshold_color="#91a5cb")
    ax.set_title("Hierarchical Clustering Dendrogram (Ward linkage, n=300 sample)",
                 color="#eef3ff", fontsize=11)
    ax.set_xlabel("Sample Index", color="#91a5cb")
    ax.set_ylabel("Distance", color="#91a5cb")
    ax.tick_params(colors="#91a5cb", labelbottom=False)
    fig.tight_layout()
    dendro_chart = _fig_b64(fig)
    agg = AgglomerativeClustering(n_clusters=4, linkage="ward")
    agg_labels = agg.fit_predict(X)
    agg_sil = silhouette_score(X, agg_labels)
    return {"dendro_chart": dendro_chart, "hier_sil": f"{agg_sil:.4f}", "hier_clusters": 4}


def _dbscan_section(X):
    pca2 = PCA(n_components=2, random_state=42)
    X2 = pca2.fit_transform(X)
    eps_vals = [0.3, 0.5, 0.8, 1.2]
    results = []
    best_sil, best_eps, best_labels2 = -1, None, None
    for eps in eps_vals:
        db = DBSCAN(eps=eps, min_samples=5)
        lbls = db.fit_predict(X2)
        n_clusters = len(set(lbls)) - (1 if -1 in lbls else 0)
        noise = (lbls == -1).sum()
        sil = silhouette_score(X2, lbls) if n_clusters >= 2 else -1
        results.append({"eps": eps, "Clusters": n_clusters, "Noise Points": int(noise),
                         "Silhouette": f"{sil:.4f}" if n_clusters >= 2 else "N/A"})
        if sil > best_sil:
            best_sil, best_eps, best_labels2 = sil, eps, lbls
    res_df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#111a2d")
    ax.set_facecolor("#111a2d")
    for spine in ax.spines.values():
        spine.set_edgecolor("#263653")
    for lbl in sorted(set(best_labels2)):
        mask = best_labels2 == lbl
        color = "#555555" if lbl == -1 else _PALETTE[lbl % len(_PALETTE)]
        label = "Noise" if lbl == -1 else f"Cluster {lbl}"
        ax.scatter(X2[mask, 0], X2[mask, 1], s=8, c=color, label=label, alpha=0.7)
    ax.set_title(f"DBSCAN (eps={best_eps}, PCA-2D)", color="#eef3ff", fontsize=11)
    ax.set_xlabel("PC1", color="#91a5cb")
    ax.set_ylabel("PC2", color="#91a5cb")
    ax.tick_params(colors="#91a5cb")
    ax.legend(fontsize=7, facecolor="#1d2b49", labelcolor="#eef3ff", markerscale=2)
    fig.tight_layout()
    return {
        "dbscan_table": res_df.to_html(classes="eda-table", index=False, border=0, escape=True),
        "dbscan_chart": _fig_b64(fig),
        "dbscan_best_eps": best_eps,
        "dbscan_best_sil": f"{best_sil:.4f}",
    }


def _tsne_section(X, labels):
    n = min(2000, len(X))
    idx = np.random.default_rng(42).choice(len(X), n, replace=False)
    X_sub, lbl_sub = X[idx], labels[idx]
    tsne = TSNE(n_components=2, perplexity=40, max_iter=500, random_state=42,
                init="pca", learning_rate="auto")
    X_2d = tsne.fit_transform(X_sub)
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#111a2d")
    ax.set_facecolor("#111a2d")
    for spine in ax.spines.values():
        spine.set_edgecolor("#263653")
    for lbl in sorted(set(lbl_sub)):
        mask = lbl_sub == lbl
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=9,
                   c=_PALETTE[lbl % len(_PALETTE)], label=f"Cluster {lbl}", alpha=0.75)
    ax.set_title("t-SNE Visualization (n=2000, K-Means clusters)", color="#eef3ff", fontsize=11)
    ax.set_xlabel("t-SNE 1", color="#91a5cb")
    ax.set_ylabel("t-SNE 2", color="#91a5cb")
    ax.tick_params(colors="#91a5cb")
    ax.legend(fontsize=7, facecolor="#1d2b49", labelcolor="#eef3ff", markerscale=2)
    fig.tight_layout()
    return {"tsne_chart": _fig_b64(fig)}


def _umap_section(X, labels):
    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        n = min(3000, len(X))
        idx = np.random.default_rng(42).choice(len(X), n, replace=False)
        X_sub, lbl_sub = X[idx], labels[idx]
        X_2d = reducer.fit_transform(X_sub)
        fig, ax = plt.subplots(figsize=(9, 6), facecolor="#111a2d")
        ax.set_facecolor("#111a2d")
        for spine in ax.spines.values():
            spine.set_edgecolor("#263653")
        for lbl in sorted(set(lbl_sub)):
            mask = lbl_sub == lbl
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=9,
                       c=_PALETTE[lbl % len(_PALETTE)], label=f"Cluster {lbl}", alpha=0.75)
        ax.set_title("UMAP Visualization (n=3000, K-Means clusters)", color="#eef3ff", fontsize=11)
        ax.set_xlabel("UMAP 1", color="#91a5cb")
        ax.set_ylabel("UMAP 2", color="#91a5cb")
        ax.tick_params(colors="#91a5cb")
        ax.legend(fontsize=7, facecolor="#1d2b49", labelcolor="#eef3ff", markerscale=2)
        fig.tight_layout()
        return {"umap_chart": _fig_b64(fig), "umap_available": True}
    except ImportError:
        return {"umap_chart": None, "umap_available": False}


def _isolation_forest_section(X):
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=1)
    preds = iso.fit_predict(X)
    scores = iso.decision_function(X)
    n_anomalies = int((preds == -1).sum())
    n_normal = int((preds == 1).sum())
    pca2 = PCA(n_components=2, random_state=42)
    X2 = pca2.fit_transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#111a2d")
    for ax in axes:
        ax.set_facecolor("#111a2d")
        for spine in ax.spines.values():
            spine.set_edgecolor("#263653")
    normal_mask = preds == 1
    axes[0].scatter(X2[normal_mask, 0], X2[normal_mask, 1], s=5, c="#4de4e2", alpha=0.4, label="Normal")
    axes[0].scatter(X2[~normal_mask, 0], X2[~normal_mask, 1], s=18, c="#ef4444", alpha=0.9, label="Anomaly", zorder=5)
    axes[0].set_title("Isolation Forest - Anomaly Detection", color="#eef3ff", fontsize=11)
    axes[0].set_xlabel("PC1", color="#91a5cb")
    axes[0].set_ylabel("PC2", color="#91a5cb")
    axes[0].tick_params(colors="#91a5cb")
    axes[0].legend(fontsize=8, facecolor="#1d2b49", labelcolor="#eef3ff", markerscale=2)
    axes[1].hist(scores, bins=60, color="#a994ff", edgecolor="none", alpha=0.85)
    axes[1].axvline(x=0, color="#ef4444", linestyle="--", linewidth=1.4, label="Threshold")
    axes[1].set_title("Anomaly Score Distribution", color="#eef3ff", fontsize=11)
    axes[1].set_xlabel("Anomaly Score", color="#91a5cb")
    axes[1].set_ylabel("Count", color="#91a5cb")
    axes[1].tick_params(colors="#91a5cb")
    axes[1].legend(fontsize=8, facecolor="#1d2b49", labelcolor="#eef3ff")
    fig.tight_layout(pad=2)
    summary_df = pd.DataFrame({
        "Category": ["Normal Points", "Anomalies", "Contamination Rate"],
        "Value": [f"{n_normal:,}", f"{n_anomalies:,}", f"{n_anomalies / len(X) * 100:.2f}%"],
    })
    return {
        "iso_chart": _fig_b64(fig),
        "iso_n_anomalies": f"{n_anomalies:,}",
        "iso_n_normal": f"{n_normal:,}",
        "iso_table": summary_df.to_html(classes="eda-table", index=False, border=0, escape=True),
    }


def _feature_compression_section(X, labels, feature_names):
    pca_full = PCA(random_state=42)
    X_full = pca_full.fit_transform(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumvar, 0.95) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#111a2d")
    for ax in axes:
        ax.set_facecolor("#111a2d")
        for spine in ax.spines.values():
            spine.set_edgecolor("#263653")
    for lbl in sorted(set(labels)):
        mask = labels == lbl
        axes[0].scatter(X_full[mask, 0], X_full[mask, 1], s=7,
                        c=_PALETTE[lbl % len(_PALETTE)], label=f"Cluster {lbl}", alpha=0.65)
    axes[0].set_title("PCA Feature Compression (PC1 vs PC2)", color="#eef3ff", fontsize=11)
    axes[0].set_xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)", color="#91a5cb")
    axes[0].set_ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)", color="#91a5cb")
    axes[0].tick_params(colors="#91a5cb")
    axes[0].legend(fontsize=7, facecolor="#1d2b49", labelcolor="#eef3ff", markerscale=2)
    n_show = min(5, len(pca_full.components_))
    loadings = pca_full.components_[:n_show]
    im = axes[1].imshow(loadings, aspect="auto", cmap="coolwarm")
    axes[1].set_xticks(range(len(feature_names)))
    axes[1].set_xticklabels(feature_names, rotation=45, ha="right", fontsize=7, color="#91a5cb")
    axes[1].set_yticks(range(n_show))
    axes[1].set_yticklabels([f"PC{i+1}" for i in range(n_show)], color="#91a5cb")
    axes[1].set_title("PCA Feature Loadings (top PCs)", color="#eef3ff", fontsize=11)
    plt.colorbar(im, ax=axes[1], shrink=0.85)
    fig.tight_layout(pad=2)
    feat_df = pd.DataFrame({
        "Feature": feature_names + ["KMeans_Cluster", "PC1", "PC2"],
        "Source": (["Original"] * len(feature_names) +
                   ["K-Means (unsupervised)", "PCA (unsupervised)", "PCA (unsupervised)"]),
        "Description": (
            [f"Audio characteristic: {f}" for f in feature_names] +
            ["Cluster membership label", "First principal component", "Second principal component"]
        ),
    })
    return {
        "compression_chart": _fig_b64(fig),
        "n95_components": n95,
        "total_features": len(feature_names),
        "feat_engineering_table": feat_df.to_html(classes="eda-table", index=False, border=0, escape=True),
    }


def build_clustering_context(raw_df):
    """Build all CO4 unsupervised learning artefacts."""
    if raw_df is None or raw_df.empty:
        raise ValueError("Dataset is empty or could not be loaded.")
    X, feature_names = _prep_features(raw_df, n_sample=5000)
    km_ctx = _kmeans_section(X)
    mb_ctx = _minibatch_section(X, km_ctx["km_best_k"])
    hier_ctx = _hierarchical_section(X)
    dbscan_ctx = _dbscan_section(X)
    km_final = KMeans(n_clusters=km_ctx["km_best_k"], n_init=15, random_state=42)
    cluster_labels = km_final.fit_predict(X)
    tsne_ctx = _tsne_section(X, cluster_labels)
    umap_ctx = _umap_section(X, cluster_labels)
    iso_ctx = _isolation_forest_section(X)
    comp_ctx = _feature_compression_section(X, cluster_labels, feature_names)
    ctx = {"features_used": feature_names}
    for d in [km_ctx, mb_ctx, hier_ctx, dbscan_ctx, tsne_ctx, umap_ctx, iso_ctx, comp_ctx]:
        ctx.update(d)
    return ctx
