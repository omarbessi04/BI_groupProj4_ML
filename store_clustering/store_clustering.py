"""
Store Clustering - Hópverkefni 4 (Gagnanám)
============================================
Connects to the PostgreSQL data warehouse from Hópverkefni 3 and clusters
stores based on sales behaviour, basket patterns, product mix, and inventory.

Requirements (pip install):
    pip install psycopg2-binary pandas scikit-learn matplotlib seaborn

Environment variables:
    DB_NAME, DB_USERNAME, DB_PASSWORD

Usage:
    1. Set env vars (or use a .env file).
    2. Run:  python store_clustering.py
    3. Outputs are saved to a timestamped folder.
"""

import warnings

warnings.filterwarnings("ignore")

import os
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import psycopg2
from dotenv import load_dotenv

# ── Configuration ────────────────────────────────────────────────────────────
load_dotenv()
DB_CONFIG = {
    "host": "vgbi2026-vh.duckdns.org",
    "port": 35053,
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USERNAME"),
    "password": os.getenv("DB_PASSWORD"),
}

MAX_K = 10  # max clusters to try in elbow / silhouette search
# ─────────────────────────────────────────────────────────────────────────────

# Output directory
dir_name = str(datetime.now()).strip().replace(" ", "_").replace(":", "-")
os.makedirs(dir_name)
print(f"[INFO] Output directory: {dir_name}/")


def fetch_data(query: str) -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return df


# ============================================================================
# 1. BUILD STORE FEATURE MATRIX
# ============================================================================


def build_features() -> pd.DataFrame:
    """
    One row per store with features derived from factSales, factInventory,
    dimStore, and dimProduct.
    """

    # --- Sales-based features ---
    sales_query = """
        SELECT
            ds.id                                       AS store_id,
            ds.storename,
            ds.city,
            ds.country,
            ds.location,
            COUNT(*)                                    AS total_rows,
            SUM(fs.unitssold)                           AS total_units,
            COUNT(DISTINCT fs.receipt)                   AS total_receipts,
            SUM(fs.unitssold * dp.priceeur)             AS total_revenue,
            SUM(fs.unitssold * dp.costeur)              AS total_cost,
            SUM(fs.unitssold * (dp.priceeur - dp.costeur)) AS total_profit,
            AVG(fs.unitssold)                           AS avg_units_per_line,
            COUNT(DISTINCT dp.category)                 AS n_categories_sold,
            COUNT(DISTINCT dp.id)                       AS n_products_sold,
            COUNT(DISTINCT dd.fulldate)                 AS active_days
        FROM public.factsales   fs
        JOIN public.dimstore    ds ON ds.id = fs.idstore
        JOIN public.dimproduct  dp ON dp.id = fs.idproduct
        JOIN public.dimdate     dd ON dd.id = fs.idcalendar
        WHERE dp.ispricemissing = FALSE
          AND dp.iscostmissing  = FALSE
        GROUP BY ds.id, ds.storename, ds.city, ds.country, ds.location;
    """

    df_sales = fetch_data(sales_query)
    print(f"[INFO] Sales features loaded for {len(df_sales)} stores")

    # Derived metrics
    df_sales["avg_basket_size"] = df_sales["total_units"] / df_sales["total_receipts"]
    df_sales["avg_basket_value"] = (
        df_sales["total_revenue"] / df_sales["total_receipts"]
    )
    df_sales["profit_margin"] = (
        df_sales["total_profit"] / df_sales["total_revenue"] * 100
    )
    df_sales["revenue_per_day"] = df_sales["total_revenue"] / df_sales["active_days"]
    df_sales["units_per_day"] = df_sales["total_units"] / df_sales["active_days"]

    # --- Inventory-based features ---
    inv_query = """
        SELECT
            ds.id                       AS store_id,
            SUM(fi.stockonhand)         AS total_stock,
            COUNT(DISTINCT fi.idproduct) AS n_products_stocked
        FROM public.factinventory fi
        JOIN public.dimstore      ds ON ds.id = fi.idstore
        GROUP BY ds.id;
    """

    df_inv = fetch_data(inv_query)
    print(f"[INFO] Inventory features loaded for {len(df_inv)} stores")

    # --- Category mix (share of units per category) ---
    cat_query = """
        SELECT
            fs.idstore                  AS store_id,
            dp.category,
            SUM(fs.unitssold)           AS cat_units
        FROM public.factsales  fs
        JOIN public.dimproduct dp ON dp.id = fs.idproduct
        GROUP BY fs.idstore, dp.category;
    """

    df_cat = fetch_data(cat_query)
    cat_totals = df_cat.groupby("store_id")["cat_units"].sum().rename("store_total")
    df_cat = df_cat.merge(cat_totals, on="store_id")
    df_cat["cat_share"] = df_cat["cat_units"] / df_cat["store_total"]

    cat_pivot = df_cat.pivot_table(
        index="store_id",
        columns="category",
        values="cat_share",
        fill_value=0,
    )
    cat_pivot.columns = [f"cat_share_{c}" for c in cat_pivot.columns]
    cat_pivot = cat_pivot.reset_index()

    # --- Merge everything ---
    features = df_sales.merge(df_inv, on="store_id", how="left")
    features = features.merge(cat_pivot, on="store_id", how="left")
    features = features.fillna(0)

    print(
        f"[INFO] Final feature matrix: {features.shape[0]} stores × {features.shape[1]} columns"
    )
    return features


# ============================================================================
# 2. FIND OPTIMAL K (Elbow + Silhouette)
# ============================================================================


def find_optimal_k(X_scaled: np.ndarray) -> int:
    """Plot elbow curve and silhouette scores, return best k."""

    K_range = range(2, min(MAX_K + 1, len(X_scaled)))
    inertias = []
    silhouettes = []

    for k in K_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(list(K_range), inertias, "o-", color="tab:blue")
    ax1.set_title("Elbow Method", fontsize=13)
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (within-cluster sum of squares)")

    ax2.plot(list(K_range), silhouettes, "s-", color="tab:green")
    ax2.set_title("Silhouette Score", fontsize=13)
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")

    fig.suptitle("Choosing the Optimal Number of Clusters", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{dir_name}/optimal_k.png", dpi=150)
    print(f"[SAVED] {dir_name}/optimal_k.png")

    best_k = list(K_range)[np.argmax(silhouettes)]
    print(
        f"[INFO] Best k by silhouette score: {best_k} (score = {max(silhouettes):.3f})"
    )
    return best_k


# ============================================================================
# 3. CLUSTER AND VISUALISE
# ============================================================================

NUMERIC_FEATURES = [
    "total_units",
    "total_receipts",
    "total_revenue",
    "total_cost",
    "total_profit",
    "avg_basket_size",
    "avg_basket_value",
    "profit_margin",
    "revenue_per_day",
    "units_per_day",
    "n_categories_sold",
    "n_products_sold",
    "active_days",
    "total_stock",
    "n_products_stocked",
]


def cluster_stores(features: pd.DataFrame, k: int) -> pd.DataFrame:
    """Run K-Means, attach labels, and produce visualisations."""

    # Include category share columns too
    cat_cols = [c for c in features.columns if c.startswith("cat_share_")]
    feat_cols = NUMERIC_FEATURES + cat_cols

    X = features[feat_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    features["cluster"] = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, features["cluster"])
    print(f"[INFO] Final clustering: k={k}, silhouette={sil:.3f}")

    # ── PCA scatter ──────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    features["pca1"] = coords[:, 0]
    features["pca2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        features["pca1"],
        features["pca2"],
        c=features["cluster"],
        cmap="Set1",
        s=80,
        edgecolors="k",
        alpha=0.8,
    )
    for _, row in features.iterrows():
        ax.annotate(
            row["storename"],
            (row["pca1"], row["pca2"]),
            fontsize=7,
            alpha=0.7,
            textcoords="offset points",
            xytext=(5, 3),
        )
    ax.legend(
        *scatter.legend_elements(),
        title="Cluster",
        loc="best",
    )
    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    ax.set_title("Store Clusters (PCA projection)", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{dir_name}/cluster_pca.png", dpi=150)
    print(f"[SAVED] {dir_name}/cluster_pca.png")

    # ── Cluster profile: radar / bar chart of means ──────────────────────
    profile_cols = [
        "revenue_per_day",
        "units_per_day",
        "avg_basket_size",
        "avg_basket_value",
        "profit_margin",
        "total_stock",
        "n_products_sold",
    ]
    profile = features.groupby("cluster")[profile_cols].mean()

    # Normalise each column to [0, 1] for a comparable bar chart
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 6))
    profile_norm.T.plot(kind="bar", ax=ax, colormap="Set1", edgecolor="k")
    ax.set_title("Cluster Profiles (normalised feature means)", fontsize=14)
    ax.set_ylabel("Normalised Value (0–1)")
    ax.set_xlabel("")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(f"{dir_name}/cluster_profiles.png", dpi=150)
    print(f"[SAVED] {dir_name}/cluster_profiles.png")

    # ── Category mix heatmap per cluster ─────────────────────────────────
    if cat_cols:
        cat_means = features.groupby("cluster")[cat_cols].mean()
        cat_means.columns = [c.replace("cat_share_", "") for c in cat_means.columns]

        fig, ax = plt.subplots(figsize=(12, 4 + 0.4 * k))
        sns.heatmap(
            cat_means,
            annot=True,
            fmt=".1%",
            cmap="YlOrRd",
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Average Category Sales Mix per Cluster", fontsize=14)
        ax.set_ylabel("Cluster")
        ax.set_xlabel("Product Category")
        fig.tight_layout()
        fig.savefig(f"{dir_name}/cluster_category_mix.png", dpi=150)
        print(f"[SAVED] {dir_name}/cluster_category_mix.png")

    # ── Boxplots of key metrics by cluster ───────────────────────────────
    box_metrics = [
        "revenue_per_day",
        "avg_basket_value",
        "profit_margin",
        "total_stock",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, col in zip(axes.flat, box_metrics):
        features.boxplot(column=col, by="cluster", ax=ax)
        ax.set_title(col.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("Cluster")
    fig.suptitle("Key Metrics by Cluster", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{dir_name}/cluster_boxplots.png", dpi=150)
    print(f"[SAVED] {dir_name}/cluster_boxplots.png")

    return features


# ============================================================================
# 4. SUMMARY TABLE
# ============================================================================


def print_summary(features: pd.DataFrame):
    """Print and save a summary of each cluster."""

    summary_cols = [
        "total_revenue",
        "total_units",
        "total_receipts",
        "revenue_per_day",
        "units_per_day",
        "avg_basket_size",
        "avg_basket_value",
        "profit_margin",
        "total_stock",
        "n_products_sold",
        "n_categories_sold",
    ]

    summary = (
        features.groupby("cluster")
        .agg(
            n_stores=("store_id", "count"),
            **{col: (col, "mean") for col in summary_cols},
        )
        .round(2)
    )

    print("\n" + "=" * 70)
    print("CLUSTER SUMMARY (mean values per cluster)")
    print("=" * 70)
    print(summary.to_string())
    print("=" * 70)

    summary.to_csv(f"{dir_name}/cluster_summary.csv")
    print(f"[SAVED] {dir_name}/cluster_summary.csv")

    # Store-level assignment
    store_assignments = features[
        ["store_id", "storename", "city", "country", "location", "cluster"]
    ]
    store_assignments = store_assignments.sort_values(["cluster", "storename"])
    store_assignments.to_csv(f"{dir_name}/store_assignments.csv", index=False)
    print(f"[SAVED] {dir_name}/store_assignments.csv")

    # Print stores per cluster
    print("\nStores per cluster:")
    for c in sorted(features["cluster"].unique()):
        stores = features[features["cluster"] == c]["storename"].tolist()
        print(f"  Cluster {c} ({len(stores)} stores): {', '.join(stores)}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  STORE CLUSTERING – Hópverkefni 4 (Gagnanám)")
    print("=" * 60)

    print("\n▸ Step 1: Building store feature matrix")
    features = build_features()

    print("\n▸ Step 2: Finding optimal number of clusters")
    cat_cols = [c for c in features.columns if c.startswith("cat_share_")]
    feat_cols = NUMERIC_FEATURES + cat_cols
    X_scaled = StandardScaler().fit_transform(features[feat_cols].values)
    best_k = find_optimal_k(X_scaled)

    print(f"\n▸ Step 3: Clustering stores (k={best_k})")
    features = cluster_stores(features, best_k)

    print("\n▸ Step 4: Summary")
    print_summary(features)

    print(f"\n✓ All done! Check outputs in: {dir_name}/")
