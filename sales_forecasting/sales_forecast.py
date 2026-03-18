"""
Sales Forecasting - Hópverkefni 4 (Gagnanám)
=============================================
Connects to the PostgreSQL data warehouse from Hópverkefni 3 and builds
a time-series sales forecast using Facebook Prophet.

Requirements (pip install):
    pip install psycopg2-binary pandas prophet matplotlib

Usage:
    1. Fill in the DB connection details below.
    2. Run:  python sales_forecast.py
    3. Outputs are saved to the current directory:
         - forecast_plot.png          (overall forecast chart)
         - forecast_components.png    (trend + weekly + yearly seasonality)
         - forecast_by_category.png   (per-category forecast)
         - forecast_results.csv       (raw forecast numbers)
"""

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from prophet import Prophet
import psycopg2
import os
from datetime import datetime
from dotenv import load_dotenv


# ── Configuration ────────────────────────────────────────────────────────────
dir_name = str(datetime.now()).strip().replace(" ", "_").replace(":", "-")
os.makedirs(dir_name)

warnings.filterwarnings("ignore")

load_dotenv()

DB_CONFIG = {
    "host": "vgbi2026-vh.duckdns.org",
    "port": 35053,
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USERNAME"),
    "password": os.getenv("DB_PASSWORD"),
}

FORECAST_DAYS = 90  # how many days into the future to forecast
# ─────────────────────────────────────────────────────────────────────────────


def fetch_data(query: str) -> pd.DataFrame:
    """Run a SQL query against the warehouse and return a DataFrame."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return df


# ============================================================================
# 1. OVERALL DAILY SALES FORECAST
# ============================================================================


def overall_forecast() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate daily total units sold and total revenue, then forecast both.
    Returns (history_df, forecast_df).
    """

    query = """
        SELECT
            dd.fulldate                         AS ds,
            SUM(fs.unitssold)                   AS units_sold,
            SUM(fs.unitssold * dp.priceeur)     AS revenue
        FROM public.factsales  fs
        JOIN public.dimdate    dd ON dd.id = fs.idcalendar
        JOIN public.dimproduct dp ON dp.id = fs.idproduct
        WHERE dp.ispricemissing = FALSE          -- exclude rows without price
        GROUP BY dd.fulldate
        ORDER BY dd.fulldate;
    """

    df = fetch_data(query)
    df["ds"] = pd.to_datetime(df["ds"])

    print(
        f"[INFO] Loaded {len(df)} days of sales history "
        f"({df['ds'].min().date()} → {df['ds'].max().date()})"
    )

    # --- Units forecast ---
    m_units = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    m_units.fit(df[["ds", "units_sold"]].rename(columns={"units_sold": "y"}))

    future = m_units.make_future_dataframe(periods=FORECAST_DAYS)
    fc_units = m_units.predict(future)

    # --- Revenue forecast ---
    m_rev = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    m_rev.fit(df[["ds", "revenue"]].rename(columns={"revenue": "y"}))
    fc_rev = m_rev.predict(future)

    # --- Plot: Units ---
    fig1 = m_units.plot(fc_units)
    ax1 = fig1.gca()
    # Prophet draws: dots = actual, line = forecast, fill = uncertainty
    ax1.get_lines()[0].set_label("Actual")
    ax1.get_lines()[1].set_label("Forecast")
    for coll in ax1.collections:
        coll.set_label("95% confidence interval")
        break  # only label the first fill
    ax1.legend(loc="upper left")
    ax1.set_title("Daily Units Sold – Forecast", fontsize=14)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Units Sold")
    fig1.tight_layout()
    fig1.savefig(f"{dir_name}/forecast_units_plot.png", dpi=150)
    print("[SAVED] forecast_units_plot.png")

    # --- Plot: Revenue ---
    fig2 = m_rev.plot(fc_rev)
    ax2 = fig2.gca()
    ax2.get_lines()[0].set_label("Actual")
    ax2.get_lines()[1].set_label("Forecast")
    for coll in ax2.collections:
        coll.set_label("95% confidence interval")
        break
    ax2.legend(loc="upper left")
    ax2.set_title("Daily Revenue (EUR) – Forecast", fontsize=14)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Revenue (EUR)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    fig2.tight_layout()
    fig2.savefig(f"{dir_name}/forecast_revenue_plot.png", dpi=150)
    print("[SAVED] forecast_revenue_plot.png")

    # --- Components (seasonality breakdown) ---
    fig3 = m_units.plot_components(fc_units)
    fig3.suptitle("Seasonality Components – Units Sold", fontsize=14, y=1.02)
    fig3.tight_layout()
    fig3.savefig(f"{dir_name}/forecast_components.png", dpi=150)
    print("[SAVED] forecast_components.png")

    # --- Save CSV ---
    out = fc_units[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    out.columns = ["date", "units_forecast", "units_lower", "units_upper"]
    rev_cols = fc_rev[["yhat", "yhat_lower", "yhat_upper"]].copy()
    rev_cols.columns = ["revenue_forecast", "revenue_lower", "revenue_upper"]
    out = pd.concat([out, rev_cols], axis=1)
    out.to_csv(f"{dir_name}/forecast_results.csv", index=False)
    print("[SAVED] forecast_results.csv")

    return df, out


# ============================================================================
# 2. PER-CATEGORY FORECAST
# ============================================================================


def category_forecast():
    """
    Break sales down by product category, forecast each independently,
    and plot them together.
    """

    query = """
        SELECT
            dd.fulldate        AS ds,
            dp.category,
            SUM(fs.unitssold)  AS units_sold
        FROM public.factsales  fs
        JOIN public.dimdate    dd ON dd.id = fs.idcalendar
        JOIN public.dimproduct dp ON dp.id = fs.idproduct
        GROUP BY dd.fulldate, dp.category
        ORDER BY dd.fulldate;
    """

    df = fetch_data(query)
    df["ds"] = pd.to_datetime(df["ds"])

    categories = sorted(df["category"].dropna().unique())
    print(f"[INFO] Found {len(categories)} product categories")

    fig, axes = plt.subplots(
        nrows=len(categories),
        ncols=1,
        figsize=(12, 4 * len(categories)),
        sharex=True,
    )
    if len(categories) == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories):
        cat_df = (
            df[df["category"] == cat]
            .groupby("ds")["units_sold"]
            .sum()
            .reset_index()
            .rename(columns={"units_sold": "y"})
        )

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        m.fit(cat_df)
        future = m.make_future_dataframe(periods=FORECAST_DAYS)
        fc = m.predict(future)

        ax.plot(cat_df["ds"], cat_df["y"], "k.", markersize=1, label="Actual")
        ax.plot(fc["ds"], fc["yhat"], color="tab:blue", label="Forecast")
        ax.fill_between(
            fc["ds"],
            fc["yhat_lower"],
            fc["yhat_upper"],
            alpha=0.2,
            color="tab:blue",
        )
        ax.set_title(f"Category: {cat}", fontsize=12)
        ax.set_ylabel("Units Sold")
        ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Sales Forecast by Product Category", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{dir_name}/forecast_by_category.png", dpi=150)
    print("[SAVED] forecast_by_category.png")


# ============================================================================
# 3. MODEL EVALUATION (Train/Test Split)
# ============================================================================


def evaluate_model():
    """
    Hold out the last 30 days of actual data, train on the rest,
    and compute MAE / MAPE so you can report forecast accuracy.
    """

    query = """
        SELECT
            dd.fulldate           AS ds,
            SUM(fs.unitssold)     AS y
        FROM public.factsales  fs
        JOIN public.dimdate    dd ON dd.id = fs.idcalendar
        GROUP BY dd.fulldate
        ORDER BY dd.fulldate;
    """

    df = fetch_data(query)
    df["ds"] = pd.to_datetime(df["ds"])

    HOLDOUT = 30
    train = df.iloc[:-HOLDOUT]
    test = df.iloc[-HOLDOUT:]

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    m.fit(train)

    future = m.make_future_dataframe(periods=HOLDOUT)
    fc = m.predict(future)

    # Merge predictions onto test set
    comparison = test.merge(
        fc[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds",
        how="left",
    )

    comparison["error"] = comparison["y"] - comparison["yhat"]
    comparison["abs_error"] = comparison["error"].abs()
    comparison["pct_error"] = (comparison["abs_error"] / comparison["y"]) * 100

    mae = comparison["abs_error"].mean()
    mape = comparison["pct_error"].mean()

    print(f"\n{'=' * 50}")
    print(f"MODEL EVALUATION  (hold-out = last {HOLDOUT} days)")
    print(f"{'=' * 50}")
    print(f"  MAE  = {mae:,.1f} units")
    print(f"  MAPE = {mape:.1f}%")
    print(f"{'=' * 50}\n")

    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(comparison["ds"], comparison["y"], "o-", label="Actual", markersize=4)
    ax.plot(
        comparison["ds"], comparison["yhat"], "s--", label="Predicted", markersize=4
    )
    ax.fill_between(
        comparison["ds"],
        comparison["yhat_lower"],
        comparison["yhat_upper"],
        alpha=0.2,
        color="tab:orange",
        label="95% interval",
    )
    ax.set_title(
        f"Forecast vs Actual (last {HOLDOUT} days)  —  MAE={mae:,.1f}, MAPE={mape:.1f}%"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{dir_name}/forecast_evaluation.png", dpi=150)
    print("[SAVED] forecast_evaluation.png")

    comparison.to_csv(f"{dir_name}/forecast_evaluation.csv", index=False)
    print("[SAVED] forecast_evaluation.csv")

    return mae, mape


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  SALES FORECASTING – Hópverkefni 4 (Gagnanám)")
    print("=" * 60)

    print("\n▸ Step 1: Overall daily forecast")
    history, forecast = overall_forecast()

    print("\n▸ Step 2: Per-category forecast")
    category_forecast()

    print("\n▸ Step 3: Model evaluation (train/test split)")
    mae, mape = evaluate_model()

    print("\n✓ All done! Check the output files in the current directory.")
