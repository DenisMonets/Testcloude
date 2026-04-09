"""
Benchmark: Feature engineering on sales dataset using Polars.

Measures performance of:
1. Lag features (1, 7, 14, 28 days)
2. Rolling features (mean, std over 7/28 days)
3. Cumulative features (cumsum)
4. Iterative features (EWM, diff, pct_change)

All features are computed per (store_id, sku_id) group, sorted by date.
"""

import polars as pl
import time
import sys

INPUT_FILE = "sales_data.parquet"


def timer(func):
    """Decorator to time a function and return (result, elapsed_seconds)."""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        return result, elapsed
    return wrapper


# ── Load data ────────────────────────────────────────────────────────────────

print(f"Loading {INPUT_FILE}...")
t0 = time.perf_counter()
df = pl.read_parquet(INPUT_FILE)
load_time = time.perf_counter() - t0
print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols in {load_time:.2f}s")
print(f"  Memory: ~{df.estimated_size('mb'):.0f} MB\n")

# Pre-sort by group + date (required for correct lag/rolling calculations)
t0 = time.perf_counter()
df = df.sort(["store_id", "sku_id", "date"])
sort_time = time.perf_counter() - t0
print(f"Sorting by (store_id, sku_id, date): {sort_time:.2f}s\n")

GROUP_COLS = ["store_id", "sku_id"]

# ── 1. Lag features ─────────────────────────────────────────────────────────

@timer
def compute_lags(df):
    return df.with_columns([
        pl.col("sales_qty")
            .shift(lag)
            .over(GROUP_COLS)
            .alias(f"sales_qty_lag_{lag}")
        for lag in [1, 7, 14, 28]
    ])

df, lag_time = compute_lags(df)
print(f"[1] Lag features (1, 7, 14, 28):        {lag_time:.2f}s")

# ── 2. Rolling features ─────────────────────────────────────────────────────

@timer
def compute_rolling(df):
    return df.with_columns([
        pl.col("sales_qty")
            .rolling_mean(window_size=7)
            .over(GROUP_COLS)
            .alias("sales_qty_rolling_mean_7"),
        pl.col("sales_qty")
            .rolling_mean(window_size=28)
            .over(GROUP_COLS)
            .alias("sales_qty_rolling_mean_28"),
        pl.col("sales_qty")
            .rolling_std(window_size=7)
            .over(GROUP_COLS)
            .alias("sales_qty_rolling_std_7"),
        pl.col("sales_qty")
            .rolling_std(window_size=28)
            .over(GROUP_COLS)
            .alias("sales_qty_rolling_std_28"),
    ])

df, rolling_time = compute_rolling(df)
print(f"[2] Rolling features (mean/std 7, 28):   {rolling_time:.2f}s")

# ── 3. Cumulative features ──────────────────────────────────────────────────

@timer
def compute_cumulative(df):
    return df.with_columns([
        pl.col("sales_qty")
            .cum_sum()
            .over(GROUP_COLS)
            .alias("sales_qty_cumsum"),
        pl.col("sales_amount")
            .cum_sum()
            .over(GROUP_COLS)
            .alias("sales_amount_cumsum"),
        (pl.col("sales_qty").cum_sum().over(GROUP_COLS)
            / pl.col("sales_qty").cum_count().over(GROUP_COLS))
            .alias("sales_qty_cummean"),
        pl.col("sales_qty")
            .cum_max()
            .over(GROUP_COLS)
            .alias("sales_qty_cummax"),
    ])

df, cumul_time = compute_cumulative(df)
print(f"[3] Cumulative features (sum/mean/max):  {cumul_time:.2f}s")

# ── 4. Iterative features (EWM, diff, pct_change) ───────────────────────────

@timer
def compute_iterative(df):
    return df.with_columns([
        pl.col("sales_qty")
            .ewm_mean(span=7)
            .over(GROUP_COLS)
            .alias("sales_qty_ewm_7"),
        pl.col("sales_qty")
            .ewm_mean(span=28)
            .over(GROUP_COLS)
            .alias("sales_qty_ewm_28"),
        pl.col("sales_qty")
            .diff(n=1)
            .over(GROUP_COLS)
            .alias("sales_qty_diff_1"),
        pl.col("sales_qty")
            .diff(n=7)
            .over(GROUP_COLS)
            .alias("sales_qty_diff_7"),
        pl.col("sales_qty")
            .pct_change(n=1)
            .over(GROUP_COLS)
            .alias("sales_qty_pct_change_1"),
        pl.col("sales_qty")
            .pct_change(n=7)
            .over(GROUP_COLS)
            .alias("sales_qty_pct_change_7"),
    ])

df, iter_time = compute_iterative(df)
print(f"[4] Iterative features (EWM/diff/pct):   {iter_time:.2f}s")

# ── Summary ──────────────────────────────────────────────────────────────────

total_features_time = lag_time + rolling_time + cumul_time + iter_time
total_time = load_time + sort_time + total_features_time

print("\n" + "=" * 55)
print(f"{'SUMMARY':^55}")
print("=" * 55)
print(f"  Data loading:              {load_time:>8.2f}s")
print(f"  Sorting:                   {sort_time:>8.2f}s")
print(f"  Lag features:              {lag_time:>8.2f}s")
print(f"  Rolling features:          {rolling_time:>8.2f}s")
print(f"  Cumulative features:       {cumul_time:>8.2f}s")
print(f"  Iterative features:        {iter_time:>8.2f}s")
print(f"  ─────────────────────────────────────")
print(f"  Feature engineering total: {total_features_time:>8.2f}s")
print(f"  Grand total:               {total_time:>8.2f}s")
print("=" * 55)
print(f"\nFinal shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"Columns: {df.columns}")
