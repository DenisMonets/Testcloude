"""
Benchmark: Feature engineering on sales dataset using Pandas.

Measures performance of:
1. Lag features (1, 7, 14, 28 days)
2. Rolling features (mean, std over 7/28 days)
3. Cumulative features (cumsum, cummean, cummax)
4. Iterative features (EWM, diff, pct_change)

All features are computed per (store_id, sku_id) group, sorted by date.
"""

import pandas as pd
import time

INPUT_FILE = "sales_data.parquet"

GROUP_COLS = ["store_id", "sku_id"]


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
df = pd.read_parquet(INPUT_FILE)
load_time = time.perf_counter() - t0
print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols in {load_time:.2f}s")
print(f"  Memory: ~{df.memory_usage(deep=True).sum() / 1024**2:.0f} MB\n")

# Pre-sort
t0 = time.perf_counter()
df = df.sort_values(GROUP_COLS + ["date"]).reset_index(drop=True)
sort_time = time.perf_counter() - t0
print(f"Sorting by (store_id, sku_id, date): {sort_time:.2f}s\n")

grouped = df.groupby(GROUP_COLS, sort=False)

# ── 1. Lag features ─────────────────────────────────────────────────────────

@timer
def compute_lags(df, grouped):
    for lag in [1, 7, 14, 28]:
        df[f"sales_qty_lag_{lag}"] = grouped["sales_qty"].shift(lag)
    return df

df, lag_time = compute_lags(df, grouped)
print(f"[1] Lag features (1, 7, 14, 28):        {lag_time:.2f}s")

# ── 2. Rolling features ─────────────────────────────────────────────────────

@timer
def compute_rolling(df, grouped):
    for window in [7, 28]:
        df[f"sales_qty_rolling_mean_{window}"] = grouped["sales_qty"].transform(
            lambda x: x.rolling(window, min_periods=window).mean()
        )
        df[f"sales_qty_rolling_std_{window}"] = grouped["sales_qty"].transform(
            lambda x: x.rolling(window, min_periods=window).std()
        )
    return df

df, rolling_time = compute_rolling(df, grouped)
print(f"[2] Rolling features (mean/std 7, 28):   {rolling_time:.2f}s")

# ── 3. Cumulative features ──────────────────────────────────────────────────

@timer
def compute_cumulative(df, grouped):
    df["sales_qty_cumsum"] = grouped["sales_qty"].cumsum()
    df["sales_amount_cumsum"] = grouped["sales_amount"].cumsum()
    df["sales_qty_cummean"] = grouped["sales_qty"].transform(
        lambda x: x.expanding().mean()
    )
    df["sales_qty_cummax"] = grouped["sales_qty"].cummax()
    return df

df, cumul_time = compute_cumulative(df, grouped)
print(f"[3] Cumulative features (sum/mean/max):  {cumul_time:.2f}s")

# ── 4. Iterative features (EWM, diff, pct_change) ───────────────────────────

@timer
def compute_iterative(df, grouped):
    for span in [7, 28]:
        df[f"sales_qty_ewm_{span}"] = grouped["sales_qty"].transform(
            lambda x: x.ewm(span=span).mean()
        )
    for n in [1, 7]:
        df[f"sales_qty_diff_{n}"] = grouped["sales_qty"].diff(n)
        df[f"sales_qty_pct_change_{n}"] = grouped["sales_qty"].pct_change(periods=n)
    return df

df, iter_time = compute_iterative(df, grouped)
print(f"[4] Iterative features (EWM/diff/pct):   {iter_time:.2f}s")

# ── Summary ──────────────────────────────────────────────────────────────────

total_features_time = lag_time + rolling_time + cumul_time + iter_time
total_time = load_time + sort_time + total_features_time

print("\n" + "=" * 55)
print(f"{'SUMMARY (Pandas)':^55}")
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
print(f"Columns: {list(df.columns)}")
