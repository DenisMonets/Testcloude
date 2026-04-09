"""
Benchmark: Feature engineering on sales dataset using pure NumPy.

Measures performance of:
1. Lag features (1, 7, 14, 28 days)
2. Rolling features (mean, std over 7/28 days)
3. Cumulative features (cumsum, cummean, cummax)
4. Iterative features (EWM, diff, pct_change)

All features are computed per (store_id, sku_id) group, sorted by date.
No DataFrame overhead — works directly with NumPy arrays.
"""

import numpy as np
import pyarrow.parquet as pq
import time

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
table = pq.read_table(INPUT_FILE)
store_id = table.column("store_id").to_numpy()
sku_id = table.column("sku_id").to_numpy()
date_col = table.column("date").to_numpy()
sales_qty = table.column("sales_qty").to_numpy().astype(np.float64)
sales_amount = table.column("sales_amount").to_numpy().astype(np.float64)
n_rows = len(store_id)
load_time = time.perf_counter() - t0
print(f"  Loaded: {n_rows:,} rows in {load_time:.2f}s")
print(f"  Memory: ~{(store_id.nbytes + sku_id.nbytes + date_col.nbytes + sales_qty.nbytes + sales_amount.nbytes) / 1024**2:.0f} MB\n")

# ── Sort by (store_id, sku_id, date) ─────────────────────────────────────────

t0 = time.perf_counter()
sort_idx = np.lexsort((date_col, sku_id, store_id))
store_id = store_id[sort_idx]
sku_id = sku_id[sort_idx]
date_col = date_col[sort_idx]
sales_qty = sales_qty[sort_idx]
sales_amount = sales_amount[sort_idx]
sort_time = time.perf_counter() - t0
print(f"Sorting by (store_id, sku_id, date): {sort_time:.2f}s\n")

# ── Find group boundaries ────────────────────────────────────────────────────
# Groups are contiguous runs of (store_id, sku_id) in sorted data

t0 = time.perf_counter()
group_key = store_id.astype(np.int64) * 1_000_000 + sku_id.astype(np.int64)
breaks = np.nonzero(np.diff(group_key))[0] + 1
group_starts = np.concatenate(([0], breaks))
group_ends = np.concatenate((breaks, [n_rows]))
n_groups = len(group_starts)
group_time = time.perf_counter() - t0
print(f"Found {n_groups:,} groups in {group_time:.4f}s\n")


# ── Helper: apply function per group ─────────────────────────────────────────

def per_group(func, *args):
    """Apply func to each group slice, return concatenated result."""
    results = []
    for i in range(n_groups):
        s, e = group_starts[i], group_ends[i]
        slices = tuple(a[s:e] for a in args)
        results.append(func(*slices))
    return np.concatenate(results)


# ── Helper: rolling mean using cumsum trick ──────────────────────────────────

def _rolling_mean(x, w):
    n = len(x)
    out = np.full(n, np.nan)
    if n >= w:
        cs = np.cumsum(x)
        out[w - 1] = cs[w - 1] / w
        out[w:] = (cs[w:] - cs[:-w]) / w
    return out


def _rolling_std(x, w):
    n = len(x)
    out = np.full(n, np.nan)
    if n >= w:
        cs = np.cumsum(x)
        cs2 = np.cumsum(x * x)
        for i in range(w - 1, n):
            s = cs[i] - (cs[i - w] if i >= w else 0.0)
            s2 = cs2[i] - (cs2[i - w] if i >= w else 0.0)
            mean_val = s / w
            out[i] = np.sqrt(s2 / (w - 1) - s * mean_val / (w - 1))
    return out


def _ewm_mean(x, span):
    alpha = 2.0 / (span + 1)
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


# ── 1. Lag features ─────────────────────────────────────────────────────────

@timer
def compute_lags():
    results = {}
    for lag in [1, 7, 14, 28]:
        def _lag(x, _lag=lag):
            out = np.full(len(x), np.nan)
            if len(x) > _lag:
                out[_lag:] = x[:-_lag]
            return out
        results[f"lag_{lag}"] = per_group(_lag, sales_qty)
    return results

lag_results, lag_time = compute_lags()
print(f"[1] Lag features (1, 7, 14, 28):        {lag_time:.2f}s")

# ── 2. Rolling features ─────────────────────────────────────────────────────

@timer
def compute_rolling():
    results = {}
    for w in [7, 28]:
        results[f"rolling_mean_{w}"] = per_group(lambda x, _w=w: _rolling_mean(x, _w), sales_qty)
        results[f"rolling_std_{w}"] = per_group(lambda x, _w=w: _rolling_std(x, _w), sales_qty)
    return results

rolling_results, rolling_time = compute_rolling()
print(f"[2] Rolling features (mean/std 7, 28):   {rolling_time:.2f}s")

# ── 3. Cumulative features ──────────────────────────────────────────────────

@timer
def compute_cumulative():
    results = {}
    results["cumsum_qty"] = per_group(np.cumsum, sales_qty)
    results["cumsum_amount"] = per_group(np.cumsum, sales_amount)
    results["cummean"] = per_group(
        lambda x: np.cumsum(x) / np.arange(1, len(x) + 1), sales_qty
    )
    results["cummax"] = per_group(np.maximum.accumulate, sales_qty)
    return results

cumul_results, cumul_time = compute_cumulative()
print(f"[3] Cumulative features (sum/mean/max):  {cumul_time:.2f}s")

# ── 4. Iterative features (EWM, diff, pct_change) ───────────────────────────

@timer
def compute_iterative():
    results = {}
    for span in [7, 28]:
        results[f"ewm_{span}"] = per_group(
            lambda x, _s=span: _ewm_mean(x, _s), sales_qty
        )
    for n in [1, 7]:
        def _diff(x, _n=n):
            out = np.full(len(x), np.nan)
            if len(x) > _n:
                out[_n:] = x[_n:] - x[:-_n]
            return out

        def _pct(x, _n=n):
            out = np.full(len(x), np.nan)
            if len(x) > _n:
                prev = x[:-_n]
                with np.errstate(divide="ignore", invalid="ignore"):
                    out[_n:] = np.where(prev != 0, (x[_n:] - prev) / prev, np.nan)
            return out

        results[f"diff_{n}"] = per_group(_diff, sales_qty)
        results[f"pct_change_{n}"] = per_group(_pct, sales_qty)
    return results

iter_results, iter_time = compute_iterative()
print(f"[4] Iterative features (EWM/diff/pct):   {iter_time:.2f}s")

# ── Summary ──────────────────────────────────────────────────────────────────

total_features = len(lag_results) + len(rolling_results) + len(cumul_results) + len(iter_results)
total_features_time = lag_time + rolling_time + cumul_time + iter_time
total_time = load_time + sort_time + total_features_time

print("\n" + "=" * 55)
print(f"{'SUMMARY (NumPy)':^55}")
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
print(f"\nComputed {total_features} feature arrays, each with {n_rows:,} elements")
print(f"Feature names: {sorted(set(list(lag_results) + list(rolling_results) + list(cumul_results) + list(iter_results)))}")
