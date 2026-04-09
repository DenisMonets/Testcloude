"""
Generate a simulated retail sales dataset.

Dimensions: store_id × sku_id × date
- 100 stores, 200 SKUs, 3 years (2023-01-01 to 2025-12-31)
- ~22M rows total

Output: sales_data.parquet
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import time
from datetime import date

# --- Parameters ---
N_STORES = 100
N_SKUS = 200
DATE_START = date(2023, 1, 1)
DATE_END = date(2025, 12, 31)
OUTPUT_FILE = "sales_data.parquet"
SEED = 42

np.random.seed(SEED)

# --- Date range ---
dates = np.arange(
    np.datetime64(DATE_START),
    np.datetime64(DATE_END) + np.timedelta64(1, "D"),
    dtype="datetime64[D]",
)
n_days = len(dates)
print(f"Period: {DATE_START} to {DATE_END} ({n_days} days)")
print(f"Stores: {N_STORES}, SKUs: {N_SKUS}")
print(f"Total rows: {N_STORES * N_SKUS * n_days:,}")

# --- Pre-generate store and SKU characteristics ---
# Base demand per SKU (some SKUs sell more than others)
sku_base_demand = np.random.lognormal(mean=1.5, sigma=0.8, size=N_SKUS)

# Base price per SKU (between 5 and 500)
sku_base_price = np.random.uniform(5, 500, size=N_SKUS)

# Store traffic multiplier (some stores are busier)
store_multiplier = np.random.lognormal(mean=0.0, sigma=0.4, size=N_STORES)

# Day-of-week seasonality (weekends are higher)
day_of_week = np.array([(d.astype("datetime64[D]").astype(int)) % 7 for d in dates])
dow_factor = np.where(
    np.isin(day_of_week, [5, 6]),  # Saturday=5, Sunday=6
    1.3,
    np.where(np.isin(day_of_week, [4]), 1.1, 1.0),  # Friday slightly higher
)

# Monthly seasonality (December boost, summer dip)
month = np.array(
    [
        (d.astype("datetime64[M]") - d.astype("datetime64[Y]")).astype(int) + 1
        for d in dates
    ]
)
month_factor = np.ones(n_days)
month_factor[month == 12] = 1.5  # December boost
month_factor[month == 11] = 1.2  # November
month_factor[(month >= 6) & (month <= 8)] = 0.85  # Summer dip
month_factor[month == 1] = 1.15  # January sales

# Combined daily seasonality factor
daily_factor = dow_factor * month_factor  # shape: (n_days,)

# --- Generate data in chunks per store ---
t0 = time.time()
writer = None

for store_idx in range(N_STORES):
    store_id = store_idx + 1
    sm = store_multiplier[store_idx]

    # For each SKU in this store, compute lambda for Poisson
    # shape: (N_SKUS, n_days)
    lambdas = (
        sku_base_demand[:, np.newaxis]  # (N_SKUS, 1)
        * sm  # scalar
        * daily_factor[np.newaxis, :]  # (1, n_days)
    )

    # Generate sales quantities (Poisson)
    sales_qty = np.random.poisson(lam=lambdas).astype(np.int32)  # (N_SKUS, n_days)

    # Price with small daily noise (±5%)
    price_noise = np.random.uniform(0.95, 1.05, size=(N_SKUS, n_days))
    prices = (sku_base_price[:, np.newaxis] * price_noise).round(2)  # (N_SKUS, n_days)

    # Revenue
    sales_amount = (sales_qty * prices).round(2)

    # Flatten to rows
    n_rows = N_SKUS * n_days
    store_ids = np.full(n_rows, store_id, dtype=np.int32)
    sku_ids = np.repeat(np.arange(1, N_SKUS + 1, dtype=np.int32), n_days)
    date_col = np.tile(dates, N_SKUS)

    table = pa.table(
        {
            "store_id": pa.array(store_ids, type=pa.int32()),
            "sku_id": pa.array(sku_ids, type=pa.int32()),
            "date": pa.array(date_col, type=pa.date32()),
            "sales_qty": pa.array(sales_qty.ravel(), type=pa.int32()),
            "price": pa.array(prices.ravel().astype(np.float32), type=pa.float32()),
            "sales_amount": pa.array(
                sales_amount.ravel().astype(np.float32), type=pa.float32()
            ),
        }
    )

    if writer is None:
        writer = pq.ParquetWriter(OUTPUT_FILE, table.schema, compression="snappy")
    writer.write_table(table)

    if (store_idx + 1) % 10 == 0:
        print(f"  Stores processed: {store_idx + 1}/{N_STORES}")

writer.close()

elapsed = time.time() - t0
file_size_mb = round(
    __import__("os").path.getsize(OUTPUT_FILE) / (1024 * 1024), 1
)
print(f"\nDone in {elapsed:.1f}s")
print(f"Output: {OUTPUT_FILE} ({file_size_mb} MB)")
