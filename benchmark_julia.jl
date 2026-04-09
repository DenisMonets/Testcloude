"""
Benchmark: Feature engineering on sales dataset using Julia DataFrames.

Measures performance of:
1. Lag features (1, 7, 14, 28 days)
2. Rolling features (mean, std over 7/28 days)
3. Cumulative features (cumsum, cummean, cummax)
4. Iterative features (EWM, diff, pct_change)

All features are computed per (store_id, sku_id) group, sorted by date.
"""

using Pkg

# Install required packages if not present
for pkg in ["DataFrames", "Arrow", "Statistics", "ShiftedArrays"]
    if !haskey(Pkg.project().dependencies, pkg)
        Pkg.add(pkg)
    end
end

using DataFrames
using Arrow
using Statistics
using ShiftedArrays
using Printf

const INPUT_FILE = "sales_data.parquet"

# ── Utility ──────────────────────────────────────────────────────────────────

function format_number(n::Integer)
    s = string(n)
    parts = String[]
    while length(s) > 3
        push!(parts, s[end-2:end])
        s = s[1:end-3]
    end
    push!(parts, s)
    return join(reverse(parts), ",")
end

# ── Helper functions ─────────────────────────────────────────────────────────

"""Rolling mean over a vector with window size `w`."""
function rolling_mean(x::AbstractVector, w::Int)
    n = length(x)
    result = Vector{Union{Missing, Float64}}(missing, n)
    @inbounds for i in w:n
        result[i] = mean(@view x[i-w+1:i])
    end
    return result
end

"""Rolling std over a vector with window size `w`."""
function rolling_std(x::AbstractVector, w::Int)
    n = length(x)
    result = Vector{Union{Missing, Float64}}(missing, n)
    @inbounds for i in w:n
        result[i] = std(@view x[i-w+1:i])
    end
    return result
end

"""Exponential weighted mean with given span."""
function ewm_mean(x::AbstractVector, span::Int)
    n = length(x)
    alpha = 2.0 / (span + 1)
    result = Vector{Float64}(undef, n)
    result[1] = Float64(x[1])
    @inbounds for i in 2:n
        result[i] = alpha * Float64(x[i]) + (1 - alpha) * result[i-1]
    end
    return result
end

"""Cumulative maximum."""
function cummax(x::AbstractVector)
    return accumulate(max, x)
end

"""Difference with lag n."""
function diff_n(x::AbstractVector, n::Int)
    result = Vector{Union{Missing, Float64}}(missing, length(x))
    @inbounds for i in (n+1):length(x)
        result[i] = Float64(x[i]) - Float64(x[i-n])
    end
    return result
end

"""Percentage change with lag n."""
function pct_change(x::AbstractVector, n::Int)
    result = Vector{Union{Missing, Float64}}(missing, length(x))
    @inbounds for i in (n+1):length(x)
        prev = Float64(x[i-n])
        result[i] = prev == 0.0 ? missing : (Float64(x[i]) - prev) / prev
    end
    return result
end

# ── Load data ────────────────────────────────────────────────────────────────

println("Loading $INPUT_FILE...")
t0 = time()
df = DataFrame(Arrow.Table(INPUT_FILE))
load_time = time() - t0
@printf("  Loaded: %s rows × %d cols in %.2fs\n",
    format_number(nrow(df)), ncol(df), load_time)
@printf("  Memory: ~%.0f MB\n\n", Base.summarysize(df) / 1024^2)

# Pre-sort
t0 = time()
sort!(df, [:store_id, :sku_id, :date])
sort_time = time() - t0
@printf("Sorting by (store_id, sku_id, date): %.2fs\n\n", sort_time)

# Group data
gdf = groupby(df, [:store_id, :sku_id])
n_groups = length(gdf)
println("Number of groups: $n_groups")

# ── 1. Lag features ─────────────────────────────────────────────────────────

t0 = time()
transform!(gdf,
    :sales_qty => (x -> ShiftedArrays.lag(x, 1))  => :sales_qty_lag_1,
    :sales_qty => (x -> ShiftedArrays.lag(x, 7))  => :sales_qty_lag_7,
    :sales_qty => (x -> ShiftedArrays.lag(x, 14)) => :sales_qty_lag_14,
    :sales_qty => (x -> ShiftedArrays.lag(x, 28)) => :sales_qty_lag_28,
)
lag_time = time() - t0
@printf("[1] Lag features (1, 7, 14, 28):        %.2fs\n", lag_time)

# ── 2. Rolling features ─────────────────────────────────────────────────────

t0 = time()
transform!(gdf,
    :sales_qty => (x -> rolling_mean(x, 7))  => :sales_qty_rolling_mean_7,
    :sales_qty => (x -> rolling_mean(x, 28)) => :sales_qty_rolling_mean_28,
    :sales_qty => (x -> rolling_std(x, 7))   => :sales_qty_rolling_std_7,
    :sales_qty => (x -> rolling_std(x, 28))  => :sales_qty_rolling_std_28,
)
rolling_time = time() - t0
@printf("[2] Rolling features (mean/std 7, 28):   %.2fs\n", rolling_time)

# ── 3. Cumulative features ──────────────────────────────────────────────────

t0 = time()
transform!(gdf,
    :sales_qty    => cumsum  => :sales_qty_cumsum,
    :sales_amount => cumsum  => :sales_amount_cumsum,
    :sales_qty    => (x -> cumsum(x) ./ (1:length(x))) => :sales_qty_cummean,
    :sales_qty    => cummax  => :sales_qty_cummax,
)
cumul_time = time() - t0
@printf("[3] Cumulative features (sum/mean/max):  %.2fs\n", cumul_time)

# ── 4. Iterative features ───────────────────────────────────────────────────

t0 = time()
transform!(gdf,
    :sales_qty => (x -> ewm_mean(x, 7))    => :sales_qty_ewm_7,
    :sales_qty => (x -> ewm_mean(x, 28))   => :sales_qty_ewm_28,
    :sales_qty => (x -> diff_n(x, 1))      => :sales_qty_diff_1,
    :sales_qty => (x -> diff_n(x, 7))      => :sales_qty_diff_7,
    :sales_qty => (x -> pct_change(x, 1))  => :sales_qty_pct_change_1,
    :sales_qty => (x -> pct_change(x, 7))  => :sales_qty_pct_change_7,
)
iter_time = time() - t0
@printf("[4] Iterative features (EWM/diff/pct):   %.2fs\n", iter_time)

# ── Summary ──────────────────────────────────────────────────────────────────

total_features_time = lag_time + rolling_time + cumul_time + iter_time
total_time = load_time + sort_time + total_features_time

println()
println("=" ^ 55)
@printf("%s\n", lpad("SUMMARY", 35))
println("=" ^ 55)
@printf("  Data loading:              %8.2fs\n", load_time)
@printf("  Sorting:                   %8.2fs\n", sort_time)
@printf("  Lag features:              %8.2fs\n", lag_time)
@printf("  Rolling features:          %8.2fs\n", rolling_time)
@printf("  Cumulative features:       %8.2fs\n", cumul_time)
@printf("  Iterative features:        %8.2fs\n", iter_time)
println("  ─────────────────────────────────────")
@printf("  Feature engineering total: %8.2fs\n", total_features_time)
@printf("  Grand total:               %8.2fs\n", total_time)
println("=" ^ 55)
@printf("\nFinal shape: %s rows × %d cols\n",
    format_number(nrow(df)), ncol(df))
println("Columns: ", names(df))
