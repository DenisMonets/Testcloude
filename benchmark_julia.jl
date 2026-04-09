"""
Benchmark: Feature engineering on sales dataset using Julia.
Optimized: works directly with sorted arrays + multithreading (Threads.@threads).

Run with: julia -t auto benchmark_julia.jl

Measures performance of:
1. Lag features (1, 7, 14, 28 days)
2. Rolling features (mean, std over 7/28 days)
3. Cumulative features (cumsum, cummean, cummax)
4. Iterative features (EWM, diff, pct_change)

All features are computed per (store_id, sku_id) group, sorted by date.
"""

using Pkg

# Install required packages if not present
for pkg in ["Arrow", "Printf"]
    try
        @eval using $(Symbol(pkg))
    catch
        Pkg.add(pkg)
    end
end

using Arrow
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

println("Julia threads: $(Threads.nthreads())")
if Threads.nthreads() == 1
    println("⚠  Running single-threaded. Use `julia -t auto` for best performance.\n")
end

# ── Load data ────────────────────────────────────────────────────────────────

println("Loading $INPUT_FILE...")
t0 = time()
tbl = Arrow.Table(INPUT_FILE)
store_id = Int32.(tbl.store_id)
sku_id   = Int32.(tbl.sku_id)
date_col = tbl.date
sales_qty    = Float64.(tbl.sales_qty)
sales_amount = Float64.(tbl.sales_amount)
n_rows = length(store_id)
load_time = time() - t0
mem_mb = (sizeof(store_id) + sizeof(sku_id) + sizeof(sales_qty) + sizeof(sales_amount)) / 1024^2
@printf("  Loaded: %s rows in %.2fs\n", format_number(n_rows), load_time)
@printf("  Memory: ~%.0f MB\n\n", mem_mb)

# ── Sort by (store_id, sku_id, date) ─────────────────────────────────────────

t0 = time()
sort_idx = sortperm(1:n_rows, by=i -> (store_id[i], sku_id[i], date_col[i]))
store_id     = store_id[sort_idx]
sku_id       = sku_id[sort_idx]
sales_qty    = sales_qty[sort_idx]
sales_amount = sales_amount[sort_idx]
sort_time = time() - t0
@printf("Sorting by (store_id, sku_id, date): %.2fs\n\n", sort_time)

# ── Find group boundaries ────────────────────────────────────────────────────

t0 = time()
group_starts = Int[1]
for i in 2:n_rows
    if store_id[i] != store_id[i-1] || sku_id[i] != sku_id[i-1]
        push!(group_starts, i)
    end
end
push!(group_starts, n_rows + 1)  # sentinel
n_groups = length(group_starts) - 1
group_time = time() - t0
@printf("Found %s groups in %.4fs\n\n", format_number(n_groups), group_time)

# ── Per-group kernel functions (fully typed, @inbounds, @simd where possible) ─

function kernel_lag!(out::Vector{Float64}, x::Vector{Float64}, s::Int, e::Int, lag::Int)
    len = e - s + 1
    @inbounds for i in 1:min(lag, len)
        out[s + i - 1] = NaN
    end
    @inbounds @simd for i in (lag+1):len
        out[s + i - 1] = x[s + i - 1 - lag]
    end
end

function kernel_rolling_mean!(out::Vector{Float64}, x::Vector{Float64}, s::Int, e::Int, w::Int)
    len = e - s + 1
    @inbounds for i in 1:(min(w, len) - 1)
        out[s + i - 1] = NaN
    end
    if len >= w
        # Cumsum trick for O(n) rolling mean
        acc = 0.0
        @inbounds for i in 0:(w-1)
            acc += x[s + i]
        end
        out[s + w - 1] = acc / w
        @inbounds for i in w:(len-1)
            acc += x[s + i] - x[s + i - w]
            out[s + i] = acc / w
        end
    end
end

function kernel_rolling_std!(out::Vector{Float64}, x::Vector{Float64}, s::Int, e::Int, w::Int)
    len = e - s + 1
    @inbounds for i in 1:(min(w, len) - 1)
        out[s + i - 1] = NaN
    end
    if len >= w
        # Online sum/sum² for O(n) rolling std
        sum_x = 0.0
        sum_x2 = 0.0
        @inbounds for i in 0:(w-1)
            v = x[s + i]
            sum_x += v
            sum_x2 += v * v
        end
        mean_val = sum_x / w
        out[s + w - 1] = sqrt((sum_x2 - sum_x * mean_val) / (w - 1))
        @inbounds for i in w:(len-1)
            old = x[s + i - w]
            new = x[s + i]
            sum_x += new - old
            sum_x2 += new * new - old * old
            mean_val = sum_x / w
            out[s + i] = sqrt(max(0.0, (sum_x2 - sum_x * mean_val) / (w - 1)))
        end
    end
end

function kernel_cumsum!(out::Vector{Float64}, x::Vector{Float64}, s::Int, e::Int)
    acc = 0.0
    @inbounds for i in s:e
        acc += x[i]
        out[i] = acc
    end
end

function kernel_cummean!(out::Vector{Float64}, x::Vector{Float64}, s::Int, e::Int)
    acc = 0.0
    cnt = 0
    @inbounds for i in s:e
        acc += x[i]
        cnt += 1
        out[i] = acc / cnt
    end
end

function kernel_cummax!(out::Vector{Float64}, x::Vector{Float64}, s::Int, e::Int)
    mx = -Inf
    @inbounds for i in s:e
        mx = max(mx, x[i])
        out[i] = mx
    end
end

function kernel_ewm!(out::Vector{Float64}, x::Vector{Float64}, s::Int, e::Int, alpha::Float64)
    @inbounds out[s] = x[s]
    one_minus_alpha = 1.0 - alpha
    @inbounds for i in (s+1):e
        out[i] = alpha * x[i] + one_minus_alpha * out[i-1]
    end
end

function kernel_diff!(out::Vector{Float64}, x::Vector{Float64}, s::Int, e::Int, n::Int)
    len = e - s + 1
    @inbounds for i in 1:min(n, len)
        out[s + i - 1] = NaN
    end
    @inbounds @simd for i in (n+1):len
        out[s + i - 1] = x[s + i - 1] - x[s + i - 1 - n]
    end
end

function kernel_pct_change!(out::Vector{Float64}, x::Vector{Float64}, s::Int, e::Int, n::Int)
    len = e - s + 1
    @inbounds for i in 1:min(n, len)
        out[s + i - 1] = NaN
    end
    @inbounds for i in (n+1):len
        prev = x[s + i - 1 - n]
        out[s + i - 1] = prev == 0.0 ? NaN : (x[s + i - 1] - prev) / prev
    end
end

# ── Parallel dispatch helper ─────────────────────────────────────────────────

function parallel_apply!(out::Vector{Float64}, x::Vector{Float64},
                         group_starts::Vector{Int}, n_groups::Int, kernel::F, args...) where F
    Threads.@threads for g in 1:n_groups
        s = group_starts[g]
        e = group_starts[g+1] - 1
        kernel(out, x, s, e, args...)
    end
end

# Pre-allocate all output arrays
function alloc()::Vector{Float64}
    return Vector{Float64}(undef, n_rows)
end

# ── 1. Lag features ─────────────────────────────────────────────────────────

t0 = time()
lag_1  = alloc(); lag_7  = alloc(); lag_14 = alloc(); lag_28 = alloc()
parallel_apply!(lag_1,  sales_qty, group_starts, n_groups, kernel_lag!, 1)
parallel_apply!(lag_7,  sales_qty, group_starts, n_groups, kernel_lag!, 7)
parallel_apply!(lag_14, sales_qty, group_starts, n_groups, kernel_lag!, 14)
parallel_apply!(lag_28, sales_qty, group_starts, n_groups, kernel_lag!, 28)
lag_time = time() - t0
@printf("[1] Lag features (1, 7, 14, 28):        %.2fs\n", lag_time)

# ── 2. Rolling features ─────────────────────────────────────────────────────

t0 = time()
rmean_7  = alloc(); rmean_28 = alloc()
rstd_7   = alloc(); rstd_28  = alloc()
parallel_apply!(rmean_7,  sales_qty, group_starts, n_groups, kernel_rolling_mean!, 7)
parallel_apply!(rmean_28, sales_qty, group_starts, n_groups, kernel_rolling_mean!, 28)
parallel_apply!(rstd_7,   sales_qty, group_starts, n_groups, kernel_rolling_std!, 7)
parallel_apply!(rstd_28,  sales_qty, group_starts, n_groups, kernel_rolling_std!, 28)
rolling_time = time() - t0
@printf("[2] Rolling features (mean/std 7, 28):   %.2fs\n", rolling_time)

# ── 3. Cumulative features ──────────────────────────────────────────────────

t0 = time()
csum_qty = alloc(); csum_amt = alloc()
cmean    = alloc(); cmax     = alloc()
parallel_apply!(csum_qty, sales_qty,    group_starts, n_groups, kernel_cumsum!)
parallel_apply!(csum_amt, sales_amount, group_starts, n_groups, kernel_cumsum!)
parallel_apply!(cmean,    sales_qty,    group_starts, n_groups, kernel_cummean!)
parallel_apply!(cmax,     sales_qty,    group_starts, n_groups, kernel_cummax!)
cumul_time = time() - t0
@printf("[3] Cumulative features (sum/mean/max):  %.2fs\n", cumul_time)

# ── 4. Iterative features ───────────────────────────────────────────────────

t0 = time()
ewm_7  = alloc(); ewm_28 = alloc()
diff_1 = alloc(); diff_7 = alloc()
pct_1  = alloc(); pct_7  = alloc()
alpha_7  = 2.0 / (7 + 1)
alpha_28 = 2.0 / (28 + 1)
parallel_apply!(ewm_7,  sales_qty, group_starts, n_groups, kernel_ewm!, alpha_7)
parallel_apply!(ewm_28, sales_qty, group_starts, n_groups, kernel_ewm!, alpha_28)
parallel_apply!(diff_1, sales_qty, group_starts, n_groups, kernel_diff!, 1)
parallel_apply!(diff_7, sales_qty, group_starts, n_groups, kernel_diff!, 7)
parallel_apply!(pct_1,  sales_qty, group_starts, n_groups, kernel_pct_change!, 1)
parallel_apply!(pct_7,  sales_qty, group_starts, n_groups, kernel_pct_change!, 7)
iter_time = time() - t0
@printf("[4] Iterative features (EWM/diff/pct):   %.2fs\n", iter_time)

# ── Summary ──────────────────────────────────────────────────────────────────

total_features_time = lag_time + rolling_time + cumul_time + iter_time
total_time = load_time + sort_time + total_features_time
n_features = 18  # 4 lags + 4 rolling + 4 cumul + 6 iterative

println()
println("=" ^ 60)
@printf("%s\n", lpad("SUMMARY (Julia, $(Threads.nthreads()) threads)", 45))
println("=" ^ 60)
@printf("  Data loading:              %8.2fs\n", load_time)
@printf("  Sorting:                   %8.2fs\n", sort_time)
@printf("  Group detection:           %8.4fs\n", group_time)
@printf("  Lag features:              %8.2fs\n", lag_time)
@printf("  Rolling features:          %8.2fs\n", rolling_time)
@printf("  Cumulative features:       %8.2fs\n", cumul_time)
@printf("  Iterative features:        %8.2fs\n", iter_time)
println("  ──────────────────────────────────────────")
@printf("  Feature engineering total: %8.2fs\n", total_features_time)
@printf("  Grand total:               %8.2fs\n", total_time)
println("=" ^ 60)
@printf("\nComputed %d feature arrays, each with %s elements\n", n_features, format_number(n_rows))
