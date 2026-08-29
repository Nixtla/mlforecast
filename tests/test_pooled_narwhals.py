# tests/test_pooled_narwhals.py
"""Unit tests for the narwhals pooled engine internals."""

import mlforecast.pooled as mp
from mlforecast._pooled_keys import _NULL_SENTINEL, add_bucket_id


def test_engine_constant_is_valid():
    assert mp.POOLED_ENGINE in ("narwhals", "numpy")


def test_shared_key_helpers_importable_from_both_engines():
    from mlforecast import _pooled_legacy

    assert _pooled_legacy.add_bucket_id is add_bucket_id
    assert _NULL_SENTINEL == "\x00__MLF_NULL__"


import narwhals as nw
import numpy as np
import polars as pl
import pytest

from mlforecast._pooled_engine import (
    PooledCtx,
    build_agg_table,
    grouped_accumulate,
    quantile_feature,
    quantile_values,
)

BACKENDS = ["polars", "pandas"]


def _panel(backend, n_buckets=3, n_times=10, n_series_per_bucket=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_buckets):
        for s in range(n_series_per_bucket):
            for t in range(n_times):
                rows.append((f"b{b}_s{s}", t, b, rng.normal(10, 2)))
    df = pl.DataFrame(rows, schema=["unique_id", "ds", "store", "y"], orient="row")
    return df if backend == "polars" else df.to_pandas()


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_accumulate_cum_sum_resets_per_bucket(backend):
    df = _panel(backend, n_buckets=2, n_times=3, n_series_per_bucket=1)
    out = grouped_accumulate(df, ["store"], ["y"], "cum_sum", ["Ey"])
    o = out if isinstance(out, pl.DataFrame) else pl.from_pandas(out)
    o = o.sort(["store", "ds"])
    for b in (0, 1):
        blk = o.filter(pl.col("store") == b)
        assert blk["Ey"].to_list() == pytest.approx(
            np.cumsum(blk["y"].to_numpy()).tolist()
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_accumulate_rejects_unknown_op(backend):
    with pytest.raises(ValueError, match="unsupported accumulate op"):
        grouped_accumulate(_panel(backend), ["store"], ["y"], "cum_prod", ["Ey"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_build_agg_table_shape_and_aggregates(backend):
    df = _panel(backend, n_buckets=3, n_times=10, n_series_per_bucket=2)
    tbl = build_agg_table(df, ["store"], "ds", "y", {None})
    o = tbl if isinstance(tbl, pl.DataFrame) else pl.from_pandas(tbl)
    assert o.height == 3 * 10, "one row per (bucket, timestamp)"
    for c in ["store", "ds", "ord", "s", "c", "q", "mn", "mx", "Es", "Ec", "Eq"]:
        assert c in o.columns
    # every timestamp had 2 contributing series
    assert o["c"].to_list() == [2.0] * 30
    # per-bucket prefix sums restart at each bucket
    first_of_bucket = o.sort(["store", "ord"]).group_by("store").first()
    assert first_of_bucket["Es"].to_list() == pytest.approx(
        first_of_bucket["s"].to_list()
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_build_agg_table_emits_no_complex_groupby_warning(backend):
    """A composite expression inside .agg() costs 250x on pandas and is
    invisible on polars. Guard it here, permanently."""
    import warnings

    df = _panel(backend, n_buckets=4, n_times=20)
    # NOTE: pytest.warns(None) was removed in pytest 8; this repo runs pytest 9.
    # catch_warnings(record=True) is the supported way to assert NO warning.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        build_agg_table(df, ["store"], "ds", "y", {None, "sum", "mean"})
    complex_gb = [w for w in record if "complex group-by expression" in str(w.message)]
    assert not complex_gb, (
        "build_agg_table fell back to narwhals' slow pandas group-by path: "
        f"{[str(w.message) for w in complex_gb]}"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_nan_targets_do_not_poison_prefix_sums(backend):
    """NaN is not null: an unguarded sum() would make one NaN target turn that
    bucket's ENTIRE prefix sum into NaN, and count() would count it as present.
    The legacy engine treats NaN as missing; so must we."""
    df = _panel(backend, n_buckets=2, n_times=5, n_series_per_bucket=2)
    o = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
    # Poison exactly one raw row: series "b0_s0" at ds=0. There are 2 series
    # per bucket in this fixture, so (store=0, ds=0) has 2 contributing rows;
    # poisoning only one of them (not both, via unique_id) leaves 1 surviving
    # observation, which is what the assertions below check against.
    o = o.with_columns(
        pl.when((pl.col("unique_id") == "b0_s0") & (pl.col("ds") == 0))
        .then(None)
        .otherwise(pl.col("y"))
        .alias("y")
    ).with_columns(pl.col("y").fill_null(float("nan")))
    src = o if isinstance(df, pl.DataFrame) else o.to_pandas()

    tbl = build_agg_table(src, ["store"], "ds", "y", {None})
    t = tbl if isinstance(tbl, pl.DataFrame) else pl.from_pandas(tbl)
    t = t.sort(["store", "ord"])
    b0 = t.filter(pl.col("store") == 0)
    assert b0["c"][0] == 1.0, "the NaN row must not be counted as an observation"
    es = b0["Es"].cast(pl.Float64).to_numpy()
    assert not np.isnan(es).any(), f"NaN leaked into the prefix sum: {es}"
    assert b0["s"][0] == pytest.approx(
        o.filter((pl.col("store") == 0) & (pl.col("ds") == 0) & ~pl.col("y").is_nan())[
            "y"
        ].sum()
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_accumulate_rejects_empty_keys(backend):
    with pytest.raises(ValueError, match="non-empty"):
        grouped_accumulate(_panel(backend), [], ["y"], "cum_sum", ["Ey"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_accumulate_ewm_mean_requires_explicit_adjust(backend):
    with pytest.raises(ValueError, match="adjust"):
        grouped_accumulate(
            _panel(backend, n_buckets=1, n_times=5, n_series_per_bucket=1),
            ["store"],
            ["y"],
            "ewm_mean",
            ["Ey"],
            alpha=0.5,
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_accumulate_ewm_mean_requires_explicit_alpha(backend):
    with pytest.raises(ValueError, match="alpha"):
        grouped_accumulate(
            _panel(backend, n_buckets=1, n_times=5, n_series_per_bucket=1),
            ["store"],
            ["y"],
            "ewm_mean",
            ["Ey"],
            adjust=False,
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_accumulate_ewm_mean_requires_explicit_ignore_nulls(backend):
    with pytest.raises(ValueError, match="ignore_nulls"):
        grouped_accumulate(
            _panel(backend, n_buckets=1, n_times=5, n_series_per_bucket=1),
            ["store"],
            ["y"],
            "ewm_mean",
            ["Ey"],
            alpha=0.5,
            adjust=False,
        )


def test_grouped_accumulate_ewm_mean_backends_agree_with_explicit_adjust():
    """With alpha, adjust, and ignore_nulls all explicit, polars and pandas
    must agree -- otherwise the shim silently reintroduces the divergence it
    exists to prevent (polars' native ewm_mean defaults adjust=True and
    ignore_nulls=False; the two backends' unadjusted-vs-adjusted recursions,
    and their null-handling, diverge without explicit values)."""
    polars_df = _panel(backend="polars", n_buckets=1, n_times=6, n_series_per_bucket=1)
    pandas_df = polars_df.to_pandas()

    pl_out = grouped_accumulate(
        polars_df,
        ["store"],
        ["y"],
        "ewm_mean",
        ["Ey"],
        alpha=0.5,
        adjust=False,
        ignore_nulls=True,
    )
    pd_out = grouped_accumulate(
        pandas_df,
        ["store"],
        ["y"],
        "ewm_mean",
        ["Ey"],
        alpha=0.5,
        adjust=False,
        ignore_nulls=True,
    )
    pl_vals = pl_out.sort("ds")["Ey"].to_list()
    pd_vals = pl.from_pandas(pd_out).sort("ds")["Ey"].to_list()
    assert pl_vals == pytest.approx(pd_vals)


def test_grouped_accumulate_ewm_mean_forward_fills_gaps_per_bucket():
    """Legacy `_ewm_from_agg` assigns the running EWM at EVERY ordinal,
    updating only where an observation exists, so gap ordinals carry the
    previous value forward. polars' raw ewm_mean leaves those rows null;
    this must be forward-filled WITHIN each bucket (never bleeding across
    buckets) to match. Two buckets with interior gaps, offset so a naive
    frame-level (rather than per-bucket) ffill would leak bucket 0's last
    value into bucket 1's leading gap and fail this test."""
    # bucket 0: [1.0, None, 3.0, 4.0, None, 6.0] -> legacy fold
    #   [1.0, 1.0, 2.0, 3.0, 3.0, 4.5]
    # bucket 1: [None, 10.0, None, None, 20.0, None] -- starts with a gap, so
    # a cross-bucket ffill would wrongly carry bucket 0's final value (4.5)
    # into this bucket's first row instead of leaving it null (no observation
    # yet in this bucket).
    rows = []
    b0_y = [1.0, None, 3.0, 4.0, None, 6.0]
    b1_y = [None, 10.0, None, None, 20.0, None]
    for t, y in enumerate(b0_y):
        rows.append(("s0", t, 0, y))
    for t, y in enumerate(b1_y):
        rows.append(("s1", t, 1, y))
    df = pl.DataFrame(rows, schema=["unique_id", "ds", "store", "y"], orient="row")

    want_b0 = [1.0, 1.0, 2.0, 3.0, 3.0, 4.5]
    want_b1 = [None, 10.0, 10.0, 10.0, 15.0, 15.0]

    for backend in BACKENDS:
        src = df if backend == "polars" else df.to_pandas()
        out = grouped_accumulate(
            src,
            ["store"],
            ["y"],
            "ewm_mean",
            ["Ey"],
            alpha=0.5,
            adjust=False,
            ignore_nulls=True,
        )
        o = out if isinstance(out, pl.DataFrame) else pl.from_pandas(out)
        o = o.sort(["store", "ds"])
        got_b0 = o.filter(pl.col("store") == 0)["Ey"].to_list()
        got_b1 = o.filter(pl.col("store") == 1)["Ey"].to_list()
        assert got_b0 == pytest.approx(want_b0), f"{backend}: bucket 0 mismatch"
        assert got_b1[0] is None, (
            f"{backend}: bucket 1 must not inherit bucket 0's tail"
        )
        assert got_b1[1:] == pytest.approx(want_b1[1:]), f"{backend}: bucket 1 mismatch"


def test_grouped_accumulate_cum_min_forward_fills_gaps_per_bucket():
    """Legacy `_expanding_min_from_agg` is `np.fmin.accumulate(agg.mins)`;
    `np.fmin` IGNORES NaN, so the running minimum carries THROUGH a gap
    ordinal. polars' `cum_min` and pandas' `cummin` instead leave that exact
    position null, so `grouped_accumulate(..., "cum_min", ...)` must
    forward-fill to match -- same defect class as the ewm_mean case above.
    Two buckets, the second STARTING with a gap, so a frame-level (rather
    than per-bucket) ffill would wrongly carry bucket 0's trailing minimum
    into bucket 1's leading gap instead of leaving it null (no observation
    yet in that bucket)."""
    # bucket 0: [5.0, None, 2.0, None, 8.0, 1.0] -> running min
    #   [5.0, 5.0, 2.0, 2.0, 2.0, 1.0]
    # bucket 1: [None, 10.0, None, None, 3.0, None] -- starts with a gap, so
    # bucket 1's first row must stay null (no observation yet in THIS
    # bucket), not inherit bucket 0's final running min of 1.0.
    rows = []
    b0_y = [5.0, None, 2.0, None, 8.0, 1.0]
    b1_y = [None, 10.0, None, None, 3.0, None]
    for t, y in enumerate(b0_y):
        rows.append(("s0", t, 0, y))
    for t, y in enumerate(b1_y):
        rows.append(("s1", t, 1, y))
    df = pl.DataFrame(rows, schema=["unique_id", "ds", "store", "y"], orient="row")

    want_b0 = [5.0, 5.0, 2.0, 2.0, 2.0, 1.0]
    want_b1 = [None, 10.0, 10.0, 10.0, 3.0, 3.0]

    for backend in BACKENDS:
        src = df if backend == "polars" else df.to_pandas()
        out = grouped_accumulate(src, ["store"], ["y"], "cum_min", ["Amn"])
        o = out if isinstance(out, pl.DataFrame) else pl.from_pandas(out)
        o = o.sort(["store", "ds"])
        got_b0 = o.filter(pl.col("store") == 0)["Amn"].to_list()
        got_b1 = o.filter(pl.col("store") == 1)["Amn"].to_list()
        assert got_b0 == pytest.approx(want_b0), f"{backend}: bucket 0 mismatch"
        assert got_b1[0] is None, (
            f"{backend}: bucket 1 must not inherit bucket 0's tail minimum"
        )
        assert got_b1[1:] == pytest.approx(want_b1[1:]), f"{backend}: bucket 1 mismatch"


@pytest.mark.parametrize("backend", BACKENDS)
def test_ctx_col_respects_time_agg_suffix(backend):  # noqa: ARG001 (parametrized for uniformity; backend-independent)
    plain = PooledCtx(keys=["store"], lag=1, min_samples=7, time_agg=None)
    agg = PooledCtx(keys=["store"], lag=1, min_samples=7, time_agg="sum")
    assert plain.col("Es") == "Es"
    assert agg.col("Es") == "Es__sum"


@pytest.mark.parametrize("backend", BACKENDS)
def test_ctx_window_matches_manual_rolling_sum(backend):
    df = _panel(backend, n_buckets=2, n_times=12, n_series_per_bucket=1)
    tbl = build_agg_table(df, ["store"], "ds", "y", {None})
    ctx = PooledCtx(keys=["store"], lag=1, min_samples=1, time_agg=None)
    t = nw.from_native(tbl, eager_only=True)
    got = t.with_columns(ctx.window("s", 3).alias("w3")).to_native()
    o = got if isinstance(got, pl.DataFrame) else pl.from_pandas(got)
    o = o.sort(["store", "ord"])
    for b in (0, 1):
        blk = o.filter(pl.col("store") == b)
        s = blk["s"].to_numpy()
        for i in range(len(s)):
            lo, hi = max(i - 1 - 3 + 1, 0), i - 1 + 1  # ordinals (t-lag-w, t-lag]
            want = s[lo:hi].sum() if hi > lo else 0.0
            assert blk["w3"][i] == pytest.approx(want, abs=1e-12), f"bucket {b} row {i}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_quantile_values_layout_is_flat_csr(backend):
    """DEVIATION from the task-7 brief's Step-1 snippet: that snippet calls
    ``quantile_values(df, ["store"], "ds", "y")`` -- 4 positional args, which
    would bind ``["store"]`` to the ``agg_native`` parameter and drop ``keys``
    entirely. This contradicts the task's own CRITICAL INTERFACE DETAIL
    (``quantile_values(df, agg_native, keys, time_col, target_col)``, with
    ``agg_native`` as the ordinal-grid authority) and Step 4's own caller
    (``quantile_values(self._df, self.agg, self.keys, self.time_col,
    self._target_col)``, 5 args). Fixed here to build the real aggregate
    table and pass it as ``agg_native``, matching the required signature and
    the only caller that actually exists in ``pooled.py``.
    """
    df = _panel(backend, n_buckets=2, n_times=4, n_series_per_bucket=3)
    agg = build_agg_table(df, ["store"], "ds", "y", {None})
    vals, offs = quantile_values(df, agg, ["store"], "ds", "y")
    assert offs[0] == 0
    assert offs[-1] == len(vals)
    assert len(offs) == 2 * 4 + 1, "one offset per (bucket, ordinal), plus a sentinel"
    assert np.diff(offs).tolist() == [3] * 8, "3 series contribute per timestamp"


def test_quantile_feature_matches_numpy_oracle():
    # 1 bucket, 5 ordinals, 2 values each
    vals = np.arange(10, dtype=float)
    offs = np.arange(0, 11, 2, dtype=np.intp)
    got = quantile_feature(vals, offs, 5, offsets=[1, 2], p=0.5, min_samples=4)
    # ordinal 2 sees ordinals 1 and 0 -> values [2,3,0,1]
    assert got[2] == pytest.approx(np.quantile([2.0, 3.0, 0.0, 1.0], 0.5))
    assert np.isnan(got[0]) and np.isnan(got[1]), "warm-up below min_samples"


def _naive_quantile_feature(values, row_offsets, n_ordinals, offsets, p, min_samples):
    """Reference oracle: always gathers per-ordinal chunks and concatenates,
    never takes the single-slice contiguous fast path. Used to prove the
    fix-round-1 performance change (single-slice for contiguous offsets) is
    bit-identical to the always-gather form it replaces, on both contiguous
    (rolling/expanding) and strided (seasonal) offset shapes."""
    out = np.full(n_ordinals, np.nan)
    for t in range(n_ordinals):
        src = [t - o for o in offsets if 0 <= t - o < n_ordinals]
        if not src:
            continue
        chunks = [
            values[row_offsets[i] : row_offsets[i + 1]]
            for i in src
            if row_offsets[i + 1] > row_offsets[i]
        ]
        if not chunks:
            continue
        win = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        if len(win) >= min_samples and len(win) > 0:
            out[t] = np.quantile(win, p, method="linear")
    return out


@pytest.mark.parametrize(
    "offsets,label",
    [
        (list(range(1, 15)), "contiguous_rolling"),
        (list(range(1, 40)), "contiguous_expanding"),
        ([1, 3, 5, 7], "strided_seasonal"),
        ([2, 5, 6, 7], "non_contiguous_arbitrary"),
    ],
)
def test_quantile_feature_contiguous_fast_path_matches_naive_gather(offsets, label):  # noqa: ARG001
    """Fix-round-1 performance change: `quantile_feature` takes a single-slice
    fast path when `offsets` is contiguous (rolling/expanding) instead of
    gathering and `np.concatenate`-ing one chunk per contributing ordinal.
    Proves BIT-IDENTICAL output against the always-gather reference
    (`_naive_quantile_feature`) on a randomized, gappy value store, for both
    a contiguous and a strided offset shape -- not just close, `atol=0.0`,
    since this must be a pure performance change, never a numerical one."""
    rng = np.random.default_rng(0)
    n_ordinals = 60
    # variable, sometimes-zero counts per ordinal (holes), like a real store
    counts = rng.integers(0, 4, size=n_ordinals)
    row_offsets = np.zeros(n_ordinals + 1, dtype=np.intp)
    np.cumsum(counts, out=row_offsets[1:])
    values = rng.normal(size=int(row_offsets[-1]))

    got_fast = quantile_feature(values, row_offsets, n_ordinals, offsets, 0.5, 2)
    got_naive = _naive_quantile_feature(
        values, row_offsets, n_ordinals, offsets, 0.5, 2
    )
    np.testing.assert_array_equal(got_fast, got_naive)


def test_should_densify_threshold():
    from mlforecast._pooled_engine import should_densify

    assert should_densify(n_buckets=2, n_calendar=100, n_sparse_rows=100)
    assert not should_densify(n_buckets=1000, n_calendar=1000, n_sparse_rows=5000)


# ---- Task 9 fix round 1: structural invariant the quantile CSR store needs ----
#
# `_quantile_columns` (mlforecast/pooled.py) indexes `self.agg` POSITIONALLY,
# assuming each bucket's rows occupy a single contiguous row range
# (`base = sel.min()`, then `values[row_offsets[base]:row_offsets[base+n_ord]]`).
# `_rebuild_tail`/`build_query_arrays` build the tail by concatenating
# [seed rows (all buckets), retained history (all buckets), one pending/query
# row per bucket] -- interleaving buckets -- so each sorts the result back to
# bucket-contiguous, ord-monotonic order before returning. Output-equality
# tests (the differential suite, the sqlite oracle) cannot pin this: every
# shipped transform has `lag >= 1`, so no transform ever reads a bucket's own
# CURRENT ordinal, and the two sort sites happen to be mutually compensating
# for that reason -- removing either one alone still passes every committed
# numeric test. This checks the structural invariant directly.


def _assert_bucket_contiguous_ord_monotonic(native, keys, msg=""):
    """Each distinct bucket-key value in ``native`` must appear as exactly
    ONE uninterrupted run of rows (in physical row order), with ``ord``
    non-decreasing within that run.
    """
    df = nw.from_native(native, eager_only=True)
    bucket_arr = (
        df.get_column(keys[0]).to_numpy() if keys else np.zeros(len(df), dtype=np.int64)
    )
    ords = df.get_column("ord").to_numpy()
    seen = set()
    prev_bucket = None
    prev_ord = None
    for b, o in zip(bucket_arr, ords):
        b = int(b)
        if b != prev_bucket:
            assert b not in seen, (
                f"{msg}: bucket {b} is split across more than one run (not contiguous)"
            )
            seen.add(b)
        else:
            assert o >= prev_ord, (
                f"{msg}: ord not monotonic within bucket {b}'s run ({prev_ord} -> {o})"
            )
        prev_bucket, prev_ord = b, o


def _run_predict_and_check_contiguity(backend, n_buckets=3, n_steps=6):
    """Drive a real recursive predict loop against a `RollingQuantile`
    pooled state and check the bucket-contiguous/ord-monotonic invariant
    after every `_rebuild_tail` call (the persisted tail) and every
    `build_query_arrays` call (the query-extended tail `latest_features`
    reads from) at every step.
    """
    import importlib
    import os

    prev = os.environ.get("MLFORECAST_POOLED_ENGINE")
    os.environ["MLFORECAST_POOLED_ENGINE"] = "narwhals"
    try:
        import mlforecast.core
        import mlforecast.pooled

        importlib.reload(mlforecast.pooled)
        importlib.reload(mlforecast.core)
        from mlforecast.core import TimeSeries
        from mlforecast.lag_transforms import RollingQuantile

        n_series_per_bucket, n_times = 2, 20
        df = _panel(
            backend,
            n_buckets=n_buckets,
            n_times=n_times,
            n_series_per_bucket=n_series_per_bucket,
            seed=7,
        )
        tfm = RollingQuantile(p=0.5, window_size=5, groupby=["store"])
        ts = TimeSeries(freq=1, lags=[1], lag_transforms={1: [tfm]})
        ts.fit_transform(
            df,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            static_features=["store"],
            dropna=False,
        )
        ts._predict_setup()
        key = ("groupby", ("store",), ())
        state = ts._pooled_states[key]
        n_series = len(ts.uids)
        for step in range(n_steps):
            ts._update_features()
            _assert_bucket_contiguous_ord_monotonic(
                state.agg, state.keys, f"step {step}: persisted tail (_rebuild_tail)"
            )
            extended = state.build_query_arrays(None, n_series)
            _assert_bucket_contiguous_ord_monotonic(
                extended,
                state.keys,
                f"step {step}: query-extended tail (build_query_arrays)",
            )
            ts._update_y(np.arange(n_series, dtype=float) + step)
            _assert_bucket_contiguous_ord_monotonic(
                state.agg,
                state.keys,
                f"step {step}: persisted tail after append_predictions",
            )
    finally:
        if prev is None:
            os.environ.pop("MLFORECAST_POOLED_ENGINE", None)
        else:
            os.environ["MLFORECAST_POOLED_ENGINE"] = prev
        import mlforecast.core
        import mlforecast.pooled

        importlib.reload(mlforecast.pooled)
        importlib.reload(mlforecast.core)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rebuild_tail_and_query_arrays_keep_buckets_contiguous(backend):
    """`_rebuild_tail` and `build_query_arrays` (mlforecast/pooled.py) must
    each leave `self.agg` bucket-contiguous and ord-monotonic -- the
    invariant `_quantile_columns`'s positional slicing depends on. Proved
    load-bearing (not incidental) by reverting each of the two `.sort(keys +
    ["ord"])` call sites INDEPENDENTLY: see the task-9 report's fix-round-1
    section for the before/after output showing this test fails at each site
    alone (and that the committed differential/oracle suite does NOT, which
    is exactly why a structural check -- not output equality -- is required
    here).

    Parametrized over both data backends like its siblings in this file
    (a prior version hardcoded ``"pandas"``, leaving polars uncovered for
    this invariant -- a reviewer confirmed separately that it holds under
    polars too, so this closes a coverage gap rather than chasing a bug).
    """
    _run_predict_and_check_contiguity(backend, n_buckets=3, n_steps=6)


# ---- Task 9 fix round 1: pin the retention margin's true minimum ----
#
# `_pooled_retention`'s window-family formula (`lag + window_size`) was
# derived analytically (see its docstring) but never pinned by a test that
# would fail if the formula were accidentally loosened AND shrunk by a
# couple of ordinals -- the committed suite's fixtures all carry enough
# spare history that an off-by-one-or-two error goes unnoticed. This proves
# the formula is EXACTLY tight for RollingMean: retention one ordinal
# smaller changes the predicted value, so there is no slack to "clean up".


def test_rolling_mean_retention_formula_margin_is_pinned(monkeypatch):
    """Pins BOTH the formula's actual margin and the true floor for
    RollingMean(window_size=W, lag=L): retention `lag + window_size` (the
    formula) and `lag + window_size - 1` (one row less -- the true minimum,
    thanks to the seed row's own exactness) both give the correct predicted
    value; `lag + window_size - 2` does not. A single bucket, `global_=True`
    scenario with a simple, hand-verifiable ramp target makes the expected
    value and the effect of shrinking retention directly checkable without
    depending on the legacy engine.
    """
    import importlib
    import os

    import pandas as pd

    prev = os.environ.get("MLFORECAST_POOLED_ENGINE")
    os.environ["MLFORECAST_POOLED_ENGINE"] = "narwhals"
    try:
        import mlforecast.core
        import mlforecast.pooled as mp2

        importlib.reload(mp2)
        importlib.reload(mlforecast.core)
        from mlforecast.core import TimeSeries
        from mlforecast.lag_transforms import RollingMean

        lag, window_size = 2, 3
        n_times = 12
        y = np.arange(1.0, n_times + 1.0)  # 1..12, a simple ramp
        df = pd.DataFrame(
            {"unique_id": ["a"] * n_times, "ds": np.arange(n_times), "y": y}
        )
        col = RollingMean(window_size, global_=True)._get_name(lag)

        def _first_predicted_feature(retention_delta):
            ts = TimeSeries(
                freq=1,
                lags=[lag],
                lag_transforms={lag: [RollingMean(window_size, global_=True)]},
            )
            ts.fit_transform(
                df,
                id_col="unique_id",
                time_col="ds",
                target_col="y",
                static_features=[],
                dropna=False,
            )
            ts._predict_setup()
            if retention_delta:
                original = mp2._pooled_retention

                def patched(tfm):
                    r = original(tfm)
                    return None if r is None else max(r - retention_delta, 0)

                monkeypatch.setattr(mp2, "_pooled_retention", patched)
            feats = ts._update_features()
            return float(np.asarray(feats[col], dtype=float)[0])

        # `_pooled_retention`'s formula (`lag + window_size`) matches the
        # hand-computed rolling mean of the window `lag` ordinals back from
        # the first predicted row.
        formula = _first_predicted_feature(0)
        expected = float(
            np.mean(y[n_times - lag - window_size + 1 : n_times - lag + 1])
        )
        assert formula == pytest.approx(expected), (formula, expected)

        # The formula's own claimed minimum is NOT the tightest possible:
        # shrinking by exactly ONE ordinal (to `lag + window_size - 1`)
        # STILL matches. This is not a bug in the test -- it's because the
        # seed row (always emitted when any history precedes the retained
        # suffix) carries the EXACT `Es`/`Ec`/`Eq` cumulative value at its
        # own ordinal, and for this family's very first predict step that
        # ordinal happens to equal exactly the window's `lo` reference, so
        # the seed satisfies it without needing one more retained raw row.
        # See `_pooled_retention`'s docstring for why the formula keeps this
        # one extra row of (here, provably inert) margin anyway.
        true_min = _first_predicted_feature(1)
        assert true_min == pytest.approx(expected), (true_min, expected)

        # One ordinal past the TRUE minimum: the seed's own ordinal is now
        # one *after* the window's `lo` reference, which is unreachable (no
        # row precedes the seed) -- `ctx.window`'s `lo` shift resolves to
        # null, filled to 0.0 ("no prior data"), and the predicted value
        # must change.
        too_short = _first_predicted_feature(2)
        assert too_short != pytest.approx(expected), (
            "shrinking retention two ordinals below the formula (one past "
            "the true minimum) left the predicted value unchanged -- either "
            "this fixture no longer exercises the boundary, or the formula "
            "has MORE untested slack than documented"
        )
    finally:
        if prev is None:
            os.environ.pop("MLFORECAST_POOLED_ENGINE", None)
        else:
            os.environ["MLFORECAST_POOLED_ENGINE"] = prev
        import mlforecast.core
        import mlforecast.pooled

        importlib.reload(mlforecast.pooled)
        importlib.reload(mlforecast.core)


# ---- Task 10 fix round 1: `history_warmup` must densify before predict ----


def test_history_warmup_partition_by_densifies_before_first_predict():
    """`history_warmup` skips computing training features entirely (its own
    docstring: "skips materializing the features dataframe"), so a
    `partition_by` state's first densification decision (`ensure_densified`,
    deliberately deferred by Task 8 until the actual transform set is known)
    used to fire for the first time from INSIDE `latest_features`, mid-predict,
    after `self._df` had already been swapped to `_tail_df`'s reduced schema.

    This crashed outright on a missing scope column (a `local`-mode state
    reads `id_col` via `_parent_scope_cols`) before this fix. Fixing only
    `_dense_skeleton` to read `self.agg` instead of `self._df` (a real, but
    BY ITSELF insufficient, improvement) stopped that crash but exposed a
    deeper ordering bug one layer in: `_make_seeds` would still compute its
    retention cutoffs against the still-SPARSE table's occurrence-count
    ordinals (not yet the eventual dense-calendar ones), and the
    `_apply_densification` rebuild `ensure_densified` triggers mid-
    `feature_frame` would then discard tail bookkeeping columns (e.g.
    `_is_query`) built against that stale shape -- a narwhals `ValueError`
    on exactly that column. `_initialize_lag_transform_states` (called by
    `history_warmup`) now settles both `ensure_time_aggs` and
    `ensure_densified` eagerly, in the same relative order `feature_frame`
    itself uses, reproducing the ordering invariant a normal `fit_transform`
    already guarantees before any predict-time code runs.

    Mirrors `tests/test_history_warmup.py::test_history_warmup_partition_by`
    (which may not be modified) but stands on its own here so this file's
    own regression suite pins the fix directly.
    """
    import importlib
    import os

    import pandas as pd
    from sklearn.linear_model import LinearRegression

    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean

    prev = os.environ.get("MLFORECAST_POOLED_ENGINE")
    os.environ["MLFORECAST_POOLED_ENGINE"] = "narwhals"
    try:
        import mlforecast.core
        import mlforecast.pooled

        importlib.reload(mlforecast.pooled)
        importlib.reload(mlforecast.core)

        n_times = 20
        df = pd.DataFrame(
            {
                "unique_id": ["a"] * n_times,
                "ds": pd.date_range("2020-01-01", periods=n_times, freq="D"),
                "y": np.arange(1.0, n_times + 1.0),
            }
        )
        df["promo"] = df["ds"].dt.dayofweek % 2

        def new_fcst():
            return MLForecast(
                models=[LinearRegression()],
                freq="D",
                lags=[1],
                lag_transforms={1: [RollingMean(3, partition_by=["promo"])]},
            )

        fitted = new_fcst()
        fitted.fit(df.copy(), static_features=[])
        h = 4
        last = df["ds"].max()
        x_df = df[df["ds"] > last - pd.Timedelta(days=h)][["unique_id", "ds"]].copy()
        x_df["ds"] = x_df["ds"] + pd.Timedelta(days=h)
        x_df["promo"] = x_df["ds"].dt.dayofweek % 2
        expected = fitted.predict(h, X_df=x_df)

        warmed = new_fcst()
        warmed.models_ = fitted.models_
        warmed.history_warmup(df.copy(), static_features=[])
        actual = warmed.predict(h, X_df=x_df)

        np.testing.assert_allclose(
            expected["LinearRegression"].to_numpy(),
            actual["LinearRegression"].to_numpy(),
        )
    finally:
        if prev is None:
            os.environ.pop("MLFORECAST_POOLED_ENGINE", None)
        else:
            os.environ["MLFORECAST_POOLED_ENGINE"] = prev
        import mlforecast.core
        import mlforecast.pooled

        importlib.reload(mlforecast.pooled)
        importlib.reload(mlforecast.core)
