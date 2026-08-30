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


from ._pooled_engine_env import pooled_engine  # noqa: E402

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
    for c in [
        "store",
        "ds",
        "ord",
        "s",
        "c",
        "sK",
        "qK",
        "mn",
        "mx",
        "Es",
        "Ec",
        "EsK",
        "EqK",
    ]:
        assert c in o.columns
    # the zero-centred `sum(y**2)` is GONE, replaced by the shifted moments:
    # nothing may reintroduce it (it is what made the std families lose the
    # variance to cancellation at large magnitude).
    assert "q" not in o.columns and "Eq" not in o.columns
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
    with pooled_engine("narwhals"):
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


def test_rolling_mean_retention_formula_margin_is_pinned():
    """Pins BOTH the formula's actual margin and the true floor for
    RollingMean(window_size=W, lag=L): retention `lag + window_size` (the
    formula) and `lag + window_size - 1` (one row less -- the true minimum,
    thanks to the seed row's own exactness) both give the correct predicted
    value; `lag + window_size - 2` does not. A single bucket, `global_=True`
    scenario with a simple, hand-verifiable ramp target makes the expected
    value and the effect of shrinking retention directly checkable without
    depending on the legacy engine.
    """
    import pandas as pd

    with pooled_engine("narwhals"):
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
                # Plain assignment, not `monkeypatch.setattr`: `pooled_engine`
                # restores `mlforecast.pooled.__dict__` wholesale on exit, and
                # monkeypatch's own teardown runs AFTER that -- it would put a
                # reloaded-generation function back into the restored module,
                # re-introducing exactly the residue this file stopped leaving.
                original = mp._pooled_retention

                def patched(tfm, _original=original, _d=retention_delta):
                    r = _original(tfm)
                    return None if r is None else max(r - _d, 0)

                mp._pooled_retention = patched
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
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean

    with pooled_engine("narwhals"):
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


# ---- Task 10 fix round 2: pin `_dense_skeleton`'s `self.agg`-not-`self._df` ----
# invariant INDEPENDENTLY of `_initialize_lag_transform_states`'s eager-settle
# ordering fix (fix round 1's "option 1"). Re-review measured that with option
# 1 in place, `_dense_skeleton` never actually runs against a swapped/reduced
# `self._df` through any real call path today (`ensure_densified` freezes its
# decision on first call, and option 1 forces that first call before
# `self._df` is ever swapped in `latest_features`) -- so reverting the
# `self.agg`-vs-`self._df` read alone ("option 2") produces ZERO diff across
# the whole suite, including the 510-case SQL oracle. This is the same "each
# fix independently masks the other's absence" trap as the two `.sort()` call
# sites pinned earlier in this plan. This test calls `_dense_skeleton`
# directly against a state whose `self._df` has been manually reduced to
# `_tail_df`'s exact schema, bypassing `_initialize_lag_transform_states`/
# `ensure_densified` entirely -- so it fails if the `self.agg`-vs-`self._df`
# fix is reverted, REGARDLESS of whether the ordering fix is in place.


def test_dense_skeleton_correct_when_df_reduced_to_tail_schema():
    """`_dense_skeleton` must derive its (bucket, time) grid from `self.agg`
    (joined with `self.groups` for the scoped branch), never from
    `self._df` -- the exact channel `latest_features` temporarily reduces
    to `_tail_df`'s schema (`keys + [time_col, target_col]`) for the
    duration of one tail evaluation, dropping e.g. a `local`-mode state's
    own scope column (`id_col`) entirely.

    Manually reduces `self._df` to that exact schema BEFORE calling
    `_dense_skeleton` -- bypassing `_initialize_lag_transform_states`'s
    eager-settle fix (fix round 1's "option 1") entirely, so this pins the
    `self.agg`-vs-`self._df` read (fix round 1's "option 2")
    independently of it. See the module comment above for why that
    independence is not exercised by any other test in this suite.
    """
    from mlforecast.pooled import NarwhalsPooledState

    df = pl.DataFrame(
        {
            "unique_id": ["a", "a", "a", "b", "b"],
            "ds": [1, 2, 3, 1, 2],
            "y": [10.0, 20.0, 30.0, 100.0, 200.0],
            "promo": [0, 1, 0, 0, 1],
        }
    )
    state = NarwhalsPooledState.from_partition(
        df,
        mode="local",
        group_cols_list=None,
        partition_cols_list=["promo"],
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        ga_data_dtype=np.float64,
        static_features=pl.DataFrame({"unique_id": ["a", "b"]}),
        n_series=2,
    )
    assert state._parent_scope_cols == ["unique_id"]
    assert not state._densified and not state._densify_declined  # not yet settled

    # Ground truth, computed from the ORIGINAL (unreduced) input: each
    # `local`-mode scope is one series' own calendar.
    expected_calendar_by_scope = {"a": {1, 2, 3}, "b": {1, 2}}
    groups = nw.from_native(state.groups, eager_only=True)
    scope_by_bucket = dict(
        zip(
            groups.get_column("_bucket_id").to_numpy(),
            groups.get_column("unique_id").to_numpy(),
        )
    )

    # Simulate `latest_features`'s swap: reduce `self._df` to `_tail_df`'s
    # EXACT schema (`keys + [time_col, target_col]`) -- dropping `unique_id`
    # (the scope column this state's `_dense_skeleton` needs) entirely.
    reduced = (
        nw.from_native(state._df, eager_only=True)
        .select(state.keys + [state.time_col, state._target_col])
        .to_native()
    )
    assert "unique_id" not in nw.from_native(reduced, eager_only=True).columns
    state._df = reduced

    skeleton = nw.from_native(state._dense_skeleton(), eager_only=True)
    bids = skeleton.get_column("_bucket_id").to_numpy()
    times = skeleton.get_column("ds").to_numpy()
    actual_by_bucket: dict = {}
    for b, t in zip(bids, times):
        actual_by_bucket.setdefault(int(b), set()).add(int(t))

    assert set(actual_by_bucket) == set(scope_by_bucket)
    for bid, times_seen in actual_by_bucket.items():
        scope = scope_by_bucket[bid]
        assert times_seen == expected_calendar_by_scope[scope], (
            f"bucket {bid} (scope {scope!r}): got {times_seen}, "
            f"expected {expected_calendar_by_scope[scope]}"
        )


# ---------------------------------------------------------------------------
# Exact (arbitrary-precision) variance oracle for the three std families.
#
# Legacy is NOT the reference here. Its own aggregate fast paths
# (`_rolling_std_from_agg` / `_expanding_std_from_agg`) compute the variance
# as `sum(x**2) - sum(x)**2/n`, which is algebraically right and numerically
# catastrophic once the mean dwarfs the spread -- so "agrees with legacy"
# would pin the defect, not the fix. float64 values ARE exact rationals, so
# `Fraction` gives the true sample variance with no rounding at all, and
# `Decimal.sqrt` at 60 digits turns it into a std we can measure a relative
# error against.
# ---------------------------------------------------------------------------

import math  # noqa: E402
from decimal import Decimal, localcontext  # noqa: E402
from fractions import Fraction  # noqa: E402

from mlforecast.lag_transforms import (  # noqa: E402
    ExpandingStd,
    RollingStd,
    SeasonalRollingStd,
)

# MEASURED, then given ~3x headroom for a backend's own summation order (the
# two backends already differ by ~1.7x below). Worst relative error observed
# over the full sweep, per family/backend:
#   expanding polars 2.35e-16   expanding pandas 2.04e-16
#   rolling   polars 1.56e-15   rolling   pandas 9.32e-16
#   seasonal  polars 2.70e-16   seasonal  pandas 2.83e-16
# against a worst NAIVE (pre-fix formula) error of 4.9e+01 .. 6.1e+01 on the
# very same windows -- i.e. the pre-fix engine was wrong by 5000%, not by a
# few ULP.
_STD_REL_TOL = 5e-15

_MAGNITUDES = (1e6, 1e9, 1e11)
_REL_SPREADS = (1e-2, 1e-4, 1e-6, 1e-9)


def _exact_sample_std(vals):
    """The TRUE sample std (ddof=1) of these float64 values, 60 digits.

    float64 is a subset of the rationals, so `Fraction(v)` is exact and the
    two-pass `sum((x - mean)**2)` below carries no rounding whatsoever.
    """
    fr = [Fraction(v) for v in vals]
    n = len(fr)
    mean = sum(fr) / n
    var = sum((x - mean) ** 2 for x in fr) / (n - 1)
    if var == 0:
        return Decimal(0)
    with localcontext() as ctx:
        ctx.prec = 60
        return (Decimal(var.numerator) / Decimal(var.denominator)).sqrt()


def _naive_sample_std(vals):
    """The two-moment formula this fix replaces: `(sum(x^2) - sum(x)^2/n)/(n-1)`.

    Used only to PROVE the fixture reaches the ill-conditioned regime -- a
    sweep where the naive formula is already accurate would pass vacuously.
    """
    a = np.asarray(vals, dtype=np.float64)
    n = a.size
    var = (float((a * a).sum()) - float(a.sum()) ** 2 / n) / (n - 1)
    return math.sqrt(max(var, 0.0))


def _rel_err(got, exact_dec):
    if exact_dec == 0:
        return 0.0 if got == 0 else float("inf")
    with localcontext() as ctx:
        ctx.prec = 60
        return float(abs(Decimal(float(got)) - exact_dec) / exact_dec)


def _std_panel(backend, mag, rel_spread, n_times=24, seed=0):
    """Two buckets, ONE series each, at magnitudes 1000x apart.

    One series per bucket makes each timestamp contribute exactly one value,
    so a window over `w` ordinals is a window over `w` raw values and the
    oracle below needs no re-derivation of the pooled aggregation.

    The 1000x magnitude gap between the buckets is deliberate: a single
    GLOBAL centring reference would be ~500x bucket 1's own scale and would
    fail this test outright, so it pins the reference as PER-BUCKET.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for b, scale in ((0, mag), (1, mag / 1000.0)):
        u = rng.uniform(-1.0, 1.0, n_times)
        y = scale * (1.0 + rel_spread * u)
        for t in range(n_times):
            rows.append((f"b{b}", t, b, float(y[t])))
    df = pl.DataFrame(rows, schema=["unique_id", "ds", "store", "y"], orient="row")
    return df if backend == "polars" else df.to_pandas()


def _window_ordinals(family, t, n_ordinals):
    """The ordinals contributing to the feature at ordinal ``t`` (lag=1)."""
    if family == "rolling":
        lo, hi = t - 1 - 6 + 1, t - 1  # window_size=6
    elif family == "expanding":
        lo, hi = 0, t - 1
    else:  # seasonal: season_length=3, window_size=4 -> offsets 1, 4, 7, 10
        return [t - o for o in (1, 4, 7, 10) if 0 <= t - o < n_ordinals]
    return [i for i in range(max(lo, 0), hi + 1) if 0 <= i < n_ordinals]


_STD_TFMS = {
    "rolling": lambda: RollingStd(6, min_samples=2, groupby=["store"]),
    "expanding": lambda: ExpandingStd(groupby=["store"]),
    "seasonal": lambda: SeasonalRollingStd(
        season_length=3, window_size=4, min_samples=2, groupby=["store"]
    ),
}


def _std_feature_by_bucket(df, tfm, min_samples):
    """``{bucket: [feature at ordinal 0, 1, ...]}`` straight off the engine."""
    tbl = build_agg_table(df, ["store"], "ds", "y", {None})
    ctx = PooledCtx(keys=["store"], lag=1, min_samples=min_samples, time_agg=None)
    out = (
        nw.from_native(tbl, eager_only=True)
        .with_columns(tfm._pooled_expr(ctx).alias("_v"))
        .to_native()
    )
    o = out if isinstance(out, pl.DataFrame) else pl.from_pandas(out)
    o = o.sort(["store", "ord"])
    return {
        int(b): o.filter(pl.col("store") == b)["_v"].cast(pl.Float64).to_list()
        for b in (0, 1)
    }


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("family", sorted(_STD_TFMS))
def test_std_families_match_the_exact_oracle(backend, family):
    """Every std family, every magnitude x spread, against the EXACT variance.

    The assertion is two-sided on purpose:

    * the engine's relative error must stay under `_STD_REL_TOL`, and
    * the naive `sum(x^2) - sum(x)^2/n` formula must, somewhere in the same
      sweep, be catastrophically wrong on the SAME windows -- otherwise the
      sweep never reaches the regime this fix exists for and would pass
      against the pre-fix code.
    """
    min_samples = 2
    worst_engine = 0.0
    worst_engine_case = None
    worst_naive = 0.0
    checked = 0
    for mag in _MAGNITUDES:
        for rel_spread in _REL_SPREADS:
            df = _std_panel(backend, mag, rel_spread)
            raw = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
            raw = raw.sort(["store", "ds"])
            by_bucket = {
                int(b): raw.filter(pl.col("store") == b)["y"].to_list() for b in (0, 1)
            }
            got = _std_feature_by_bucket(df, _STD_TFMS[family](), min_samples)
            for b, vals in by_bucket.items():
                n_ord = len(vals)
                for t in range(n_ord):
                    idxs = _window_ordinals(family, t, n_ord)
                    if len(idxs) < max(min_samples, 2):
                        continue
                    win = [vals[i] for i in idxs]
                    exact = _exact_sample_std(win)
                    assert got[b][t] is not None, (
                        f"{family} {mag:g}/{rel_spread:g} bucket {b} ord {t}: "
                        "engine returned null where the window is full"
                    )
                    err = _rel_err(got[b][t], exact)
                    checked += 1
                    if err > worst_engine:
                        worst_engine, worst_engine_case = err, (mag, rel_spread, b, t)
                    worst_naive = max(
                        worst_naive, _rel_err(_naive_sample_std(win), exact)
                    )
    assert checked > 0, "sweep never evaluated a full window"
    assert worst_naive > 0.5, (
        "precondition failed: the naive two-moment formula is still accurate "
        f"on this sweep (worst relative error {worst_naive:g}) -- the test "
        "would pass against the pre-fix code"
    )
    assert worst_engine < _STD_REL_TOL, (
        f"{family}/{backend}: worst relative error {worst_engine:g} at "
        f"(mag, rel_spread, bucket, ord)={worst_engine_case} "
        f"exceeds {_STD_REL_TOL:g} over {checked} windows"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_time_agg_family_has_its_own_centring_reference(backend):
    """One reference per column family, each on ITS OWN scale.

    ``compute_kref`` must cover every value ``time_agg`` accepts -- a family
    with no reference would centre on 0 (i.e. keep the defect) or, worse,
    fail at build time -- and the references must NOT all be the target's
    magnitude. ``count`` is the case that proves it: its column holds the
    number of rows per timestamp (3 here), so reusing the target's reference
    (1e11) for it would manufacture exactly the cancellation the shift
    exists to remove.
    """
    from mlforecast._pooled_engine import _KREF_SUFFIXES, compute_kref
    from mlforecast.lag_transforms import _TIME_AGGS

    assert set(_KREF_SUFFIXES) == {""} | {f"__{a}" for a in _TIME_AGGS}, (
        "a time_agg exists with no centring reference"
    )

    n_times, per_ts = 10, 3
    rows = [
        (f"s{j}", t, 0, 1e11 + t + j) for t in range(n_times) for j in range(per_ts)
    ]
    df = pl.DataFrame(rows, schema=["unique_id", "ds", "store", "y"], orient="row")
    df = df if backend == "polars" else df.to_pandas()
    kref = compute_kref(df, ["store"], "ds", "y")
    assert kref.columns == ["store"] + [f"K{s}" for s in _KREF_SUFFIXES]
    got = {c: float(kref.get_column(c).to_numpy()[0]) for c in kref.columns[1:]}
    assert got["K__count"] == pytest.approx(per_ts), (
        f"the count family must be centred on the row count, got {got['K__count']}"
    )
    assert got["K__sum"] == pytest.approx(per_ts * got["K"], rel=1e-9), (
        "the sum family must be centred on the per-timestamp SUM's scale"
    )
    for fam in ("K", "K__mean", "K__min", "K__max"):
        assert got[fam] == pytest.approx(1e11, rel=1e-6), (fam, got[fam])
