"""narwhals expression machinery for the pooled lag-transform engine.

Everything narwhals cannot express on both backends is confined to this module
and named explicitly. See the design spec, section 4.
"""

from dataclasses import dataclass
from typing import List, Optional

import narwhals as nw
import numpy as np
import polars as pl

from ._pooled_keys import _order_preserving_left_join

_BASE_AGGS = ("s", "c", "q", "mn", "mx")
_PREFIX_AGGS = ("s", "c", "q")

# narwhals rejects non-elementary `.over(partition_by)` on pandas-like backends
# ("Only elementary expressions are supported"). `shift` is elementary; the
# accumulate family is not. Both backends support these natively, so dispatch.
_PANDAS_OPS = {"cum_sum": "cumsum", "cum_min": "cummin", "cum_max": "cummax"}


def grouped_accumulate(frame_native, keys, cols, op, out_names, **kw):
    """Per-group accumulate, dispatched to each backend's native implementation.

    ``op`` is one of ``cum_sum``, ``cum_min``, ``cum_max``, ``ewm_mean``. The
    frame must already be sorted by ``keys`` plus its ordering column.
    ``keys`` must be non-empty -- both backends raise on an empty grouping key
    list, so this is rejected up front with a clear message. (Callers in
    ``global_`` mode, where there is a single implicit bucket, must bypass
    this function entirely rather than call it with ``keys=[]``.)

    Prefix sums MUST be per-group: a global accumulate with a per-group baseline
    subtracted drifts to 6.85e-10 on a 146k-row table and fails the pooled
    suite's atol=1e-10.

    For ``op == "ewm_mean"``, ``alpha``, ``adjust``, and ``ignore_nulls`` must
    all be passed explicitly in ``**kw``. No soft defaults: the two backends
    disagree on BOTH ewm defaults, and each disagreement silently produces
    wrong numbers rather than an error.
      * ``adjust``: polars' and pandas' native default is ``True``, but the
        legacy ``_ewm_from_agg`` fold is UNADJUSTED -> callers must pass
        ``False``.
      * ``ignore_nulls``: polars' native default is ``False``, which yields
        ``3.1667`` where legacy yields ``3.0`` on ``[1, null, 3, 4]`` with
        ``alpha=0.5``. Legacy skips missing entries, i.e. ignore-nulls
        semantics -> callers must pass ``True``.
    Both backends also forward-fill the per-ordinal EWM value within each
    bucket: legacy assigns the running EWM at EVERY ordinal, updating only
    where an observation exists, so gap ordinals carry the previous value.
    polars leaves those rows null without an explicit forward-fill; pandas
    already carries forward (the ffill there is then a no-op, kept for
    parity/documentation). Verified: with ``ignore_nulls=True`` plus this
    fill, BOTH backends equal the legacy fold exactly on
    ``[1.0, None, 3.0, 4.0, None, 6.0]`` -> ``[1.0, 1.0, 2.0, 3.0, 3.0, 4.5]``.
    """
    if op not in _PANDAS_OPS and op != "ewm_mean":
        raise ValueError(
            f"unsupported accumulate op {op!r}; "
            f"expected one of {sorted(set(_PANDAS_OPS) | {'ewm_mean'})}"
        )
    if len(cols) != len(out_names):
        raise ValueError("cols and out_names must be the same length")
    if not keys:
        raise ValueError(
            "grouped_accumulate requires a non-empty `keys`; callers in "
            "global_ mode (a single implicit bucket) must bypass this "
            "function rather than call it with keys=[]"
        )
    if op == "ewm_mean":
        # No soft defaults: the two backends disagree on BOTH ewm defaults, and
        # each disagreement silently produces wrong numbers rather than an error.
        #   * `adjust`: polars' native default is True, pandas' is True, but the
        #     legacy `_ewm_from_agg` fold is UNADJUSTED -> callers must pass False.
        #   * `ignore_nulls`: polars' default is False, which yields 3.1667 where
        #     legacy yields 3.0 on [1, null, 3, 4]. Legacy skips missing entries,
        #     i.e. ignore-nulls semantics -> callers must pass True.
        # Require all three explicitly and pass identical values to both branches.
        for required in ("alpha", "adjust", "ignore_nulls"):
            if required not in kw:
                raise ValueError(
                    f"grouped_accumulate(op='ewm_mean') requires an explicit "
                    f"{required!r}; a per-backend default would diverge silently."
                )

    if isinstance(frame_native, pl.DataFrame):
        if op == "ewm_mean":
            # forward_fill: legacy assigns the running EWM at EVERY ordinal,
            # updating only where an observation exists, so gap ordinals carry
            # the previous value. polars leaves them null; pandas already carries
            # forward (the ffill is then a no-op). Verified: with ignore_nulls=True
            # plus this fill, BOTH backends equal the legacy fold exactly.
            exprs = [
                pl.col(c).ewm_mean(**kw).forward_fill().over(keys).alias(o)
                for c, o in zip(cols, out_names)
            ]
        else:
            # forward_fill for cum_min/cum_max too: legacy uses
            # np.fmin.accumulate / np.fmax.accumulate, which IGNORE NaN and so
            # carry the running extremum THROUGH a gap ordinal. polars'
            # cum_min/cum_max and pandas' cummin/cummax leave that position
            # null, emitting a null feature exactly one row after any gap.
            # Same defect class as the ewm path.
            exprs = [
                getattr(pl.col(c), op)().forward_fill().over(keys).alias(o)
                for c, o in zip(cols, out_names)
            ]
        return frame_native.with_columns(exprs)

    # pandas-like
    out = frame_native.copy()
    gb = out.groupby(keys, sort=False, observed=True)
    if op == "ewm_mean":
        alpha, adjust, ignore_na = kw["alpha"], kw["adjust"], kw["ignore_nulls"]
        for c, o in zip(cols, out_names):
            out[o] = gb[c].transform(
                lambda s: s.ewm(alpha=alpha, adjust=adjust, ignore_na=ignore_na)
                .mean()
                .ffill()
            )
    else:
        method = _PANDAS_OPS[op]
        acc = getattr(gb[list(cols)], method)()
        for c, o in zip(cols, out_names):
            out[o] = acc[c].to_numpy()
        # forward-fill within each group, same reason as the polars branch
        out[list(out_names)] = out.groupby(keys, sort=False, observed=True)[
            list(out_names)
        ].ffill()
    return out


def _time_agg_value_expr(time_agg):
    """The single value ``v_t`` a timestamp contributes under ``time_agg``.

    Mirrors ``_pooled_legacy._time_agg_values``: an all-null timestamp is
    unobserved for sum/mean/min/max (null), while ``count`` treats 0 as a real
    observation.
    """
    if time_agg == "count":
        return nw.col("c")
    if time_agg == "sum":
        return nw.when(nw.col("c") > 0).then(nw.col("s"))
    if time_agg == "mean":
        return nw.when(nw.col("c") > 0).then(nw.col("s") / nw.col("c"))
    if time_agg == "min":
        return nw.col("mn")
    if time_agg == "max":
        return nw.col("mx")
    raise ValueError(f"unknown time_agg {time_agg!r}")


def build_agg_table(df, keys, time_col, target_col, time_aggs):
    """One row per (bucket, timestamp) with aggregates and per-bucket prefix sums.

    ``time_aggs`` is the set of ``time_agg`` values used by the transforms in
    this state (``None`` for raw-row transforms). Each non-None value gets its
    own suffixed column family, since one state may hold transforms with
    different ``time_agg``s.

    Every aggregation below is a bare sum/count/min/max. A composite expression
    inside ``.agg()`` makes narwhals abandon pandas' native group-by for a
    per-group apply -- measured 250x slower (1.628s -> 0.006s) -- and the
    warning is invisible on the polars backend. Derived columns are therefore
    materialized first.
    """
    keys = list(keys)
    d = nw.from_native(df, eager_only=True)
    # NaN IS NOT NULL. On polars, `sum()` over a group containing NaN returns
    # NaN -- which then poisons that bucket's ENTIRE prefix sum -- and `count()`
    # counts the NaN as present. The legacy engine treats NaN as MISSING
    # (`_build_ts_aggs` masks with `~np.isnan(y_b)` before summing). Normalize
    # NaN -> null ONCE, before any aggregation or derived column, so every
    # aggregate below inherits the legacy engine's missing-value semantics.
    # Cast first: `is_nan()` is only valid on float dtypes.
    d = d.with_columns(nw.col(target_col).cast(nw.Float64).alias(target_col))
    d = d.with_columns(
        nw.when(nw.col(target_col).is_nan())
        .then(None)
        .otherwise(nw.col(target_col))
        # `.then(None)` on the pandas-like backend produces an object-dtype
        # column (float 1.0 next to python `None`), which then breaks native
        # pandas groupby cumsum/cummin/cummax ("not implemented for dtype
        # object"). Casting back to Float64 restores a real float column
        # where pandas' own null representation (NaN) carries the meaning
        # "missing" -- which is exactly what we want after this point, since
        # every aggregation below already ignores nulls.
        .cast(nw.Float64)
        .alias(target_col)
    )
    d = d.with_columns((nw.col(target_col) ** 2).alias("_y2"))
    tbl = (
        d.group_by(keys + [time_col])
        .agg(
            nw.col(target_col).sum().alias("s"),
            nw.col(target_col).count().alias("c"),
            nw.col("_y2").sum().alias("q"),
            nw.col(target_col).min().alias("mn"),
            nw.col(target_col).max().alias("mx"),
        )
        .sort(keys + [time_col])
    )
    # integer ordinal per bucket over its own sorted calendar
    tbl = tbl.with_columns(nw.col("c").cast(nw.Float64))
    tbl = _add_ordinals(tbl, keys, time_col)

    derived, prefix_cols, prefix_outs = [], [], []
    for a in sorted(x for x in time_aggs if x is not None):
        v = _time_agg_value_expr(a)
        obs = v.is_null().__invert__()
        derived += [
            nw.when(obs).then(v).otherwise(0.0).alias(f"s__{a}"),
            obs.cast(nw.Float64).alias(f"c__{a}"),
            nw.when(obs).then(v * v).otherwise(0.0).alias(f"q__{a}"),
            v.alias(f"mn__{a}"),
            v.alias(f"mx__{a}"),
        ]
    if derived:
        tbl = tbl.with_columns(derived)

    # `ewm` is the per-timestamp mean of the bucket -- the value `_ewm_from_agg`
    # folds over. Not part of the prefix-sum family below (it's consumed via
    # `ensure_accumulates`'s `ewm_mean` shim, not a cumulative sum), so it is
    # materialized here, once for the raw table and once per `time_agg` family
    # (mirroring the "mean" derivation already used for the `time_agg="mean"`
    # column: EWM's own `time_agg` defaults to "mean", so most callers select
    # the suffixed variant, not this bare one).
    tbl = tbl.with_columns(
        nw.when(nw.col("c") > 0).then(nw.col("s") / nw.col("c")).alias("ewm")
    )
    for a in sorted(x for x in time_aggs if x is not None):
        tbl = tbl.with_columns(
            nw.when(nw.col(f"c__{a}") > 0)
            .then(nw.col(f"s__{a}") / nw.col(f"c__{a}"))
            .alias(f"ewm__{a}")
        )

    for suffix in [""] + [
        f"__{a}" for a in sorted(x for x in time_aggs if x is not None)
    ]:
        for base in _PREFIX_AGGS:
            prefix_cols.append(f"{base}{suffix}")
            prefix_outs.append(f"E{base}{suffix}")

    native = tbl.to_native()
    if keys:
        native = grouped_accumulate(native, keys, prefix_cols, "cum_sum", prefix_outs)
    else:
        n = nw.from_native(native, eager_only=True)
        native = n.with_columns(
            [nw.col(c).cum_sum().alias(o) for c, o in zip(prefix_cols, prefix_outs)]
        ).to_native()
    return native


def _add_ordinals(tbl, keys, time_col):
    """Dense 0-based ordinal per bucket over its own sorted distinct timestamps.

    The frame is already sorted by ``keys + [time_col]`` and has one row per
    (bucket, timestamp), so the ordinal is the within-bucket row number. Uses a
    cumulative count -- not a Python loop over buckets.
    """
    native = tbl.to_native()
    if not keys:
        return nw.from_native(native, eager_only=True).with_columns(
            (nw.col(time_col).cum_count() - 1).cast(nw.Int64).alias("ord")
        )
    native = grouped_accumulate(
        nw.from_native(native, eager_only=True)
        .with_columns(nw.lit(1.0).alias("_one"))
        .to_native(),
        keys,
        ["_one"],
        "cum_sum",
        ["ord"],
    )
    return (
        nw.from_native(native, eager_only=True)
        .with_columns((nw.col("ord") - 1).cast(nw.Int64).alias("ord"))
        .drop("_one")
    )


@dataclass
class PooledCtx:
    """Everything a transform needs to build its pooled expression.

    ``keys`` is empty in ``global_`` mode (a single bucket), in which case the
    window helpers take the un-partitioned branch -- ``over([])`` is invalid.
    """

    keys: List[str]
    lag: int
    min_samples: int
    time_agg: Optional[str] = None

    @property
    def _suffix(self) -> str:
        return "" if self.time_agg is None else f"__{self.time_agg}"

    def col(self, base: str) -> str:
        """Resolve a base column name to its ``time_agg`` variant."""
        return f"{base}{self._suffix}"

    def shift(self, base: str, k: int) -> nw.Expr:
        """``base`` shifted ``k`` ordinals back, within the bucket."""
        e = nw.col(self.col(base)).shift(k)
        return e.over(self.keys) if self.keys else e

    def window(self, base: str, w: Optional[int]) -> nw.Expr:
        """Per-bucket sum of ``base`` over ordinals ``(t - lag - w, t - lag]``.

        ``w=None`` gives the expanding sum (no lower bound). Derived from the
        prefix-sum column, so cost is O(1) in ``w``.

        Both shifts are null-filled to 0.0: a shift landing before the bucket's
        first ordinal means "no data observed yet", i.e. an empty prefix sum,
        which is 0 -- not null. Without this, `hi - lo` stays null whenever
        `t - lag` itself predates the bucket's history (e.g. row 0 with
        lag=1), even though the true windowed sum there is 0.
        """
        hi = self.shift(f"E{base}", self.lag).fill_null(0.0)
        if w is None:
            return hi
        lo = self.shift(f"E{base}", self.lag + w).fill_null(0.0)
        return hi - lo


def quantile_values(df, agg_native, keys, time_col, target_col):
    """Non-null target values grouped by (bucket, ordinal), flat + offsets.

    Returns ``(values, row_offsets)`` where the values for the aggregate
    table's row ``i`` are ``values[row_offsets[i]:row_offsets[i + 1]]``. One
    contiguous slice per row, so a window is a small set of slice ranges and
    needs no mask scan.

    ``agg_native`` is the ORDINAL-GRID AUTHORITY: ``row_offsets`` has one
    entry per row of ``agg_native`` (plus a trailing sentinel), IN
    ``agg_native``'s OWN ROW ORDER -- not recomputed from ``df``. Every raw
    row of ``df`` is matched to its ``(bucket, ord)`` grid cell via a left
    join on ``keys + [time_col]`` against ``agg_native``, so a grid row with
    no matching raw data (a densified hole -- see Task 8) gets an empty
    slice, and the store stays aligned with ``agg_native`` whether or not the
    state was densified for RANGE windows.

    DEVIATION from the plan's Step-3 code: that snippet accepted
    ``agg_native`` but never used it -- it recomputed the (bucket, timestamp)
    grid straight from ``df`` via ``_dense_codes``, which happens to coincide
    with ``agg_native``'s row order today (no state is densified yet in this
    task's scope) but is exactly the misalignment this docstring (and the
    task's own "CRITICAL INTERFACE DETAIL") warns against for Task 8. Fixed
    to actually treat ``agg_native`` as the grid authority.

    Quantiles have no sufficient statistic -- they need the raw values -- so
    this store is built ONLY when a quantile transform is present, and must
    be invalidated (``self._qvalues = None``) whenever ``agg`` changes shape
    (densify, trim, append).
    """
    keys = list(keys)
    grid_cols = keys + [time_col]
    a = nw.from_native(agg_native, eager_only=True)
    n_groups = len(a)
    grid = a.select(grid_cols).with_row_index(name="_qv_grid_idx")

    d = nw.from_native(df, eager_only=True)
    d = d.with_columns(nw.col(target_col).cast(nw.Float64).alias(target_col))
    joined = _order_preserving_left_join(
        d.select(grid_cols + [target_col]), grid, on=grid_cols
    )

    y = joined.get_column(target_col).to_numpy().astype(float)
    codes = joined.get_column("_qv_grid_idx").to_numpy().astype(np.int64)
    valid = ~np.isnan(y)
    codes_v, y_v = codes[valid], y[valid]

    counts = np.bincount(codes_v, minlength=n_groups)
    row_offsets = np.zeros(n_groups + 1, dtype=np.intp)
    np.cumsum(counts, out=row_offsets[1:])
    order = np.argsort(codes_v, kind="stable")
    return y_v[order], row_offsets


def quantile_feature(values, row_offsets, n_ordinals, offsets, p, min_samples):
    """Quantile per ordinal over the union of the ordinals at ``offsets`` back.

    ``offsets`` is the list of ordinal distances contributing to each window --
    ``range(lag, lag + w)`` for rolling, the seasonal strides for seasonal, and
    ``range(lag, n_ordinals)`` for expanding (the caller filters to ``<= t``
    via the ``0 <= t - o`` bound below). Contiguity means each contributing
    ordinal is one slice, so a window is a small list of slices concatenated
    once.
    """
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
