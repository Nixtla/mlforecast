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

# `sK`/`qK` are the SHIFTED moments -- `sum(x - K)` and `sum((x - K)**2)` for
# a per-bucket reference `K` (see `compute_kref`) -- and they REPLACE the
# plain `sum(x**2)` this table used to carry. The variance every std family
# reads is `(sum_win qK - (sum_win sK)**2 / n) / (n - 1)`: algebraically the
# same statistic as `sum(x**2) - sum(x)**2/n`, but both of its terms are
# proportional to the SPREAD rather than to `K**2`, so the catastrophic
# cancellation that made `RollingStd`/`ExpandingStd`/`SeasonalRollingStd`
# silently return 0.0 (or a wildly large value) on large-magnitude,
# small-spread data cannot happen. Measured on 16 values around 1e9 with the
# plain formula: spread ~1 -> 0.000000 against a true std of 0.852011.
_BASE_AGGS = ("s", "c", "sK", "qK", "mn", "mx")
_PREFIX_AGGS = ("s", "c", "sK", "qK")

# One centring reference per column family: `""` is the raw-row family, the
# rest are the `time_agg` families `_time_agg_value_expr` derives. Each family
# needs its OWN reference because its values live on its own scale -- a
# `time_agg="count"` column holds small integers, and centring THAT on the
# target's magnitude (1e9, say) would manufacture exactly the cancellation
# this whole mechanism exists to remove.
_KREF_SUFFIXES = ("", "__sum", "__mean", "__count", "__min", "__max")

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


def apply_accumulate(native, keys, col, op, out, **kw):
    """Single-column accumulate, grouped if ``keys`` else over the whole table.

    Factors out the ``keys``/no-``keys`` (``global_``) branch that
    ``NarwhalsPooledState.ensure_accumulates`` (fit) and the predict tail
    rebuild (``_rebuild_tail``, Task 9) both need for one ``(col, op)`` pair
    at a time.
    """
    if keys:
        return grouped_accumulate(native, keys, [col], op, [out], **kw)
    n = nw.from_native(native, eager_only=True)
    return n.with_columns(
        getattr(nw.col(col), op)(**kw).fill_null(strategy="forward").alias(out)
    ).to_native()


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


def _normalize_target(d, target_col):
    """NaN -> null on ``target_col``, cast to Float64. See ``build_agg_table``."""
    d = d.with_columns(nw.col(target_col).cast(nw.Float64).alias(target_col))
    return d.with_columns(
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


def compute_kref(df, keys, time_col, target_col):
    """One row per bucket holding the frozen centring reference ``K`` per family.

    The variance of a window is computed as
    ``(sum qK - (sum sK)**2 / n) / (n - 1)`` where ``sK``/``qK`` are the first
    and second moments of ``x - K``. ANY ``K`` is algebraically correct; the
    numerics only care that ``K`` sits near the data, so that both terms scale
    with the SPREAD instead of with the magnitude. This picks the bucket mean.

    THE REFERENCE MUST BE FROZEN AND CARRIED, never re-derived from whatever
    rows the table happens to hold later. ``trim_to_last`` drops a prefix and
    ``append_observations`` adds a suffix; either would move a re-derived mean
    and thereby silently invalidate every ``qK`` already stored against the old
    one (they are only summable with each other because they share one ``K``).
    That is why this returns a standalone per-bucket frame the state owns
    (``NarwhalsPooledState._kref``) rather than a column on the aggregate
    table, and why every later writer passes the stored frame back in.

    Every family's reference comes out of ONE bare group-by over the raw
    frame -- mean/count/n_unique, all elementary aggregations (a composite
    expression inside ``.agg()`` costs 250x on pandas, see
    ``build_agg_table``):

    * the raw-row family and ``mean``/``min``/``max`` all take values on the
      target's own scale, so the bucket mean ``m`` serves all four;
    * ``sum`` collapses a timestamp to ``sum(y)`` over its rows, i.e. ``m``
      times the mean number of rows per timestamp (``n / n_unique(time)``);
    * ``count`` collapses it to that row count itself, which lives on a
      completely different (small-integer) scale -- centring it on the
      target's magnitude would CREATE catastrophic cancellation where there
      was none.

    All six references are computed unconditionally, whichever ``time_agg``
    families the state happens to need today: they cost one group-by between
    them, and a state that later grows a family (``ensure_time_aggs``) must
    not have to recompute -- and thereby move -- the ones already in use.
    """
    return _kref_from_normalized(
        _normalize_target(nw.from_native(df, eager_only=True), target_col),
        keys,
        time_col,
        target_col,
    )


def _kref_from_normalized(d, keys, time_col, target_col):
    """``compute_kref`` over an ALREADY NaN-normalized frame.

    Split out purely so ``build_agg_and_kref`` can share one normalization
    pass with the table build instead of paying for two: that pass is a
    ``when/then/cast`` over every raw row and costs 30.6 ms per 1.46M rows on
    the pandas backend (1.1 ms on polars), i.e. a third of the whole
    reference computation.
    """
    keys = list(keys)
    aggs = (
        nw.col(target_col).mean().alias("_m"),
        nw.col(target_col).count().alias("_n"),
        nw.col(time_col).n_unique().alias("_nt"),
    )
    g = d.group_by(keys).agg(*aggs) if keys else d.select(*aggs)
    g = g.with_columns(
        nw.col("_m").fill_null(0.0).cast(nw.Float64).alias("_m"),
        nw.col("_n").cast(nw.Float64).alias("_n"),
        nw.col("_nt").cast(nw.Float64).alias("_nt"),
    )
    # rows-per-timestamp; `_nt >= 1` for any group that exists at all, so the
    # guard below only covers a caller handing in an empty frame.
    g = g.with_columns(
        nw.when(nw.col("_nt") > 0)
        .then(nw.col("_n") / nw.col("_nt"))
        .otherwise(0.0)
        .alias("_avgc")
    )
    out = [
        nw.col("_m").alias("K"),
        nw.col("_m").alias("K__mean"),
        nw.col("_m").alias("K__min"),
        nw.col("_m").alias("K__max"),
        (nw.col("_m") * nw.col("_avgc")).alias("K__sum"),
        nw.col("_avgc").alias("K__count"),
    ]
    return g.with_columns(out).select(keys + [f"K{s}" for s in _KREF_SUFFIXES])


def build_agg_and_kref(df, keys, time_col, target_col, time_aggs, kref=None):
    """``(aggregate table, the kref it was built against)``.

    The entry point a state uses at ``_build``: it needs the frozen reference
    back to store, and getting it this way costs one NaN-normalization pass
    over the raw frame instead of two (see ``_kref_from_normalized``).
    """
    d = _normalize_target(nw.from_native(df, eager_only=True), target_col)
    if kref is None:
        kref = _kref_from_normalized(d, keys, time_col, target_col).to_native()
    return _build_agg_impl(d, keys, time_col, target_col, time_aggs, kref), kref


def attach_kref(frame, kref, keys, cols):
    """Broadcast ``kref``'s ``cols`` onto ``frame``, one value per bucket.

    A bucket absent from ``kref`` (only reachable for one that appeared after
    the reference was frozen -- a new ``partition_by`` bucket during predict)
    falls back to ``K = 0``, i.e. exactly the un-centred formula: no better
    than before this fix for that bucket, but self-consistent, which is the
    part that matters (mixing two references within one bucket would make its
    prefix sums meaningless).

    The result is NOT order-preserving -- callers that care re-sort.
    """
    keys = list(keys)
    cols = list(cols)
    kr = nw.from_native(kref, eager_only=True).select(keys + cols)
    out = frame.join(kr, on=keys, how="left") if keys else frame.join(kr, how="cross")
    return out.with_columns([nw.col(c).fill_null(0.0).cast(nw.Float64) for c in cols])


def build_agg_table(df, keys, time_col, target_col, time_aggs, kref=None):
    """One row per (bucket, timestamp) with aggregates and per-bucket prefix sums.

    ``kref`` is the frozen per-bucket centring reference (``compute_kref``).
    Callers holding a state MUST pass the one they stored -- recomputing it
    here would silently re-centre a table whose surviving rows were centred on
    the old value (see ``compute_kref``). ``None`` computes a fresh one from
    ``df``, which is correct only for a self-contained one-shot build.

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
    # NaN IS NOT NULL. On polars, `sum()` over a group containing NaN returns
    # NaN -- which then poisons that bucket's ENTIRE prefix sum -- and `count()`
    # counts the NaN as present. The legacy engine treats NaN as MISSING
    # (`_build_ts_aggs` masks with `~np.isnan(y_b)` before summing). Normalize
    # NaN -> null ONCE, before any aggregation or derived column, so every
    # aggregate below inherits the legacy engine's missing-value semantics.
    d = _normalize_target(nw.from_native(df, eager_only=True), target_col)
    if kref is None:
        kref = _kref_from_normalized(d, keys, time_col, target_col).to_native()
    return _build_agg_impl(d, keys, time_col, target_col, time_aggs, kref)


def _build_agg_impl(d, keys, time_col, target_col, time_aggs, kref):
    """``build_agg_table``'s body, over an already-normalized frame."""
    keys = list(keys)
    # `y - K` and `(y - K)**2` are materialized per RAW ROW, before the
    # group-by, for two independent reasons. (1) The aggregation must stay a
    # bare `sum` (see above). (2) `sK` must be summed from the shifted values,
    # never reconstructed afterwards as `s - c*K`: that subtraction is between
    # two quantities both of size `n*K`, i.e. it reintroduces exactly the
    # cancellation the shift exists to remove.
    d = attach_kref(d, kref, keys, ["K"])
    d = d.with_columns((nw.col(target_col) - nw.col("K")).alias("_dy"))
    d = d.with_columns((nw.col("_dy") ** 2).alias("_dy2"))
    tbl = (
        d.group_by(keys + [time_col])
        .agg(
            nw.col(target_col).sum().alias("s"),
            nw.col(target_col).count().alias("c"),
            nw.col("_dy").sum().alias("sK"),
            nw.col("_dy2").sum().alias("qK"),
            nw.col(target_col).min().alias("mn"),
            nw.col(target_col).max().alias("mx"),
        )
        .sort(keys + [time_col])
    )
    # integer ordinal per bucket over its own sorted calendar
    tbl = tbl.with_columns(nw.col("c").cast(nw.Float64))
    tbl = _add_ordinals(tbl, keys, time_col)
    tbl = _derive_time_agg_family(
        tbl, time_aggs, keys=keys, kref=kref, time_col=time_col
    )
    native = _add_prefix_sums(tbl.to_native(), keys, time_aggs)
    return native


def _derive_time_agg_family(tbl, time_aggs, keys=(), kref=None, time_col=None):
    """Add the suffixed ``{s,c,sK,qK,mn,mx}__<agg>`` family plus ``ewm``/``ewm__<agg>``.

    Factored out of :func:`build_agg_table` so the pooled predict tail
    (``NarwhalsPooledState._rebuild_tail``, Task 9) can derive the identical
    per-``time_agg`` columns for a synthetic pending row (one already-
    aggregated row per bucket per new timestamp) without duplicating this
    derivation. ``tbl`` must already carry the bare ``s``/``c``/``sK``/``qK``/
    ``mn``/``mx`` columns (from a raw-row aggregation at fit, or directly
    assigned at predict).

    ``kref`` (required whenever a non-``None`` ``time_agg`` is in play) is the
    frozen per-bucket centring reference: each family's ``sK__<agg>``/
    ``qK__<agg>`` is centred on ITS OWN ``K__<agg>``, since a ``count`` family
    holds small integers where a ``sum`` family holds the target's magnitude.
    Unlike the raw-row family there is no summation here -- a timestamp
    contributes exactly ONE value ``v_t`` -- so ``v_t - K`` is the shifted
    first moment directly.
    """
    tas = sorted(x for x in time_aggs if x is not None)
    keys = list(keys)
    if tas:
        if kref is None and any(f"K__{a}" not in tbl.columns for a in tas):
            raise ValueError(
                "_derive_time_agg_family needs the frozen `kref` to centre the "
                f"{tas} family(ies); see compute_kref"
            )
        # `NarwhalsPooledState._pending_agg_frame` attaches its own `K__<agg>`
        # columns positionally (numpy, one row per bucket) rather than by a
        # join it would then have to re-sort -- only fetch what is missing.
        k_cols = [f"K__{a}" for a in tas if f"K__{a}" not in tbl.columns]
        if k_cols:
            tbl = attach_kref(tbl, kref, keys, k_cols)
            # the join is not order-preserving, and every downstream reader
            # (`_add_ordinals`' running count, `_add_prefix_sums`' cum_sum) is
            # row-order dependent.
            if time_col is not None:
                tbl = tbl.sort(keys + [time_col])
    derived = []
    for a in tas:
        v = _time_agg_value_expr(a)
        obs = v.is_null().__invert__()
        dv = v - nw.col(f"K__{a}")
        derived += [
            nw.when(obs).then(v).otherwise(0.0).alias(f"s__{a}"),
            obs.cast(nw.Float64).alias(f"c__{a}"),
            nw.when(obs).then(dv).otherwise(0.0).alias(f"sK__{a}"),
            nw.when(obs).then(dv * dv).otherwise(0.0).alias(f"qK__{a}"),
            v.alias(f"mn__{a}"),
            v.alias(f"mx__{a}"),
        ]
    if derived:
        tbl = tbl.with_columns(derived).drop([f"K__{a}" for a in tas])

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
    for a in tas:
        tbl = tbl.with_columns(
            nw.when(nw.col(f"c__{a}") > 0)
            .then(nw.col(f"s__{a}") / nw.col(f"c__{a}"))
            .alias(f"ewm__{a}")
        )
    return tbl


def _prefix_sum_names(time_aggs):
    """``(prefix_cols, prefix_outs)`` for `_PREFIX_AGGS` and their ``time_agg`` suffixes."""
    prefix_cols, prefix_outs = [], []
    for suffix in [""] + [
        f"__{a}" for a in sorted(x for x in time_aggs if x is not None)
    ]:
        for base in _PREFIX_AGGS:
            prefix_cols.append(f"{base}{suffix}")
            prefix_outs.append(f"E{base}{suffix}")
    return prefix_cols, prefix_outs


def _add_prefix_sums(native, keys, time_aggs):
    """Cumulative per-bucket sum of every prefix-summed base column.

    Shared by :func:`build_agg_table` (fit) and the predict tail rebuild
    (Task 9): both need the identical ``E``-prefixed cumulative sums over
    ``s``/``c``/``sK``/``qK`` and their ``time_agg`` suffixes, just over a different
    (full-history vs. seed+tail) row set.
    """
    prefix_cols, prefix_outs = _prefix_sum_names(time_aggs)
    if keys:
        return grouped_accumulate(native, keys, prefix_cols, "cum_sum", prefix_outs)
    n = nw.from_native(native, eager_only=True)
    return n.with_columns(
        [nw.col(c).cum_sum().alias(o) for c, o in zip(prefix_cols, prefix_outs)]
    ).to_native()


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


def quantile_values_collapsed(agg_native, time_agg):
    """CSR values+offsets for a ``time_agg``-collapsed quantile.

    Not in the task-7 brief (added in a fix round): under ``time_agg``, every
    row sharing a (bucket, timestamp) collapses to exactly ONE contributing
    value ``v_t`` before the window statistic runs -- the same ``v_t``
    ``build_agg_table``'s ``s__<agg>``/``c__<agg>``/... derived family is
    built from (``_time_agg_value_expr``), and the same convention legacy's
    ``_time_agg_values`` uses: NaN means "unobserved timestamp" for
    ``sum``/``mean``/``min``/``max`` (an empty slice here, exactly like a
    dropped NaN row in the raw-row store built by ``quantile_values``), while
    ``count`` is never NaN (0 is a genuine observation).

    Returned in ``agg_native``'s own row order (one offset per row, plus a
    trailing sentinel) -- the same CSR contract ``quantile_values`` returns,
    so ``quantile_feature``/``_quantile_columns``'s per-bucket slicing is
    identical for either store.
    """
    a = nw.from_native(agg_native, eager_only=True)
    n = len(a)
    v = (
        a.with_columns(_time_agg_value_expr(time_agg).alias("_qv_collapsed"))
        .get_column("_qv_collapsed")
        .to_numpy()
        .astype(float)
    )
    valid = ~np.isnan(v)
    row_offsets = np.zeros(n + 1, dtype=np.intp)
    np.cumsum(valid.astype(np.intp), out=row_offsets[1:])
    return v[valid], row_offsets


# Memory bound for the dense (bucket x parent-calendar) grid.
#
# MEASURED (`tmp/measure_densify.py`, this fix wave): a densified aggregate
# table costs 96.4 bytes/row on polars and 104.0 bytes/row on pandas for the
# 12-column base schema (1 int64 bucket id + 1 datetime + 10 float64), i.e.
# ~100 B/row; each additional `time_agg` family adds 3 float64 columns
# (+24 B/row). 20M dense rows is therefore ~1.9 GiB on the base schema --
# the point past which materializing the grid is a memory problem in its own
# right regardless of how fast it runs.
_MAX_DENSE_ROWS = 20_000_000


def should_densify(
    n_buckets, n_calendar, n_sparse_rows, k=64, max_dense_rows=_MAX_DENSE_ROWS
):
    """Whether a dense (bucket x parent-calendar) grid is worth materializing.

    Densifying turns a RANGE window into a row window, but costs
    ``n_buckets * n_calendar`` rows. Two bounds, both about MEMORY:

    * ``dense <= k * n_sparse_rows`` -- the grid must stay proportionate to
      the data it is derived from;
    * ``dense <= max_dense_rows`` -- and proportionate is not enough on its
      own, since a ratio bound is unbounded in absolute terms.

    ``k = 64``, RECALIBRATED in this fix wave. It was 4, which refused
    ordinary documented usage: when the ``partition_by`` values are mutually
    exclusive over the calendar (every calendar-derived partition is), each
    bucket observes ``n_calendar / cardinality`` timestamps, so
    ``dense / sparse == cardinality`` EXACTLY. ``k = 4`` therefore refused
    every partition of cardinality >= 5 -- day-of-week included. Verified
    before the change: ``RollingMean(7, global_=True, partition_by=["dow"])``
    raised ``NotImplementedError`` (dense grid 7x90 against 90 sparse rows,
    ratio 7) while the legacy engine computed it, and
    ``groupby=["store"], partition_by=["dow"]`` raised the same way (2520
    against 360, ratio 7). ``k = 64`` clears every ordinary calendar
    cardinality with headroom -- day-of-week 7, month 12, hour-of-day 24,
    day-of-month 31, week-of-year 53 -- and still refuses a partition whose
    grid is more than 64x its own data (day-of-year, 365, is the first
    calendar field past the bound: each bucket then holds roughly one
    observation per year, which is the pathological shape this guard exists
    for).

    ``k`` is NOT a performance breakeven: none exists in the measured range.
    Task 8's scratch benchmark (a 2000 series x 200 day panel, 400k rows,
    ``RollingMean(28, global_=True, partition_by=["promo"])``, ``promo``
    cardinality swept 400 -> 32000 to push the ratio from ~1x to ~16.5x, with
    the guard forced True) found densifying beat the legacy engine at EVERY
    ratio tried, by a WIDENING margin: 6x faster at ratio 1x (0.054s vs
    0.324s), 9x at 1.6x, 17x at 2.5x, 19x at 4.5x, 20x at 8.5x, 21x at 16.5x.
    The legacy engine's per-bucket Python loop scales worse with bucket count
    than the vectorized dense-grid join does, so higher cardinality favours
    densifying. Both bounds here are memory bounds and nothing else.
    """
    dense = n_buckets * n_calendar
    if dense > max_dense_rows:
        return False
    if n_sparse_rows == 0:
        return True
    return dense <= k * n_sparse_rows


def densify_to_parent(agg_native, keys, time_col, parent_keys_native):
    """Left-join a bucket's sparse aggregates onto its full parent calendar.

    ``parent_keys_native`` is the FULLY EXPANDED dense skeleton the caller
    wants materialized: one row per (bucket, parent-calendar timestamp) pair,
    already carrying the bucket key column(s) (``keys``) -- e.g. one row per
    ``(_bucket_id, ds)`` for every ``ds`` in THAT bucket's own parent scope's
    calendar. On a dense grid a ROW-based window IS a RANGE window -- which is
    what the legacy engine's parent-calendar ordinal bookkeeping exists to
    emulate. Holes (skeleton rows with no matching sparse row) get
    zero-filled counts/sums so they occupy an ordinal without contributing
    observations; the per-timestamp ``mn``/``mx``/``ewm`` family and any
    ``time_agg``-suffixed variants stay null at a hole (no value was
    observed), exactly like an absent legacy ``unique_times`` entry.

    DEVIATION from the plan's Step-3 snippet: that version built the
    skeleton itself via ``buckets.join(cal, how="cross")`` against a single
    flat calendar (``cal`` holding only ``time_col``). That is only correct
    when every bucket shares ONE calendar -- the ``global_ + partition_by``
    case. ``local`` and ``groupby + partition_by`` give each bucket its OWN
    parent scope's calendar (mirroring legacy's per-scope
    ``_parent_time_grids``): a blanket cross join would leak another scope's
    calendar rows into a bucket that never observed them. The scoped
    expansion is therefore done by the caller
    (``NarwhalsPooledState._dense_skeleton``, an equi-join on the parent
    scope columns -- the vectorized form of the same "cross join per scope"
    the brief's snippet did unconditionally for the single-scope case); this
    function's only job is the final left-join-and-zero-fill against that
    already-scoped skeleton.

    A second deviation: the brief's ``zero_fill`` list only matched the bare
    ``s``/``c``/``q`` columns (today ``s``/``c``/``sK``/``qK``). A state
    needing more than one ``time_agg``
    family also carries suffixed derived columns (``s__mean``, ``c__mean``,
    ...) built by ``build_agg_table`` from those same three aggregates --
    left unfilled, a hole's ``c__mean`` stays null instead of 0, and the
    hole is silently treated as "unknown" rather than "no observation",
    corrupting e.g. a ``time_agg="mean"`` EWM's forward-fill-through-holes
    invariant. Matched by base name (split on the first ``__``) instead.
    """
    keys = list(keys)
    a = nw.from_native(agg_native, eager_only=True)
    cal = nw.from_native(parent_keys_native, eager_only=True).sort(keys + [time_col])
    out = cal.join(a, on=keys + [time_col], how="left")
    zero_fill = [c for c in out.columns if c.split("__", 1)[0] in _PREFIX_AGGS]
    if zero_fill:
        out = out.with_columns([nw.col(c).fill_null(0.0) for c in zero_fill])
    # A left join is not guaranteed order-preserving on every backend (see
    # `_order_preserving_left_join`'s docstring) -- re-sort defensively so
    # `_add_ordinals`'s row-order-dependent cumulative count is correct
    # regardless of how the backend actually implemented the join.
    out = out.sort(keys + [time_col])
    return out.to_native()


def quantile_feature(values, row_offsets, n_ordinals, offsets, p, min_samples):
    """Quantile per ordinal over the union of the ordinals at ``offsets`` back.

    ``offsets`` is the list of ordinal distances contributing to each window --
    ``range(lag, lag + w)`` for rolling, the seasonal strides for seasonal, and
    ``range(lag, n_ordinals)`` for expanding (the caller filters to ``<= t``
    via the ``0 <= t - o`` bound below).

    PERFORMANCE (fix round 1): when ``offsets`` is CONTIGUOUS (rolling and
    expanding -- both pass a plain ``range(...)``; only seasonal's strided
    offsets are not), the set of contributing ordinals for a given ``t`` is
    also a contiguous run, and the flat ``values`` store lays consecutive
    ordinals out back-to-back (both ``quantile_values`` and
    ``quantile_values_collapsed`` build it in ordinal order) -- so the whole
    window is ONE slice, ``values[row_offsets[lo]:row_offsets[hi+1]]``, with
    no ``np.concatenate`` at all. Measured on a ~1000-series/365-day/100-bucket
    panel (``RollingQuantile(window_size=28)``): this ran `np.concatenate`
    once per ordinal (~36.5k times for that panel's aggregate table), which
    made the narwhals engine 10-14% SLOWER than the legacy engine it replaces
    (0.831s/1.037s vs 0.755s/0.910s, polars/pandas) despite computing the
    identical numbers -- see the task-7 report's benchmark section for the
    full before/after. The multi-chunk gather path is kept, unchanged, for
    the strided (seasonal) case, which genuinely needs it.

    Bit-identical either way: a "hole" ordinal inside a contiguous run (an
    empty per-ordinal slice, from a densified gap or an all-NaN timestamp)
    contributes a zero-width sub-range within the single slice, exactly as it
    would contribute nothing to a concatenation of the same slices -- so this
    is a pure performance change, not a numerical one; the `atol=0.0`
    differential tests are the proof.
    """
    out = np.full(n_ordinals, np.nan)
    if not offsets:
        return out
    o_min, o_max = min(offsets), max(offsets)
    contiguous = sorted(offsets) == list(range(o_min, o_max + 1))
    for t in range(n_ordinals):
        if contiguous:
            lo = max(0, t - o_max)
            hi = min(n_ordinals - 1, t - o_min)
            if lo > hi:
                continue
            win = values[row_offsets[lo] : row_offsets[hi + 1]]
        else:
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
