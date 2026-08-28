"""Pooled lag-transform engine dispatch.

``MLFORECAST_POOLED_ENGINE`` selects the implementation:

- ``narwhals`` (default): the aggregate-table engine in this module.
- ``numpy``: the original engine in ``_pooled_legacy``, retained for the
  differential tests and the A/B benchmark.

Both the environment variable and the numpy engine are removed in iteration two.
"""

import os
from typing import Any, Dict, List, Optional

import narwhals as nw
import numpy as np
import utilsforecast.processing as ufp

from ._pooled_engine import PooledCtx, build_agg_table
from ._pooled_keys import (  # noqa: F401
    _attach_bucket_id,
    _dedupe_preserve_order,
    _encode_join_keys,
    _extend_groups,
    _order_preserving_left_join,
    add_bucket_id,
    lookup_bucket_ids,
)
from ._pooled_legacy import (  # noqa: F401
    PooledState as _LegacyPooledState,
    _build_ts_aggs,
    _collapse_rows_by_time,
    _compute_idsorted_to_bucket_pos,
    _reaggregate_ts_aggs,
)
from ._pooled_legacy import compute_pooled_features as _legacy_compute

# core.py:51 imports the name ``PooledState`` from this module, and Tasks 3/8/10
# branch per engine with ``isinstance(state, PooledState)``. Keep the name bound
# to the legacy class; the narwhals state is ``NarwhalsPooledState``.
PooledState = _LegacyPooledState

__all__ = [
    "PooledState",
    "NarwhalsPooledState",
    "compute_pooled_features",
    "POOLED_ENGINE",
]

_VALID_ENGINES = ("narwhals", "numpy")

POOLED_ENGINE = os.environ.get("MLFORECAST_POOLED_ENGINE", "numpy")
if POOLED_ENGINE not in _VALID_ENGINES:
    raise ValueError(
        f"MLFORECAST_POOLED_ENGINE must be one of {_VALID_ENGINES}; "
        f"got {POOLED_ENGINE!r}."
    )


def _resolve_ctx(tfm, keys):
    from .lag_transforms import _resolve_min_samples

    min_samples = _resolve_min_samples(tfm) if hasattr(tfm, "window_size") else 1
    return PooledCtx(
        keys=list(keys),
        lag=tfm._get_configured_lag(),
        min_samples=min_samples,
        time_agg=getattr(tfm, "time_agg", None),
    )


def _iter_leaf_tfms(tfm):
    """Yield the pooled leaf transform(s) a composite wrapper delegates to.

    ``Offset`` and ``Combine`` compute their pooled expression by calling
    into inner transform(s), each of which may carry its OWN ``time_agg`` /
    ``_pooled_accumulate`` requirement -- independent of the wrapper's own
    attributes, which always read as the class default (``time_agg=None``,
    no ``_pooled_accumulate``). ``ensure_time_aggs``/``ensure_accumulates``
    must see every inner transform's real requirement, not just the outer
    wrapper's always-trivial one, or the suffixed column family an inner
    transform needs is silently never built. Recurses to handle arbitrary
    nesting (e.g. ``Offset(Combine(...))``).
    """
    tfm1 = getattr(tfm, "tfm1", None)
    tfm2 = getattr(tfm, "tfm2", None)
    if tfm1 is not None and tfm2 is not None:
        yield from _iter_leaf_tfms(tfm1)
        yield from _iter_leaf_tfms(tfm2)
        return
    inner = getattr(tfm, "tfm", None)
    if inner is not None:
        yield from _iter_leaf_tfms(inner)
        return
    yield tfm


class NarwhalsPooledState:
    """Pooled state as a single aggregate table.

    One row per (bucket, timestamp) with per-timestamp aggregates and
    per-bucket prefix sums. Replaces the legacy flat row-arrays, ``bucket_df``,
    ``_idsorted_to_bucket_pos`` and the per-bucket ``_ts_aggs`` dict.
    """

    def __init__(
        self,
        agg,
        groups,
        group_cols: Optional[List[str]],
        series_bucket_id: np.ndarray,
        join_cols: List[str],
        keys: List[str],
        time_col: str,
        mode: str = "nonlocal",
    ):
        self.agg = agg
        self.groups = groups
        self.group_cols = group_cols
        self.series_bucket_id = series_bucket_id
        self.join_cols = join_cols
        self.keys = keys
        self.time_col = time_col
        self.mode = mode
        # mutated during recursive prediction (Task 9); initialized here so
        # snapshot()/restore() never touch a missing attribute
        self._pending: list = []
        self._next_ord = None
        self._seeds = None
        # partition_by / densification bookkeeping (Task 8). Harmless no-ops
        # for `global_`/`groupby` states (`mode` is always in
        # `_KNOWN_DENSE_MODES` for those, so `ensure_densified` returns
        # immediately and none of this is ever read).
        self.key_cols: Optional[List[str]] = None
        self.partition_cols: Optional[List[str]] = None
        self._parent_scope_cols: Optional[List[str]] = None
        self._densified = False
        self._densify_declined = False
        self._skeleton = None
        self._n_buckets: Optional[int] = None
        self._n_calendar_est: Optional[int] = None
        self._n_sparse_rows: Optional[int] = None

    @classmethod
    def from_global(
        cls, sorted_df, id_col, time_col, target_col, ga_data_dtype, n_series
    ):
        keep = _dedupe_preserve_order([id_col, time_col, target_col])
        df = ufp.drop_index_if_pandas(sorted_df[keep])
        return cls(
            agg=None,
            groups=None,
            group_cols=None,
            series_bucket_id=np.zeros(n_series, dtype=np.int64),
            join_cols=[id_col, time_col],
            keys=[],
            time_col=time_col,
            mode="global",
        )._build(df, time_col, target_col, ga_data_dtype)

    @classmethod
    def from_groupby(
        cls,
        df_for_group,
        group_cols_list,
        id_col,
        time_col,
        target_col,
        ga_data_dtype,
        static_features,
    ):
        keep = _dedupe_preserve_order(
            [id_col] + list(group_cols_list) + [time_col, target_col]
        )
        df = ufp.drop_index_if_pandas(df_for_group[keep])
        df, groups = add_bucket_id(df, list(group_cols_list))
        sbid = lookup_bucket_ids(static_features, groups, list(group_cols_list)).astype(
            np.int64, copy=False
        )
        return cls(
            agg=None,
            groups=groups,
            group_cols=list(group_cols_list),
            series_bucket_id=sbid,
            join_cols=[id_col, time_col],
            keys=["_bucket_id"],
            time_col=time_col,
            mode="groupby",
        )._build(df, time_col, target_col, ga_data_dtype)

    @classmethod
    def from_partition(
        cls,
        sorted_df,
        mode: str,
        group_cols_list: Optional[List[str]],
        partition_cols_list: List[str],
        id_col: str,
        time_col: str,
        target_col: str,
        ga_data_dtype,
        static_features,
        n_series: int,
    ):
        """Build a partition_by state; densification is decided lazily.

        Mirrors legacy ``PooledState.from_partition``'s key/parent-scope
        derivation exactly (see ``_pooled_legacy.py``):

        - ``mode == "local"``: bucket key is ``(id_col, *partition_cols)``;
          the parent scope is each series' own calendar (``[id_col]``).
        - ``groupby + partition_by``: bucket key is
          ``(*group_cols, *partition_cols)``; the parent scope is the
          group's shared calendar (``group_cols``).
        - ``global_ + partition_by``: bucket key is ``(*partition_cols,)``;
          the parent scope is the single global calendar
          (``parent_scope_cols=None``).

        Builds the SPARSE aggregate table (one row per observed
        ``(bucket, timestamp)``) up front -- identical to ``from_groupby``
        -- and caches the (bucket count, parent-calendar size, sparse row
        count) needed for the ``should_densify`` size guard. The grid is
        NOT densified here: whether a state may/must densify depends on
        which TRANSFORMS share it (`LookupLag` forbids it; everything else
        needs it -- rulings F17/F25), and that set isn't known at
        construction time in the legacy call signature this mirrors -- and
        `core.py`'s single call site (`_pooled_state_cls().from_partition(...)`)
        is shared with the legacy engine's `PooledState.from_partition`, so
        this signature can't grow a `tfms` parameter the legacy one doesn't
        have. See ``ensure_densified``, invoked lazily from ``feature_frame``
        once the actual transforms for this state are visible (the same
        pattern ``ensure_time_aggs``/``ensure_accumulates`` already use).
        """
        if mode == "local":
            key_cols = _dedupe_preserve_order([id_col] + list(partition_cols_list))
            parent_scope_cols: Optional[List[str]] = [id_col]
        elif group_cols_list:
            key_cols = _dedupe_preserve_order(
                list(group_cols_list) + list(partition_cols_list)
            )
            parent_scope_cols = list(group_cols_list)
        else:
            key_cols = _dedupe_preserve_order(list(partition_cols_list))
            parent_scope_cols = None

        keep = _dedupe_preserve_order([id_col] + key_cols + [time_col, target_col])
        df = ufp.drop_index_if_pandas(sorted_df[keep])
        df, groups = add_bucket_id(df, key_cols)
        sf_cols = set(static_features.columns)
        if set(key_cols).issubset(sf_cols):
            sbid = lookup_bucket_ids(static_features, groups, key_cols).astype(
                np.int64, copy=False
            )
        else:
            sbid = np.zeros(n_series, dtype=np.int64)

        state = cls(
            agg=None,
            groups=groups,
            group_cols=group_cols_list,
            series_bucket_id=sbid,
            join_cols=[id_col, time_col],
            keys=["_bucket_id"],
            time_col=time_col,
            mode=mode,
        )
        state.key_cols = key_cols
        state.partition_cols = list(partition_cols_list)
        state._parent_scope_cols = parent_scope_cols
        built = state._build(df, time_col, target_col, ga_data_dtype)
        built._compute_density_estimate()
        return built

    def _compute_density_estimate(self):
        """Cache (bucket count, parent-calendar size, sparse row count).

        Cheap: group-by counts only, no cross product materialized. Used by
        `ensure_densified`'s `should_densify` call. Scopes can have
        DIFFERENT calendar lengths (a `local` state's series may start/end
        at different dates; a `groupby + partition_by` state's groups may
        span different date ranges) -- a single ``(n_buckets, n_calendar)``
        pair can't represent that exactly, so the scoped branch folds the
        EXACT total dense-row count into ``n_calendar`` with ``n_buckets``
        pinned to 1, rather than reporting a possibly-misleading average or
        an underestimate that could pick "densify" for a state whose real
        dense grid is much bigger than a naive per-bucket-average implies.
        """
        self._n_sparse_rows = len(nw.from_native(self.agg, eager_only=True))
        d = nw.from_native(self._df, eager_only=True)
        if self._parent_scope_cols is None:
            self._n_buckets = len(nw.from_native(self.groups, eager_only=True))
            self._n_calendar_est = len(d.select(self.time_col).unique())
            return
        scope_cols = self._parent_scope_cols
        enc_cols = [f"__enc_{c}" for c in scope_cols]
        # Null-safe: a `groupby + partition_by` scope column (e.g. `brand`)
        # can itself be null for some series (mirrors legacy's own
        # sentinel-encoding of `parent_scope_cols` in `PooledState.
        # from_partition` -- "a null/NaN scope value ... matches itself and
        # only itself"). A plain `group_by`/`join` on the raw column would
        # treat two null scope values as NOT equal (standard null
        # semantics), silently splitting one scope's buckets/calendar across
        # what looks like two different (mismatched, unmatched) groups.
        groups_nw = nw.from_native(self.groups, eager_only=True)
        left_enc, right_enc = _encode_join_keys(
            groups_nw.select(scope_cols + ["_bucket_id"]),
            d.select(scope_cols + [self.time_col]).unique(),
            scope_cols,
        )
        bucket_counts = left_enc.group_by(enc_cols).agg(
            nw.col("_bucket_id").count().alias("_n_buckets_in_scope")
        )
        cal_lens = right_enc.group_by(enc_cols).agg(
            nw.col(self.time_col).count().alias("_cal_len")
        )
        joined = bucket_counts.join(cal_lens, on=enc_cols, how="left")
        total = (
            joined.with_columns(
                (nw.col("_n_buckets_in_scope") * nw.col("_cal_len").fill_null(0)).alias(
                    "_prod"
                )
            )
            .get_column("_prod")
            .sum()
        )
        self._n_buckets = 1
        self._n_calendar_est = int(total) if total is not None else 0

    def _dense_skeleton(self):
        """Every (bucket, parent-calendar-timestamp) pair to materialize.

        Built lazily (only when a state actually needs densifying) and
        cached: computed once per state, reused if `ensure_time_aggs`
        rebuilds `self.agg` from scratch later and must re-densify.
        """
        if self._skeleton is not None:
            return self._skeleton
        d = nw.from_native(self._df, eager_only=True)
        groups_nw = nw.from_native(self.groups, eager_only=True)
        if self._parent_scope_cols is None:
            cal = d.select([self.time_col]).unique()
            buckets = groups_nw.select(["_bucket_id"]).unique()
            skeleton = buckets.join(cal, how="cross")
        else:
            scope_cols = self._parent_scope_cols
            enc_cols = [f"__enc_{c}" for c in scope_cols]
            cal = d.select(scope_cols + [self.time_col]).unique()
            bucket_scope = groups_nw.select(["_bucket_id"] + scope_cols)
            # Null-safe join -- see `_compute_density_estimate`'s comment: a
            # scope column can be null (e.g. a null `groupby` key), and two
            # null scope values must match each other, not fail to match at
            # all (standard join null semantics).
            left_enc, right_enc = _encode_join_keys(bucket_scope, cal, scope_cols)
            skeleton = left_enc.join(
                right_enc.select(enc_cols + [self.time_col]), on=enc_cols, how="inner"
            ).select(["_bucket_id", self.time_col])
        self._skeleton = skeleton.to_native()
        return self._skeleton

    def _apply_densification(self):
        """Rebuild `self.agg` as the dense (bucket x parent-calendar) grid.

        The `E`-prefixed prefix-sum columns and `ord` were computed over the
        SPARSE row order (in `build_agg_table`, called from `_build`/
        `ensure_time_aggs`) and are stale for the new dense row set: a
        densified hole changes which row is `lag` positions back from any
        given row. Rebuilt from scratch here, mirroring exactly what
        `build_agg_table` itself does after aggregation (assign ordinals,
        then a per-bucket cumulative sum of every base aggregate column) --
        just against the dense base table instead of the sparse one.
        """
        from ._pooled_engine import (
            _add_ordinals,
            _PREFIX_AGGS,
            densify_to_parent,
            grouped_accumulate,
        )

        skeleton = self._dense_skeleton()
        a = nw.from_native(self.agg, eager_only=True)
        base_cols = [c for c in a.columns if c != "ord" and not c.startswith("E")]
        dense = densify_to_parent(
            a.select(base_cols).to_native(), self.keys, self.time_col, skeleton
        )
        t = _add_ordinals(
            nw.from_native(dense, eager_only=True).sort(self.keys + [self.time_col]),
            self.keys,
            self.time_col,
        )
        prefix_cols, prefix_outs = [], []
        for suffix in [""] + [
            f"__{a_}" for a_ in sorted(x for x in self._time_aggs if x is not None)
        ]:
            for base in _PREFIX_AGGS:
                prefix_cols.append(f"{base}{suffix}")
                prefix_outs.append(f"E{base}{suffix}")
        self.agg = grouped_accumulate(
            t.to_native(), self.keys, prefix_cols, "cum_sum", prefix_outs
        )
        # F4/F5: densification changes the aggregate table's row count, and
        # `quantile_values` emits one offset per row -- stale offsets would
        # silently misalign every partition_by quantile.
        self._qvalues = None

    def ensure_densified(self, leaf_tfms):
        """Densify this partition_by state so a row-shift IS a RANGE window.

        No-op for `global_`/`groupby` states (`self.mode` is structurally
        dense already, see `_KNOWN_DENSE_MODES`) and once this state has
        already settled its decision (`_densified` / `_densify_declined`).

        Two families pull in OPPOSITE directions (rulings F17/F25):

        - `LookupLag` must NEVER see the dense grid: its lag counts
          OCCURRENCES via a positional shift over the table
          (`LookupLag._pooled_expr`'s own docstring), and a densified hole
          would silently convert that occurrence lag into an ordinal lag.
        - Every other pooled family needs the opposite. `PooledCtx.shift`/
          `.window` (window transforms), the accumulate shim
          (`ensure_accumulates`, e.g. `ExpandingMin`/`Max`/EWM), and the
          quantile store's `ord` column are all ROW-position operations,
          which equal a calendar-RANGE window only when the grid has no
          gaps -- true for `global_`/`groupby` states (legacy renumbers
          `unique_times` to `0..n-1`) but NOT for a sparse `partition_by`
          state, whose ordinals are real, possibly-gapped parent-calendar
          positions. `ExponentiallyWeightedMean` at `lag > 1` is simply the
          family this was *measured* on (11.0/31.0 vs legacy's 16.0/36.0 on
          a gapped bucket) -- the same row-vs-ordinal mismatch applies to
          every other non-`LookupLag` family, not only EWM.

        A state needing both is a real conflict this architecture can't
        satisfy (one shared aggregate table can't be simultaneously sparse
        for one family and dense for another) -- refused loudly rather than
        computed silently wrong. Likewise if the dense grid is simply too
        big (`should_densify` declines) for a state that isn't pure
        `LookupLag`.
        """
        if self.mode in self._KNOWN_DENSE_MODES:
            return
        if self.key_cols is None:
            # Not a state actually built by `from_partition` (e.g. a state
            # constructed directly, as some unit tests do, to probe
            # `_guard_ewm_positional_shift` in isolation) -- no density
            # bookkeeping (`_n_buckets`/`_n_calendar_est`/`_n_sparse_rows`,
            # `_dense_skeleton`) was computed for it, and there is no size
            # guard or LookupLag/EWM conflict to resolve. Fall through to
            # `_guard_ewm_positional_shift`, which still fires for that case.
            return
        if self._densified or self._densify_declined:
            return
        from .lag_transforms import LookupLag

        has_lookup = any(isinstance(t, LookupLag) for t in leaf_tfms)
        others = sorted(
            {type(t).__name__ for t in leaf_tfms if not isinstance(t, LookupLag)}
        )
        if has_lookup and others:
            raise NotImplementedError(
                "partition_by state mixes LookupLag (requires the sparse, "
                f"occurrence-indexed table) with {others} (require the "
                "dense parent-calendar grid for correct RANGE-window "
                "semantics) -- the two can't share one aggregate table. "
                "Give LookupLag its own partition_by grouping (a separate "
                "lag_transforms entry with no other transform sharing its "
                "exact mode/groupby/partition_by combination), or run this "
                "key with MLFORECAST_POOLED_ENGINE=numpy."
            )
        if has_lookup:
            self._densify_declined = True
            return
        from ._pooled_engine import should_densify

        if should_densify(self._n_buckets, self._n_calendar_est, self._n_sparse_rows):
            self._apply_densification()
            self._densified = True
            return
        self._densify_declined = True
        ewm_names = sorted(
            {
                type(t).__name__
                for t in leaf_tfms
                if getattr(t, "_pooled_accumulate", None) == ("ewm", "ewm_mean")
                and t._get_configured_lag() > 1
            }
        )
        if ewm_names:
            raise NotImplementedError(
                f"partition_by state needs densification for {ewm_names} "
                "(lag > 1) but the dense grid "
                f"({self._n_buckets}x{self._n_calendar_est} rows) exceeds "
                f"the should_densify size guard against "
                f"{self._n_sparse_rows} sparse rows; refusing rather than "
                "computing a row-position shift against sparse ordinals "
                "(measured divergence: 11.0/31.0 vs legacy's 16.0/36.0 at "
                "lag=2 on a gapped bucket). Run this key with "
                "MLFORECAST_POOLED_ENGINE=numpy."
            )
        raise NotImplementedError(
            "partition_by state needs the dense parent-calendar grid for "
            f"correct RANGE-window semantics ({sorted({type(t).__name__ for t in leaf_tfms})}) "
            f"but the dense grid ({self._n_buckets}x{self._n_calendar_est} "
            f"rows) exceeds the should_densify size guard against "
            f"{self._n_sparse_rows} sparse rows. Run this key with "
            "MLFORECAST_POOLED_ENGINE=numpy."
        )

    def _build(self, df, time_col, target_col, ga_data_dtype):
        # cast through the model's working dtype before float, matching the
        # legacy engine so numerics stay bit-identical with the model's
        # working dtype (e.g. float32 rounding). Round-tripping through numpy
        # (rather than a narwhals dtype cast) mirrors the legacy engine
        # exactly and works for any ``ga_data_dtype``, not just float32/64.
        d = nw.from_native(df, eager_only=True)
        y = d.get_column(target_col).to_numpy().astype(ga_data_dtype).astype(float)
        d = d.with_columns(
            nw.new_series(
                target_col, y, dtype=nw.Float64, backend=nw.get_native_namespace(d)
            )
        )
        self._df = ufp.drop_index_if_pandas(d.to_native())
        self._target_col = target_col
        self._time_aggs = {None}
        self._qvalues = None  # quantile value store, built on demand (Task 7)
        self.agg = build_agg_table(
            self._df, self.keys, time_col, target_col, self._time_aggs
        )
        return self

    def ensure_time_aggs(self, time_aggs):
        """Rebuild the table if a transform needs a ``time_agg`` family absent from it."""
        missing = set(time_aggs) - self._time_aggs
        if missing:
            self._time_aggs |= set(time_aggs)
            self.agg = build_agg_table(
                self._df, self.keys, self.time_col, self._target_col, self._time_aggs
            )
            # `build_agg_table` always rebuilds from the SPARSE `self._df` --
            # re-apply densification (Task 8) if this state had already
            # densified, or the fresh rebuild would silently revert to the
            # sparse grid. Row count changes either way -> invalidate qvalues
            # (F4/F5: one offset per aggregate-table row).
            self._qvalues = None
            if self._densified:
                self._apply_densification()

    def ensure_accumulates(self, leaf_tfms):
        """Materialize ``A<col>`` columns for transforms needing a running min/max/ewm.

        Not expressible as a narwhals expression (``cum_min``/``cum_max`` cannot
        take ``.over()`` on pandas), so it goes through the shim.

        The accumulate op alone doesn't identify the column: ``ewm_mean`` also
        needs ``alpha``/``adjust``/``ignore_nulls`` (see
        ``ExponentiallyWeightedMean._pooled_accumulate_kwargs`` and
        ``grouped_accumulate``'s docstring for why no default is safe), so the
        dedup key includes the transform's kwargs alongside the column and op.

        ``leaf_tfms`` is an iterable of already-flattened leaf transforms (see
        ``_iter_leaf_tfms``): a composite (``Offset``/``Combine``) transform
        itself never carries ``_pooled_accumulate``, only its inner
        transform(s) do.
        """
        from ._pooled_engine import grouped_accumulate

        need = {}
        for tfm in leaf_tfms:
            spec = getattr(tfm, "_pooled_accumulate", None)
            if spec is None:
                continue
            base, op = spec
            suffix = (
                "" if getattr(tfm, "time_agg", None) is None else f"__{tfm.time_agg}"
            )
            kw = (
                tfm._pooled_accumulate_kwargs()
                if hasattr(tfm, "_pooled_accumulate_kwargs")
                else {}
            )
            need[(f"{base}{suffix}", op, tuple(sorted(kw.items())))] = (
                f"A{base}{suffix}"
            )
        present = set(nw.from_native(self.agg, eager_only=True).columns)
        for (col, op, kw_items), out in need.items():
            if out in present:
                continue
            kw = dict(kw_items)
            if self.keys:
                self.agg = grouped_accumulate(
                    self.agg, self.keys, [col], op, [out], **kw
                )
            else:
                # Single implicit bucket (global_ mode, no keys): apply the op
                # directly and forward-fill, mirroring `grouped_accumulate`'s
                # documented invariant (gap ordinals carry the previous
                # running value) for the un-partitioned case too.
                self.agg = (
                    nw.from_native(self.agg, eager_only=True)
                    .with_columns(
                        getattr(nw.col(col), op)(**kw)
                        .fill_null(strategy="forward")
                        .alias(out)
                    )
                    .to_native()
                )

    # Modes for which `NarwhalsPooledState`'s per-bucket `ord` column is
    # STRUCTURALLY dense (0..n-1 with no calendar gaps): legacy renumbers
    # `unique_times` the same way for `global_`/`groupby` buckets. A
    # `partition_by` state ("nonlocal"/"local") keeps real, non-renumbered
    # parent-calendar ordinals and can be gapped -- UNLESS densified (Task 8,
    # `ensure_densified`/`_densified`). See `_guard_ewm_positional_shift` and
    # `ExponentiallyWeightedMean._pooled_expr`.
    _KNOWN_DENSE_MODES = ("global", "groupby")

    def _guard_ewm_positional_shift(self, leaf_tfms):
        """Refuse rather than silently mis-compute EWM on a possibly-gapped state.

        `ExponentiallyWeightedMean._pooled_expr` shifts by row POSITION,
        which only matches legacy's calendar-ORDINAL threshold fold when the
        bucket is dense or `lag == 1`. That information -- this state's
        `mode` (and, since Task 8, whether it was densified) -- isn't
        visible from inside `_pooled_expr` (`PooledCtx` carries only
        `keys`/`lag`/`min_samples`/`time_agg`, not how the bucket's ordinals
        were built), so the check lives here instead, where `self.mode` /
        `self._densified` are available.

        A `partition_by` state reaching `feature_frame` with an EWM(lag>1)
        leaf has already been forced through `ensure_densified` (which
        raises first if densifying isn't possible), so `self._densified`
        is true by the time this runs -- this guard is then a no-op backstop
        for that path, and the one that actually fires for a state that
        DECLINED densification (pure `LookupLag`, no EWM present -- and if
        an EWM ever were present in that mix, `ensure_densified` would have
        already raised before this method is ever reached).
        """
        if self.mode in self._KNOWN_DENSE_MODES or self._densified:
            return
        for tfm in leaf_tfms:
            if getattr(tfm, "_pooled_accumulate", None) != ("ewm", "ewm_mean"):
                continue
            lag = tfm._get_configured_lag()
            if lag > 1:
                raise NotImplementedError(
                    f"{type(tfm).__name__}._pooled_expr uses a row-position "
                    f"shift that only matches the legacy engine's calendar-"
                    f"ordinal threshold fold when the bucket is dense or "
                    f"lag == 1 (this state's mode={self.mode!r}, lag={lag}). "
                    "Measured divergence: 11.0/31.0 (this expression) vs "
                    "16.0/36.0 (legacy) at lag=2 on a gapped partition "
                    "bucket. This state must be densified (or this "
                    "transform refused) before evaluating it -- not yet "
                    "wired for partition_by states."
                )

    def _quantile_columns(self, transforms):
        """Quantile features computed over the flat values+offsets store.

        Quantiles have no sufficient statistic (no aggregate reconstructs
        them), so this is the one family that reads raw values instead of an
        expression over `self.agg`'s aggregate columns.

        Two stores, selected per transform by its own `time_agg`:
          * `time_agg is None`: the raw per-row store (`quantile_values`),
            cached on `self._qvalues` -- must be invalidated (set back to
            ``None``) anywhere `self.agg` changes shape (see
            `quantile_values`'s docstring) -- currently only `_build`, since
            Task 7's states are never densified/trimmed/appended.
          * `time_agg` set: every row sharing a (bucket, timestamp) first
            collapses to ONE value (the time_agg aggregate `v_t`) before the
            window statistic runs -- exactly what legacy's own
            `_compute_bucket_feature_collapsed` does for this family
            (`quantile_values_collapsed`). Not cached across calls (cheap: a
            single expression pass over `self.agg`, no join), but computed at
            most once per distinct `time_agg` value within one call, shared
            by every transform that needs it.
        Both stores share the identical CSR (values, row_offsets) contract,
        so the per-bucket windowing loop below is unchanged either way.
        """
        from ._pooled_engine import (
            quantile_feature,
            quantile_values,
            quantile_values_collapsed,
        )

        t = nw.from_native(self.agg, eager_only=True)
        bucket_of = (
            t.get_column(self.keys[0]).to_numpy()
            if self.keys
            else np.zeros(len(t), dtype=np.int64)
        )
        ords = t.get_column("ord").to_numpy()

        stores = {}
        for tfm in transforms.values():
            ta = tfm.time_agg
            if ta in stores:
                continue
            if ta is None:
                if self._qvalues is None:
                    # agg is the ordinal-grid authority -- see quantile_values'
                    # docstring.
                    self._qvalues = quantile_values(
                        self._df, self.agg, self.keys, self.time_col, self._target_col
                    )
                stores[ta] = self._qvalues
            else:
                stores[ta] = quantile_values_collapsed(self.agg, ta)

        cols = {}
        for name, tfm in transforms.items():
            values, row_offsets = stores[tfm.time_agg]
            ctx = _resolve_ctx(tfm, self.keys)
            out = np.full(len(t), np.nan)
            for b in np.unique(bucket_of):
                sel = np.flatnonzero(bucket_of == b)
                n_ord = int(ords[sel].max()) + 1
                base = int(sel.min())  # rows are bucket-contiguous and ord-sorted
                offs = tfm._pooled_quantile_offsets(ctx, n_ord)
                # DEVIATION from the brief's verbatim Step-4 code: it passed
                # the FULL GLOBAL `values` array together with LOCALIZED
                # offsets (`row_offsets[...] - row_offsets[base]`), so
                # `quantile_feature`'s `values[row_offsets[i]:row_offsets[i+1]]`
                # indexed the wrong slice for every bucket after the first
                # (`base > 0`) -- verified: on a 5-bucket panel this produced
                # wrong values for the ~80% of rows outside bucket 0 (bucket
                # 0's `base == 0` makes the localized and global offsets
                # coincide, which is exactly why the bug is invisible on a
                # single-bucket/global_ fixture). `values` must be sliced to
                # the same local window as the offsets.
                local_values = values[row_offsets[base] : row_offsets[base + n_ord]]
                vals = quantile_feature(
                    local_values,
                    row_offsets[base : base + n_ord + 1] - row_offsets[base],
                    n_ord,
                    offs,
                    tfm.p,
                    ctx.min_samples,
                )
                out[sel] = vals[ords[sel]]
            cols[name] = out
        return cols

    def feature_frame(self, transforms: Dict[str, Any]):
        """Evaluate every transform's expression in one pass over the table."""
        # Flatten composites (Offset/Combine) so their INNER transforms' own
        # time_agg/accumulate requirements are seen -- the wrapper itself
        # always reports the trivial default. See `_iter_leaf_tfms`.
        leaves = [leaf for t in transforms.values() for leaf in _iter_leaf_tfms(t)]
        self.ensure_time_aggs({getattr(t, "time_agg", None) for t in leaves})
        # Must run before `ensure_accumulates`/the quantile store: a
        # partition_by state's row order only means "calendar order" once
        # densified (Task 8), and both of those read `self.agg` in row
        # order. Also decides the LookupLag-vs-everything-else conflict
        # (rulings F17/F25) -- see `ensure_densified`'s docstring.
        self.ensure_densified(leaves)
        self.ensure_accumulates(leaves)
        self._guard_ewm_positional_shift(leaves)
        # Quantiles have no narwhals expression (no sufficient statistic --
        # see `_quantile_columns`); split them out and compute separately.
        quantile_tfms = {
            n: tfm
            for n, tfm in transforms.items()
            if getattr(tfm, "_pooled_quantile", False)
        }
        expr_tfms = {n: tfm for n, tfm in transforms.items() if n not in quantile_tfms}
        t = nw.from_native(self.agg, eager_only=True)
        exprs = []
        for name, tfm in expr_tfms.items():
            expr = tfm._pooled_expr(_resolve_ctx(tfm, self.keys))
            exprs.append(expr.alias(name))
        if exprs:
            t = t.with_columns(exprs)
        if quantile_tfms:
            qcols = self._quantile_columns(quantile_tfms)
            backend = nw.get_native_namespace(t)
            t = t.with_columns(
                *[
                    nw.new_series(name, vals, dtype=nw.Float64, backend=backend)
                    for name, vals in qcols.items()
                ]
            )
        return t.to_native()

    def join_to_panel(self, df_sorted, transforms, _id_col, time_col):
        """Feature values aligned positionally with ``df_sorted``'s rows.

        ``_id_col`` is accepted for call-site symmetry with the legacy engine
        (and the caller's ``(id_col, time_col)`` pair) but unused here: the
        join key is ``keys + [time_col]`` -- values are correctly broadcast to
        every id sharing a (bucket, timestamp), which is the pooled semantics.
        """
        feats = nw.from_native(self.feature_frame(transforms), eager_only=True)
        names = list(transforms)
        on = ([] if not self.keys else self.keys) + [time_col]
        left = nw.from_native(df_sorted, eager_only=True)
        if self.keys:
            # attach _bucket_id to the panel rows via the group registry.
            # `key_cols` (set by `from_partition`) is the bucket key
            # actually used to build `self.groups` -- `(*group_cols,
            # *partition_cols)`, `(id_col, *partition_cols)`, or
            # `(*partition_cols,)` -- which for a `partition_by` state is
            # NOT the same as `self.group_cols` (the plain `groupby=` cols,
            # None for `local`/`global_ + partition_by`). `from_groupby`
            # states never set `key_cols`, so fall back to `group_cols`.
            bucket_key_cols = (
                self.key_cols if self.key_cols is not None else (self.group_cols)
            )
            left_native = _attach_bucket_id(
                left.select(bucket_key_cols + [time_col]).to_native(),
                self.groups,
                bucket_key_cols,
            )
            left = nw.from_native(left_native, eager_only=True)
        joined = _order_preserving_left_join(left, feats.select(on + names), on=on)
        out = nw.to_native(joined)
        return {n: nw.from_native(out, eager_only=True)[n].to_numpy() for n in names}


def compute_pooled_features(state, transforms, query_arrays=None):
    """Dispatch each transform to the engine that can compute it.

    A transform whose ``_pooled_expr`` returns ``None`` has no narwhals
    expression yet and is delegated to the numpy engine. This fallback is
    iteration-one scaffolding; Task 14 asserts it is unused.
    """
    if POOLED_ENGINE == "numpy" or isinstance(state, _LegacyPooledState):
        return _legacy_compute(state, transforms, query_arrays=query_arrays)
    fast, slow = {}, {}
    for name, tfm in transforms.items():
        if getattr(tfm, "_pooled_quantile", False):
            # No `_pooled_expr` at all (no sufficient statistic) -- routed
            # through `NarwhalsPooledState._quantile_columns` instead, inside
            # `feature_frame`/`join_to_panel`.
            fast[name] = tfm
            continue
        probe = tfm._pooled_expr(_resolve_ctx(tfm, getattr(state, "keys", [])))
        (fast if probe is not None else slow)[name] = tfm
    if slow:
        raise NotImplementedError(
            "narwhals engine reached a transform with no _pooled_expr: "
            f"{sorted(type(t).__name__ for t in slow.values())}. "
            "Fallback is wired at the core.py seam, not here."
        )
    out: Dict[str, np.ndarray] = state.join_to_panel(
        state._df, fast, state.join_cols[0], state.time_col
    )
    return out
