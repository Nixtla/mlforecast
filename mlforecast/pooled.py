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

from ._pooled_engine import (
    PooledCtx,
    _add_ordinals,
    _add_prefix_sums,
    _derive_time_agg_family,
    apply_accumulate,
    build_agg_table,
)
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


def _pooled_retention(tfm) -> Optional[int]:
    """Historical ordinals (relative to the query row) a leaf transform's
    pooled expression can reach back into, i.e. the minimum tail retention
    that keeps its feature value correct once the earlier history is
    discarded (Task 9's seed mechanism).

    * ``window_size`` + ``season_length`` (Seasonal* family): the strided
      offsets are ``lag, lag+season_length, ..., lag+(window_size-1)*season_length``
      -- the last is the furthest back reference.
    * ``window_size`` alone (Rolling* / RollingQuantile): ``ctx.window``'s
      ``lo`` shift reaches ``lag + window_size`` back (one further than the
      window itself, the boundary the ``fill_null(0.0)`` relies on); RollingMin/
      RollingMax's own direct shifts only reach ``lag + window_size - 1``, so
      this is a safe (by exactly one row) upper bound for them too.
    * Neither (Expanding*/EWM/LookupLag): these read a CUMULATIVE prefix sum
      or accumulate column (``Es``/``Ec``/``Eq``/``Amn``/``Amx``/``Aewm``) or,
      for ``LookupLag``, a raw positional shift over the (sparse) occurrence
      table -- in every case exactly ``lag`` ordinals back, made correct
      beyond that by the seed row's carried baseline rather than more
      retained raw history.
    * Quantile transforms with neither (``ExpandingQuantile``): no sufficient
      statistic exists for an unbounded window (unlike ``ExpandingMean``'s
      prefix sum), so no seed can compress arbitrarily old history -- signals
      "keep everything" via ``None``.
    """
    lag = tfm._get_configured_lag()
    window_size = getattr(tfm, "window_size", None)
    season_length = getattr(tfm, "season_length", None)
    if window_size is not None and season_length is not None:
        return lag + season_length * (window_size - 1)
    if window_size is not None:
        return lag + window_size
    if getattr(tfm, "_pooled_quantile", False):
        return None
    return max(lag, 0)


def _accumulate_specs(leaf_tfms):
    """``{(col, op, sorted kwargs items): out_name}`` for every leaf transform's
    ``_pooled_accumulate`` requirement.

    Shared by ``ensure_accumulates`` (fit: materialize once against the full
    aggregate table) and ``NarwhalsPooledState._make_seeds``/``_rebuild_tail``
    (predict: re-derive the same columns from a seed baseline over the small
    tail) so the two call sites can never drift on which ``(col, op, kwargs)``
    a transform needs.
    """
    need = {}
    for tfm in leaf_tfms:
        spec = getattr(tfm, "_pooled_accumulate", None)
        if spec is None:
            continue
        base, op = spec
        suffix = "" if getattr(tfm, "time_agg", None) is None else f"__{tfm.time_agg}"
        kw = (
            tfm._pooled_accumulate_kwargs()
            if hasattr(tfm, "_pooled_accumulate_kwargs")
            else {}
        )
        need[(f"{base}{suffix}", op, tuple(sorted(kw.items())))] = f"A{base}{suffix}"
    return need


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
        self._pending_raw: list = []
        self._next_ord = None
        self._seeds = None
        # Predict-time tail machinery (Task 9), computed once lazily on the
        # first `latest_features` call and reused for the rest of this
        # predict run (and every subsequent model's, since they only depend
        # on the transforms/fit-time history, never on appended predictions):
        # `_retention` (int, or None for "keep everything" -- see
        # `_pooled_retention`), `_seed_rows`/`_hist_suffix` (native frames,
        # `self.agg`'s own schema), `_hist_suffix_df` (native frame,
        # `self._df`'s own schema, for the raw quantile store) and
        # `_accum_specs` (see `_accumulate_specs`).
        self._retention: Optional[int] = None
        self._seed_rows = None
        self._hist_suffix = None
        self._hist_suffix_df = None
        self._accum_specs: Dict[Any, str] = {}
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
        need = _accumulate_specs(leaf_tfms)
        present = set(nw.from_native(self.agg, eager_only=True).columns)
        for (col, op, kw_items), out in need.items():
            if out in present:
                continue
            kw = dict(kw_items)
            # `apply_accumulate` dispatches on `self.keys`: grouped per-bucket
            # when non-empty, else a single un-partitioned accumulate with the
            # same forward-fill-through-gaps invariant (see its docstring and
            # `grouped_accumulate`'s).
            self.agg = apply_accumulate(self.agg, self.keys, col, op, out, **kw)

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

    # ---- Predict path (Task 9): tail evaluation + per-bucket seed rows ----
    #
    # Correctness needs only a bounded window per transform (`_pooled_retention`):
    # a FINITE window (Rolling*/Seasonal*/RollingQuantile) needs its own
    # `lag + window` raw history, since its feature is a DIFFERENCE of two
    # prefix sums (or, for quantile, a union of raw-value slices) that cancels
    # any constant baseline before the retained window -- truncating history
    # never changes the answer as long as both endpoints stay inside the
    # retained suffix. A PREFIX-DEPENDENT transform (Expanding*/EWM/LookupLag)
    # instead reads an ABSOLUTE cumulative value (a prefix sum, an
    # accumulate-column value, or a raw positional shift over the sparse
    # occurrence table) with no such cancellation -- so one SEED row per
    # bucket carries the exact state as of the ordinal just before the
    # retained suffix (`Es`/`Ec`/`Eq` and any consumed accumulate column's
    # value), and a fresh cumsum/accumulate over
    # ``[seed, retained history, pending predictions]`` reconstructs the true
    # absolute value at every ordinal from there on -- this is the "carried
    # accumulator" the legacy engine lacks (see `_make_seeds`'s docstring).
    #
    # `_pending`/`_pending_raw` grow by one entry per recursive predict step;
    # `_rebuild_tail` concatenates the (fixed-size) seed + retained history
    # with all of `_pending` and reruns the ordinal/prefix-sum/accumulate
    # machinery once over that small frame -- O(buckets + retention +
    # pending), never O(full history), replacing the legacy engine's six
    # ``np.append`` calls per bucket per step.
    #
    # Fit (`feature_frame`, above) and predict (`latest_features`, below)
    # evaluate the IDENTICAL `_pooled_expr`/`_quantile_columns` -- `latest_features`
    # only ever swaps which (seed+tail vs. full-history) aggregate table
    # `feature_frame` runs over, never reimplements a transform's statistic.

    _APPEND_FIELDS = ("agg", "_pending", "_pending_raw", "_next_ord")

    def snapshot(self):
        """Reference copy of the mutated fields.

        Prediction replaces frames wholesale rather than mutating in place, so
        references suffice -- no per-bucket dict walk, unlike the legacy engine.
        """
        return {f: getattr(self, f) for f in self._APPEND_FIELDS}

    def restore(self, snap):
        for k, v in snap.items():
            setattr(self, k, v)

    def _leaf_tfms(self, transforms):
        return [leaf for t in transforms.values() for leaf in _iter_leaf_tfms(t)]

    def _ensure_predict_init(self, transforms):
        """Lazily compute the retention/seed/tail machinery and, the first
        time this runs for the CURRENT model's predict loop, reduce
        ``self.agg`` from the full fit-time table down to the small persisted
        tail. Retention/seeds are pure functions of the fit-time history and
        the (fixed, per pooled key) transform set, so they are computed once
        ever and simply reused by every later model -- ``TimeSeries._backup``
        resets ``self.agg``/``_pending``/``_next_ord`` to their pre-model
        values between models (see ``snapshot``/``restore``), which is
        exactly the ``not self._pending`` signal this checks: a restored (or
        genuinely fresh) state always has empty ``_pending``, and it is only
        empty again mid-model if this is the very first step.
        """
        if self._pending:
            return
        if self._seeds is None:
            self._make_seeds(transforms)
            self._seeds = True
        self._rebuild_tail()

    def _make_seeds(self, transforms):
        """Compute (once) ``_retention``, ``_accum_specs``, the per-bucket
        seed row (``_seed_rows``) and the retained historical suffix
        (``_hist_suffix``/``_hist_suffix_df``) that ``_rebuild_tail`` then
        concatenates with ``_pending``/``_pending_raw`` at every step.

        A bucket's own last historical ordinal is ``last_ord``; the boundary
        ordinal is ``last_ord - retention``. Rows with ``ord >`` that boundary
        are the retained suffix, copied verbatim (their aggregate/accumulate
        columns are already correct, computed over the FULL history at fit
        time -- rerunning the same cumsum/accumulate from the seed forward
        reproduces them identically, see the accumulate-column argument
        below). The row AT the boundary (if any -- a bucket shorter than
        ``retention`` has none) becomes the seed, with its raw ``s``/``c``/
        ``q`` (+ ``time_agg`` suffixes) columns overwritten by their own
        ``E``-prefixed cumulative value, and any accumulate-consumed column
        (``mn``/``mx``/``ewm`` (+suffixes), e.g. for ``ExpandingMin``/EWM)
        overwritten by its already-computed ``A``-prefixed running value --
        so a fresh cumsum/accumulate over ``[seed, suffix, pending]``
        reconstructs the true ABSOLUTE state at every subsequent ordinal, not
        a value merely relative to where the tail happens to start. A bucket
        with no boundary row (its whole history already fits within
        ``retention``, or ``retention is None`` -- see
        ``_pooled_retention``) gets a synthetic empty seed instead (zero
        prefix-sum contribution, null accumulate baseline): nothing precedes
        the retained suffix for it, so the seed must contribute nothing.

        No expression ever actually reads a seed row through a RAW (not
        ``E``/``A``-prefixed) column: every leaf's own retention is an upper
        bound on its raw-column reach (by construction, see
        ``_pooled_retention``), and the query row sits at
        ``retention + 1 + len(pending)`` -- one past the retained suffix --
        so shifting back by at most ``retention`` from there lands at
        position ``>= 1``, never at the seed (position 0). Seed rows are
        therefore safe to give a nonsense raw value where no accumulate
        substitution applies (nothing ever reads it).
        """
        leaves = self._leaf_tfms(transforms)
        retentions = [_pooled_retention(t) for t in leaves]
        self._retention = (
            None
            if any(r is None for r in retentions)
            else (max(retentions) if retentions else 0)
        )
        self._accum_specs = _accumulate_specs(leaves)

        a = nw.from_native(self.agg, eager_only=True)
        backend = nw.get_native_namespace(a)
        keys = self.keys
        n = len(a)
        full_cols = list(a.columns)
        if n == 0:
            self._seed_rows = a.to_native()
            self._hist_suffix = a.to_native()
            on_cols = (keys + [self.time_col]) if keys else [self.time_col]
            self._hist_suffix_df = (
                nw.from_native(self._df, eager_only=True)
                .select(on_cols + [self._target_col])
                .head(0)
                .to_native()
            )
            return

        ords = a.get_column("ord").to_numpy()
        bucket_arr = (
            a.get_column(keys[0]).to_numpy() if keys else np.zeros(n, dtype=np.int64)
        )
        order = np.argsort(bucket_arr, kind="stable")
        sb, so = bucket_arr[order], ords[order]
        uniq_b, start_idx = np.unique(sb, return_index=True)
        end_idx = np.append(start_idx[1:], len(sb))
        last_ord = {int(b): int(so[e - 1]) for b, e in zip(uniq_b, end_idx)}
        if self._retention is None:
            baseline = {b: -1 for b in last_ord}
        else:
            baseline = {b: last_ord[b] - self._retention for b in last_ord}
        baseline_arr = np.array([baseline[int(b)] for b in bucket_arr])

        hist_mask = ords > baseline_arr
        boundary_mask = ords == baseline_arr
        a_tagged = a.with_columns(
            nw.new_series("_hist_mask", hist_mask, dtype=nw.Boolean, backend=backend),
            nw.new_series(
                "_boundary_mask", boundary_mask, dtype=nw.Boolean, backend=backend
            ),
        )
        self._hist_suffix = (
            a_tagged.filter(nw.col("_hist_mask"))
            .drop(["_hist_mask", "_boundary_mask"])
            .select(full_cols)
            .to_native()
        )
        boundary_rows = a_tagged.filter(nw.col("_boundary_mask")).drop(
            ["_hist_mask", "_boundary_mask"]
        )

        subs = []
        for base in ("s", "c", "q"):
            for suffix in [""] + [
                f"__{ta}" for ta in sorted(x for x in self._time_aggs if x is not None)
            ]:
                col = f"{base}{suffix}"
                if col in boundary_rows.columns:
                    subs.append(nw.col(f"E{col}").alias(col))
        for (col, _op, _kw), out in self._accum_specs.items():
            if col in boundary_rows.columns and out in boundary_rows.columns:
                subs.append(nw.col(out).alias(col))
        if subs:
            boundary_rows = boundary_rows.with_columns(subs)

        # A bucket with no boundary row (its whole history already fits
        # within `retention`, or `retention is None`) simply gets NO seed
        # row at all -- `_hist_suffix` already carries its ENTIRE history
        # starting at ordinal 0, so a fresh cumsum/accumulate over
        # `[(no seed), full history, pending]` starts from an implicit empty
        # baseline, exactly like `build_agg_table` does for that bucket. This
        # also sidesteps needing a null placeholder `time_col` (pandas can't
        # hold a null in a plain int64 column, and any real placeholder value
        # risks colliding with an actual observation in the raw quantile
        # join) -- a bucket simply absent from `self._seed_rows` needs none.
        self._seed_rows = boundary_rows.select(full_cols).to_native()

        # Reduced to exactly the columns `quantile_values` reads
        # (`keys + [time_col, target_col]`) -- matches `_pending_raw_frame`'s
        # schema exactly, so `_rebuild_tail`'s vertical concat of the two
        # never hits a width/column mismatch (polars' `concat` requires an
        # exact schema match, unlike pandas' outer-join fallback).
        on_cols = (keys + [self.time_col]) if keys else [self.time_col]
        grid = (
            nw.from_native(self._hist_suffix, eager_only=True).select(on_cols).unique()
        )
        df_nw = nw.from_native(self._df, eager_only=True)
        self._hist_suffix_df = (
            df_nw.join(grid, on=on_cols, how="inner")
            .select(on_cols + [self._target_col])
            .to_native()
        )

    def _pending_agg_frame(self, entry, backend, time_dtype):
        """One synthetic (already-aggregated) row per bucket for one new
        pending timestamp, in ``self.agg``'s bare-column shape (``s``/``c``/
        ``q``/``mn``/``mx`` plus their ``time_agg`` derivations) --
        ``_rebuild_tail`` pads it with placeholder ``E``/``A``-prefixed
        columns and recomputes those fresh over the whole tail.
        """
        ts, s, c, q, mn, mx = entry
        n_b = len(s)
        data: Dict[str, Any] = {}
        if self.keys:
            data[self.keys[0]] = np.arange(n_b, dtype=np.int64)
        data[self.time_col] = np.full(n_b, ts)
        data["s"] = np.asarray(s, dtype=np.float64)
        data["c"] = np.asarray(c, dtype=np.float64)
        data["q"] = np.asarray(q, dtype=np.float64)
        data["mn"] = np.asarray(mn, dtype=np.float64)
        data["mx"] = np.asarray(mx, dtype=np.float64)
        tbl = nw.from_dict(data, backend=backend)
        tbl = tbl.with_columns(
            nw.col(self.time_col).cast(time_dtype),
            nw.lit(0).cast(nw.Int64).alias("ord"),
        )
        tbl = _derive_time_agg_family(tbl, self._time_aggs)
        return tbl

    def _pending_raw_frame(self, bids_arr, y_arr, ts, backend, time_dtype):
        """One raw row per SERIES prediction for one pending timestamp, in
        ``self._df``'s own schema -- feeds the raw quantile store
        (``quantile_values``) the individual (not bucket-aggregated) values a
        window over un-collapsed raw observations needs.
        """
        data: Dict[str, Any] = {}
        if self.keys:
            data[self.keys[0]] = np.asarray(bids_arr, dtype=np.int64)
        data[self.time_col] = np.full(len(y_arr), ts)
        data[self._target_col] = np.asarray(y_arr, dtype=np.float64)
        tbl = nw.from_dict(data, backend=backend)
        tbl = tbl.with_columns(nw.col(self.time_col).cast(time_dtype))
        return tbl.to_native()

    def _rebuild_tail(self):
        """Reconstruct ``self.agg`` as
        ``[seed rows, retained history, *pending]`` and recompute the
        ordinal/prefix-sum/accumulate columns fresh over that (small) frame --
        called once per ``append_predictions`` (extending ``_pending`` by one
        entry) and once, lazily, before the very first ``latest_features``
        call for a model (see ``_ensure_predict_init``).
        """
        seed_nw = nw.from_native(self._seed_rows, eager_only=True)
        backend = nw.get_native_namespace(seed_nw)
        time_dtype = seed_nw.schema[self.time_col]
        full_cols = list(seed_nw.columns)

        parts = [self._seed_rows, self._hist_suffix]
        for entry in self._pending:
            pend = self._pending_agg_frame(entry, backend, time_dtype)
            missing_cols = [c for c in full_cols if c not in pend.columns]
            if missing_cols:
                pend = pend.with_columns([nw.lit(0.0).alias(c) for c in missing_cols])
            parts.append(pend.select(full_cols).to_native())

        combined = nw.from_native(ufp.vertical_concat(parts), eager_only=True)
        combined = _add_ordinals(combined, self.keys, self.time_col)
        # `_quantile_columns` indexes `self.agg` positionally, assuming each
        # bucket's rows occupy a CONTIGUOUS row range (`base = sel.min()`,
        # `values[row_offsets[base]:row_offsets[base+n_ord]]`) -- true of the
        # fit-time table (built bucket-sorted) but NOT of the physical
        # concatenation order above (seed rows for every bucket, then history
        # for every bucket, then one pending row per bucket per step
        # interleaves buckets). Sorting by (bucket, ord) restores bucket
        # contiguity while preserving each bucket's own chronological order
        # (ord was assigned in exactly that order), matching the fit-time
        # invariant every downstream reader of `self.agg` relies on.
        sort_cols = (self.keys + ["ord"]) if self.keys else ["ord"]
        combined = combined.sort(sort_cols)
        combined_native = _add_prefix_sums(
            combined.to_native(), self.keys, self._time_aggs
        )
        for (col, op, kw_items), out in self._accum_specs.items():
            combined_native = apply_accumulate(
                combined_native, self.keys, col, op, out, **dict(kw_items)
            )
        self.agg = combined_native
        self._qvalues = None

        raw_parts = [self._hist_suffix_df]
        for bids_arr, y_arr, ts in self._pending_raw:
            raw_parts.append(
                self._pending_raw_frame(bids_arr, y_arr, ts, backend, time_dtype)
            )
        self._tail_df = (
            ufp.vertical_concat(raw_parts) if len(raw_parts) > 1 else raw_parts[0]
        )

        a = nw.from_native(self.agg, eager_only=True)
        bucket_arr = (
            a.get_column(self.keys[0]).to_numpy()
            if self.keys
            else np.zeros(len(a), dtype=np.int64)
        )
        ords = a.get_column("ord").to_numpy()
        next_ord = {}
        for b in np.unique(bucket_arr):
            next_ord[int(b)] = int(ords[bucket_arr == b].max()) + 1
        self._next_ord = next_ord

    def append_predictions(self, curr_dates, predictions, _n_series):
        """Append one row per bucket for the new timestamp.

        Rows accumulate in ``_pending``/``_pending_raw`` and are concatenated
        once per step by ``_rebuild_tail``, replacing the legacy engine's six
        ``np.append`` calls per bucket per step. Builds a NEW list each call
        (rather than mutating ``self._pending`` in place) so an earlier
        ``snapshot()``'s reference copy is unaffected -- ``.append()`` would
        corrupt it, since both would then point at the identical, now-mutated,
        list object; see ``test_two_models_do_not_leak_state``.

        ``_n_series`` is accepted for call-site symmetry with the legacy
        engine (``PooledState.append_predictions``'s same signature): the
        bucket count is derived from ``series_bucket_id`` instead.
        """
        new_ts = np.asarray(curr_dates)[:1]
        y = np.asarray(predictions, dtype=float)
        bids = self.series_bucket_id
        n_b = int(bids.max()) + 1 if len(bids) else 1
        s = np.bincount(bids, weights=np.nan_to_num(y), minlength=n_b)
        valid = ~np.isnan(y)
        c = np.bincount(bids, weights=valid.astype(float), minlength=n_b)
        q = np.bincount(bids, weights=np.nan_to_num(y) ** 2, minlength=n_b)
        mn = np.full(n_b, np.nan)
        mx = np.full(n_b, np.nan)
        if valid.any():
            np.fmin.at(mn, bids[valid], y[valid])
            np.fmax.at(mx, bids[valid], y[valid])
        self._pending = self._pending + [(new_ts[0], s, c, q, mn, mx)]
        self._pending_raw = self._pending_raw + [(bids.copy(), y.copy(), new_ts[0])]
        self._rebuild_tail()

    def build_query_arrays(self, curr_dates, _n_series):
        """The persisted tail extended by one query row per bucket at the
        next ordinal (raw contribution zero/null -- no observation yet),
        with ordinals/prefix sums/accumulate columns recomputed fresh over
        that extension. Returns the extended native frame with an extra
        boolean ``_is_query`` column marking the newly appended query rows
        (a column, rather than a positional mask, survives the
        bucket-contiguity sort below).

        ``_n_series`` is accepted for call-site symmetry with the legacy
        engine's same-named method: the bucket count is derived from
        ``series_bucket_id`` instead.
        """
        del curr_dates  # every bucket shares one new timestamp; see append_predictions
        n_b = int(self.series_bucket_id.max()) + 1 if len(self.series_bucket_id) else 1
        zeros = np.zeros(n_b)
        nans = np.full(n_b, np.nan)
        # placeholder timestamp: never read (query rows are identified by
        # `_is_query`, not by time_col), but must satisfy the schema/dtype --
        # in particular it must not be null, or casting it to an integer
        # `time_col` dtype (e.g. a plain `ds=0,1,2,...` calendar) raises. A
        # bucket whose entire history fits within the retention window gets a
        # synthetic seed with a NULL time_col (`_empty_seed_rows`), so this is
        # read from `self.agg` (seed + retained history + pending), not from
        # `self._seed_rows` alone, and the first NON-null value is taken.
        agg_nw = nw.from_native(self.agg, eager_only=True)
        backend = nw.get_native_namespace(agg_nw)
        time_dtype = agg_nw.schema[self.time_col]
        ts_col = agg_nw.get_column(self.time_col).drop_nulls()
        placeholder_ts = ts_col.to_numpy()[0] if len(ts_col) else None
        query = self._pending_agg_frame(
            (placeholder_ts, zeros, zeros, zeros, nans, nans), backend, time_dtype
        )
        full_cols = list(nw.from_native(self.agg, eager_only=True).columns)
        missing_cols = [c for c in full_cols if c not in query.columns]
        if missing_cols:
            query = query.with_columns([nw.lit(0.0).alias(c) for c in missing_cols])
        base = nw.from_native(self.agg, eager_only=True).select(full_cols)
        base = base.with_columns(
            nw.new_series(
                "_is_query",
                np.zeros(len(base), dtype=bool),
                dtype=nw.Boolean,
                backend=backend,
            )
        )
        query = query.select(full_cols).with_columns(
            nw.new_series(
                "_is_query", np.ones(n_b, dtype=bool), dtype=nw.Boolean, backend=backend
            )
        )
        combined = nw.from_native(
            ufp.vertical_concat([base.to_native(), query.to_native()]),
            eager_only=True,
        )
        combined = _add_ordinals(combined, self.keys, self.time_col)
        # See `_rebuild_tail`'s identical comment: restore bucket contiguity
        # (base rows for every bucket, then one query row per bucket,
        # interleaves buckets) so `_quantile_columns`'s positional slicing
        # stays correct.
        sort_cols = (self.keys + ["ord"]) if self.keys else ["ord"]
        combined = combined.sort(sort_cols)
        combined_native = _add_prefix_sums(
            combined.to_native(), self.keys, self._time_aggs
        )
        for (col, op, kw_items), out in self._accum_specs.items():
            combined_native = apply_accumulate(
                combined_native, self.keys, col, op, out, **dict(kw_items)
            )
        return combined_native

    def latest_features(self, transforms, n_series):
        """Feature values at the next (not-yet-observed) ordinal, one entry
        per bucket broadcast through ``series_bucket_id`` to ``n_series``.

        Temporarily extends the persisted tail with one query row per bucket
        (``build_query_arrays``), evaluates the SAME ``feature_frame`` used at
        fit time over that extension, reads off the query rows, then restores
        the persisted (non-extended) tail so the next ``append_predictions``
        rebuilds from the correct base.
        """
        self._ensure_predict_init(transforms)
        persisted_agg, persisted_df, persisted_qvalues = (
            self.agg,
            self._df,
            self._qvalues,
        )
        extended_agg = self.build_query_arrays(None, n_series)
        self.agg = extended_agg
        self._df = self._tail_df
        self._qvalues = None
        try:
            feats_native = self.feature_frame(transforms)
        finally:
            self.agg, self._df, self._qvalues = (
                persisted_agg,
                persisted_df,
                persisted_qvalues,
            )
        feats = nw.from_native(feats_native, eager_only=True)
        names = list(transforms)
        bucket_col = self.keys[0] if self.keys else None
        query_rows = feats.filter(nw.col("_is_query"))
        bids = (
            query_rows.get_column(bucket_col).to_numpy()
            if bucket_col
            else np.zeros(len(query_rows), dtype=np.int64)
        )
        out: Dict[str, np.ndarray] = {}
        for name in names:
            vals = query_rows.get_column(name).to_numpy()
            max_bid = max(
                int(bids.max()) if len(bids) else -1,
                int(self.series_bucket_id.max()) if len(self.series_bucket_id) else -1,
            )
            lookup = np.full(max_bid + 1, np.nan)
            lookup[bids] = vals
            out[name] = lookup[self.series_bucket_id]
        return out


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
