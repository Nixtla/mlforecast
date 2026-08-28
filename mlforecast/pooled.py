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

    def feature_frame(self, transforms: Dict[str, Any]):
        """Evaluate every transform's expression in one pass over the table."""
        # Flatten composites (Offset/Combine) so their INNER transforms' own
        # time_agg/accumulate requirements are seen -- the wrapper itself
        # always reports the trivial default. See `_iter_leaf_tfms`.
        leaves = [leaf for t in transforms.values() for leaf in _iter_leaf_tfms(t)]
        self.ensure_time_aggs({getattr(t, "time_agg", None) for t in leaves})
        self.ensure_accumulates(leaves)
        t = nw.from_native(self.agg, eager_only=True)
        exprs = []
        for name, tfm in transforms.items():
            expr = tfm._pooled_expr(_resolve_ctx(tfm, self.keys))
            exprs.append(expr.alias(name))
        return t.with_columns(exprs).to_native()

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
            # attach _bucket_id to the panel rows via the group registry
            left_native = _attach_bucket_id(
                left.select(self.group_cols + [time_col]).to_native(),
                self.groups,
                self.group_cols,
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
